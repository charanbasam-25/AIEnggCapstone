import json
from pathlib import Path
from time import perf_counter

from src.orchestration.telemetry import record_retrieval

from src.retrieval.retrieval_config import (
    RERANK_CANDIDATE_K,
    RETRIEVAL_TOP_K,
)
from src.tls_trust import enable_os_trust_store

# Same reason as in semantic_retriever.py: this has to happen before the
# cross-encoder weights are fetched. Both are guarded because either
# module can be the first one imported.
enable_os_trust_store()

from sentence_transformers import CrossEncoder  # noqa: E402

from src.retrieval.semantic_retriever import (  # noqa: E402
    MODEL_NAME as EMBEDDING_MODEL_NAME, SemanticRetriever,
)


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class SemanticReranker:

    def __init__(
        self,
        chunks: list[dict],
        candidate_k: int = RERANK_CANDIDATE_K,
    ):
        self.candidate_k = candidate_k

        # First-stage semantic retrieval
        self.retriever = SemanticRetriever(chunks)

        # Second-stage cross-encoder reranking
        self.reranker = CrossEncoder(MODEL_NAME)

    def retrieve(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> list[dict]:
        """
        Retrieve candidates using semantic search,
        then rerank them using a cross-encoder.
        """

        started = perf_counter()
        candidates, reranked = [], []
        semantic_seconds, rerank_seconds, error_type = None, None, None
        try:
            semantic_started = perf_counter()
            candidates = self.retriever.retrieve(query, top_k=self.candidate_k)
            semantic_seconds = perf_counter() - semantic_started
            if not candidates:
                return []
            rerank_started = perf_counter()
            pairs = [[query, candidate["text"]] for candidate in candidates]
            scores = self.reranker.predict(pairs)
            for candidate, score in zip(candidates, scores):
                result = candidate.copy()
                result["reranker_score"] = float(score)
                reranked.append(result)
            reranked.sort(key=lambda result: result["reranker_score"], reverse=True)
            rerank_seconds = perf_counter() - rerank_started
            return reranked[:top_k]
        except Exception as exception:
            error_type = type(exception).__name__
            raise
        finally:
            record_retrieval(
                query=query, candidates=candidates, reranked=reranked, top_k=top_k,
                candidate_k=self.candidate_k, semantic_seconds=semantic_seconds,
                rerank_seconds=rerank_seconds, elapsed_seconds=perf_counter() - started,
                embedding_model=EMBEDDING_MODEL_NAME, reranker_model=MODEL_NAME,
                error_type=error_type,
            )


def load_chunks(file_path: str) -> list[dict]:
    chunks = []

    with Path(file_path).open("r", encoding="utf-8") as file:
        for line in file:
            chunks.append(json.loads(line))

    return chunks


if __name__ == "__main__":

    chunks = load_chunks(
        "data/processed/chunks.jsonl"
    )

    reranker = SemanticReranker(chunks)

    query = "How is the President of India elected?"

    results = reranker.retrieve(query)

    print(f"Query: {query}")
    print()

    for rank, result in enumerate(results, start=1):
        print(f"--- Result {rank} ---")
        print(f"Reranker score: {result['reranker_score']:.4f}")
        print(f"Source: {result['source']}")
        print(f"Page: {result['page']}")
        print()
        print(result["text"][:500])
        print()
