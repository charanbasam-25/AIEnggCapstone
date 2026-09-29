import json
from pathlib import Path

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

from src.retrieval.semantic_retriever import SemanticRetriever  # noqa: E402


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

        candidates = self.retriever.retrieve(
            query,
            top_k=self.candidate_k,
        )

        pairs = [
            [query, candidate["text"]]
            for candidate in candidates
        ]

        scores = self.reranker.predict(pairs)

        reranked = []

        for candidate, score in zip(candidates, scores):
            result = candidate.copy()
            result["reranker_score"] = float(score)
            reranked.append(result)

        reranked.sort(
            key=lambda result: result["reranker_score"],
            reverse=True,
        )

        return reranked[:top_k]


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
        