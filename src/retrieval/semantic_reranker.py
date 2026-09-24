import json
from pathlib import Path

from sentence_transformers import CrossEncoder

from src.retrieval.semantic_retriever import SemanticRetriever


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class SemanticReranker:

    def __init__(self, chunks: list[dict], candidate_k: int = 20):
        self.candidate_k = candidate_k

        # First-stage semantic retrieval
        self.retriever = SemanticRetriever(chunks)

        # Second-stage cross-encoder reranking
        self.reranker = CrossEncoder(MODEL_NAME)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
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

    reranker = SemanticReranker(
        chunks,
        candidate_k=20,
    )

    query = "How is the President of India elected?"

    results = reranker.retrieve(
        query,
        top_k=5,
    )

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
        