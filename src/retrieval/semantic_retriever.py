import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_chunks(file_path: str) -> list[dict]:
    """Load chunk records from JSONL."""
    chunks = []

    with Path(file_path).open("r", encoding="utf-8") as file:
        for line in file:
            chunks.append(json.loads(line))

    return chunks


class SemanticRetriever:

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks

        print(f"Loading embedding model: {MODEL_NAME}")
        self.model = SentenceTransformer(MODEL_NAME)

        print(f"Embedding {len(chunks)} chunks...")
        self.embeddings = self.model.encode(
            [chunk["text"] for chunk in chunks],
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve the most semantically similar chunks."""

        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )

        scores = np.dot(self.embeddings, query_embedding)

        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in ranked_indices:
            result = self.chunks[index].copy()
            result["score"] = float(scores[index])
            results.append(result)

        return results


if __name__ == "__main__":

    chunks = load_chunks("data/processed/chunks.jsonl")

    retriever = SemanticRetriever(chunks)

    query = "What are the Fundamental Rights guaranteed by the Constitution?"

    results = retriever.retrieve(query, top_k=5)

    print()
    print(f"Query: {query}")
    print()

    for rank, result in enumerate(results, start=1):
        print(f"--- Result {rank} ---")
        print(f"Score: {result['score']:.4f}")
        print(f"Source: {result['source']}")
        print(f"Page: {result['page']}")
        print()
        print(result["text"][:500])
        print()
