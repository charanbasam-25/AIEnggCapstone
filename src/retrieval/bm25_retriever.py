import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Convert text into tokens for BM25."""

    return re.findall(r"\b\w+\b", text.lower())


def load_chunks(file_path: str) -> list[dict]:
    """Load chunk records from JSONL."""

    chunks = []

    with Path(file_path).open("r", encoding="utf-8") as file:
        for line in file:
            chunks.append(json.loads(line))

    return chunks


class BM25Retriever:

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks

        self.tokenized_corpus = [
            tokenize(chunk["text"])
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Retrieve the most relevant chunks."""

        query_tokens = tokenize(query)

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        results = []

        for index in ranked_indices:
            result = self.chunks[index].copy()
            result["score"] = float(scores[index])
            results.append(result)

        return results


if __name__ == "__main__":

    chunks = load_chunks(
        "data/processed/chunks.jsonl"
    )

    retriever = BM25Retriever(chunks)

    query = "What are the fundamental rights guaranteed by the Constitution?"

    results = retriever.retrieve(
        query,
        top_k=5,
    )

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