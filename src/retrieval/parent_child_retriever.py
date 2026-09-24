import json
from collections import defaultdict
from pathlib import Path

from sentence_transformers import CrossEncoder

from src.retrieval.semantic_retriever import SemanticRetriever


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_chunks(file_path: str) -> list[dict]:
    chunks = []

    with Path(file_path).open("r", encoding="utf-8") as file:
        for line in file:
            chunks.append(json.loads(line))

    return chunks


class ParentChildRetriever:

    def __init__(
        self,
        chunks: list[dict],
        candidate_k: int = 20,
    ):
        self.chunks = chunks
        self.candidate_k = candidate_k

        # Child-level semantic retrieval
        self.semantic_retriever = SemanticRetriever(chunks)

        # Cross-encoder for parent-page reranking
        self.reranker = CrossEncoder(RERANKER_MODEL)

        # Build parent pages from child chunks.
        self.parents = self._build_parent_pages()

    def _build_parent_pages(self) -> dict[tuple[str, int], dict]:
        """
        Build parent documents from child chunks.

        Parent identity:
            (source, page)

        Each page contains all chunks belonging to that page.
        """

        grouped = defaultdict(list)

        for chunk in self.chunks:
            parent_key = (
                chunk["source"],
                chunk["page"],
            )

            grouped[parent_key].append(chunk)

        parents = {}

        for parent_key, child_chunks in grouped.items():

            child_chunks = sorted(
                child_chunks,
                key=lambda chunk: chunk["chunk_index"],
            )

            parent_text = "\n\n".join(
                chunk["text"]
                for chunk in child_chunks
            )

            parents[parent_key] = {
                "source": parent_key[0],
                "page": parent_key[1],
                "text": parent_text,
                "child_count": len(child_chunks),
            }

        return parents

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:

        # -------------------------------------------------
        # Stage 1: retrieve child chunks semantically
        # -------------------------------------------------

        child_results = self.semantic_retriever.retrieve(
            query,
            top_k=self.candidate_k,
        )

        # -------------------------------------------------
        # Stage 2: map children to unique parent pages
        # -------------------------------------------------

        candidate_parent_keys = []

        for child in child_results:

            parent_key = (
                child["source"],
                child["page"],
            )

            if parent_key not in candidate_parent_keys:
                candidate_parent_keys.append(parent_key)

        parent_candidates = [
            self.parents[key]
            for key in candidate_parent_keys
        ]

        if not parent_candidates:
            return []

        # -------------------------------------------------
        # Stage 3: rerank parent pages
        # -------------------------------------------------

        pairs = [
            [query, parent["text"]]
            for parent in parent_candidates
        ]

        scores = self.reranker.predict(pairs)

        results = []

        for parent, score in zip(
            parent_candidates,
            scores,
        ):
            result = parent.copy()
            result["reranker_score"] = float(score)
            results.append(result)

        # -------------------------------------------------
        # Stage 4: return top parent pages
        # -------------------------------------------------

        results.sort(
            key=lambda result: result["reranker_score"],
            reverse=True,
        )

        return results[:top_k]


if __name__ == "__main__":

    chunks = load_chunks(
        "data/processed/chunks.jsonl"
    )

    retriever = ParentChildRetriever(
        chunks,
        candidate_k=20,
    )

    query = "How is the President of India elected?"

    results = retriever.retrieve(
        query,
        top_k=5,
    )

    print(f"Query: {query}")
    print()

    for rank, result in enumerate(results, start=1):

        print(f"--- Result {rank} ---")
        print(
            f"Reranker score: "
            f"{result['reranker_score']:.4f}"
        )
        print(f"Source: {result['source']}")
        print(f"Page: {result['page']}")
        print(f"Child chunks: {result['child_count']}")
        print()

        print(result["text"][:1000])
        print()