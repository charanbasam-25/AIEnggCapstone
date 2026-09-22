import json
from pathlib import Path

from src.retrieval.bm25_retriever import BM25Retriever, load_chunks as load_bm25_chunks
from src.retrieval.semantic_retriever import SemanticRetriever


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """
    Combine multiple ranked result lists using Reciprocal Rank Fusion.

    RRF score:
        score = sum(1 / (k + rank))

    Rank starts at 1.
    """

    fused_scores = {}
    documents = {}

    for ranked_list in ranked_lists:
        for rank, document in enumerate(ranked_list, start=1):
            document_id = document["id"]

            fused_scores[document_id] = (
                fused_scores.get(document_id, 0.0)
                + 1.0 / (k + rank)
            )

            documents[document_id] = document

    ranked_ids = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True,
    )

    results = []

    for document_id in ranked_ids:
        result = documents[document_id].copy()
        result["rrf_score"] = fused_scores[document_id]
        results.append(result)

    return results


class HybridRetriever:

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks

        print("Initializing BM25 retriever...")
        self.bm25 = BM25Retriever(chunks)

        print("Initializing semantic retriever...")
        self.semantic = SemanticRetriever(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> list[dict]:
        """
        Retrieve using BM25 + semantic retrieval,
        then combine rankings using RRF.
        """

        bm25_results = self.bm25.retrieve(
            query,
            top_k=candidate_k,
        )

        semantic_results = self.semantic.retrieve(
            query,
            top_k=candidate_k,
        )

        fused_results = reciprocal_rank_fusion(
            [bm25_results, semantic_results]
        )

        return fused_results[:top_k]


if __name__ == "__main__":

    chunks = load_bm25_chunks(
        "data/processed/chunks.jsonl"
    )

    retriever = HybridRetriever(chunks)

    query = "What are the Fundamental Rights guaranteed by the Constitution?"

    results = retriever.retrieve(
        query,
        top_k=5,
    )

    print()
    print(f"Query: {query}")
    print()

    for rank, result in enumerate(results, start=1):
        print(f"--- Result {rank} ---")
        print(f"RRF Score: {result['rrf_score']:.6f}")
        print(f"Source: {result['source']}")
        print(f"Page: {result['page']}")
        print()
        print(result["text"][:500])
        print()
