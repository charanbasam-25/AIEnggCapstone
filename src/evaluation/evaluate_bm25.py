import json
from pathlib import Path

from src.retrieval.bm25_retriever import (
    BM25Retriever,
    load_chunks,
)
from src.retrieval.retrieval_config import RETRIEVAL_TOP_K


QUERIES_PATH = "src/evaluation/retrieval_queries_gold.json"
CHUNKS_PATH = "data/processed/chunks.jsonl"


def load_queries(path: str) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def is_relevant(result: dict, query: dict) -> bool:
    return (
        result["source"] == query["expected_source"]
        and result["page"] in query["expected_pages"]
    )


def evaluate_recall_at_k(
    retriever: BM25Retriever,
    queries: list[dict],
    k: int = RETRIEVAL_TOP_K,
) -> float:

    hits = 0

    for query in queries:

        results = retriever.retrieve(
            query["query"],
            top_k=k,
        )

        hit = any(
            is_relevant(result, query)
            for result in results
        )

        print(
            f"{query['query_id']}: "
            f"{'HIT' if hit else 'MISS'}"
        )

        if not hit:
            print(f"  Query: {query['query']}")

            for rank, result in enumerate(results, start=1):
                print(
                    f"  Rank {rank}: "
                    f"{result['source']} "
                    f"page {result['page']} "
                    f"score={result['score']:.2f}"
                )

        if hit:
            hits += 1

    recall = hits / len(queries)

    return recall

if __name__ == "__main__":

    chunks = load_chunks(CHUNKS_PATH)
    queries = load_queries(QUERIES_PATH)

    retriever = BM25Retriever(chunks)

    recall = evaluate_recall_at_k(
        retriever,
        queries,
        k=RETRIEVAL_TOP_K,
    )

    print()
    print(f"Recall@{RETRIEVAL_TOP_K}: {recall:.2%}")