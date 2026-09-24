import json
from pathlib import Path

from src.retrieval.parent_child_retriever import (
    ParentChildRetriever,
    load_chunks,
)


QUERIES_FILE = "src/evaluation/retrieval_queries_gold.json"
CHUNKS_FILE = "data/processed/chunks.jsonl"


def load_queries(file_path: str) -> list[dict]:
    with Path(file_path).open("r", encoding="utf-8") as file:
        return json.load(file)


def is_hit(result: dict, expected: dict) -> bool:
    return (
        result["source"] == expected["expected_source"]
        and result["page"] in expected["expected_pages"]
    )


def main():

    chunks = load_chunks(CHUNKS_FILE)
    queries = load_queries(QUERIES_FILE)

    retriever = ParentChildRetriever(
        chunks,
        candidate_k=20,
    )

    hits = 0

    print("Parent-Child Retrieval Evaluation")
    print()

    for item in queries:

        query_id = item["query_id"]
        query = item["query"]

        results = retriever.retrieve(
            query,
            top_k=5,
        )

        hit = any(
            is_hit(result, item)
            for result in results
        )

        if hit:
            hits += 1
            print(f"{query_id}: HIT")
        else:
            print(f"{query_id}: MISS")
            print(f"  Query: {query}")
            print("  Top results:")

            for rank, result in enumerate(results, start=1):
                print(
                    f"    Rank {rank}: "
                    f"{result['source']} "
                    f"page {result['page']} "
                    f"score={result['reranker_score']:.4f}"
                )

        print()

    recall = hits / len(queries)

    print(f"Hits: {hits}")
    print(f"Queries: {len(queries)}")
    print(f"Recall@5: {recall:.2%}")


if __name__ == "__main__":
    main()