import json

from src.retrieval.semantic_retriever import (
    SemanticRetriever,
    load_chunks,
)


def load_queries(file_path: str) -> list[dict]:
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def is_hit(result: dict, query: dict) -> bool:
    return (
        result["source"] == query["expected_source"]
        and result["page"] in query["expected_pages"]
    )


def main():
    chunks = load_chunks("data/processed/chunks.jsonl")
    queries = load_queries("src/evaluation/retrieval_queries_gold.json")

    retriever = SemanticRetriever(chunks)

    top_k = 5
    hits = 0

    print()
    print("Semantic Retrieval Evaluation")
    print("=" * 60)

    for query in queries:
        results = retriever.retrieve(
            query["query"],
            top_k=top_k,
        )

        hit = any(
            is_hit(result, query)
            for result in results
        )

        if hit:
            hits += 1
            print(f"{query['query_id']}: HIT")
        else:
            print(f"{query['query_id']}: MISS")

            print("Top results:")

            for rank, result in enumerate(results, start=1):
                print(
                    f"  {rank}. "
                    f"{result['source']} "
                    f"page {result['page']} "
                    f"score={result['score']:.4f}"
                )

        print()

    recall = hits / len(queries)

    print("=" * 60)
    print(f"Hits: {hits}")
    print(f"Queries: {len(queries)}")
    print(f"Recall@{top_k}: {recall:.2%}")


if __name__ == "__main__":
    main()
