import json
from pathlib import Path


CHUNKS_FILE = "data/processed/chunks.jsonl"
OUTPUT_FILE = "src/evaluation/retrieval_queries_expanded.json"


def load_chunks():
    chunks = []

    with Path(CHUNKS_FILE).open("r", encoding="utf-8") as file:
        for line in file:
            chunks.append(json.loads(line))

    return chunks


def find_pages(chunks, phrases):
    pages = set()

    for chunk in chunks:
        if chunk["source"] != "constitution":
            continue

        text = chunk["text"].lower()

        if all(phrase.lower() in text for phrase in phrases):
            pages.add(chunk["page"])

    return sorted(pages)


def main():

    chunks = load_chunks()

    evaluation_topics = [
        {
            "query_id": "q1",
            "query": "What are the Fundamental Rights guaranteed by the Constitution?",
            "phrases": ["fundamental rights"],
        },
        {
            "query_id": "q2",
            "query": "What are the Directive Principles of State Policy?",
            "phrases": ["directive principles"],
        },
        {
            "query_id": "q3",
            "query": "What are the Fundamental Duties of citizens?",
            "phrases": ["fundamental duties"],
        },
        {
            "query_id": "q4",
            "query": "How is the President of India elected?",
            "phrases": ["president", "elected"],
        },
        {
            "query_id": "q5",
            "query": "What is the procedure for amending the Constitution?",
            "phrases": ["amendment", "constitution"],
        },
        {
            "query_id": "q6",
            "query": "What are the powers and functions of the President of India?",
            "phrases": ["president"],
        },
        {
            "query_id": "q7",
            "query": "What is the constitutional position of the Vice-President of India?",
            "phrases": ["vice-president"],
        },
        {
            "query_id": "q8",
            "query": "What are the constitutional provisions relating to the Council of Ministers?",
            "phrases": ["council of ministers"],
        },
        {
            "query_id": "q9",
            "query": "What are the powers and functions of the Prime Minister?",
            "phrases": ["prime minister"],
        },
        {
            "query_id": "q10",
            "query": "What are the constitutional provisions relating to Parliament?",
            "phrases": ["parliament"],
        },
        {
            "query_id": "q11",
            "query": "What are the constitutional provisions relating to the Lok Sabha?",
            "phrases": ["lok sabha"],
        },
        {
            "query_id": "q12",
            "query": "What are the constitutional provisions relating to the Rajya Sabha?",
            "phrases": ["rajya sabha"],
        },
        {
            "query_id": "q13",
            "query": "What are the constitutional provisions relating to the Supreme Court?",
            "phrases": ["supreme court"],
        },
        {
            "query_id": "q14",
            "query": "What are the constitutional provisions relating to High Courts?",
            "phrases": ["high courts"],
        },
        {
            "query_id": "q15",
            "query": "What are the Emergency Provisions in the Constitution?",
            "phrases": ["emergency provisions"],
        },
        {
            "query_id": "q16",
            "query": "What are the constitutional provisions relating to the Election Commission?",
            "phrases": ["election commission"],
        },
        {
            "query_id": "q17",
            "query": "What are the constitutional provisions relating to the Finance Commission?",
            "phrases": ["finance commission"],
        },
        {
            "query_id": "q18",
            "query": "What are the constitutional provisions relating to the Union Public Service Commission?",
            "phrases": ["union public service commission"],
        },
        {
            "query_id": "q19",
            "query": "What are the constitutional provisions relating to the Comptroller and Auditor-General of India?",
            "phrases": ["comptroller and auditor-general"],
        },
        {
            "query_id": "q20",
            "query": "What are the constitutional provisions relating to the distribution of legislative powers between the Union and the States?",
            "phrases": ["distribution of legislative powers"],
        },
    ]

    results = []

    print()
    print("Finding expected Constitution pages...")
    print("=" * 70)

    for item in evaluation_topics:

        pages = find_pages(
            chunks,
            item["phrases"],
        )

        print()
        print(item["query_id"])
        print(item["query"])
        print("Expected pages:", pages)

        if not pages:
            print("WARNING: No matching Constitution page found.")

        results.append(
            {
                "query_id": item["query_id"],
                "query": item["query"],
                "expected_source": "constitution",
                "expected_pages": pages,
            }
        )

    valid_results = [
        result
        for result in results
        if result["expected_pages"]
    ]

    Path(OUTPUT_FILE).write_text(
        json.dumps(valid_results, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print(f"Created: {OUTPUT_FILE}")
    print(f"Queries with ground-truth pages: {len(valid_results)}")
    print(f"Total requested queries: {len(results)}")


if __name__ == "__main__":
    main()
