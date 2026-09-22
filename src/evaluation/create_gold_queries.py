import json
from pathlib import Path


CHUNKS_FILE = "data/processed/chunks.jsonl"
OUTPUT_FILE = "src/evaluation/retrieval_queries_gold.json"


# Each query is tied to a specific constitutional provision or section.
# We search for the provision's distinctive text in the actual corpus.
GOLD_QUERIES = [
    {
        "query_id": "q1",
        "query": "What are the Fundamental Rights guaranteed by the Constitution?",
        "anchors": ["PART III", "FUNDAMENTAL RIGHTS"],
        "topic": "Fundamental Rights",
    },
    {
        "query_id": "q2",
        "query": "What are the Directive Principles of State Policy?",
        "anchors": ["DIRECTIVE PRINCIPLES OF STATE POLICY"],
        "topic": "Directive Principles",
    },
    {
        "query_id": "q3",
        "query": "What are the Fundamental Duties of citizens?",
        "anchors": ["FUNDAMENTAL DUTIES"],
        "topic": "Fundamental Duties",
    },
    {
        "query_id": "q4",
        "query": "How is the President of India elected?",
        "anchors": ["Election of President"],
        "topic": "President",
    },
    {
        "query_id": "q5",
        "query": "What is the constitutional procedure for amending the Constitution?",
        "anchors": ["368.", "Power of Parliament to amend the Constitution"],
        "topic": "Constitutional Amendment",
    },
    {
        "query_id": "q6",
        "query": "What are the constitutional provisions relating to the Vice-President?",
        "anchors": ["THE VICE-PRESIDENT"],
        "topic": "Vice-President",
    },
    {
        "query_id": "q7",
        "query": "What are the constitutional provisions relating to the Council of Ministers?",
        "anchors": ["Council of Ministers"],
        "topic": "Council of Ministers",
    },
    {
        "query_id": "q8",
        "query": "What are the constitutional provisions relating to the Prime Minister?",
        "anchors": ["Prime Minister"],
        "topic": "Prime Minister",
    },
    {
        "query_id": "q9",
        "query": "What are the constitutional provisions relating to Parliament?",
        "anchors": ["PARLIAMENT"],
        "topic": "Parliament",
    },
    {
        "query_id": "q10",
        "query": "What are the constitutional provisions relating to the Lok Sabha?",
        "anchors": ["HOUSE OF THE PEOPLE"],
        "topic": "Lok Sabha",
    },
    {
        "query_id": "q11",
        "query": "What are the constitutional provisions relating to the Rajya Sabha?",
        "anchors": ["COUNCIL OF STATES"],
        "topic": "Rajya Sabha",
    },
    {
        "query_id": "q12",
        "query": "What are the constitutional provisions relating to the Supreme Court?",
        "anchors": ["THE SUPREME COURT OF INDIA"],
        "topic": "Supreme Court",
    },
    {
        "query_id": "q13",
        "query": "What are the constitutional provisions relating to High Courts?",
        "anchors": ["HIGH COURTS IN THE STATES"],
        "topic": "High Courts",
    },
    {
        "query_id": "q14",
        "query": "What are the Emergency Provisions in the Constitution?",
        "anchors": ["EMERGENCY PROVISIONS"],
        "topic": "Emergency",
    },
    {
        "query_id": "q15",
        "query": "What are the constitutional provisions relating to the Election Commission?",
        "anchors": ["Election Commission"],
        "topic": "Election Commission",
    },
    {
        "query_id": "q16",
        "query": "What are the constitutional provisions relating to the Finance Commission?",
        "anchors": ["Finance Commission"],
        "topic": "Finance Commission",
    },
    {
        "query_id": "q17",
        "query": "What are the constitutional provisions relating to the Union Public Service Commission?",
        "anchors": ["Union Public Service Commission"],
        "topic": "UPSC",
    },
    {
        "query_id": "q18",
        "query": "What are the constitutional provisions relating to the Comptroller and Auditor-General of India?",
        "anchors": ["Comptroller and Auditor-General"],
        "topic": "CAG",
    },
    {
        "query_id": "q19",
        "query": "How are legislative powers distributed between the Union and the States?",
        "anchors": ["Distribution of Legislative Powers"],
        "topic": "Federalism",
    },
    {
        "query_id": "q20",
        "query": "What are the constitutional provisions relating to the procedure for elections?",
        "anchors": ["Elections to the Parliament"],
        "topic": "Elections",
    },
]


def load_chunks():
    chunks = []

    with Path(CHUNKS_FILE).open("r", encoding="utf-8") as file:
        for line in file:
            chunks.append(json.loads(line))

    return chunks


def find_matching_pages(chunks, anchors):
    pages = set()

    for chunk in chunks:
        if chunk["source"] != "constitution":
            continue

        text = chunk["text"].lower()

        if all(anchor.lower() in text for anchor in anchors):
            pages.add(chunk["page"])

    return sorted(pages)


def main():
    chunks = load_chunks()

    results = []

    print()
    print("Creating gold retrieval evaluation set")
    print("=" * 70)

    for item in GOLD_QUERIES:

        pages = find_matching_pages(
            chunks,
            item["anchors"],
        )

        print()
        print(f"{item['query_id']} - {item['topic']}")
        print(f"Query: {item['query']}")
        print(f"Anchors: {item['anchors']}")
        print(f"Matching Constitution pages: {pages}")

        if not pages:
            print("WARNING: No exact anchor match found.")

        results.append(
            {
                "query_id": item["query_id"],
                "query": item["query"],
                "expected_source": "constitution",
                "expected_pages": pages,
                "topic": item["topic"],
            }
        )

    Path(OUTPUT_FILE).write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print(f"Created: {OUTPUT_FILE}")
    print(f"Queries: {len(results)}")


if __name__ == "__main__":
    main()
