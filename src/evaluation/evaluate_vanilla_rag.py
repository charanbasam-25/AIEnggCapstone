import json
from pathlib import Path

from src.rag.vanilla_rag import VanillaRAG
from src.retrieval.semantic_reranker import load_chunks


DATASET_PATH = "data/evaluation/pyq_2025_polity.json"
CHUNKS_PATH = "data/processed/chunks.jsonl"
OUTPUT_PATH = "data/evaluation/vanilla_rag_results.json"


def build_mcq_prompt(question: dict) -> str:
    options = question["options"]

    return f"""
Answer the following UPSC Indian Polity multiple-choice question.

Question:
{question["question_text"]}

Options:
A. {options["A"]}
B. {options["B"]}
C. {options["C"]}
D. {options["D"]}

State the correct option clearly as A, B, C, or D, and explain why
using the available source evidence.
""".strip()


def main() -> None:
    dataset = json.loads(
        Path(DATASET_PATH).read_text(encoding="utf-8")
    )

    chunks = load_chunks(CHUNKS_PATH)

    rag = VanillaRAG(chunks)

    results = []

    for index, question in enumerate(dataset["questions"], start=1):

        print(
            f"Evaluating {index}/{len(dataset['questions'])}: "
            f"Q{question['q_number']}"
        )

        query = build_mcq_prompt(question)

        result = rag.answer(query)

        results.append(
            {
                "q_number": question["q_number"],
                "question": question["question_text"],
                "options": question["options"],
                "official_answer": question["official_answer"],
                "selected_option": result["selected_option"],
                "rag_answer": result["answer"],
                "sources": result["sources"],
            }
        )

    Path(OUTPUT_PATH).write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(f"Saved results to: {OUTPUT_PATH}")
    print(f"Evaluated questions: {len(results)}")


if __name__ == "__main__":
    main()
