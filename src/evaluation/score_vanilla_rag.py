import json
import re
from pathlib import Path


INPUT_PATH = "data/evaluation/vanilla_rag_results.json"
OUTPUT_PATH = "data/evaluation/vanilla_rag_scored.json"


def main() -> None:
    results = json.loads(
        Path(INPUT_PATH).read_text(encoding="utf-8")
    )

    scored = []

    for result in results:
        predicted = result.get("selected_option")

        official = result["official_answer"]

        if predicted is None:
            status = "NO_ANSWER"
        elif predicted == official:
            status = "CORRECT"
        else:
            status = "INCORRECT"

        scored.append(
            {
                "q_number": result["q_number"],
                "official_answer": official,
                "predicted_answer": predicted,
                "status": status,
                "rag_answer": result["rag_answer"],
                "sources": result["sources"],
            }
        )

    correct = sum(
        item["status"] == "CORRECT"
        for item in scored
    )

    incorrect = sum(
        item["status"] == "INCORRECT"
        for item in scored
    )

    no_answer = sum(
        item["status"] == "NO_ANSWER"
        for item in scored
    )

    total = len(scored)

    accuracy = correct / total if total else 0.0

    output = {
        "system": "Vanilla RAG",
        "dataset": "UPSC 2025 Polity PYQs",
        "total_questions": total,
        "correct": correct,
        "incorrect": incorrect,
        "no_answer": no_answer,
        "accuracy": accuracy,
        "results": scored,
    }

    Path(OUTPUT_PATH).write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== VANILLA RAG EVALUATION ===")
    print(f"Total:      {total}")
    print(f"Correct:    {correct}")
    print(f"Incorrect:  {incorrect}")
    print(f"No answer:  {no_answer}")
    print(f"Accuracy:   {accuracy:.2%}")

    print("\n=== QUESTION RESULTS ===")

    for item in scored:
        print(
            f"Q{item['q_number']}: "
            f"official={item['official_answer']} "
            f"predicted={item['predicted_answer']} "
            f"status={item['status']}"
        )

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
