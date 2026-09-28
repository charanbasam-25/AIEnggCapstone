"""
Paired selective-prediction comparison between System A and System B.

Relationship to system_b_metrics.py
-----------------------------------

These two modules look like duplicates and are not. system_b_metrics.py
owns the arm table: one row per architecture, each scored over all 13
questions. This module owns the one comparison that table cannot make,
the paired one restricted to the questions System B actually attempted.

That subset matters because System B abstains. Comparing 13-question
accuracy against a baseline that always answers compares two systems on
two different question sets, and the abstaining one is charged for
questions it never claimed to answer. `paired_comparison` below is the
only place in the project where the two systems are scored on identical
questions.

Everything else here (`score`, `classify_abstention`,
`abstention_breakdown`, `statement_verdict_counts`) is superseded by
`selective_metrics`, `abstention_taxonomy` and `verdict_distributions` in
system_b_metrics.py, which handle more arms and report an error rate as
well. Prefer those; this module is kept for the paired comparison.

Note also that `classify_abstention` reads `complete_evidence` from each
record's claim_mapping. That field is now emitted by a compatibility line
in evaluate_verified_pyqs.py rather than computed natively, so this
module keeps working across the mapping refactor.

System B may decline to answer. When no option can be matched from the
retrieved evidence it returns predicted_answer = None.

Scoring an abstention as a wrong answer conflates two different
behaviours:

    "the system committed to a wrong option"
    "the system declined to commit"

A verification pipeline is supposed to do the second thing. Charging it
as the first makes the headline accuracy unusable for answering the
project's research question.

This module therefore reports:

    coverage                 answered / total
    precision_when_answered  correct / answered
    accuracy_overall         correct / total      (abstention = wrong)

and a paired comparison restricted to the questions System B actually
attempted, which is the only subset where the two systems are directly
comparable.
"""

import json
from pathlib import Path


VERIFIED_PATH = "data/evaluation/verified_pyq_results.json"
VANILLA_PATH = "data/evaluation/vanilla_rag_scored.json"
OUTPUT_PATH = "data/evaluation/selective_metrics.json"


def load_verified(file_path: str) -> dict[int, dict]:
    """Load System B results keyed by question number."""

    records = json.loads(
        Path(file_path).read_text(encoding="utf-8")
    )

    return {
        record["q_number"]: {
            "official_answer": record["official_answer"],
            "predicted_answer": record["predicted_answer"],
            "claim_mapping": record.get("claim_mapping", {}),
        }
        for record in records
    }


def load_vanilla(file_path: str) -> dict[int, dict]:
    """Load System A results keyed by question number."""

    payload = json.loads(
        Path(file_path).read_text(encoding="utf-8")
    )

    return {
        record["q_number"]: {
            "official_answer": record["official_answer"],
            "predicted_answer": record["predicted_answer"],
        }
        for record in payload["results"]
    }


def score(records: dict[int, dict]) -> dict:
    """
    Compute selective-prediction metrics for one system.

    An answer is counted as correct only when the system committed to an
    option and that option matches the official UPSC answer key.
    """

    total = len(records)

    answered = [
        q_number
        for q_number, record in records.items()
        if record["predicted_answer"] is not None
    ]

    abstained = [
        q_number
        for q_number, record in records.items()
        if record["predicted_answer"] is None
    ]

    correct = [
        q_number
        for q_number in answered
        if records[q_number]["predicted_answer"]
        == records[q_number]["official_answer"]
    ]

    return {
        "total": total,
        "answered": len(answered),
        "abstained": len(abstained),
        "correct": len(correct),
        "coverage": len(answered) / total if total else None,
        "precision_when_answered": (
            len(correct) / len(answered) if answered else None
        ),
        "accuracy_overall": len(correct) / total if total else None,
        "answered_questions": sorted(answered),
        "abstained_questions": sorted(abstained),
        "correct_questions": sorted(correct),
    }


def classify_abstention(claim_mapping: dict) -> str:
    """
    Explain why System B declined to answer.

    Two distinct causes are possible and they call for different fixes:

    incomplete_evidence
        At least one statement came back INSUFFICIENT, so the mapping
        layer refused to evaluate any option. This is a retrieval or
        claim-construction problem.

    no_option_matched
        Every statement resolved, but no option corresponded to the
        resulting set of statements. This is a mapping-logic problem.
    """

    if not claim_mapping:
        return "unknown"

    if not claim_mapping.get("complete_evidence", False):
        return "incomplete_evidence"

    if not claim_mapping.get("matching_options"):
        return "no_option_matched"

    return "answered"


def abstention_breakdown(records: dict[int, dict]) -> dict:
    """Group System B's abstentions by cause."""

    breakdown: dict[str, list[int]] = {}

    for q_number, record in sorted(records.items()):

        if record["predicted_answer"] is not None:
            continue

        cause = classify_abstention(record["claim_mapping"])

        breakdown.setdefault(cause, []).append(q_number)

    return breakdown


def statement_verdict_counts(records: dict[int, dict]) -> dict[str, int]:
    """Count SUPPORTED / CONTRADICTED / INSUFFICIENT across statements."""

    counts: dict[str, int] = {}

    for record in records.values():

        statuses = record["claim_mapping"].get(
            "statement_statuses", {}
        )

        for verdict in statuses.values():
            counts[verdict] = counts.get(verdict, 0) + 1

    return counts


def paired_comparison(
    verified: dict[int, dict],
    vanilla: dict[int, dict],
) -> dict:
    """
    Compare both systems on the questions System B attempted.

    Overall accuracy across all questions is not a fair comparison while
    System B abstains, because the two systems are answering different
    question sets. This restricts the comparison to the shared subset.
    """

    attempted = sorted(
        q_number
        for q_number, record in verified.items()
        if record["predicted_answer"] is not None
        and q_number in vanilla
    )

    def correct_on(records: dict[int, dict]) -> list[int]:
        return [
            q_number
            for q_number in attempted
            if records[q_number]["predicted_answer"]
            == records[q_number]["official_answer"]
        ]

    verified_correct = correct_on(verified)
    vanilla_correct = correct_on(vanilla)

    return {
        "subset": attempted,
        "subset_size": len(attempted),
        "system_b_correct": len(verified_correct),
        "system_a_correct": len(vanilla_correct),
        "system_b_accuracy": (
            len(verified_correct) / len(attempted)
            if attempted
            else None
        ),
        "system_a_accuracy": (
            len(vanilla_correct) / len(attempted)
            if attempted
            else None
        ),
        "system_b_correct_questions": verified_correct,
        "system_a_correct_questions": vanilla_correct,
    }


def percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def build_report(
    verified: dict[int, dict],
    vanilla: dict[int, dict],
) -> dict:

    return {
        "dataset": "UPSC 2025 Prelims Polity, Q54-Q66",
        "system_a_vanilla_rag": score(vanilla),
        "system_b_verified": score(verified),
        "system_b_abstention_causes": abstention_breakdown(verified),
        "system_b_statement_verdicts": statement_verdict_counts(
            verified
        ),
        "paired_comparison_on_attempted": paired_comparison(
            verified,
            vanilla,
        ),
    }


def print_report(report: dict) -> None:

    system_a = report["system_a_vanilla_rag"]
    system_b = report["system_b_verified"]

    print("=== SELECTIVE PREDICTION COMPARISON ===")
    print(f"Dataset: {report['dataset']}")
    print()

    header = f"{'metric':<26}{'System A':>12}{'System B':>12}"
    print(header)
    print("-" * len(header))

    rows = [
        ("questions", "total"),
        ("answered", "answered"),
        ("abstained", "abstained"),
        ("correct", "correct"),
    ]

    for label, key in rows:
        print(f"{label:<26}{system_a[key]:>12}{system_b[key]:>12}")

    rate_rows = [
        ("coverage", "coverage"),
        ("precision when answered", "precision_when_answered"),
        ("accuracy (abstain=wrong)", "accuracy_overall"),
    ]

    for label, key in rate_rows:
        print(
            f"{label:<26}"
            f"{percent(system_a[key]):>12}"
            f"{percent(system_b[key]):>12}"
        )

    print()
    print("=== WHY SYSTEM B ABSTAINED ===")

    for cause, questions in sorted(
        report["system_b_abstention_causes"].items()
    ):
        listed = ", ".join(f"Q{q}" for q in questions)
        print(f"{cause:<22}{len(questions):>3}  {listed}")

    print()
    print("=== STATEMENT VERDICTS (System B) ===")

    verdicts = report["system_b_statement_verdicts"]
    verdict_total = sum(verdicts.values())

    for verdict, count in sorted(
        verdicts.items(),
        key=lambda item: -item[1],
    ):
        share = count / verdict_total if verdict_total else 0
        print(f"{verdict:<16}{count:>4}  {share:.1%}")

    print()
    print("=== PAIRED COMPARISON ON ATTEMPTED SUBSET ===")

    paired = report["paired_comparison_on_attempted"]
    subset = ", ".join(f"Q{q}" for q in paired["subset"])

    print(f"Subset ({paired['subset_size']} questions): {subset}")
    print(
        f"System A correct: {paired['system_a_correct']}"
        f"/{paired['subset_size']} "
        f"({percent(paired['system_a_accuracy'])})"
    )
    print(
        f"System B correct: {paired['system_b_correct']}"
        f"/{paired['subset_size']} "
        f"({percent(paired['system_b_accuracy'])})"
    )


def main() -> None:

    verified = load_verified(VERIFIED_PATH)
    vanilla = load_vanilla(VANILLA_PATH)

    report = build_report(verified, vanilla)

    print_report(report)

    Path(OUTPUT_PATH).write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print(f"Saved metrics to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
