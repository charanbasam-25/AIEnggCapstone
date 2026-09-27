"""
Replay saved statement verdicts through the corrected answer mapping.

This does not call an LLM and does not touch the retriever. It reuses
the fact-verification verdicts already stored in
verified_pyq_results.json, so it isolates the effect of the mapping
logic and the abstention policy from any change in verification.

Because the verdicts are held fixed, any difference in accuracy reported
here is attributable to the mapping layer alone.

Run:

    python -m src.evaluation.replay_answer_mapping
"""

import json
from pathlib import Path

from src.evaluation.answer_mapping import (
    CLOSED_WORLD,
    ELIMINATION,
    STRICT,
    count_option_survives,
    explicit_option_survives,
    map_answer,
    parse_count_option,
    parse_statement_set,
)


RESULTS_PATH = "data/evaluation/verified_pyq_results.json"
OUTPUT_PATH = "data/evaluation/answer_mapping_replay.json"


def check_parsing_fixes() -> None:
    """
    Guard the two parsing defects this module was written to fix.

    These run on every invocation so a regression surfaces immediately
    rather than silently reintroducing an abstention.
    """

    # Defect 2: "None" previously parsed as neither a set nor a count.
    assert parse_count_option("None") == 0, "None should count as zero"

    # Defect 1: negation must be reported so the caller can use it.
    statements, negated = parse_statement_set("Neither I nor II")
    assert statements == {"I", "II"}, statements
    assert negated is True, "Neither ... nor ... must be negated"

    statements, negated = parse_statement_set("I and III only")
    assert statements == {"I", "III"}, statements
    assert negated is False, "plain option must not be negated"

    assert parse_count_option("All the three") == 3
    assert parse_statement_set("Only two") is None, (
        "a pure count option names no statements"
    )

    check_elimination_logic()


def check_elimination_logic() -> None:
    """
    Guard the elimination policy's soundness properties.

    Two things must hold for it to be usable. It must never be swayed by
    an unresolved statement, since that would smuggle in the closed-world
    assumption it exists to avoid; and with nothing unresolved it must
    behave exactly like STRICT, so it cannot be a regression.
    """

    # An unresolved statement must not eliminate an option that includes
    # it, nor one that omits it.
    assert explicit_option_survives({"I"}, False, {"I"}, set())
    assert explicit_option_survives({"I", "II"}, False, {"I"}, set())

    # A statement established as being in the target set must appear.
    assert not explicit_option_survives({"II"}, False, {"I"}, set())

    # A statement established as being outside it must not appear.
    assert not explicit_option_survives(
        {"I", "II"}, False, {"I"}, {"II"}
    )

    # "Neither I nor II" survives only while nothing is established.
    assert explicit_option_survives(set(), True, set(), {"I", "II"})
    assert not explicit_option_survives(set(), True, {"I"}, set())

    # A count must lie between the established count and that count plus
    # the number of unresolved statements.
    assert count_option_survives(1, {"I"}, set())
    assert count_option_survives(2, {"I"}, {"II"})
    assert not count_option_survives(0, {"I"}, {"II"})
    assert not count_option_survives(3, {"I"}, {"II"})

    # With nothing unresolved, elimination must agree with STRICT.
    options = {
        "a": "I only",
        "b": "II only",
        "c": "Both I and II",
        "d": "Neither I nor II",
    }

    for statuses in (
        {"I": "SUPPORTED", "II": "CONTRADICTED"},
        {"I": "CONTRADICTED", "II": "SUPPORTED"},
        {"I": "SUPPORTED", "II": "SUPPORTED"},
        {"I": "CONTRADICTED", "II": "CONTRADICTED"},
    ):

        strict_answer, _ = map_answer(
            statuses, options, "Which are correct?", STRICT
        )
        elimination_answer, _ = map_answer(
            statuses, options, "Which are correct?", ELIMINATION
        )

        assert strict_answer == elimination_answer, (
            statuses,
            strict_answer,
            elimination_answer,
        )


def replay(records: list[dict], policy: str) -> dict:
    """Score every saved question under one abstention policy."""

    rows = []
    answered = 0
    correct = 0

    for record in records:

        statement_statuses = record["claim_mapping"][
            "statement_statuses"
        ]

        predicted, debug = map_answer(
            statement_statuses=statement_statuses,
            options=record["options"],
            question_text=record["question"],
            policy=policy,
        )

        official = record["official_answer"]
        is_correct = predicted is not None and predicted == official

        if predicted is not None:
            answered += 1

        if is_correct:
            correct += 1

        rows.append(
            {
                "q_number": record["q_number"],
                "official_answer": official,
                "predicted_answer": predicted,
                "status": (
                    "CORRECT"
                    if is_correct
                    else "ABSTAINED"
                    if predicted is None
                    else "INCORRECT"
                ),
                "abstention_reason": debug["abstention_reason"],
                "statement_statuses": statement_statuses,
            }
        )

    total = len(records)

    return {
        "policy": policy,
        "total": total,
        "answered": answered,
        "abstained": total - answered,
        "correct": correct,
        "coverage": answered / total if total else None,
        "precision_when_answered": (
            correct / answered if answered else None
        ),
        "accuracy_overall": correct / total if total else None,
        "rows": rows,
    }


def percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def print_policy(result: dict, baseline_accuracy: float) -> None:

    print(f"--- policy: {result['policy']} ---")
    print(
        f"answered {result['answered']}/{result['total']}   "
        f"correct {result['correct']}/{result['total']}"
    )
    print(
        f"coverage {percent(result['coverage'])}   "
        f"precision {percent(result['precision_when_answered'])}   "
        f"accuracy {percent(result['accuracy_overall'])}"
    )

    delta = result["accuracy_overall"] - baseline_accuracy

    print(
        f"vs vanilla RAG ({percent(baseline_accuracy)}): "
        f"{delta:+.2%}"
    )
    print()

    for row in result["rows"]:

        marker = {
            "CORRECT": "ok  ",
            "INCORRECT": "X   ",
            "ABSTAINED": "--  ",
        }[row["status"]]

        reason = (
            f"  [{row['abstention_reason']}]"
            if row["abstention_reason"]
            else ""
        )

        print(
            f"  {marker}Q{row['q_number']:<3} "
            f"predicted={str(row['predicted_answer']):<5} "
            f"official={row['official_answer']}{reason}"
        )

    print()


def main() -> None:

    check_parsing_fixes()
    print("parsing fixes verified\n")

    records = json.loads(
        Path(RESULTS_PATH).read_text(encoding="utf-8")
    )

    # Vanilla RAG scored 5/13 on this same question set.
    baseline_accuracy = 5 / 13

    report = {}

    for policy in (STRICT, ELIMINATION, CLOSED_WORLD):
        result = replay(records, policy)
        report[policy] = result
        print_policy(result, baseline_accuracy)

    Path(OUTPUT_PATH).write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved replay to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
