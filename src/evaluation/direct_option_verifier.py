"""
System B2: answer the question directly from evidence, with no answer leak.

Why this exists
---------------

Two problems in the existing evaluation motivated this arm.

1. Answer leakage in the existing answer-key verifier.

   `AnswerKeyVerifier` prints DECLARED ANSWER into its prompt
   (answer_key_verifier.py), and `pyq_to_mcq` sets that field to the
   official UPSC answer. Rule 7 of that prompt tells the model not to
   treat the declared answer as evidence, but a prompt instruction is not
   an information barrier: the correct letter is in the context window.

   That component selects exactly the official option on 8/13 of the
   evaluation set. That number is contaminated and must not be reported
   as a result. This module re-runs the same job with the answer removed,
   which is the only way to find out what the component is actually worth.

2. System B throws away a working signal.

   System B predicts by verifying each numbered statement and then
   mapping the set of SUPPORTED statements onto an option. That mapping is
   all-or-nothing: one INSUFFICIENT statement and the whole question
   abstains, which is why coverage is 3/13. Meanwhile the option-level
   analysis runs on every question and is then discarded.

   Verifying options directly is a shorter inference chain than verifying
   statements and reconstructing the option from them, so it should be
   less brittle. This arm measures whether that is true.

What is held constant
---------------------

Same corpus, same retrieved evidence (replayed from the stored run, not
re-retrieved), same model family as the rest of the pipeline. The only
changes are the removal of the declared answer and the switch from
statement-set mapping to direct option support.

Any difference in accuracy is therefore attributable to the inference
path, not to retrieval.

Evidence conditions
-------------------

answer_evidence
    The top-5 passages retrieved for the question as a whole. This is
    exactly what the leaked verifier saw, so it isolates the leak.

union
    Those passages plus every passage retrieved per-statement. Statement
    retrieval often surfaces the specific provision the question turns on,
    so this separates "wrong inference" from "never saw the provision".

Abstention is preserved. Zero or several supported options means the
system declines rather than guesses, so this arm stays comparable with
System B on the selective-prediction metrics.

Run:

    python -m src.evaluation.direct_option_verifier
"""

import json
import time
import urllib.error
from pathlib import Path

from src.evaluation.judge_llm import (
    format_evidence,
    load_api_key,
    post_chat_completion,
)


# The same generator model both other arms use, so the comparison is not
# confounded by model capability.
MODEL_NAME = "gpt-4o-mini"

VERIFIED_PATH = "data/evaluation/verified_pyq_results.json"
REPLAY_PATH = "data/evaluation/answer_mapping_replay.json"
OUTPUT_PATH = "data/evaluation/direct_option_results.json"

ANSWER_EVIDENCE = "answer_evidence"
UNION_EVIDENCE = "union"

CONDITIONS = (ANSWER_EVIDENCE, UNION_EVIDENCE)

# Vanilla RAG answers every question, so its accuracy is its precision.
BASELINE_ACCURACY = 5 / 13

MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 4


OPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "supported_options": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["A", "B", "C", "D"],
            },
        },
        "per_option_reasoning": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "option": {
                        "type": "string",
                        "enum": ["A", "B", "C", "D"],
                    },
                    "supported": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["option", "supported", "reason"],
                "additionalProperties": False,
            },
        },
        "cited_pages": {
            "type": "array",
            "items": {"type": "integer"},
        },
        "evidence_is_sufficient": {"type": "boolean"},
        "reasoning": {"type": "string"},
    },
    "required": [
        "supported_options",
        "per_option_reasoning",
        "cited_pages",
        "evidence_is_sufficient",
        "reasoning",
    ],
    "additionalProperties": False,
}


PROMPT_RULES = """
You are an independent answer-analysis component for a UPSC Indian Polity
question. Decide which of the four options the SOURCE EVIDENCE supports
as the correct answer.

You are NOT told the official answer. Do not guess what it might be from
the shape of the options, from which option looks most examinable, or from
background knowledge. Work only from the evidence.

RULES

1. Use ONLY the provided source evidence. Do not use outside knowledge,
   even if you are confident about Indian constitutional law.
2. A semantically related passage does not establish an option. The
   evidence must actually settle the question.
3. Include an option in supported_options only when the evidence
   explicitly establishes it as the correct answer.
4. If the evidence does not settle the question, return an EMPTY
   supported_options list and set evidence_is_sufficient to false. An
   empty list is a valid and expected answer. Declining is better than
   guessing.
5. If the evidence genuinely supports more than one option, include all
   of them. Do not pick one to appear decisive.
6. Many of these questions ask which of several numbered statements are
   correct. Evidence that merely MENTIONS the subject of a statement does
   not establish that the statement is true. Read for the specific
   proposition, including any numbers, time limits, age thresholds,
   authorities and exceptions.
7. Where a statement asserts a specific number or threshold and the
   evidence states a different number for the same provision, that
   statement is false, and options requiring it to be true are not
   supported.
8. cited_pages must contain only pages you actually relied on.
9. Set evidence_is_sufficient to true only if the evidence settles the
   question. It is independent of whether you found exactly one option.
"""


def analyse(api_key: str, prompt: str) -> dict:
    """
    Call the model, retrying transient failures on the same model.

    The model is never downgraded on failure: a silent model switch
    mid-run would make the arm incomparable with the others, which is the
    one thing this evaluation depends on.
    """

    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):

        try:
            return post_chat_completion(
                api_key=api_key,
                model=MODEL_NAME,
                prompt=prompt,
                schema_name="option_analysis",
                schema=OPTION_SCHEMA,
            )

        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            KeyError,
        ) as error:
            last_error = error

        if attempt < MAX_ATTEMPTS:
            wait = RETRY_BACKOFF_SECONDS * attempt

            print(
                f"    attempt {attempt} failed "
                f"({type(last_error).__name__}), "
                f"retrying in {wait}s"
            )

            time.sleep(wait)

    raise RuntimeError(
        f"Option analysis failed after {MAX_ATTEMPTS} attempts: "
        f"{last_error}"
    )


def gather_evidence(record: dict, condition: str) -> list[dict]:
    """Select the evidence set for one condition."""

    items = list(record.get("answer_evidence", []))

    if condition == UNION_EVIDENCE:
        for passages in record.get(
            "claim_evidence", {}
        ).values():
            items.extend(passages)

    return items


def build_prompt(record: dict, condition: str) -> str:
    """Assemble the option-analysis prompt, with no declared answer."""

    options = "\n".join(
        f"  {letter}. {text}"
        for letter, text in sorted(record["options"].items())
    )

    evidence = format_evidence(
        gather_evidence(record, condition)
    )

    return f"""{PROMPT_RULES}

========================================
QUESTION
========================================

{record["question"]}

OPTIONS

{options}

========================================
SOURCE EVIDENCE
========================================

{evidence}
"""


def predict(analysis: dict) -> tuple[str | None, str | None]:
    """
    Commit to an option only when exactly one is supported.

    Mirrors System B's abstention rule so the two arms stay comparable:
    ambiguity is reported, never resolved by taking the first candidate.
    """

    supported = [
        option
        for option in analysis["supported_options"]
        if option in {"A", "B", "C", "D"}
    ]

    supported = list(dict.fromkeys(supported))

    if len(supported) == 1:
        return supported[0], None

    if not supported:
        return None, "no_option_supported"

    return None, "multiple_options_supported"


def run_condition(
    records: list[dict],
    condition: str,
    api_key: str,
) -> dict:
    """Score every question under one evidence condition."""

    rows = []

    for index, record in enumerate(records, start=1):
        q_number = record["q_number"]

        print(
            f"  [{condition}] {index}/{len(records)}: Q{q_number}"
        )

        analysis = analyse(
            api_key, build_prompt(record, condition)
        )

        predicted, abstention_reason = predict(analysis)

        official = record["official_answer"]

        rows.append(
            {
                "q_number": q_number,
                "official_answer": official,
                "predicted_answer": predicted,
                "status": (
                    "ABSTAINED"
                    if predicted is None
                    else "CORRECT"
                    if predicted == official
                    else "INCORRECT"
                ),
                "abstention_reason": abstention_reason,
                "supported_options": analysis[
                    "supported_options"
                ],
                "evidence_is_sufficient": analysis[
                    "evidence_is_sufficient"
                ],
                "cited_pages": analysis["cited_pages"],
                "reasoning": analysis["reasoning"],
                "per_option_reasoning": analysis[
                    "per_option_reasoning"
                ],
            }
        )

    total = len(rows)
    answered = [row for row in rows if row["status"] != "ABSTAINED"]
    correct = [row for row in rows if row["status"] == "CORRECT"]

    return {
        "condition": condition,
        "evidence": (
            "top-5 question-level passages"
            if condition == ANSWER_EVIDENCE
            else "question-level plus per-statement passages"
        ),
        "total": total,
        "answered": len(answered),
        "abstained": total - len(answered),
        "correct": len(correct),
        "coverage": len(answered) / total if total else None,
        "precision_when_answered": (
            len(correct) / len(answered) if answered else None
        ),
        "accuracy_overall": (
            len(correct) / total if total else None
        ),
        "rows": rows,
    }


def percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def load_system_b_reference() -> dict | None:
    """
    Read System B's strict-policy figures from the mapping replay.

    Returns None when the replay has not been run, so this arm still
    reports its own numbers instead of failing on a missing file.
    """

    replay_file = Path(REPLAY_PATH)

    if not replay_file.exists():
        return None

    replay = json.loads(replay_file.read_text(encoding="utf-8"))

    strict = replay.get("strict")

    if strict is None:
        return None

    # Drop the per-question rows; only the headline metrics are needed
    # here, and carrying the rows would duplicate the replay file.
    return {
        key: value
        for key, value in strict.items()
        if key != "rows"
    }


def print_condition(result: dict) -> None:
    print()
    print(f"--- condition: {result['condition']} ---")
    print(f"evidence: {result['evidence']}")
    print(
        f"answered {result['answered']}/{result['total']}   "
        f"correct {result['correct']}/{result['total']}"
    )
    print(
        f"coverage {percent(result['coverage'])}   "
        f"precision {percent(result['precision_when_answered'])}   "
        f"accuracy {percent(result['accuracy_overall'])}"
    )
    print(
        f"vs vanilla RAG {percent(BASELINE_ACCURACY)}: "
        f"{result['accuracy_overall'] - BASELINE_ACCURACY:+.2%}"
    )
    print()

    for row in result["rows"]:
        marker = {
            "CORRECT": "ok  ",
            "INCORRECT": "X   ",
            "ABSTAINED": "--  ",
        }[row["status"]]

        note = (
            f"  [{row['abstention_reason']}]"
            if row["abstention_reason"]
            else ""
        )

        print(
            f"  {marker}Q{row['q_number']:<3} "
            f"predicted={str(row['predicted_answer']):<5} "
            f"official={row['official_answer']}  "
            f"supported={row['supported_options']}{note}"
        )


def main() -> None:
    records = json.loads(
        Path(VERIFIED_PATH).read_text(encoding="utf-8")
    )

    api_key = load_api_key()

    report = {
        "dataset": "UPSC 2025 Prelims Polity, Q54-Q66",
        "model": MODEL_NAME,
        "arm": (
            "System B2: direct option verification, "
            "declared answer withheld"
        ),
        "note": (
            "Replays stored evidence. Retrieval is held fixed, so "
            "differences against System B reflect the inference path "
            "only. The existing AnswerKeyVerifier result is not "
            "comparable because its prompt contains the official "
            "answer."
        ),
        "reference_accuracy": {
            "system_a_vanilla_rag": BASELINE_ACCURACY,
            "system_b_statement_mapping": load_system_b_reference(),
        },
        "conditions": {},
    }

    for condition in CONDITIONS:
        print(f"Running condition: {condition}")

        result = run_condition(records, condition, api_key)

        report["conditions"][condition] = result

        print_condition(result)

    print()
    print("=== SUMMARY ===")

    header = (
        f"{'arm':<34}{'cov':>8}{'prec':>9}{'acc':>9}"
    )
    print(header)
    print("-" * len(header))

    print(
        f"{'System A vanilla RAG':<34}"
        f"{'100.00%':>8}{percent(BASELINE_ACCURACY):>9}"
        f"{percent(BASELINE_ACCURACY):>9}"
    )

    # Read System B's figures from the replay rather than restating them,
    # so this table cannot drift out of date when the mapping changes.
    system_b = load_system_b_reference()

    if system_b is not None:
        print(
            f"{'System B statement mapping':<34}"
            f"{percent(system_b['coverage']):>8}"
            f"{percent(system_b['precision_when_answered']):>9}"
            f"{percent(system_b['accuracy_overall']):>9}"
        )
    else:
        print(
            f"{'System B statement mapping':<34}"
            f"{'(run replay_answer_mapping first)':>26}"
        )

    for condition in CONDITIONS:
        result = report["conditions"][condition]

        print(
            f"{'System B2 ' + condition:<34}"
            f"{percent(result['coverage']):>8}"
            f"{percent(result['precision_when_answered']):>9}"
            f"{percent(result['accuracy_overall']):>9}"
        )

    Path(OUTPUT_PATH).write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print(f"Saved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
