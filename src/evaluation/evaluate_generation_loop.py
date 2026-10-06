"""Measure the current source-grounded generate → verify → revise workflow.

Streamed updates preserve each attempt's checks before revision clears them.
Simple MCQs use option comparison; statement MCQs use resolved truth patterns,
where CONTRADICTED is an intentional false statement rather than a defect.
Current reports are separate from historical generation-loop experiments and
include the source snapshot and generation-policy version.

First-attempt block rates and repair rates measure the system's own checks.
They do not establish question correctness or provide an independent accuracy
benchmark. Accepted questions still need expert adjudication for that claim.
"""

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src.generation.mcq_generator import (
    FORMAT_SIMPLE,
    FORMAT_STATEMENTS,
    FORMATS,
)
from src.orchestration.graph import build_graph
from src.orchestration.nodes import corpus_hash
from src.orchestration.telemetry import GENERATION_POLICY
from src.verification.learning_notes import (
    citation_issues, explanation_review_passes, notes_issues,
)
from src.verification.mcq_quality_auditor import quality_passes

# One file per arm, because the two arms are not the same measurement and
# overwriting one with the other is exactly the stale-report failure that
# run_all.py's freshness check exists to prevent.
OUTPUT_PATHS = {
    FORMAT_SIMPLE: "data/evaluation/generation_grounded_results.json",
    FORMAT_STATEMENTS: (
        "data/evaluation/generation_grounded_results.statements.json"
    ),
}

# Fixed topics for comparable workflow sampling. A retrieval miss or a gap
# in source coverage can block generation; blocked is not necessarily wrong.
TOPICS = [
    "Fundamental Rights",
    "Directive Principles of State Policy",
    "Fundamental Duties",
    "Amendment of the Constitution",
    "Emergency Provisions",
    "The President of India",
    "Parliament of India",
    "The Supreme Court of India",
    "Union Public Service Commission",
    "Election Commission of India",
    "Panchayats",
    "Municipalities",
    "Finance Commission",
    "Official Language",
    "Citizenship",
]

DIFFICULTY = "medium"
MAX_RETRIES = 2

# Source retrieval plus three attempts through seven candidate nodes.
RECURSION_LIMIT = 50

# A 15-topic run makes roughly 250 API calls through an intercepting
# corporate proxy, and the first full statements-arm run lost three topics
# to transient "Connection error." The whole topic is retried rather than
# the failed call, because a half-finished topic has no valid attempt
# history and merging one would corrupt the per-attempt record that is the
# entire point of this harness. Retried topics cost their API calls again;
# that is the price of a clean record.
TOPIC_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 15


def run_topic(
    workflow,
    topic: str,
    question_format: str = FORMAT_SIMPLE,
) -> dict:
    """
    Run one topic through the loop, returning per-attempt history.

    Attempts are delimited by `generate_mcq` updates. Gate results seen
    after the Nth generation belong to attempt N.
    """
    attempts = []
    current = None
    state = {}

    config = {"recursion_limit": RECURSION_LIMIT}

    initial_state = {
        "topic": topic,
        "difficulty": DIFFICULTY,
        "question_format": question_format,
        "retry_count": 0,
        "max_retries": MAX_RETRIES,
    }

    for update in workflow.stream(
        initial_state,
        config=config,
        stream_mode="updates",
    ):
        for node, payload in update.items():
            state.update(payload)
            if node == "retrieve_sources":
                continue
            if node == "generate_mcq":
                # A new generation starts a new attempt record. The
                # feedback that produced it is whatever the previous
                # attempt's decide step recorded, which is already stored.
                current = {
                    "attempt": payload.get("retry_count"),
                    "mcq": None,
                    "fact_verdicts": None,
                    "answer_verdict": None,
                    "quality": None,
                    "quality_issues": None,
                    "decision": None,
                    "failure_reasons": None,
                    "generation_policy": GENERATION_POLICY,
                    "source_count": len(state.get("generation_evidence", [])),
                    "structural_issues": None,
                    "fact_review_complete": False,
                    "answer_passed": False,
                    "quality_passed": False,
                    "explanations_passed": False,
                }

                mcq = payload.get("mcq")

                if mcq is not None:
                    current["mcq"] = {
                        "question": mcq.question,
                        "option_a": mcq.option_a,
                        "option_b": mcq.option_b,
                        "option_c": mcq.option_c,
                        "option_d": mcq.option_d,
                        "correct_answer": mcq.correct_answer,
                        "explanation": mcq.explanation,
                    }

                attempts.append(current)

            elif current is None:
                # Defensive: a gate update before any generation would
                # mean the graph shape changed. Fail loudly rather than
                # attribute it to a nonexistent attempt.
                raise RuntimeError(
                    f"received '{node}' update before any generate_mcq; "
                    f"graph shape has changed"
                )

            elif node == "extract_claims":
                current["structural_issues"] = list(payload.get("structural_issues", []))

            elif node == "verify_claims":
                verdicts = payload.get("fact_verifications") or {}

                current["fact_verdicts"] = {
                    claim_id: result.verdict
                    for claim_id, result in verdicts.items()
                }
                current["fact_review_complete"] = all(
                    (result := verdicts.get(claim.claim_id)) is not None
                    and result.verdict in ("SUPPORTED", "CONTRADICTED")
                    and result.independently_reviewed
                    and not citation_issues(result.citations, result.checked_evidence)
                    for claim in state.get("claims", [])
                )

            elif node == "verify_answer_key":
                result = payload.get("answer_verification")

                if result is not None:
                    current["answer_verdict"] = result.verdict
                    direct = state.get("direct_verification")
                    current["answer_passed"] = (
                        result.verdict == "VALID" and result.exactly_one_correct
                        and result.supported_options == [state["mcq"].correct_answer]
                        and (bool(state.get("claims")) or (
                            direct is not None and direct.status == "ANSWERED"
                            and direct.independently_reviewed
                        ))
                    )

            elif node == "audit_quality":
                result = payload.get("quality_audit")

                if result is not None:
                    current["quality"] = result.overall_quality
                    current["quality_issues"] = list(result.issues)
                current["quality_passed"] = quality_passes(result)

            elif node == "explain_question":
                result = payload.get("learning_notes")
                current["explanations_passed"] = bool(
                    result is not None and result.verdict == "PASS"
                    and result.notes is not None and result.review is not None
                    and explanation_review_passes(result.review)
                    and not notes_issues(result.notes, state.get("answer_evidence", []), state["mcq"])
                )

            elif node == "decide":
                current["decision"] = payload.get("decision")
                current["failure_reasons"] = list(
                    payload.get("failure_reasons") or []
                )

    return {
        "topic": topic,
        "attempts": attempts,
        "terminal_decision": (
            attempts[-1]["decision"] if attempts else None
        ),
        "attempt_count": len(attempts),
    }


def gates_fired(attempt: dict) -> dict:
    """Attribute current checks while retaining historical report semantics."""
    if attempt.get("generation_policy") == GENERATION_POLICY:
        return {
            "sources": not attempt.get("source_count"),
            "format": attempt.get("structural_issues") != [],
            "fact": not attempt.get("fact_review_complete", False),
            "answer_key": not attempt.get("answer_passed", False),
            "quality": not attempt.get("quality_passed", False),
            "explanations": not attempt.get("explanations_passed", False),
        }
    verdicts = attempt.get("fact_verdicts") or {}

    return {
        "fact": any(
            verdict != "SUPPORTED"
            for verdict in verdicts.values()
        ),
        "answer_key": (
            attempt.get("answer_verdict") is not None
            and attempt["answer_verdict"] != "VALID"
        ),
        "quality": (
            attempt.get("quality") is not None
            and attempt["quality"] != "PASS"
        ),
    }


def summarise(results: list[dict]) -> dict:
    """
    Turn per-attempt history into the reported metrics.

    Kept separate from the run so the JSON on disk can be re-summarised
    without spending API calls, the same split as
    replay_answer_mapping.py.
    """
    completed = [r for r in results if not r.get("error")]
    errored = [r for r in results if r.get("error")]

    terminal = Counter(r["terminal_decision"] for r in completed)
    attempt_counts = Counter(r["attempt_count"] for r in completed)

    # Gate attribution over every attempt that was blocked. Gates are not
    # mutually exclusive, so "blocked" counts overlaps and "sole" counts
    # the attempts where exactly one gate fired. Reporting only the first
    # would overstate whichever gate is checked first.
    blocked = Counter()
    sole = Counter()
    blocked_attempts = 0

    # Attempt 1 across all topics is the unverified-generator arm.
    baseline_clean = 0
    baseline_gates = Counter()

    fact_verdicts = Counter()
    claims_per_attempt = []

    for result in completed:
        for attempt in result["attempts"]:
            fired = gates_fired(attempt)
            names = [name for name, hit in fired.items() if hit]

            verdicts = attempt.get("fact_verdicts") or {}
            fact_verdicts.update(verdicts.values())
            claims_per_attempt.append(len(verdicts))

            if names:
                blocked_attempts += 1
                blocked.update(names)

                if len(names) == 1:
                    sole[names[0]] += 1

            if attempt.get("attempt") == 1:
                if names:
                    baseline_gates.update(names)
                else:
                    baseline_clean += 1

    # Repair rate: of topics whose first attempt was blocked, how many
    # ended up ACCEPTed. This is the only number that says whether
    # feeding failure_reasons back into the prompt actually does anything.
    needed_repair = [
        r for r in completed
        if r["attempts"] and any(gates_fired(r["attempts"][0]).values())
    ]

    repaired = [
        r for r in needed_repair
        if r["terminal_decision"] == "ACCEPT"
    ]

    total = len(completed)

    def pct(numerator: int, denominator: int) -> float:
        if not denominator:
            return 0.0

        return round(100.0 * numerator / denominator, 2)

    return {
        "topics_run": len(results),
        "topics_completed": total,
        "topics_errored": len(errored),
        "errors": [
            {"topic": r["topic"], "error": r["error"]}
            for r in errored
        ],
        "terminal_decisions": {
            "ACCEPT": terminal.get("ACCEPT", 0),
            "REJECT": terminal.get("REJECT", 0),
            "accept_rate": pct(terminal.get("ACCEPT", 0), total),
            "reject_rate": pct(terminal.get("REJECT", 0), total),
        },
        "attempts": {
            "distribution": dict(sorted(attempt_counts.items())),
            "total_attempts": sum(
                r["attempt_count"] for r in completed
            ),
            "blocked_attempts": blocked_attempts,
        },
        "gate_attribution": {
            "blocked_by": dict(blocked),
            "sole_blocker": dict(sole),
            "note": (
                "gates are not mutually exclusive; blocked_by counts "
                "overlaps, sole_blocker counts attempts where exactly "
                "one gate fired"
            ),
        },
        "unverified_baseline": {
            "description": (
                "First drafts assessed by this system's own checks; "
                "a block is not an independently established error."
            ),
            "n": total,
            "clean": baseline_clean,
            "defective": total - baseline_clean,
            "defect_rate": pct(total - baseline_clean, total),
            "block_rate": pct(total - baseline_clean, total),
            "gates_that_would_have_caught_it": dict(baseline_gates),
        },
        "repair": {
            "first_attempt_blocked": len(needed_repair),
            "eventually_accepted": len(repaired),
            "repair_rate": pct(len(repaired), len(needed_repair)),
        },
        "claims": {
            "verdict_distribution": dict(fact_verdicts),
            "total_claims_verified": sum(claims_per_attempt),
            "mean_claims_per_attempt": (
                round(
                    sum(claims_per_attempt) / len(claims_per_attempt),
                    2,
                )
                if claims_per_attempt
                else 0.0
            ),
        },
    }


def print_summary(summary: dict) -> None:
    print("\n=== GENERATION LOOP: OUTCOMES ===\n")

    decisions = summary["terminal_decisions"]

    print(
        f"Topics:   {summary['topics_completed']} completed"
        f"  ({summary['topics_errored']} errored)"
    )
    print(
        f"ACCEPT:   {decisions['ACCEPT']}"
        f"  ({decisions['accept_rate']}%)"
    )
    print(
        f"REJECT:   {decisions['REJECT']}"
        f"  ({decisions['reject_rate']}%)"
    )

    print("\nAttempts per topic:")

    for count, topics in summary["attempts"]["distribution"].items():
        print(f"  {count} attempt(s): {topics} topic(s)")

    print(
        f"\nTotal attempts:   "
        f"{summary['attempts']['total_attempts']}"
    )
    print(
        f"Blocked attempts: "
        f"{summary['attempts']['blocked_attempts']}"
    )

    print("\n=== GATE ATTRIBUTION ===\n")

    blocked = summary["gate_attribution"]["blocked_by"]
    sole = summary["gate_attribution"]["sole_blocker"]

    print(f"{'gate':<14}{'blocked':>9}{'sole':>7}")

    for gate in ("sources", "format", "fact", "answer_key", "quality", "explanations"):
        print(
            f"{gate:<14}"
            f"{blocked.get(gate, 0):>9}"
            f"{sole.get(gate, 0):>7}"
        )

    print("\n=== UNVERIFIED GENERATOR (attempt 1) ===\n")

    baseline = summary["unverified_baseline"]

    print(f"n:            {baseline['n']}")
    print(f"Passed checks: {baseline['clean']}")
    print(
        f"Blocked:      {baseline['defective']}"
        f"  ({baseline['defect_rate']}%)"
    )

    print("\n=== REPAIR ===\n")

    repair = summary["repair"]

    print(
        f"First attempt blocked: "
        f"{repair['first_attempt_blocked']}"
    )
    print(
        f"Eventually accepted:   "
        f"{repair['eventually_accepted']}"
        f"  ({repair['repair_rate']}%)"
    )

    print("\n=== CLAIMS ===\n")

    claims = summary["claims"]

    print(
        f"Verified:            "
        f"{claims['total_claims_verified']}"
    )
    print(
        f"Mean per attempt:    "
        f"{claims['mean_claims_per_attempt']}"
    )

    for verdict, count in sorted(
        claims["verdict_distribution"].items()
    ):
        print(f"  {verdict:<14} {count}")


def main() -> dict:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the generate-and-gate LangGraph loop."
        ),
    )

    parser.add_argument(
        "--format",
        dest="question_format",
        choices=FORMATS,
        default=FORMAT_SIMPLE,
        help=(
            "'simple' uses option comparison; 'statements' uses "
            "individual claim checks and deterministic mapping"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "run only the first N topics; each accepted item uses "
            "several paid generation, verification and explanation calls"
        ),
    )

    parser.add_argument(
        "--summarise-only",
        action="store_true",
        help=(
            "re-summarise the existing results file without spending "
            "any API calls"
        ),
    )

    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    results_path = OUTPUT_PATHS[args.question_format]
    output_path = Path(results_path)

    if args.summarise_only:
        if not output_path.exists():
            raise SystemExit(
                f"{results_path} does not exist; run without "
                f"--summarise-only first"
            )

        with output_path.open(encoding="utf-8") as handle:
            stored = json.load(handle)

        summary = summarise(stored["results"])
        print_summary(summary)

        return summary

    topics = TOPICS[: args.limit] if args.limit else TOPICS

    workflow = build_graph()
    source_hash = corpus_hash()

    results = []

    print(f"Arm: question_format={args.question_format}\n")

    for index, topic in enumerate(topics, start=1):
        print(f"[{index}/{len(topics)}] {topic}")

        result = None
        last_error = None

        for tries in range(1, TOPIC_ATTEMPTS + 1):
            try:
                result = run_topic(
                    workflow,
                    topic,
                    question_format=args.question_format,
                )

                break

            except Exception as error:
                last_error = error

                print(
                    f"    transient failure "
                    f"({tries}/{TOPIC_ATTEMPTS}): {type(error).__name__}"
                )

                if tries < TOPIC_ATTEMPTS:
                    time.sleep(RETRY_BACKOFF_SECONDS * tries)

        if result is None:
            # One topic failing should not discard the other fourteen
            # topics' worth of API spend. The error is recorded in the
            # output file so a partial run cannot be mistaken for a
            # complete one.
            print(f"    ERROR: giving up on {topic}")

            results.append(
                {
                    "topic": topic,
                    "attempts": [],
                    "terminal_decision": None,
                    "attempt_count": 0,
                    "error": type(last_error).__name__,
                }
            )

            continue

        print(
            f"    {result['terminal_decision']} after "
            f"{result['attempt_count']} attempt(s)"
        )

        results.append(result)

    summary = summarise(results)

    payload = {
        "config": {
            "generation_policy": GENERATION_POLICY,
            "corpus_hash": source_hash,
            "corpus_changed_during_run": corpus_hash() != source_hash,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "topics": topics,
            "difficulty": DIFFICULTY,
            "question_format": args.question_format,
            "max_retries": MAX_RETRIES,
        },
        "summary": summary,
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print_summary(summary)

    print(f"\nSaved results to: {results_path}")

    return summary


if __name__ == "__main__":
    main()
