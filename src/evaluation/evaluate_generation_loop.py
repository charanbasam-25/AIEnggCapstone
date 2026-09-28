"""
Measure the generate-and-gate loop in src/orchestration/.

Why this module exists
----------------------

The LangGraph loop was the most elaborate piece of orchestration in the
project and it was the only piece with no metrics attached. Every artifact
in data/evaluation/ measured the verifier *answering* the 13 PYQs; nothing
measured it *gating* a generated question. `run_workflow.py` ran one topic
and printed, so the loop's behaviour was only ever observed anecdotally.

That is a real gap and not a cosmetic one, because the loop is where the
project's headline claim is supposed to pay off. If verification catches
defects in generated questions, the rate at which it does so is the number
that says so.

Two design notes, because both were wrong in the obvious implementation
-----------------------------------------------------------------------

1. The loop is driven with `stream(..., stream_mode="updates")`, not
   `invoke()`. `generate_mcq` clears `failure_reasons` at the start of
   every attempt (nodes.py:88) and the graph returns only terminal state,
   so `invoke()` can tell you that a topic was ACCEPTed on attempt 3 but
   not what was wrong with attempts 1 and 2. The per-attempt history is
   the entire point of measuring a revision loop, and it exists only in
   the stream.

2. Gate attribution is computed from the gate *objects*, not by parsing
   `failure_reasons` strings. The three gates format their reasons
   differently - the fact gate prefixes a claim_id (nodes.py:176), the
   others use literal "answer_key: " and "quality: " tags - so string
   prefixes would attribute the fact gate by whatever a claim_id happens
   to look like. Reading `fact_verifications`, `answer_verification` and
   `quality_audit` out of the stream instead means a change to a message
   format cannot silently corrupt the attribution table.

The free baseline arm
---------------------

Attempt 1 of every topic *is* the unverified generator: an MCQ produced by
a RAG generator with no gate in front of it. It costs nothing extra to
record, and the fraction of attempt-1 questions carrying at least one
detectable defect is the closest analogue in the generation mode to System
A's error rate. Reporting it is the comparative-evaluation arm for this
flow.

The two format arms, and why the first one is not enough
-------------------------------------------------------

The first run of this harness produced a result that says more about the
architecture than about the gates: a generated question decomposed into
exactly **one** claim, and the loop accepted it on the first attempt. That
is not a gate working well, it is a gate with almost nothing to hold. The
generator's eleven requirements never ask for the multi-statement form, so
every generated question is a single-fact recall item - and the machinery
this project is actually built out of never runs. question_parser's
STATEMENTS/STEM_DISTRIBUTED/PAIRS classification, claim_builder's predicate
binding, and answer_mapping's subset and count parsing are all unreachable
from a FORMAT_SIMPLE question.

So the generate-and-gate mode had been routing around the verifier's only
interesting capability, which is a large part of why the project's centre
of gravity drifted to the PYQ-answering mode. Two arms are therefore run:

  --format simple       what the loop shipped with; one claim per question
  --format statements   multi-statement UPSC form; exercises the real path

Comparing gate-firing rates between the arms is the comparative evaluation
for this flow, and it quantifies the format mismatch rather than asserting
it. Results go to separate files so neither arm overwrites the other.

What is deliberately not claimed
--------------------------------

This measures how often the gates fire and whether revision repairs what
they caught. It does not establish that the accepted questions are
*correct*, because the gate that judged them is the only judge in the
room. A human-reviewed sample, or the LLM-as-a-judge harness pointed at
accepted-vs-attempt-1 pairs, would be needed for that, and neither is run
here. n = 15 topics is also small; the attempt-level counts are the
reliable part and the per-gate splits are indicative.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from src.generation.mcq_generator import (
    FORMAT_SIMPLE,
    FORMAT_STATEMENTS,
    FORMATS,
)
from src.orchestration.graph import build_graph

# One file per arm, because the two arms are not the same measurement and
# overwriting one with the other is exactly the stale-report failure that
# run_all.py's freshness check exists to prevent.
OUTPUT_PATHS = {
    FORMAT_SIMPLE: "data/evaluation/generation_loop_results.json",
    FORMAT_STATEMENTS: (
        "data/evaluation/generation_loop_results.statements.json"
    ),
}

# Fixed, ordered, and checked into the repo so a re-run measures the same
# thing. Every topic is decidable from the corpus described in DESIGN.md
# section 2 - Constitution full text plus NCERT Polity - because a topic
# the corpus cannot settle would make the fact gate fire for a reason that
# has nothing to do with generation quality.
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

# Worst case is MAX_RETRIES + 1 attempts x 6 nodes = 18 steps. The default
# LangGraph limit of 25 would hold, but it is stated rather than relied on:
# raising MAX_RETRIES without noticing the limit is exactly the kind of
# silent truncation this project has been bitten by before.
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

            elif node == "verify_claims":
                verdicts = payload.get("fact_verifications") or {}

                current["fact_verdicts"] = {
                    claim_id: result.verdict
                    for claim_id, result in verdicts.items()
                }

            elif node == "verify_answer_key":
                result = payload.get("answer_verification")

                if result is not None:
                    current["answer_verdict"] = result.verdict

            elif node == "audit_quality":
                result = payload.get("quality_audit")

                if result is not None:
                    current["quality"] = result.overall_quality
                    current["quality_issues"] = list(result.issues)

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
    """
    Which of the three gates blocked this attempt.

    Computed from the gate results, not from failure_reasons text. The
    conditions mirror nodes.py decide() exactly: a non-SUPPORTED fact
    verdict, a non-VALID answer-key verdict, a non-PASS quality audit.
    """
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
                "attempt 1 of every topic, i.e. what a RAG generator "
                "with no gate in front of it would have shipped"
            ),
            "n": total,
            "clean": baseline_clean,
            "defective": total - baseline_clean,
            "defect_rate": pct(total - baseline_clean, total),
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

    for gate in ("fact", "answer_key", "quality"):
        print(
            f"{gate:<14}"
            f"{blocked.get(gate, 0):>9}"
            f"{sole.get(gate, 0):>7}"
        )

    print("\n=== UNVERIFIED GENERATOR (attempt 1) ===\n")

    baseline = summary["unverified_baseline"]

    print(f"n:            {baseline['n']}")
    print(f"Clean:        {baseline['clean']}")
    print(
        f"Defective:    {baseline['defective']}"
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
            "question format arm; 'simple' is what the loop shipped "
            "with, 'statements' is the multi-statement UPSC form that "
            "actually exercises the claim machinery"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "run only the first N topics; the full list costs roughly "
            "8 LLM calls per attempt"
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
                    f"({tries}/{TOPIC_ATTEMPTS}): {error}"
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
                    "error": (
                        f"{type(last_error).__name__}: {last_error}"
                    ),
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
