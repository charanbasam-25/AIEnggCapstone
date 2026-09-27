"""
Measure how repeatable the fact verifier's verdicts are.

Why this module exists
----------------------

Every accuracy number in this project is one run of the pipeline. That is
only meaningful if a second run would produce the same verdicts. It would
not have: `FactVerifier.verify` never set `temperature`, and the Responses
API defaults to 1.0, so the verifier was a stochastic classifier and every
reported figure was a single draw from an unseen distribution. The baseline
in rag/vanilla_rag.py *was* pinned, so the headline comparison put a
sampling system against a deterministic one and attributed the difference
entirely to verification.

That is now fixed. This module exists to put a number on what the defect
was worth, because "we pinned temperature" is not a finding and "N of 37
claims were unstable, touching M of 13 questions" is.

What it measures
----------------

Two conditions over the same 37 claims, replaying the evidence actually
retrieved for each claim (read from claim_evidence in the results file, so
no retrieval runs and the comparison is exact):

  unpinned  temperature omitted, i.e. the behaviour that produced every
            number currently written up
  pinned    temperature=0, i.e. the shipped behaviour now

For each claim and condition it records the verdict from each of REPEATS
samples and reports the claim as stable if all samples agree. A claim that
disagrees with itself is a claim whose verdict is not a property of the
evidence.

Reading the output honestly
---------------------------

Stability is not correctness. A claim can be unanimously and repeatably
wrong, and pinning temperature converts a coin flip into a fixed answer
without making that answer right. The pinned condition is therefore
expected to show near-total stability and that is not a quality result -
it is a precondition for the other measurements meaning anything.

The interesting quantity is the unpinned instability rate, and which
questions it lands on. An unstable claim under a question whose verdict
decides the predicted option means that question's contribution to
accuracy was luck. Those questions are listed explicitly.

Cost note: REPEATS * 37 * 2 calls to gpt-4o-mini. Requests are issued
concurrently because they are independent; WORKERS is deliberately modest
to stay well inside rate limits.
"""

import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.evaluation.answer_mapping import (
    STRICT,
    _Claim,
    _Verdict,
    determine_statement_statuses,
    map_answer,
)
from src.evaluation.judge_llm import load_api_key
from src.retrieval.lexical_index import load_chunks
from src.verification.fact_verifier import FactVerifier, MODEL_NAME

RESULTS_PATH = "data/evaluation/verified_pyq_results.json"
CHUNKS_PATH = "data/processed/chunks.jsonl"
OUTPUT_PATH = "data/evaluation/verdict_stability.json"

REPEATS = 5
WORKERS = 8

CONDITION_UNPINNED = "unpinned"
CONDITION_PINNED = "pinned"

# None means "omit temperature from the request", which is what the
# verifier did before it was pinned. See FactVerifier.verify.
CONDITIONS = {
    CONDITION_UNPINNED: None,
    CONDITION_PINNED: 0,
}


def load_records() -> list[dict]:
    return json.loads(
        Path(RESULTS_PATH).read_text(encoding="utf-8")
    )


def collect_claims(records: list[dict]) -> list[dict]:
    """
    Flatten to one entry per claim, carrying the evidence that claim was
    actually verified against.

    A claim with no stored evidence is skipped rather than re-retrieved.
    Re-retrieving would silently change what is being measured.
    """

    claims = []

    for record in records:

        evidence_by_claim = record.get("claim_evidence") or {}

        for claim in record["claims"]:

            evidence = evidence_by_claim.get(claim["claim_id"])

            if not evidence:
                continue

            claims.append(
                {
                    "q_number": record["q_number"],
                    "claim_id": claim["claim_id"],
                    "statement_number": claim["statement_number"],
                    "claim": claim["claim"],
                    "evidence": evidence,
                    "pages": [
                        chunk["page"] for chunk in evidence
                    ],
                }
            )

    return claims


def sample_verdicts(
    verifier: FactVerifier,
    claim: dict,
    temperature: float | None,
) -> list[str]:
    """
    Verify one claim REPEATS times and return the verdicts in order.

    The negative-claim checker inside FactVerifier can short-circuit
    without an API call. That is part of the shipped path, so it is left
    in: a claim resolved lexically is genuinely stable, and reporting it
    as such is correct rather than flattering.
    """

    verdicts = []

    for _ in range(REPEATS):
        result = verifier.verify(
            claim["claim"],
            claim["evidence"],
            temperature=temperature,
        )
        verdicts.append(result.verdict)

    return verdicts


def run_condition(
    verifier: FactVerifier,
    claims: list[dict],
    condition: str,
) -> list[dict]:

    temperature = CONDITIONS[condition]

    print(
        f"  {condition}: {len(claims)} claims x {REPEATS} samples",
        flush=True,
    )

    def work(claim: dict) -> dict:
        verdicts = sample_verdicts(
            verifier, claim, temperature
        )
        counts = Counter(verdicts)
        return {
            "q_number": claim["q_number"],
            "claim_id": claim["claim_id"],
            "statement_number": claim["statement_number"],
            "claim": claim["claim"],
            "pages": claim["pages"],
            "verdicts": verdicts,
            "distinct_verdicts": sorted(counts),
            "modal_verdict": counts.most_common(1)[0][0],
            "stable": len(counts) == 1,
        }

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        return list(pool.map(work, claims))


def summarise(rows: list[dict]) -> dict:

    unstable = [row for row in rows if not row["stable"]]

    return {
        "claims": len(rows),
        "repeats": REPEATS,
        "stable_claims": len(rows) - len(unstable),
        "unstable_claims": len(unstable),
        "stability_rate": (
            (len(rows) - len(unstable)) / len(rows)
            if rows
            else 0.0
        ),
        "unstable_questions": sorted(
            {row["q_number"] for row in unstable}
        ),
        # Keyed by question as well as claim id: claim ids restart at
        # claim_1 within each question, so the id alone is ambiguous.
        "three_way_claims": [
            f"Q{row['q_number']}:{row['claim_id']}"
            for row in rows
            if len(row["distinct_verdicts"]) == 3
        ],
    }


def resample_runs(
    records: list[dict],
    rows: list[dict],
) -> dict:
    """
    Turn REPEATS samples per claim into REPEATS whole-pipeline runs, and
    report the spread of the headline metrics across them.

    Each claim was sampled independently, so taking draw i from every
    claim yields one run that is statistically as valid as the run in
    verified_pyq_results.json. That gives REPEATS independent observations
    of coverage, precision, accuracy and error rate for the price of the
    sampling already done, and turns "the pipeline scores 30.77%" into a
    range.

    This reuses the shipped mapping functions rather than reimplementing
    them, so the spread is the spread of the real answer mapping. Labels
    are the audited ones and the policy is STRICT, matching the headline
    figures.
    """

    verdicts_by_claim = {
        (row["q_number"], row["claim_id"]): row["verdicts"]
        for row in rows
    }

    runs = []

    for draw in range(REPEATS):

        answered = 0
        correct = 0
        wrong = 0

        per_question = {}

        for record in records:

            claims = [
                _Claim(
                    claim["claim_id"],
                    claim["statement_number"],
                )
                for claim in record["claims"]
            ]

            sampled = {}
            complete = True

            for claim in record["claims"]:
                key = (record["q_number"], claim["claim_id"])
                if key not in verdicts_by_claim:
                    complete = False
                    break
                sampled[claim["claim_id"]] = _Verdict(
                    verdicts_by_claim[key][draw]
                )

            # A question with any unsampled claim is left out of every
            # draw rather than scored on partial verdicts, which would
            # make the spread incomparable to the headline figures.
            if not complete:
                continue

            statuses = determine_statement_statuses(
                claims, sampled
            )

            predicted, _ = map_answer(
                statuses,
                record["options"],
                record["question"],
                policy=STRICT,
            )

            gold = (
                record.get("audited_answer")
                or record["official_answer"]
            )

            per_question[record["q_number"]] = predicted

            if predicted is None:
                continue

            answered += 1

            if predicted == gold:
                correct += 1
            else:
                wrong += 1

        total = len(per_question)

        runs.append(
            {
                "draw": draw,
                "questions": total,
                "answered": answered,
                "correct": correct,
                "wrong": wrong,
                "coverage": answered / total if total else 0.0,
                "precision": (
                    correct / answered if answered else None
                ),
                "accuracy": correct / total if total else 0.0,
                "error_rate": wrong / total if total else 0.0,
                "predictions": per_question,
            }
        )

    def spread(metric: str) -> dict:
        values = [
            run[metric]
            for run in runs
            if run[metric] is not None
        ]
        return {
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": (
                sum(values) / len(values) if values else None
            ),
            "values": values,
        }

    return {
        "policy": STRICT,
        "labels": "audited",
        "runs": runs,
        "spread": {
            metric: spread(metric)
            for metric in (
                "coverage",
                "precision",
                "accuracy",
                "error_rate",
            )
        },
    }


def print_resample(condition: str, resample: dict) -> None:

    print()
    print(
        f"--- {condition}: {REPEATS} independent pipeline runs "
        f"(strict, audited labels) ---"
    )

    header = (
        f"{'draw':>5s} {'cov':>8s} {'prec':>8s} "
        f"{'acc':>8s} {'err':>8s}"
    )
    print(header)

    for run in resample["runs"]:
        precision = (
            f"{run['precision'] * 100:.2f}%"
            if run["precision"] is not None
            else "n/a"
        )
        print(
            f"{run['draw']:>5d} "
            f"{run['coverage'] * 100:>7.2f}% "
            f"{precision:>8s} "
            f"{run['accuracy'] * 100:>7.2f}% "
            f"{run['error_rate'] * 100:>7.2f}%"
        )

    for metric in ("coverage", "precision", "accuracy", "error_rate"):
        stats = resample["spread"][metric]
        if stats["min"] is None:
            continue
        print(
            f"  {metric:<11s} range "
            f"{stats['min'] * 100:.2f}% - {stats['max'] * 100:.2f}% "
            f"(mean {stats['mean'] * 100:.2f}%, "
            f"spread {(stats['max'] - stats['min']) * 100:.2f} pts)"
        )


def print_condition(condition: str, summary: dict) -> None:

    print()
    print(f"--- {condition} ---")
    print(
        f"stable claims        : "
        f"{summary['stable_claims']}/{summary['claims']} "
        f"({summary['stability_rate'] * 100:.2f}%)"
    )
    print(
        f"unstable claims      : {summary['unstable_claims']}"
    )
    print(
        f"questions affected   : {summary['unstable_questions']}"
    )
    print(
        f"all-three-verdict    : {summary['three_way_claims']}"
    )


def print_unstable(rows: list[dict]) -> None:

    unstable = [row for row in rows if not row["stable"]]

    if not unstable:
        print("  (none)")
        return

    for row in sorted(unstable, key=lambda r: r["q_number"]):
        counts = Counter(row["verdicts"])
        spread = "  ".join(
            f"{verdict} x{count}"
            for verdict, count in counts.most_common()
        )
        print(
            f"  Q{row['q_number']} {row['statement_number']:>3s}  "
            f"pages {row['pages']}  {spread}"
        )


def main() -> None:

    # Import for the side effect of populating os.environ from .env, the
    # same way the rest of the evaluation entry points do.
    api_key = load_api_key()
    os.environ.setdefault("OPENAI_API_KEY", api_key)

    records = load_records()
    claims = collect_claims(records)

    print("=== VERDICT STABILITY ===")
    print(f"model    : {MODEL_NAME}")
    print(f"source   : {RESULTS_PATH}")
    print(f"claims   : {len(claims)}")
    print(f"repeats  : {REPEATS}")
    print()

    # The corpus is supplied so that the negative-claim checker is live,
    # because it is live in the shipped pipeline. An earlier version of
    # this module built FactVerifier() with no chunks; the resampled runs
    # then abstained on Q58 in every draw while the pipeline answered it,
    # and the spread was not the spread of anything that had been
    # reported. Scanning is lexical - no embeddings, no model - so this
    # costs a file read and does not pull in retrieval.
    chunks = load_chunks(CHUNKS_PATH)
    verifier = FactVerifier(chunks)

    report = {
        "model": MODEL_NAME,
        "repeats": REPEATS,
        "claims": len(claims),
        "conditions": {},
    }

    for condition in (CONDITION_UNPINNED, CONDITION_PINNED):

        rows = run_condition(verifier, claims, condition)
        summary = summarise(rows)
        resample = resample_runs(records, rows)

        report["conditions"][condition] = {
            "temperature": CONDITIONS[condition],
            "summary": summary,
            "resample": resample,
            "rows": rows,
        }

        print_condition(condition, summary)
        print_unstable(rows)
        print_resample(condition, resample)

    unpinned = report["conditions"][CONDITION_UNPINNED][
        "summary"
    ]
    pinned = report["conditions"][CONDITION_PINNED]["summary"]

    print()
    print("--- what pinning bought ---")
    print(
        f"unstable claims: {unpinned['unstable_claims']} "
        f"-> {pinned['unstable_claims']}"
    )
    print(
        f"questions whose verdicts were a coin flip: "
        f"{unpinned['unstable_questions']}"
    )
    print(
        "Stability is not correctness. A pinned verdict can be "
        "repeatably wrong."
    )

    Path(OUTPUT_PATH).write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print(f"Saved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
