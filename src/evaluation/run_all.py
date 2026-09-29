"""
Run the evaluation modules in dependency order, and prove they are in sync.

Why this exists
---------------

The evaluation modules read each other's output files. Nothing enforced
the order, and running them in the wrong one does not fail - it silently
produces a report built from a previous run's inputs. That happened twice
while producing the numbers in EVALUATION.md:

  1. system_b_metrics ran before replay_answer_mapping, so its
     B_statement_mapping_strict row disagreed with B_mapping_fixed_strict
     even though both come from the same mapping code. The two arms
     matching is supposed to be a consistency check; instead the check
     appeared to fail.

  2. direct_option_verifier ran before replay_answer_mapping, so its
     summary table printed System B at 38.46/60.00/23.08 while the
     freshly-written results file said 30.77/100.00/30.77.

Neither produced an error. Both produced a document-ready table of wrong
numbers, which is the worst failure mode available to an evaluation
harness. So the order is written down here once, as data, and checked.

How the check works
-------------------

Each stage declares the files it reads and the files it writes. After the
stages run, every output must be at least as new as every one of its
inputs. If an output is older than something it was derived from, the
report on disk was not built from the data currently on disk and the run
fails.

This is mtime-based, which is coarse: it catches stale-ordering, which is
the failure that actually happened, and it does not catch a file that was
rewritten with identical content. Content hashing would be stricter and
is not worth it here, because the modules are not idempotent across runs
anyway - they call an LLM.

A second, non-mtime check runs alongside it. Changing RETRIEVAL_TOP_K in
src/retrieval/retrieval_config.py invalidates every number in
data/evaluation/ without modifying anything in it, so the ordering check
would pass while the reports had quietly stopped describing the system -
the same shape as the project's worst defect, where the pipeline ran at
k=3 and the benchmark that justified it was tuned at k=5. So
check_retrieval_depth compares the claim_top_k recorded in
verified_pyq_results.json against the constant the code now imports, and
fails if they differ. See its docstring.

Usage
-----

    python -m src.evaluation.run_all            # run everything, then check
    python -m src.evaluation.run_all --check    # check only, run nothing
    python -m src.evaluation.run_all --only system_b_metrics selective_metrics

--check is the one to run before quoting any number in a document: it
answers "is everything in data/evaluation/ mutually consistent right now"
without spending a single API call.

What is deliberately not in here
--------------------------------

system_c_pipeline is excluded. It reads a frozen archive
(verified_pyq_results.k3_verbatim.json) on purpose, so it does not share
this dependency chain, and it is slow and cached separately. It is run by
hand.

vanilla_rag is excluded for the opposite reason: its outputs
(vanilla_rag_results.json, vanilla_rag_scored.json) are inputs to this
chain but are treated as fixed baseline artifacts. Regenerating the
baseline is a deliberate act, not part of a routine re-run, because every
comparison in the project is against it.

evaluate_generation_loop is excluded because it measures a different flow.
Every stage below measures the verifier *answering* the 13 PYQs; the
generation loop measures it *gating* a generated question, shares no output
file with this chain, and costs roughly 250 API calls per arm across its
two arms. It is named here rather than left undiscoverable, because an
unfindable measurement is very nearly as bad as a missing one:

    python -m src.evaluation.evaluate_generation_loop --format simple
    python -m src.evaluation.evaluate_generation_loop --format statements
    python -m src.evaluation.evaluate_generation_loop --summarise-only
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.retrieval.retrieval_config import RETRIEVAL_TOP_K

VERIFIED = "data/evaluation/verified_pyq_results.json"
REPLAY = "data/evaluation/answer_mapping_replay.json"
DIRECT = "data/evaluation/direct_option_results.json"
VANILLA = "data/evaluation/vanilla_rag_results.json"
VANILLA_SCORED = "data/evaluation/vanilla_rag_scored.json"
CHUNKS = "data/processed/chunks.jsonl"
DATASET = "data/evaluation/pyq_2025_polity.json"

# Topologically ordered. The list order IS the run order; the inputs and
# outputs are what the check uses. Keep these in sync with the *_PATH
# constants in each module.
STAGES = [
    {
        "module": "evaluate_verified_pyqs",
        "reads": [DATASET, CHUNKS],
        "writes": [VERIFIED],
        "note": "the pipeline itself; retrieval + verification, slowest stage",
    },
    {
        "module": "replay_answer_mapping",
        "reads": [VERIFIED],
        "writes": [REPLAY],
        "note": "re-maps stored verdicts through the single mapping module",
    },
    {
        "module": "direct_option_verifier",
        "reads": [VERIFIED, REPLAY],
        "writes": [DIRECT],
        "note": "System B2; reads REPLAY only to print System B's row",
    },
    {
        "module": "system_b_metrics",
        "reads": [VERIFIED, VANILLA, REPLAY, DIRECT, CHUNKS],
        "writes": ["data/evaluation/system_b_metrics.json"],
        "note": "the aggregate report; must be last of the metric stages",
    },
    {
        "module": "selective_metrics",
        "reads": [VERIFIED, VANILLA_SCORED],
        "writes": ["data/evaluation/selective_metrics.json"],
        "note": "A vs B selective prediction comparison",
    },
    {
        "module": "verdict_stability",
        "reads": [VERIFIED, CHUNKS],
        "writes": ["data/evaluation/verdict_stability.json"],
        "note": "resamples the verifier; costs 2 * REPEATS * 37 LLM calls",
    },
    {
        "module": "judge_llm",
        "reads": [VERIFIED, VANILLA],
        "writes": ["data/evaluation/judge_results.json"],
        "note": "LLM-as-a-judge, blind and slot-swapped",
    },
]


def run_stage(stage: dict) -> None:

    module = stage["module"]
    log_path = Path(f"data/evaluation/{module}.log")

    print(f"--- {module}")
    print(f"    {stage['note']}")

    completed = subprocess.run(
        [sys.executable, "-m", f"src.evaluation.{module}"],
        capture_output=True,
        text=True,
    )

    log_path.write_text(
        completed.stdout + completed.stderr,
        encoding="utf-8",
    )

    if completed.returncode != 0:
        # Printed rather than swallowed: a stage that fails silently would
        # leave the next stage reading the previous run's file, which is
        # exactly the failure this module exists to prevent.
        print(completed.stdout[-2000:])
        print(completed.stderr[-2000:])
        raise SystemExit(
            f"{module} failed with exit code {completed.returncode}; "
            f"see {log_path}"
        )

    print(f"    ok -> {log_path}")


def check_freshness(stages: list[dict]) -> list[str]:
    """
    Return a list of staleness complaints; empty means consistent.

    Missing files are reported too. A missing input is a harder error than
    a stale one, so it is named as such.
    """

    problems = []

    for stage in stages:

        module = stage["module"]

        for path in stage["reads"] + stage["writes"]:
            if not Path(path).exists():
                role = (
                    "input"
                    if path in stage["reads"]
                    else "output"
                )
                problems.append(
                    f"{module}: missing {role} {path}"
                )

        readable = [
            path
            for path in stage["reads"]
            if Path(path).exists()
        ]

        for output in stage["writes"]:

            if not Path(output).exists():
                continue

            output_mtime = Path(output).stat().st_mtime

            for source in readable:

                source_mtime = Path(source).stat().st_mtime

                if source_mtime > output_mtime:
                    age = source_mtime - output_mtime
                    problems.append(
                        f"{module}: {output} is {age:.0f}s older than "
                        f"{source} it was derived from - rerun {module}"
                    )

    return problems


def check_retrieval_depth() -> list[str]:
    """
    Return complaints if the stored results were produced at a retrieval
    depth other than the one the code now uses.

    This is a different failure from staleness and mtimes cannot catch it.
    Editing RETRIEVAL_TOP_K invalidates every number in data/evaluation/
    without touching a single file in it, so the reports stay newer than
    their inputs and the freshness check passes while the numbers have
    quietly stopped describing the system. That is the exact shape of the
    project's worst defect - the pipeline ran at k=3 while the retrieval
    benchmark that justified it was tuned at k=5 - so the invariant is
    asserted here rather than left to memory.

    evaluate_verified_pyqs.py records claim_top_k on every question, so
    the depth a run was produced at is recoverable from the artifact
    itself. Costs no API calls.

    verified_pyq_results.k3_verbatim.json is deliberately not checked: it
    is a frozen k=3 archive that system_c_pipeline.py replays as the C0
    arm, and it is supposed to disagree.
    """

    problems = []
    path = Path(VERIFIED)

    if not path.exists():
        # Already reported as a missing output by check_freshness.
        return problems

    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        return [f"retrieval depth: cannot parse {VERIFIED}: {error}"]

    recorded = {
        record.get("claim_top_k")
        for record in records
        if isinstance(record, dict)
    }

    # Three distinct states, worth distinguishing because they call for
    # different actions: no field at all means the file predates the
    # field (the k3_verbatim archive looks like this), while a field
    # present on some records only means two runs got merged.
    if not recorded or recorded == {None}:
        return [
            f"retrieval depth: no question in {VERIFIED} records a "
            f"claim_top_k, so the depth it was produced at cannot be "
            f"verified. Either it predates the field or it is an archive "
            f"that does not belong here - rerun evaluate_verified_pyqs"
        ]

    if None in recorded:
        problems.append(
            f"retrieval depth: some but not all questions in {VERIFIED} "
            f"record a claim_top_k, so the file merges two runs - rerun "
            f"evaluate_verified_pyqs"
        )
        recorded.discard(None)

    if len(recorded) > 1:
        problems.append(
            f"retrieval depth: {VERIFIED} contains more than one depth "
            f"{sorted(recorded)}, so it is a mix of runs - rerun "
            f"evaluate_verified_pyqs"
        )

    stale = sorted(depth for depth in recorded if depth != RETRIEVAL_TOP_K)

    if stale:
        problems.append(
            f"retrieval depth: {VERIFIED} was produced at claim_top_k="
            f"{stale[0] if len(stale) == 1 else stale} but "
            f"RETRIEVAL_TOP_K is now {RETRIEVAL_TOP_K} "
            f"(src/retrieval/retrieval_config.py). Every number derived "
            f"from this file describes a configuration the code no longer "
            f"runs - rerun evaluate_verified_pyqs, or put the depth back"
        )

    return problems


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run the evaluation chain in dependency order and verify "
            "that every report is newer than its inputs."
        )
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="only verify freshness; run no stages and spend no tokens",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="MODULE",
        help=(
            "run just these modules, in the declared order. Freshness is "
            "still checked across the whole chain afterwards, so a "
            "partial run that leaves the chain inconsistent still fails."
        ),
    )
    arguments = parser.parse_args()

    known = [stage["module"] for stage in STAGES]

    if arguments.only:
        unknown = set(arguments.only) - set(known)
        if unknown:
            raise SystemExit(
                f"Unknown module(s): {sorted(unknown)}. "
                f"Known: {known}"
            )
        selected = [
            stage
            for stage in STAGES
            if stage["module"] in arguments.only
        ]
    else:
        selected = STAGES

    if not arguments.check:
        print("=== RUNNING EVALUATION CHAIN ===")
        print(
            f"order: {' -> '.join(s['module'] for s in selected)}"
        )
        print()
        for stage in selected:
            run_stage(stage)
        print()

    print("=== CONSISTENCY CHECK (whole chain) ===")

    problems = check_freshness(STAGES) + check_retrieval_depth()

    if not problems:
        print(
            f"every report is at least as new as its inputs, and the "
            f"stored results were produced at the current retrieval depth "
            f"(claim_top_k={RETRIEVAL_TOP_K}); the numbers in "
            f"data/evaluation/ are mutually consistent"
        )
        return

    for problem in problems:
        print(f"  {problem}")

    raise SystemExit(
        f"\n{len(problems)} consistency problem(s). Do not quote numbers "
        f"from data/evaluation/ until these are resolved."
    )


if __name__ == "__main__":
    main()
