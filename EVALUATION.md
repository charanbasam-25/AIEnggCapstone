# Evaluation record

This file records the historical measurements that shaped the current design. It is deliberately separate from the learner product. New generated-question accuracy is not measured by the workflow's own PASS labels.

## Current report inventory

| Task | Artifact | What it answers |
|---|---|---|
| Mixed official benchmark | data/evaluation/benchmark | Regression behaviour on 33 development and 42 test questions |
| Retrieval | data/evaluation/retrieval_results.json | Whether an annotated source page appears in the top five |
| Legacy selective verification | data/evaluation/system_b_metrics.json and related files | Coverage, precision, abstention and historical grounding |
| Generation gate | data/evaluation/generation_loop_results*.json | Candidate acceptance, revisions and blocking gates |
| Practice runs | data/practice/practice.sqlite3 | Actual request telemetry, retrieval traces and accepted records |

## Mixed benchmark

The preserved v3 source-verifier snapshots contain:

| Split | Questions | Correct | Wrong | Abstained | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|
| Development | 33 | 2 | 0 | 31 | 6.1% | 100.0% |
| Test | 42 | 0 | 0 | 42 | 0.0% | undefined |

These are single saved regression runs. Two attempts are too few to establish broad accuracy, and zero observed wrong answers does not mean future answers cannot be wrong. The test split has informed debugging and is not an untouched holdout.

## Retrieval

The current 16-query page benchmark records 14/16 hits for semantic search and 14/16 for semantic plus cross-encoder reranking. The older comparison recorded BM25 12/16, hybrid RRF 13/16, semantic 14/16 and parent-child 14/16. These values measure page retrieval only; they do not measure answer correctness or entailment.

## Legacy 13-question selective experiment

The earlier UPSC 2025 Q54–Q66 regression contains 37 numbered claims. Its saved strict-policy comparison recorded high abstention and lower coverage than vanilla RAG. The archived artefacts include the exact model traces, label audit, grounding checks, judge scores, mapping-policy replay and stability repeats. Read those JSON files when quoting a number; do not reconstruct results from this summary.

The main lessons were:

- reporting accuracy without coverage hides selective behaviour;
- an exact quotation proves provenance, not that the quotation proves the proposition;
- blind reviews can share model errors;
- a ranked top-five search cannot settle a universal absence claim;
- claim construction and deterministic option mapping make failures inspectable;
- model temperature 0 improves repeatability but does not guarantee determinism.

The label audit disputed one stored answer (Q58) and preserved stored and audited scoring separately. Several historical reports are intentionally retained even when they exposed pipeline defects; deleting them would hide the instrument's limits.

## Generation gate experiment

The saved 15-topic simple-format experiment accepted 8 candidates after 34 attempts; the first attempt was defective for 11 of 15 topics. The statements-format experiment accepted 0 after 45 attempts and recorded 22 CONTRADICTED, 22 INSUFFICIENT and 105 SUPPORTED claim verdicts. These are gate outcomes, not expert correctness. A zero acceptance rate can indicate a poorly calibrated generator as well as a useful safety gate.

The current workflow uses source-grounded practice policy v4 and has not been given an independent expert-labelled generated-MCQ benchmark. The next evaluation must label key correctness, ambiguity, distractors, every option explanation, source support, topic fit and expected withholding.

## Metrics

For N questions, C correct and W wrong:

- coverage = (C + W) / N;
- precision when answered = C / (C + W), undefined when there are no answers;
- overall accuracy = C / N;
- overall error rate = W / N;
- abstentions and service errors are reported separately.

Gate yield is accepted candidates divided by requested slots. Cost per accepted item includes rejected and revised work. Latency is observed preparation time, not a service-level promise.

## Reproducibility

    .venv/bin/python -m pytest
    .venv/bin/python -m src.evaluation.evaluate_benchmark --check
    .venv/bin/python -m src.evaluation.run_all --check

Model runs require an explicit output file and provider key. Reports record source, dataset, code and protocol metadata. The current source files, frozen reports and historical development log remain the evidence; this document does not upgrade an experiment into a guarantee.
