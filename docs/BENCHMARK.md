# Mixed Polity and governance regression benchmark

The frozen benchmark contains 75 official UPSC questions from 2019–2023: 33 development questions and 42 regression-test questions. It includes numbered statements, direct choices, best-answer questions and assertion/reason formats. It is broader than the current static Polity practice corpus and is retained for regression diagnostics, not as a 75/75 product target.

The current source-verifier snapshots are v3 regression runs:

| Split | Answered | Correct | Wrong | Abstained | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|
| Development (33) | 2 | 2 | 0 | 31 | 6.1% | 100.0% |
| Test (42) | 0 | 0 | 0 | 42 | 0.0% | undefined |

Only two questions were answered. A zero-error observed run does not prove future accuracy, and the test split has already informed debugging, so it is not an untouched holdout. A fresh reviewed static set is needed for an independent estimate.

## Protocol

Answering receives the question stem and options without official keys, label-audit notes or evaluation worksheets. Python grades predictions afterwards. Reports record the dataset hash, corpus hash, code fingerprint, model, temperature, split, retrieval depth and service errors. The source verifier uses per-claim or per-option retrieval, full pages, source-reference binding and blind review. Unsupported paired formats abstain before answer mapping.

The question papers and keys are kept outside the retrieval corpus. Current benchmark records use the Constitution and selected NCERT source snapshot; a source gap is not silently filled by web search.

## Reproduce or validate

    .venv/bin/python -m src.evaluation.evaluate_benchmark --check
    .venv/bin/python -m src.evaluation.evaluate_benchmark --dry-run --split development

New model runs require a deliberately new output path:

    .venv/bin/python -m src.evaluation.evaluate_benchmark --run --systems verified --split development --output data/evaluation/benchmark/development_reference_context_v2.json

This command calls the configured OpenAI model. Do not overwrite a preserved report after changing code, corpus, model or protocol. A smoke run with --limit is not a full-split result.

The complete historical snapshots and source links remain under data/evaluation/benchmark. The evidence worksheet in data/evaluation/evidence_review_75.csv is optional diagnostic material and still needs human source review.
