# Evaluation Studio

The practice app's **How it works → Evals** view and `evals_app.py` share the same dashboard.
Both read the project's saved reports. Viewing, filtering, refreshing and
downloading reports do not initialize answering models or call an API.

Run the tutor from the project root:

```bash
.venv/bin/python -m streamlit run app.py
```

Open **How it works** in the sidebar, then choose **Evals**. The default
**About this site** view explains the sources and question checks. **Guardrails**
describes the five required publication gates, their enforcement, failure
examples and additional practice safeguards. It is a read-only description of
implemented rules, rather than a new evaluation run or correctness score.
**Privacy & PII**, **System design** and **Error handling** explain data boundaries,
the current architecture/tradeoffs and bounded recovery. **Cost & latency**
shows recorded preparation work and dated model-cost estimates. Practice and
My Practice have their own sidebar pages, so technical metrics stay outside
the quiz. To open evaluations as a separate app:

```bash
.venv/bin/python -m streamlit run evals_app.py --server.port 8502
```

The standalone app is available at `http://localhost:8502` while that command runs.

## Dashboard sections

| Section | What it shows |
|---|---|
| Overview | Benchmark composition, report availability, source links and metric definitions |
| Task specification | Product and evaluation inputs/outputs, scoring units, gold references, metric definitions and the generated-MCQ expert-evaluation gap |
| Answer quality | Accuracy, precision, coverage, correct/wrong/abstained counts, system comparison and individual question traces |
| RAG / Retrieval | Actual practice queries, semantic candidates, reranked selections, expanded source context, timing, outcomes and a separate annotated source-page benchmark |
| Grounding & faithfulness | Page citation coverage, verdict counts, model-judge scores and question-level feedback |
| Generation | Acceptance and repair by format, blocking gates and the revision history for each topic |
| Stability | Repeated-verdict consistency and claims whose verdicts changed |
| Reports | Save dates, CSV/JSON downloads, saved snapshots and reproducibility commands |

## Result scope

The new 75-question benchmark has 33 development and 42 test questions. Its
reports are read from `data/evaluation/benchmark/development_results.json` and
`test_results.json`. Other JSON snapshots in that directory, including smoke
runs, remain available under **Reports**.

The current defaults are the 3 October v3 source-only regression runs:
development **2 correct, 0 wrong, 31 abstentions**; test **0 correct, 0 wrong,
42 abstentions**. Test precision is undefined because no answers were attempted.
The failed v2 test run remains in its own snapshot. The test split has informed
debugging, so its current scores are regression results; the dashboard displays
that scope. Earlier verifier/vanilla comparisons remain separate snapshots.

The answer-quality view labels scores from an earlier verification pipeline.
New runs use quotation references, named Article context and a coarse screen for
wholly unrelated direct-answer citations. The 7 October first-five development
snapshot exposed a correct prediction with unrelated quotations; inspect the
manual grounding note in that report. Smoke snapshots do not establish full
benchmark accuracy or expert-reviewed generated-question quality.

Full benchmark comparisons check the recorded dataset and corpus hashes, split,
question IDs and metric totals. An incomplete or error-containing full-split
run is labelled as preliminary. A smoke run does not replace a full-split score.
Missing or unreadable reports show a status message instead of invented scores.
**Refresh reports** reads files again, including files created while the app
was open.

The earlier 13-question regression results have their own selector. Grounding,
faithfulness and stability currently refer to that regression set; generation
reports refer to the generation workflow. The UI does not attribute those
measurements to the new benchmark.

Citation coverage checks whether a verdict cites pages. It does not measure
entailment. Faithfulness scores are model assessments. Generation acceptance
records the workflow's own gate decisions, and observed stability does not
establish correctness.

Current source-grounded generation reports require matching corpus and
generation-policy metadata. Earlier generation-loop reports are labelled
historical. **How it works → Cost & latency** shows measured preparation runs,
policy versions and supported model-cost estimates; these are not correctness scores.

## Producing reports

The **Reports** section includes an offline benchmark-validation button and
commands for the existing evaluation harnesses. To produce a new development
comparison:

```bash
.venv/bin/python -m src.evaluation.evaluate_benchmark --run --split development
```

That command calls the configured OpenAI model. After it finishes, refresh the
dashboard. The UI never starts a model evaluation automatically.

## RAG and retrieval diagnostics

New practice generation runs store the actual semantic candidate pool and
cross-encoder ranking for every search, associated with the preparation slot,
draft and workflow stage. The view shows the top 20 candidates and top five
selections (or the actual smaller counts), their scores, source excerpts and
expanded evidence pages. Accepted question records link to the reviewed key
and quotations used in learner explanations. Details for an active quiz unlock
after submission, including older runs containing the same bank question.

Reuse makes no new search or model call. Earlier runs without instrumentation
are labelled as lacking a trace; their rankings are not reconstructed from
final evidence. Viewing the tab only reads the database and report files.

Save a fresh comparison of semantic search and semantic plus cross-encoder:

```bash
.venv/bin/python -m src.evaluation.evaluate_retrieval
```

This uses local retrieval models, with no LLM API calls. It saves
`data/evaluation/retrieval_results.json`, including corpus/query hashes,
depths, model names, timings, candidates and per-query outcomes. The 16 query
labels identify expected source pages. The legacy name Recall@5 means the
fraction of queries with at least one annotated page in the top five; it does
not measure answer correctness or exhaustive passage relevance.

The semantic comparison reuses the same first-stage candidate search, so its
duration is the semantic stage only; reranked duration includes both stages.
Initial model loading is recorded separately. Changed sources, labels or
depths, partial runs and inconsistent metrics cannot appear as a current full
benchmark. `--limit 1` saves a separate smoke report under Reports.
