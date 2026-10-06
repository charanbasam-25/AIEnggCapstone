# UPSC Practice — source-grounded Polity MCQs

Choose a static Polity topic and generate UPSC-style practice questions. Attempt the
quiz first, then see the checked answer, an explanation for every option and
quotations from the sources. Geography, History, Economy and Environment are
shown as locked subjects while Polity is available.

The agreed capstone objective is useful static Polity practice with checked
answers, source-supported explanations and measured quality. Evaluation focuses
on delivered MCQs, including errors, ambiguity, grounding, yield, latency and
cost. The mixed 75-question set remains a broader regression diagnostic; a
perfect score on it is not a project acceptance criterion. See the
[current next steps](docs/BENCHMARK_NEXT_STEPS.md).

The sidebar separates practice from the project details:

- **Practice:** generate a set by topic, question style, requested difficulty and size, then attempt the quiz.
- **My Practice:** recent results and a notebook of incorrect/skipped questions.
- **How it works:** problem and scope, **Privacy & PII**, **Guardrails**, **System design**, **Evals**, **Error handling**, and **Cost & latency**.

The home page focuses on questions. Project diagnostics load when opened, and
moving between sidebar pages preserves topic choices and unfinished answers.
Generated questions are verified automatically before release; there is no
separate question-verification screen in the learner UI.
**How it works → Guardrails** explains the five mandatory publication checks,
failure examples, bounded revisions, safe reuse and the limits of automated review.

Generation retrieves source passages **before** drafting. Format validation,
answer verification, question quality and quoted explanation review must all
pass before a question enters a practice set. Resolved false statements are
allowed in statement questions; unresolved statements block publication.
Failed candidates are revised within a fixed limit or withheld. The generator's
draft explanation is replaced with separately written and reviewed notes.

Checked questions are stored locally in `data/practice/practice.sqlite3` and
reused only with the same source hash and generation-policy version. Learner
history is kept in Streamlit server memory for the current session, up to 20 completed sets.
Requested difficulty has not been calibrated against student performance.
Automatic reviews reduce errors; they do not guarantee correctness. Expert
adjudication of generated questions is still needed before a student release.

Open **How it works → Evals → RAG / Retrieval** to inspect new practice searches: their queries,
semantic candidates, reranked passages, expanded source pages, timings and
final cited evidence. Active quiz details unlock after submission. The saved
retrieval benchmark compares semantic and reranked search on annotated source
pages; it measures retrieval hits, not MCQ correctness. Reproduce it with
`.venv/bin/python -m src.evaluation.evaluate_retrieval` (no LLM API calls).

Run the learner UI:

```bash
.venv/bin/python -m streamlit run app.py
```

Restart Streamlit after code changes. File watching is disabled to keep its
module scanner from importing unrelated Transformers image/video dependencies.

See the [project brief](docs/PROJECT_BRIEF.md) for the problem, scope, privacy boundaries,
architecture and tradeoffs, evaluation task contracts, error recovery, cost and latency.
**How it works → System design** includes a downloadable architecture image;
**Evals → Task specification** separates the existing benchmarks from the missing
expert-labeled generated-MCQ evaluation. Cost estimates use recorded tokens and
dated supported prices; missing usage is not presented as zero cost.
See [Practice system design](docs/PRACTICE_DESIGN.md) for the current flow and
publication rules. The results below are **earlier verification experiments**;
they do not measure the new generation workflow's correctness.

## Verification CLI and historical experiments

A retrieval system that answers UPSC Prelims Polity multiple-choice questions
**only when the source material supports an answer**, and says so explicitly
when it does not.
The design decision the whole project rests on: **the language model never
picks the option letter.** For numbered questions, it issues a SUPPORTED / CONTRADICTED /
INSUFFICIENT verdict on one statement at a time, each with a page citation.
Python maps those verdicts to an option. If the verdicts do not determine a
single option, the system abstains rather than guessing.
Direct MCQs use a separate evidence-grounded option comparison. The model
assesses how each option answers the exact question; Python selects a letter
only when one is supported and all alternatives are ruled out. This new path
has not been benchmarked by the 13-question results below.
---
## Historical 13-question verification result
UPSC 2025 Prelims Polity, Q54–Q66 (n = 13).
| | Vanilla RAG | This system |
|---|---|---|
| Questions answered | 13 of 13 | 4 of 13 |
| Correct **when it answered** | **38.5%** | **100%** |
| Wrong answers presented as correct | **8** | **0** |
On the 4 questions both systems attempted, this system got 4 right; vanilla
RAG got 2.
The trade is deliberate and it is the point. Vanilla RAG answers everything
and is wrong 61.5% of the time, with no signal telling you which answers to
distrust. For preparing from a source of truth, eight confident wrong answers
are worse than nine honest refusals.
**Read that 100% as "1.00 on this run", not as a property of the system.**
Re-running the verifier five times at temperature 0 gives coverage between
23.1% and 38.5%, and precision-when-answered of 1.00 in four draws and 0.75
in the fifth — one wrong answer did get through. The honest summary is
*precision 0.95 ± 0.11 over 5 draws, versus vanilla RAG's 0.38*. Temperature
0 is not determinism; 35 of 37 claim verdicts were stable when pinned, 28 of
37 unpinned. Details in [EVALUATION.md §0.1](EVALUATION.md).
## Why the answer is decomposed
The rejected architecture asked the model to judge a whole question in one
call. On Q54 it returned **three of the four options as supported** — it
decided nothing, expensively. The same model, asked about one statement at a
time, produced clean verdicts with page citations.
That comparison is the justification for the architecture, not a
post-hoc rationalisation of it. Full numbers in
[EVALUATION.md §2](EVALUATION.md).
## Retrieval
Historical comparison of five configurations on a 16-query gold benchmark, Recall@5:
| Retriever | Recall@5 |
|---|---|
| BM25 (lexical baseline) | 75.00% |
| Hybrid (RRF) | 81.25% |
| Semantic (`bge-small-en-v1.5`) | 87.50% |
| Parent-child + reranker | 87.50% |
| **Semantic + cross-encoder reranker** | **93.75%** |
The baseline came first and the reranker was added because the baseline's
misses were measured, not assumed. Hybrid RRF and parent–child were both
built and both lost, and were rejected on the measurement rather than kept
because they were more sophisticated.
93.75% is 15/16, so one query is 6.25 points. That is enough to reject hybrid
and parent–child; it is not enough to claim the reranker's margin over bare
semantic is real rather than noise.
`Recall@5` and the k the pipeline queries at are the same constant —
`RETRIEVAL_TOP_K` in `src/retrieval/retrieval_config.py`. They have to be: the
metric is a claim about the consumer's evidence window, so measuring at one
depth and running at another measures nothing.
---
## Run it
```bash
pip install -r requirements.txt
# create .env in the repo root with your key:
#   OPENAI_API_KEY=sk-...
# optional, for the System C ablation's lexical arm:
#   SYSTEM_C_RETRIEVER=bm25
.venv/bin/python -m streamlit run app.py
```
The first verification or generation request loads retrieval models and
embeds the corpus. Later requests reuse cached model and index objects.
Or from the command line:
```bash
# a stored 2025 question
python -m src.verify_cli --pyq 54
# your own question
python -m src.verify_cli --file myquestion.txt --show-evidence
```
Numbered questions use the existing statement-decomposition path:
```
Consider the following statements:
I.  The Governor is appointed by the President.
II. The Governor holds office for a term of five years.
Which of the statements given above is/are correct?
A) I only
B) II only
C) Both I and II
D) Neither I nor II
```
The verification CLI supports direct questions with four options, including
single-fact and "best answer" questions. Use it for offline checks and benchmark
debugging:

```bash
python -m src.verify_cli --file data/examples/direct_constitution_2023.txt --show-evidence
python -m src.verify_cli --file data/examples/direct_governor.txt --save /tmp/governor_result.json
```

The direct path retrieves top-5 evidence for the stem and for each option,
deduplicates the passages, restores complete source pages and compares the
options. A candidate answer needs a second blind evidence review that selects
the same answer with validated quotations. For a "chief purpose" question, a true secondary function can be ruled
out as the answer without being declared false. Missing evidence does not
rule out an option. Unresolved or competing answers cause abstention.
Non-insufficient judgments require quotations that match the checked pages;
this checks provenance, not whether the model's reasoning is correct.
Direct MCQs always use this strict decision rule; the alternative `--policy`
settings apply to numbered-statement questions only. Open-ended questions
without A–D options are not supported, and there is no web-search fallback.
Paired Statement-I/II and Assertion/Reason labels currently produce an explicit
abstention: they need separate statement and explanatory-relationship checks.

## Larger benchmark

The project now includes **75 additional official UPSC Polity/governance PYQs
from 2019–2023**: **33 development** questions and **42 test** questions. The
existing 13 questions remain regression cases. Official answers are matched
by exam year, booklet series and question number, with source PDFs, page
references and a frozen dataset hash. The new set includes both statement
questions and direct/best-answer MCQs. **The latest source-verifier runs
completed on 3 October 2026:** development **2 correct, 0 wrong, 31 abstentions**;
test **0 correct, 0 wrong, 42 abstentions**. Both completed without service errors.
Only two of 75 questions were answered, so zero observed wrong answers does
not establish broad reliability. The test split has now informed debugging;
both fixed splits serve as regression sets. Fresh untouched data is needed
for an independent accuracy estimate. The current source-only reports are in
`data/evaluation/benchmark/development_results.json` and `test_results.json`.
Earlier verifier/vanilla comparisons and the failed reviewed test run are
preserved as separate snapshots. See [the reliability note](docs/VERIFIER_RELIABILITY.md)
for both observed error cases and the acceptance checks.
The performance table above still describes the earlier 13-question run;
see [the benchmark guide](docs/BENCHMARK.md) for the new results and their scope.

```bash
.venv/bin/python -m src.evaluation.evaluate_benchmark --check
.venv/bin/python -m src.evaluation.evaluate_benchmark --dry-run --split development
# These commands call the configured OpenAI model:
.venv/bin/python -m src.evaluation.evaluate_benchmark --run --systems verified --split development
.venv/bin/python -m src.evaluation.evaluate_benchmark --run --systems verified --split test
```

The runner compares the verifier with vanilla RAG, saves resumable checkpoints,
and reports correct/wrong answers, abstentions and API errors separately, with
breakdowns by format, topic and year. The UI exposes development questions and
the new benchmark inventory under **Evals**. See [the benchmark guide](docs/BENCHMARK.md)
for smoke runs, source links, split rules and the pending corpus-evidence audit.

## Evaluation Studio

Open **Practice → Evals** for dedicated views of answer
quality, grounding and faithfulness, question generation, verdict stability,
and saved reports. Inspect individual questions, refresh results, and download
CSV tables or full JSON reports. The dashboard labels the new benchmark and
earlier regression results separately.

You can also run the same dashboard as a separate app:

```bash
.venv/bin/python -m streamlit run evals_app.py --server.port 8502
```

Open `http://localhost:8502`. See [the Evals UI guide](docs/EVALS_UI.md) for report
scope, validation and reproduction commands.

## Reproduce the numbers
```bash
python -m src.evaluation.run_all --check   # verify every report is
                                           # newer than its inputs; 0 API calls
python -m src.evaluation.run_all           # re-run the full chain
```
`--check` exists because the evaluation modules read each other's output
files, and running them out of order does not fail — it silently produces a
document-ready table of wrong numbers. That happened twice. See
`src/evaluation/run_all.py`.
It asserts two different things. The first is ordering: every report must be at
least as new as every file it was derived from. The second is not about file
times at all — editing `RETRIEVAL_TOP_K` in `src/retrieval/retrieval_config.py`
invalidates every number in `data/evaluation/` **without modifying a single file
in it**, so the ordering check passes while the reports have quietly stopped
describing the system. `check_retrieval_depth` reads the `claim_top_k` recorded
in `verified_pyq_results.json` back off disk and fails if it no longer matches
the constant the code imports.
That second check exists because the failure it catches already happened: the
retriever was selected on Recall@5 and the pipeline queried it at k=3. Both
depths now come from one module, so the invariant holds because there is one
value rather than because someone remembered to keep twenty literals in step.
---
## Documentation
| Document | What's in it |
|---|---|
| [DESIGN.md](DESIGN.md) | Problem, data surface, architecture, core flow, comparative evaluation |
| [EVALUATION.md](EVALUATION.md) | Every measurement, with noise bands, threats to validity, and 10 failure analyses |
| [docs/BENCHMARK.md](docs/BENCHMARK.md) | New 75-question dataset, official sources, fixed splits and resumable runner |
| [docs/EVALS_UI.md](docs/EVALS_UI.md) | Dedicated evaluation dashboard, standalone app and report scope |
| [docs/ARCHITECTURE_DIAGRAMS.md](docs/ARCHITECTURE_DIAGRAMS.md) | Current overall architecture, verification, generation, evaluation and deployment diagrams |
| [docs/VERIFIER_RELIABILITY.md](docs/VERIFIER_RELIABILITY.md) | Development error analysis, source quotation validation and full-page context checks |
| [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) | Module-level design and interfaces |
| [docs/retrieval_baseline.md](docs/retrieval_baseline.md) | The retrieval ablation in detail |
**Read EVALUATION.md §0 first if you intend to quote any number from this
repo.** It states which run produced which figure and what the measurement
noise is. At n = 13, one question moves a percentage by 7.7 points, and
verdicts are not fully reproducible even at temperature 0 (35 of 37 stable
when pinned, 28 of 37 unpinned).
## Scope
The contribution is **verification**. A generation loop
(`src/orchestration/`) is included as a second consumer of the same verifier
— it generates a question, then gates it through the same claim
verification before accepting it. It is a stress test on unseen input, not a
second product. Its results are in
[EVALUATION.md](EVALUATION.md).
## Known limitations
- **n = 13.** Small. Every percentage carries a wide interval, stated
  wherever one is quoted.
- **Coverage is 30.8%.** Nine of thirteen abstentions; six are traceable to
  pipeline defects rather than genuine evidence gaps, and are itemised in
  EVALUATION.md §6.
- **One domain, one corpus.** Indian Polity, 1,149 chunks. Nothing here has
  been tested outside it.
