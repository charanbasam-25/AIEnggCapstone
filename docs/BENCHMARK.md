# Mixed Polity and governance regression benchmark

`upsc_polity_v1` adds **75 official UPSC Polity and governance questions** from
2019–2023. The existing 13 questions remain a separate regression set. The
new questions include numbered statements, direct choices, best-answer
questions, and assertion/reason questions.

The capstone is now scoped to static Polity practice. This mixed set remains
useful for regression diagnostics, with its recorded scope and scores preserved.
A perfect 75-question score is not an acceptance criterion. The next evaluation
work prioritises independently reviewed generated static MCQs and a smaller,
source-annotated static verification benchmark. See the
[current next steps](BENCHMARK_NEXT_STEPS.md). Neither new evaluation has been
curated or run yet.

**The latest full source-verifier runs completed on 3 October 2026.** Both full
splits contain no pending questions or service errors. An observed test error
has now informed the implementation, so these fixed splits are regression
sets. Fresh untouched questions are needed for an independent estimate.
Offline tests validate data, routing and acceptance rules separately from
model answer quality.

New evaluations record `source-references-article-context-v2`: reviewers select
numbered source quotations, Python binds those references to exact original
text, and numbered statements receive full pages for explicitly named Articles,
including continuations and footnotes. Each committed model verdict still needs
an agreeing blind review. These changes do not establish missing facts or make
the saved full-split results scores for the current implementation. Development
smoke runs are separate snapshots under **How it works → Evals → Reports**.

The first 7 October five-question development trial selected one correct answer
and abstained on four questions, with no service errors. Manual inspection found
that its correct answer cited unrelated border provisions, despite agreement
between the two reviews. That snapshot is diagnostic evidence of a grounding
failure, not a demonstration of a grounded answer. A subsequent direct-citation
screen rejects a committed judgment when its excerpts share no informative
question/option terms. This is a coarse rejection rule: shared words do not prove
entailment, and synonyms can cause unnecessary abstentions.

The subsequent one-question development run under the revised policy abstained
on that mission question with no service error. Its snapshot is
`development_reference_context_v2_first1_2026_10_07.json`. The 244 offline tests
pass, including rejection of recorded-pattern unrelated citations in both
reviews. A new full 75-question evaluation has not been run.

## Latest source-verifier results: v3 regression runs

| Split | Answered | Correct | Wrong | Abstained | Coverage | Precision when answered | Overall accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Development | 2 / 33 | 2 | 0 | 31 | 6.1% | 100.0% | 6.1% |
| Test | 0 / 42 | 0 | 0 | 42 | 0.0% | — | 0.0% |

These are actual model runs of the source verifier, not a replay or a correction
of earlier predictions. Both use `quoted-evidence-review-v3`, identical source
code/corpus hashes, `gpt-4o-mini` at temperature 0, 20 candidates and top 5 per
query. All four unsupported assertion/reason questions remain in the test
denominator and abstain before model calls. The v3 runs selected only the
verifier; no new vanilla-RAG comparison was run for this revision.

Only **two of 75 questions were answered**. Development precision has a
denominator of two; test precision is undefined because there were no test
attempts. These results do not establish accuracy on arbitrary future
questions or that every abstention was necessary. Valid quotations plus blind
agreement still leave semantic interpretation errors possible. See
[the reliability note](VERIFIER_RELIABILITY.md) for the observed failures and
remaining evidence-fit gaps.

The dashboard reads
[development_results.json](../data/evaluation/benchmark/development_results.json)
and [test_results.json](../data/evaluation/benchmark/test_results.json), which
are byte-identical copies of the preserved
[development_reliability_v3.json](../data/evaluation/benchmark/development_reliability_v3.json)
and [test_reliability_v3.json](../data/evaluation/benchmark/test_reliability_v3.json).
Open **Answer quality** in Evaluation Studio and refresh the reports.

## Frozen v2 comparison, before the test error was inspected

| Split | System | Answered | Correct | Wrong | Abstained |
|---|---|---:|---:|---:|---:|
| Development | Reviewed source verifier | 1 / 33 | 1 | 0 | 32 |
| Development | Vanilla RAG | 33 / 33 | 26 | 7 | 0 |
| Test | Reviewed source verifier | 1 / 42 | 0 | 1 | 41 |
| Test | Vanilla RAG | 42 / 42 | 24 | 18 | 0 |

The unchanged frozen pipeline still selected a wrong test answer despite both
review calls agreeing and quoting actual source text. It confused adoption of
the Constitution with formation of the Drafting Committee. This failure led
to the general paired-format guard. The actual failed results are retained in
[test_reliability_v2.json](../data/evaluation/benchmark/test_reliability_v2.json),
alongside [development_reliability_v2.json](../data/evaluation/benchmark/development_reliability_v2.json).
Temperature 0 did not prevent variation across runs: the development format
guard does not apply to any development question, yet the v3 rerun accepted
two answers instead of one. That difference does not measure a guard effect.

## Original comparison: 2 October 2026

| Split | System | Answered | Correct | Wrong | Abstained | Coverage | Precision when answered | Overall accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Development | Source verifier | 4 / 33 | 3 | 1 | 29 | 12.1% | 75.0% | 9.1% |
| Development | Vanilla RAG | 33 / 33 | 26 | 7 | 0 | 100.0% | 78.8% | 78.8% |
| Test | Source verifier | 2 / 42 | 1 | 1 | 40 | 4.8% | 50.0% | 2.4% |
| Test | Vanilla RAG | 42 / 42 | 24 | 18 | 0 | 100.0% | 57.1% | 57.1% |

These are single-run results scored against the matching official answer keys.
Both runs recorded identical dataset, corpus and source-code hashes, model
settings and retrieval depths: `gpt-4o-mini`, temperature 0, 20 candidates and
top 5 per query. The pipeline was not changed between development and test.

The verifier's high abstention rate is the main finding. An abstention is not
a run error, and the unreviewed corpus-answerability labels do not establish
whether each abstention was necessary. Precision is based on only four
development attempts and two test attempts. On the four development questions
both systems answered, the verifier got three correct and vanilla got four;
on the two shared test attempts, both got one correct. These small subsets
do not establish a general performance advantage.

The original full traces, breakdowns and configuration metadata remain in
[development_original_2026_10_02.json](../data/evaluation/benchmark/development_original_2026_10_02.json)
and [test_original_2026_10_02.json](../data/evaluation/benchmark/test_original_2026_10_02.json).
The existing grounding, faithfulness and stability reports still refer
to the earlier 13-question regression set.

## Fixed splits

| Question type | Development | Test | Total |
|---|---:|---:|---:|
| Numbered statements | 15 | 26 | 41 |
| Direct options, including assertion/reason | 16 | 13 | 29 |
| Best answer | 2 | 3 | 5 |
| **Total** | **33** | **42** | **75** |

2019–2020 questions are for development. 2021–2023 questions are for testing,
except 2023 booklet A Q33 (chief purpose of a Constitution), which was already
used in discussion and a project example and therefore belongs to development.
The split was frozen before model evaluation. Its IDs and membership remain
unchanged. The test failure has now guided a format guard, so the current test
set is explicitly used for regression/development. Collect fresh untouched
questions for independent evaluation after settling subsequent changes. Do
not remove difficult or unsupported questions from the recorded denominators.

## Official sources and label audit

| Exam year | Selected questions | Question paper | Answer key |
|---|---:|---|---|
| 2019 | 14 | [UPSC GS I](https://www.upsc.gov.in/sites/default/files/csp-p1.pdf) | [UPSC key](https://www.upsc.gov.in/sites/default/files/AnsKeyCSP-19-GS_I.pdf) |
| 2020 | 18 | [UPSC GS I](https://www.upsc.gov.in/sites/default/files/CSP_2020_GS_Paper-1.pdf) | [UPSC key](https://www.upsc.gov.in/sites/default/files/AnsKey-CSP-20-Paper-I-091121.pdf) |
| 2021 | 18 | [UPSC GS I](https://www.upsc.gov.in/sites/default/files/QP-CSP-21-GeneralStudiesPaper-I-121021.pdf) | [UPSC key](https://www.upsc.gov.in/sites/default/files/Anskey-CSP-21-GS-I-300522.pdf) |
| 2022 | 11 | [UPSC GS I](https://www.upsc.gov.in/sites/default/files/GENERAL%20STUDIES%20PAPER%20I.pdf) | [UPSC key](https://www.upsc.gov.in/sites/default/files/AnsKey-CSP-2022-Paper-I-040723.pdf) |
| 2023 | 14 | [UPSC GS I](https://www.upsc.gov.in/sites/default/files/QP_CS_Pre_Exam_2023_280523.pdf) | [UPSC key](https://www.upsc.gov.in/sites/default/files/AnsKey-CSP-2023-Paper-I-090524.pdf) |

The downloaded papers are **booklet A**. Every gold answer comes from the
matching **series A**, page 1 of that year's official key. `X` means the
question was dropped. The dropped 2021 Q80 and 2023 Q34 are excluded.
The complete series A answer tables are transcribed in `upsc_sources.json`;
the builder joins answers by **year, booklet series and question number**.
It never asks an LLM to create gold answers.

English pages were transcribed with OCR assistance and scan artifacts cleaned.
Each question retains its source PDF page numbers, original statement IDs and
original options. PDF references are **one-based file pages**. For example,
2019 Q56 is on page 25; OCR misread its number as 66, which was corrected in
the transcription and recorded in its note.

An ordered Arabic statement list (`1.`, `2.`, …) is converted to Roman IDs
(`I.`, `II.`, …), together with references in coded options, for the existing
parser. Facts, article numbers, amounts and dates in the question remain
unchanged. Assertion/reason options compare an explanatory relationship and
use the direct option comparator. The five-item development question required
extending the existing answer mapper's statement token set through `V`.

## Run commands

Run these from the project root. No virtual-environment activation is needed.

```bash
# Validate labels, split integrity, routing, frozen bytes and source PDF hashes.
# No model loading or API calls.
.venv/bin/python -m src.evaluation.evaluate_benchmark --check

# Preview a run. Omitting --run also defaults to this offline mode.
.venv/bin/python -m src.evaluation.evaluate_benchmark --dry-run --split development

# Small setup check using real answering models; calls OpenAI.
.venv/bin/python -m src.evaluation.evaluate_benchmark --run --systems verified --split development --limit 3

# Full development comparison; calls OpenAI.
.venv/bin/python -m src.evaluation.evaluate_benchmark --run --systems verified --split development \
  --output data/evaluation/benchmark/development_reference_context_v2.json

# After completing improvements, evaluate the full test set; calls OpenAI.
.venv/bin/python -m src.evaluation.evaluate_benchmark --run --systems verified --split test \
  --output data/evaluation/benchmark/test_reference_context_v2.json
```

To run a new two-system comparison, use both systems and a new filename:

```bash
.venv/bin/python -m src.evaluation.evaluate_benchmark --run \
  --systems verified vanilla --split development \
  --output data/evaluation/benchmark/development_comparison_reference_context_v2.json
```

The default files currently contain older source-only runs. Their configuration
cannot be resumed with the current implementation or a two-system comparison;
the new output preserves the recorded measurements. Choose another filename
after changing source code, corpus or settings.

The default compares `verified` and `vanilla` on the same selected questions.
To run one arm, add `--systems verified` or `--systems vanilla`. Use a distinct
`--output` file when changing systems or settings. `--limit` is available only
for development; a smoke run is marked `is_full_split: false`.

Full reports are saved to
`data/evaluation/benchmark/development_results.json` and
`data/evaluation/benchmark/test_results.json`. Smoke runs have separate names,
such as `development_first3_results.json`. Old 13-question reports are not
overwritten. A checkpoint is saved after every system/question result.
Repeat the identical command to resume pending work. Add `--retry-errors` to
retry failed calls; completed answers and abstentions stay cached. Dataset,
corpus, source code, model and retrieval settings are recorded. A checkpoint
from different settings cannot be mixed into the current run.

The UI's **How it works → Evals → Answer quality** displays the saved full
development and test results with an individual-question explorer. Named new
snapshots appear under **Reports** and do not replace those historical totals.

## What is measured

The verifier uses strict statement mapping, full-page context, quotation
validation, blind review and a gate for unsupported paired formats. In a
two-system comparison, both share the same corpus, semantic model, reranker,
candidate depth 20, top-5 depth per query, and `gpt-4o-mini` at temperature 0.
The verifier retrieves per claim or combines five direct-question queries;
vanilla RAG retrieves once for the full question and options. This comparison
measures **end-to-end systems**, not the isolated contribution of verification.
Gold answers and answer-key PDFs never enter the answering function. They are
used only after predictions are produced. The downloaded benchmark papers and
keys are not added to the retrieval corpus.

For each system, the report includes counts and metrics overall and by question
type, topic and year:

- Coverage: answered / all selected questions.
- Precision when answered: correct / answered; `null` when none are answered.
- Accuracy overall: correct / all selected questions.
- Error rate overall: wrong / all selected questions.
- Abstentions, service errors and pending questions as separate counts.
- Practice marks: +2 per correct answer, −2/3 per wrong answer.
- A paired comparison restricted to questions both systems answered.

API failures are not evidence-based abstentions. Incomplete/error-containing
runs must be resolved before quoting performance. A high precision from a few
attempts still has a small denominator; the test set contains only three
best-answer questions.

## Limits and the next evidence audit

Official exam answers are the gold labels; corpus answerability and gold
evidence chunks remain **unreviewed**. Do not report retrieval recall or call
an abstention necessary merely because this benchmark records an official
answer. Such claims need a separate manual audit of the Constitution/NCERT
corpus. The current corpus is not an archived corpus for every exam year;
historical statements involving "now" or "recent" need special attention.
Public PYQs may also have appeared in model training data. Evidence-only
prompts reduce reliance on memorization but do not establish its absence.

The benchmark is a selected Polity/governance set across 13 topics, not a random
sample of all GS I questions or a measurement of the full UPSC syllabus.
Topic counts are uneven. Its 41 numbered questions produce 119 claims with
the current builder; 21 still fail the lexical propositional check. That check
is a diagnostic, not a human truth-value annotation or an accuracy result.
Binding and evidence gaps should be investigated using development examples.

To audit evidence later, record whether every claim/option can be settled by
the frozen corpus and cite exact chunk IDs and quotations. Do not infer these
labels from the system's own model verdicts.

## Files

- `data/benchmarks/curated_questions.json`: source transcriptions.
- `data/benchmarks/upsc_sources.json`: official URLs, booklet-matched keys and PDF hashes.
- `data/benchmarks/polity_v1.json`: normalized inputs, labels, provenance and splits.
- `data/benchmarks/polity_v1.lock.json`: dataset/input hashes and frozen split IDs.
- `data/raw/benchmark/`: the ten original official PDFs.
- `src/evaluation/benchmark.py`: validation, input isolation and scoring.
- `src/evaluation/build_benchmark.py`: deterministic rebuild.
- `src/evaluation/evaluate_benchmark.py`: offline checks, execution and resume.

`python -m src.evaluation.build_benchmark` must reproduce the frozen v1 bytes.
Intentional data/split changes should create a new benchmark version with a
new lock and new output files.
