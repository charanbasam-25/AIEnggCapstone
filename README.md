# UPSC Practice — Source-Grounded Polity MCQs

A source-grounded AI tutor for **UPSC Civil Services Preliminary Examination — Indian Polity**.

The system generates and verifies UPSC-style multiple-choice questions using a controlled source corpus. It retrieves evidence **before generation or verification**, validates answers against the retrieved sources, and abstains when the available evidence does not support a unique answer.

The goal is not to make the model answer every question.

The goal is to make it **answer only when the sources support the answer**.

---

## What the Project Does

The application provides a learner-focused practice environment for static Indian Polity.

Users can:

- Select a Polity topic.
- Choose question style, difficulty and question count.
- Attempt a generated quiz.
- Review verified answers.
- Read explanations for every option.
- Inspect source quotations and evidence.
- Review previous practice attempts and incorrect/skipped questions.
- Explore the system's retrieval, verification and evaluation diagnostics.

Other UPSC subjects such as Geography, History, Economy and Environment are shown as locked subjects.

The application is intentionally scoped to **static Indian Polity** rather than current affairs.

---

## Core Objective

The capstone focuses on delivering:

> **Useful static Polity practice with checked answers, source-supported explanations and measurable quality.**

Evaluation considers:

- Answer correctness
- Wrong-answer rate
- Abstention behavior
- Question ambiguity
- Source grounding
- Retrieval quality
- Generation yield
- Latency
- Cost

The mixed 75-question benchmark is primarily a broader regression diagnostic. A perfect score on that benchmark is **not** the project's acceptance criterion.

See [`docs/BENCHMARK_NEXT_STEPS.md`](docs/BENCHMARK_NEXT_STEPS.md) for the current next steps.

---

# Key Design Principle

## The LLM does not choose the final answer

The central design decision is to keep the language model away from the final option-selection step.

For numbered statement questions, the model evaluates each statement independently as:

- `SUPPORTED`
- `CONTRADICTED`
- `INSUFFICIENT`

Each verdict must include a source citation.

Python then maps the verified statement verdicts to the final option.

If the evidence does not uniquely determine an answer, the system **abstains instead of guessing**.

For direct four-option MCQs, the system uses a separate evidence-grounded option-comparison path. Python selects an answer only when the evidence supports one option and rules out the alternatives.

This separation reduces the risk of confident but unsupported answers.

---

# Guardrails

Generated questions are automatically verified before they are released to a learner.

The publication pipeline requires:

1. Source retrieval before drafting.
2. Format validation.
3. Answer verification.
4. Question-quality validation.
5. Quoted-explanation validation.

A generated question is published only when all mandatory checks pass.

Resolved false statements are permitted in statement-based questions. Unresolved statements block publication.

Failed candidates are either revised within a fixed limit or withheld.

The generator's draft explanation is replaced with separately written and reviewed notes.

Automatic verification reduces errors, but it **does not guarantee correctness**. Expert review is still required before releasing generated questions to students.

---

# Architecture

At a high level, the system follows:

```text
                     ┌──────────────────┐
                     │    Learner UI    │
                     │    Streamlit     │
                     └────────┬─────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │ Practice / Generation │
                  │       Pipeline        │
                  └───────────┬───────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │   Source Retrieval    │
                  │ Semantic + Reranker   │
                  └───────────┬───────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │   Question / Claim    │
                  │      Verification     │
                  └───────────┬───────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │ Publication Gate      │
                  │ Pass / Revise / Block │
                  └───────────┬───────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │ Practice Set    │
                     │ + Evidence      │
                     └─────────────────┘
```

The generation pipeline is a second consumer of the same verification infrastructure.

The primary contribution of the project is **verification**, while generation serves as a stress test against unseen questions.

---

# Source Corpus

The system is source-grounded and does not use web search as a fallback.

The corpus contains:

- Constitution of India
- NCERT Polity material
- Official UPSC Preliminary Examination questions
- Official UPSC answer keys

The retrieval pipeline operates only over the configured corpus.

Open-ended questions without A–D options are not supported.

Paired Statement-I/Statement-II and Assertion/Reason questions currently result in an explicit abstention because they require additional relationship-specific verification.

---

# Retrieval

The retrieval pipeline was evaluated using a 16-query gold benchmark.

### Recall@5

| Retriever | Recall@5 |
|---|---:|
| BM25 | 75.00% |
| Hybrid RRF | 81.25% |
| Semantic (`bge-small-en-v1.5`) | 87.50% |
| Parent-child + reranker | 87.50% |
| **Semantic + cross-encoder reranker** | **93.75%** |

The final configuration uses semantic retrieval followed by a cross-encoder reranker.

The more sophisticated alternatives were not retained simply because they were more complex. They were compared experimentally and rejected based on the measured results.

`Recall@5 = 93.75%` corresponds to 15/16 queries, meaning a single query changes the metric by 6.25 percentage points.

Therefore, the result demonstrates a strong observed retrieval configuration, but it is not sufficient to establish a statistically meaningful margin over the bare semantic retriever.

---

# Verification Results

## Historical 13-Question Experiment

The original verification experiment used UPSC 2025 Preliminary Polity questions Q54–Q66.

| Metric | Vanilla RAG | Verified System |
|---|---:|---:|
| Questions answered | 13/13 | 4/13 |
| Correct when answered | 38.5% | **100%** |
| Wrong answers presented as correct | 8 | **0** |

On the four questions both systems attempted:

- Verified system: **4/4 correct**
- Vanilla RAG: **2/4 correct**

The trade-off is deliberate.

Vanilla RAG attempts every question, but in this experiment it produced eight confident wrong answers.

The verified system answered fewer questions and abstained when the evidence was insufficient.

For a source-of-truth learning system, an honest abstention is preferable to a confidently incorrect answer.

### Important qualification

The 100% precision result means **1.00 on this experiment**, not that the system is guaranteed to achieve 100% accuracy.

Five repeated runs at temperature 0 produced:

- Coverage: 23.1%–38.5%
- Precision when answered: 1.00 in four runs
- Precision when answered: 0.75 in one run
- Overall precision: **0.95 ± 0.11**

Temperature 0 also did not produce complete determinism:

- 35/37 claim verdicts were stable when pinned.
- 28/37 were stable when unpinned.

See [`EVALUATION.md`](EVALUATION.md) for the detailed analysis.

---

# Why Verification Is Decomposed

An earlier design asked the model to evaluate the entire MCQ in a single call.

On one test question, the model marked three of four options as supported, effectively failing to identify a unique answer.

The decomposed approach instead evaluates individual statements and produces explicit evidence-backed verdicts.

This makes the verification process:

- More auditable
- Easier to debug
- Easier to cite
- Less dependent on the model's implicit option-selection reasoning

The decomposition was therefore adopted based on an observed failure mode rather than as a purely theoretical architecture choice.

---

# Larger Benchmark

The project includes an additional **75 official UPSC Polity/governance PYQs from 2019–2023**:

- 33 development questions
- 42 test questions

The original 13 questions remain regression cases.

The dataset includes:

- Statement-based questions
- Direct MCQs
- Best-answer questions
- Official answers
- Exam year
- Booklet series
- Question number
- Source references
- Frozen dataset hash

### Latest source-verifier run

Completed on **3 October 2026**:

| Split | Correct | Wrong | Abstentions |
|---|---:|---:|---:|
| Development | 2 | 0 | 31 |
| Test | 0 | 0 | 42 |

Both runs completed without service errors.

Only two of the 75 questions were answered, so the zero observed wrong-answer count **does not establish broad reliability**.

The fixed development and test splits are now used as regression datasets. Fresh untouched data is required for an independent accuracy estimate.

Results are stored in:

```text
data/evaluation/benchmark/development_results.json
data/evaluation/benchmark/test_results.json
```

See [`docs/BENCHMARK.md`](docs/BENCHMARK.md) and [`docs/VERIFIER_RELIABILITY.md`](docs/VERIFIER_RELIABILITY.md).

---

# Practice Application

The learner interface is built with Streamlit.

The sidebar contains:

### Practice

Generate a practice set by:

- Topic
- Question style
- Difficulty
- Number of questions

Then attempt the quiz and review the verified answers.

### My Practice

View:

- Recent results
- Incorrect questions
- Skipped questions
- Practice history

### How it works

Explore:

- Problem and scope
- Privacy & PII
- Guardrails
- System design
- Evaluation
- Error handling
- Cost and latency

Project diagnostics are loaded only when requested.

Moving between sidebar pages preserves topic selections and unfinished answers.

---

# Run the Application

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the repository root:

```bash
OPENAI_API_KEY=sk-...
```

Optional configuration for the lexical retrieval ablation:

```bash
SYSTEM_C_RETRIEVER=bm25
```

Start the learner application:

```bash
.venv/bin/python -m streamlit run app.py
```

Or:

```bash
streamlit run app.py
```

Restart Streamlit after code changes.

File watching is disabled to prevent the module scanner from importing unrelated Transformers image/video dependencies.

---

# Verification CLI

Verify a stored UPSC question:

```bash
python -m src.verify_cli --pyq 54
```

Verify your own question:

```bash
python -m src.verify_cli \
  --file myquestion.txt \
  --show-evidence
```

Example numbered-statement question:

```text
Consider the following statements:

I.  The Governor is appointed by the President.
II. The Governor holds office for a term of five years.

Which of the statements given above is/are correct?

A) I only
B) II only
C) Both I and II
D) Neither I nor II
```

Direct MCQs can also be verified:

```bash
python -m src.verify_cli \
  --file data/examples/direct_constitution_2023.txt \
  --show-evidence
```

Save a verification result:

```bash
python -m src.verify_cli \
  --file data/examples/direct_governor.txt \
  --save /tmp/governor_result.json
```

The direct path:

1. Retrieves top-5 evidence for the question stem.
2. Retrieves evidence for each option.
3. Deduplicates passages.
4. Restores complete source pages.
5. Compares the options against the evidence.
6. Performs a second blind evidence review.
7. Validates supporting quotations.
8. Returns an answer only when the evidence supports a unique choice.

Missing evidence does not automatically make an option false.

Competing or unresolved answers result in abstention.

---

# Evaluation Studio

The project includes a dedicated evaluation dashboard.

Start it with:

```bash
.venv/bin/python -m streamlit run evals_app.py --server.port 8502
```

Open:

```text
http://localhost:8502
```

The Evaluation Studio provides views for:

- Answer quality
- Grounding
- Faithfulness
- Question generation
- Verdict stability
- Saved evaluation reports
- Individual question inspection
- Retrieval diagnostics

Reports can be downloaded as CSV or JSON.

The dashboard keeps new benchmark results separate from historical regression results.

---

# Reproduce the Evaluation

Run the full evaluation pipeline:

```bash
python -m src.evaluation.run_all --check
```

The `--check` mode validates that reports are consistent with their inputs without making API calls.

Run the full evaluation chain:

```bash
python -m src.evaluation.run_all
```

For the larger benchmark:

```bash
.venv/bin/python -m src.evaluation.evaluate_benchmark --check

.venv/bin/python -m src.evaluation.evaluate_benchmark \
  --dry-run \
  --split development

.venv/bin/python -m src.evaluation.evaluate_benchmark \
  --run \
  --systems verified \
  --split development

.venv/bin/python -m src.evaluation.evaluate_benchmark \
  --run \
  --systems verified \
  --split test
```

The benchmark runner:

- Compares the verifier with vanilla RAG.
- Saves resumable checkpoints.
- Separates correct answers, wrong answers, abstentions and API errors.
- Breaks results down by format, topic and year.

---

# Evaluation Integrity

The evaluation pipeline contains checks to prevent stale or incompatible results from being presented as current.

Reports are checked against:

- Their source files
- Their generation timestamps
- The retrieval configuration
- The recorded retrieval depth

For example, changing `RETRIEVAL_TOP_K` invalidates previously generated evaluation numbers even if none of the evaluation files themselves changed.

The pipeline therefore verifies that the recorded `claim_top_k` matches the current retrieval configuration.

This protects against a subtle but important failure mode:

> Measuring retrieval at one evidence depth while running the production pipeline at another.

---

# Documentation

| Document | Purpose |
|---|---|
| [`DESIGN.md`](DESIGN.md) | Problem, data surface, architecture and comparative evaluation |
| [`EVALUATION.md`](EVALUATION.md) | Measurements, uncertainty, threats to validity and failure analysis |
| [`docs/BENCHMARK.md`](docs/BENCHMARK.md) | 75-question dataset, sources, splits and benchmark runner |
| [`docs/EVALS_UI.md`](docs/EVALS_UI.md) | Evaluation dashboard and report scope |
| [`docs/ARCHITECTURE_DIAGRAMS.md`](docs/ARCHITECTURE_DIAGRAMS.md) | Architecture, verification, generation, evaluation and deployment diagrams |
| [`docs/VERIFIER_RELIABILITY.md`](docs/VERIFIER_RELIABILITY.md) | Error analysis and source quotation validation |
| [`docs/SYSTEM_DESIGN.md`](docs/SYSTEM_DESIGN.md) | Module-level design and interfaces |
| [`docs/retrieval_baseline.md`](docs/retrieval_baseline.md) | Retrieval ablation experiments |
| [`docs/PRACTICE_DESIGN.md`](docs/PRACTICE_DESIGN.md) | Practice generation flow and publication rules |

**Before quoting any evaluation number, read `EVALUATION.md §0`.**

At `n = 13`, a single question changes a percentage by approximately 7.7 percentage points. Verdicts are also not completely reproducible even at temperature 0.

---

# Scope

The primary contribution of this project is **source-grounded verification**.

The generation loop in `src/orchestration/` is an additional consumer of the same verifier:

```text
Generate question
       ↓
Extract claims
       ↓
Retrieve evidence
       ↓
Verify claims
       ↓
Accept / Revise / Reject
```

The generation pipeline is therefore treated as a stress test of the verification architecture rather than as a separate product.

---

# Known Limitations

### Small historical evaluation

The original comparison contains only 13 questions.

Percentages therefore have wide uncertainty.

### High abstention rate

The historical system answered only 4/13 questions.

In the larger benchmark, it answered only 2/75 questions.

Some abstentions are genuine evidence gaps, while others are caused by pipeline defects.

### Limited domain

The system has been evaluated only on:

> **Indian Polity**

The corpus currently contains approximately 1,149 chunks.

Performance outside this domain has not been established.

### Generated-question evaluation

Automatic verification reduces the risk of bad generated questions, but it does not replace expert adjudication.

A human review stage is still required before student-facing release.

### Difficulty calibration

Requested question difficulty has not yet been calibrated against actual learner performance.

### Retrieval benchmark size

The retrieval benchmark contains only 16 annotated queries.

The 93.75% Recall@5 result should therefore be interpreted as an observed benchmark result rather than a definitive statistical claim.

---

# Project Philosophy

This project deliberately prioritizes:

**Evidence over confidence.**

**Abstention over unsupported answers.**

**Measurement over architectural complexity.**

**Reproducibility over impressive-looking metrics.**

A RAG system that answers every question is not necessarily a reliable educational system.

For this project, the important question is:

> **Can the system determine when its source material actually supports an answer — and refuse to invent one when it does not?**