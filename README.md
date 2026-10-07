# UPSC Practice — Source-Grounded Polity MCQs

> **An evidence-first RAG system that generates UPSC Prelims Polity MCQs and refuses to publish questions it cannot prove from its source corpus.**

UPSC Practice is a local web application for generating practice questions on **static Indian Polity**.

The important part is not that an LLM can write an MCQ. It can.

The important part is that **every generated question is treated as an untrusted candidate** and must pass independent verification before it is shown to a learner.

If the system cannot establish a defensible answer from the source corpus, it **abstains**.

That is a deliberate product decision, not a failure.

---

## Why this exists

A language model can easily generate a plausible UPSC-style question. Plausibility, however, is not correctness.

A generated question can fail in several ways:

- The stem can invent a constitutional rule.
- The proposed answer can be incorrect.
- Two options can be defensible.
- An explanation can cite a real Article without actually proving the claim.
- A question can appear authoritative even though the underlying evidence is insufficient.

For an exam-preparation tool, confidently presenting such a question is dangerous because learners may memorise the error.

UPSC Practice therefore follows a stricter rule:

> **No evidence → no publication.**

Every generated MCQ goes through retrieval, claim verification, answer derivation, quality review, and explanation verification before it reaches the learner.

---

## What the application does

### In scope

- Static Indian Polity
- Constitution of India
- Selected NCERT Grade 7 Polity content
- 13 Polity topics
- 1, 5, or 10 questions per quiz
- Three question styles
- Three difficulty levels
- Four-option MCQs
- Per-option explanations after submission
- Source-backed evidence for generated questions
- Offline evaluation and regression testing

### Deliberately out of scope

- **Current affairs** — the corpus is a fixed snapshot.
- **Web search** — the application has no internet fallback for answering questions.
- **Other UPSC subjects** — Geography, History, Economy, and Environment are currently locked/forthcoming.
- **Free-form question answering** — this is a practice-question generator, not a general-purpose answer engine.

The system is intentionally constrained so that the model cannot silently escape the evidence boundary.

---

# Core architecture

The system is implemented as a **fixed eight-stage LangGraph workflow**.

It is not an autonomous agent that can decide to browse the web, invent tools, or change the workflow.

```text
                    ┌─────────────────────┐
                    │   Learner Request   │
                    │ topic/style/diff.   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  1. Source Retrieval │
                    │ mandatory RAG        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ 2. Candidate MCQ     │
                    │ structured generation│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ 3. Claim Extraction │
                    │ atomic propositions  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ 4. Claim Verification│
                    │ support / contradict │
                    │ / insufficient       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ 5. Answer Derivation│
                    │ Python, not the LLM  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ 6. Quality Audit     │
                    │ ambiguity/distractor │
                    │/topic checks         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ 7. Explanation       │
                    │ generation + review  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ 8. Publication Gate │
                    │ publish / revise /  │
                    │ abstain              │
                    └─────────────────────┘
```

See [`docs/SYSTEM_DESIGN.md`](docs/SYSTEM_DESIGN.md) for the detailed implementation.

---

# How one question is produced

## 1. Retrieve evidence before generation

The system first searches the local corpus for relevant passages.

Generation cannot begin without retrieved evidence.

This prevents the model from being asked to generate a question purely from its pretrained knowledge.

```text
Learner request
      ↓
Semantic retrieval
      ↓
Reranking
      ↓
Top evidence
      ↓
Only then → LLM generation
```

---

## 2. Generate an untrusted candidate

The model receives the retrieved passages and produces a structured MCQ containing:

- question stem
- four options
- proposed answer
- rationale

The candidate is **not trusted**.

It is simply the first draft.

---

## 3. Break the question into claims

Statement-based UPSC questions often contain multiple propositions.

For example:

> Which of the following statements are correct?

The system converts the statements into standalone claims that can independently be checked.

Pronouns and shared subjects are resolved so that every proposition can be evaluated without relying on surrounding context.

---

## 4. Verify every claim

Each claim is independently retrieved against the corpus.

A verifier assigns one of three verdicts:

```text
SUPPORTED
CONTRADICTED
INSUFFICIENT
```

The verifier must also provide evidence.

Python then checks that the returned quotation actually exists inside the cited source passage.

This creates a distinction between:

```text
LLM says → "This is supported."
```

and:

```text
LLM says → "This is supported."
Python verifies → "The quoted evidence actually exists."
```

---

## 5. Python derives the answer

The model does **not** get the final authority to choose the answer key.

Suppose the four claims resolve to:

```text
Statement 1 → SUPPORTED
Statement 2 → CONTRADICTED
Statement 3 → SUPPORTED
```

The system converts that truth pattern into the corresponding option using ordinary Python logic.

If any claim is:

```text
INSUFFICIENT
```

the answer remains unresolved.

The question cannot be published.

This is one of the central design decisions of the project:

> **Use the LLM for language; use deterministic code for decisions that can be deterministic.**

---

## 6. Audit the question itself

A separate quality reviewer checks:

- Is the question clear?
- Is exactly one option defensible?
- Are the distractors plausible but wrong?
- Is the question actually about the requested topic?
- Does the wording introduce ambiguity?
- Does the question satisfy the requested style and difficulty?

---

## 7. Generate explanations independently

After the question has passed answer verification, a fresh writer generates:

- a concise explanation of the correct answer
- an explanation for each option

A second reviewer checks every explanation against its cited evidence.

This prevents the explanation from simply inheriting unsupported reasoning from the earlier generation stage.

---

## 8. Publish, revise, or abstain

There are five required gates:

```text
sources
   ↓
format
   ↓
answer
   ↓
quality
   ↓
explanations
```

All five must pass.

Otherwise the candidate is:

1. revised, up to two times; or
2. withheld.

Every revision discards the previous gate results.

A revised question therefore cannot accidentally combine:

```text
old question
+
new evidence
+
old verification
```

The complete candidate is re-evaluated.

---

# Abstention is a feature

If the learner requests five questions and the system can only prove three, it may return:

> **3 questions**

rather than inventing two more.

This is intentional.

The system distinguishes between:

- successful generation
- verified publication
- abstention
- crashes
- timeouts

An abstention means:

> **The system could not establish enough evidence to safely publish the question.**

It does not mean the application failed.

---

# RAG pipeline

The active corpus is stored in:

```text
data/processed/chunks.jsonl
```

Current corpus:

| Source | Chunks |
|---|---:|
| Constitution of India | 1,107 |
| NCERT Grade 7 — selected chapter | 42 |
| **Total** | **1,149** |

Chunks are approximately:

- **1,000 characters**
- **150 characters overlap**

Each chunk contains metadata identifying:

- source
- document
- page
- chunk index

### Retrieval

The retrieval pipeline is:

```text
Query
  │
  ▼
Embedding search
  │
  ▼
20 candidate chunks
  │
  ▼
Cross-encoder reranking
  │
  ▼
Top 5 chunks
  │
  ▼
Page-aware context restoration
  │
  ▼
LLM
```

The page-aware context step restores the surrounding page so that important provisos, qualifications, and footnotes are less likely to disappear because of chunk boundaries.

---

# Embeddings and reranking

### Embeddings

```text
BAAI/bge-small-en-v1.5
```

The corpus is embedded into a normalised NumPy index.

### Reranker

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The retriever initially finds 20 candidates.

The reranker reads the query and each candidate together and selects the strongest five for downstream reasoning.

---

# Source freshness

Every saved question records a SHA-256 hash of the active corpus.

This means a previously generated question is reusable only when its source corpus still matches.

Conceptually:

```text
Question
   │
   ├── topic
   ├── style
   ├── difficulty
   ├── policy
   └── corpus SHA-256
```

Changing the corpus automatically makes questions generated against the previous corpus ineligible for reuse.

### Known issue

The current hash is calculated over raw bytes.

On a Windows checkout with CRLF line endings, `chunks.jsonl` can therefore produce a different hash from the LF-normalised value stored in evaluation reports, even when the logical content is identical.

The corpus content is unaffected; the freshness comparison is byte-sensitive.

---

# Technology

| Concern | Technology |
|---|---|
| UI | Streamlit |
| Workflow | LangGraph |
| Typed contracts | Pydantic |
| LLM | OpenAI Responses API |
| Model | `gpt-4o-mini` |
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| PDF extraction | PyMuPDF |
| Storage | SQLite |
| Tests | pytest |
| Evaluation | Custom offline evaluation pipeline |

The LangGraph workflow is deliberately **fixed** rather than autonomous.

The model cannot:

- browse the internet
- add an arbitrary workflow step
- bypass verification
- publish an unresolved question

---

# Evaluation

The project treats evaluation as a first-class component rather than relying on examples that "look correct."

## Retrieval

On a manually annotated set of 16 queries:

| Retrieval configuration | Recall@5 |
|---|---:|
| Semantic search | 14/16 — **87.5%** |
| Semantic + reranking | 14/16 — **87.5%** |

The reranker therefore produced **no measured Recall@5 improvement on this benchmark**.

Both approaches missed the same two queries:

- q06 — Parliament, p.67
- q09 — High Courts, p.130

The reranker still ships because it is relatively inexpensive and the benchmark is too small to establish that it provides no value in general.

At 16 queries:

> **One query = 6.25 percentage points.**

Small differences should therefore not be overinterpreted.

---

## Source-verifier snapshot

### Development split

```text
33 examples

Correct:     2
Wrong:       0
Abstained:  31
```

### Test split

```text
42 examples

Correct:     0
Wrong:       0
Abstained:  42
```

These numbers primarily demonstrate **selective answering**.

The system is much more willing to abstain than to publish an unsupported answer.

The very high abstention rate also means that the correctness figures are based on very few answered examples.

---

## Vanilla RAG baseline

A 13-question baseline produced:

```text
Correct:     5
Incorrect:   8
Abstained:   0

Accuracy: 38.46%
```

This baseline illustrates the central motivation for the verification pipeline:

> Retrieval followed directly by generation is not enough for a high-stakes practice-question workflow.

---

## Test suite

The repository currently contains:

```text
137 pytest test functions
```

These tests run without model calls.

```bash
.venv/bin/python -m pytest
```

---

# What these numbers do NOT prove

The project deliberately avoids claiming more than the evidence supports.

### There is no expert-labelled MCQ accuracy benchmark

This is the largest open evaluation gap.

Gate pass rates tell us whether the workflow accepted a question.

They do **not** tell us whether a UPSC examiner would judge the question as correct, unambiguous, and appropriately difficult.

### Provenance is not entailment

A genuine quotation proves that the quotation exists in the source.

It does not automatically prove that the quotation logically entails the generated claim.

### Model reviewers can share blind spots

Several verification stages use the same model family.

Independent prompts and stages reduce some correlated failures, but they do not make the system epistemically independent.

### Literal absence is not conceptual absence

The system can detect that a literal term does not appear in the corpus.

That does not prove that the underlying concept is absent.

The result depends on:

- corpus completeness
- PDF extraction quality
- wording
- search behaviour

### The system does not claim zero factual errors

The goal is not:

> "The model can never be wrong."

The goal is:

> **Make unsupported generation difficult to publish, make evidence inspectable, and abstain when the available evidence is insufficient.**

---

# Caching

Once a question has passed all verification gates, it can be reused for the same:

- topic
- style
- difficulty
- corpus
- policy

This avoids unnecessary model calls when the same practice request is made again.

---

# Privacy

The application is designed as a local practice tool.

Learner attempts are stored only for the current session.

No attempt history is written to disk.

The application does not require an account or external learner profile.

---

# Project structure

```text
.
├── app.py
├── evals_app.py
│
├── data/
│   ├── processed/
│   │   └── chunks.jsonl
│   └── evaluation/
│
├── docs/
│   ├── SYSTEM_DESIGN.md
│   ├── ARCHITECTURE_DIAGRAMS.md
│   ├── PRACTICE_DESIGN.md
│   ├── PROJECT_BRIEF.md
│   └── diagrams/
│
├── src/
│   ├── ingestion/
│   ├── retrieval/
│   ├── generation/
│   ├── orchestration/
│   ├── verification/
│   ├── practice/
│   ├── ui/
│   └── evaluation/
│
├── tests/
│
├── DESIGN.md
├── EVALUATION.md
├── README.md
└── requirements.txt
```

### Module responsibilities

| Directory | Responsibility |
|---|---|
| `ingestion/` | PDF extraction and chunking |
| `retrieval/` | Semantic, BM25, hybrid, parent-child and reranking |
| `generation/` | Candidate MCQ generation |
| `orchestration/` | LangGraph workflow and telemetry |
| `verification/` | Claim parsing, fact checking, quality and explanation verification |
| `practice/` | Request contracts, service, SQLite storage and cost tracking |
| `ui/` | Streamlit application pages |
| `evaluation/` | Benchmarks, baselines, graders and offline checks |

---

# Quick start

## Requirements

- Python 3
- OpenAI API key

## macOS / Linux

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

## Windows

```powershell
py -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app.py
```

Create `.env` in the project root:

```env
OPENAI_API_KEY=sk-...
```

Never commit `.env`.

The application will be available at:

```text
http://localhost:8501
```

### First run

The embedding and reranking models are downloaded and loaded locally on first use.

The initial startup can therefore take tens of seconds.

Subsequent runs reuse the cached models, and loaded models remain cached for the lifetime of the application process.

---

# Evaluation dashboard

The offline evaluation dashboard runs separately:

```bash
.venv/bin/python -m streamlit run evals_app.py --server.port 8502
```

Open:

```text
http://localhost:8502
```

---

# Offline checks

These commands do not call a model and do not require an API key.

### Test suite

```bash
.venv/bin/python -m pytest
```

### Benchmark validation

```bash
.venv/bin/python -m src.evaluation.evaluate_benchmark --check
```

### Full offline evaluation

```bash
.venv/bin/python -m src.evaluation.run_all --check
```

### Rebuild architecture diagrams

```bash
.venv/bin/python docs/diagrams/render_diagrams.py
```

---

# Design principles

The project is built around a few simple rules.

### 1. Evidence before generation

The model should not start writing before relevant source material has been retrieved.

### 2. Generation is untrusted

A generated question is a candidate, not an answer.

### 3. Claims are smaller than questions

Complex questions are decomposed into independently verifiable propositions.

### 4. Deterministic logic beats model judgement when possible

The LLM does not choose the final answer when Python can derive it from verified truth values.

### 5. Provenance must be checkable

Returned evidence is checked against the actual retrieved passage.

### 6. Verification must be independent

The explanation writer does not simply inherit the previous reasoning.

### 7. Abstention is better than unsupported confidence

If evidence is insufficient, the system does not fill the gap with model knowledge.

### 8. Evaluation must include failure

Abstentions, incorrect answers, retrieval misses, and crashes are all measurable outcomes.

---

# What makes this different from a typical RAG chatbot?

A conventional RAG application often looks like:

```text
Question
   ↓
Retrieve documents
   ↓
LLM
   ↓
Answer
```

This project instead treats generation itself as an object that requires verification:

```text
Request
   ↓
Retrieve
   ↓
Generate candidate
   ↓
Extract claims
   ↓
Verify claims
   ↓
Derive answer
   ↓
Audit question
   ↓
Generate explanations
   ↓
Verify explanations
   ↓
Publish OR abstain
```

The system is therefore less concerned with:

> **"Can the LLM generate something plausible?"**

and more concerned with:

> **"Can the system establish enough evidence to safely publish what the LLM generated?"**

---

# Further reading

| Topic | Document |
|---|---|
| Design decisions and rejected alternatives | [`DESIGN.md`](DESIGN.md) |
| Detailed architecture and failure handling | [`docs/SYSTEM_DESIGN.md`](docs/SYSTEM_DESIGN.md) |
| Architecture diagrams | [`docs/ARCHITECTURE_DIAGRAMS.md`](docs/ARCHITECTURE_DIAGRAMS.md) |
| Learner-facing behaviour | [`docs/PRACTICE_DESIGN.md`](docs/PRACTICE_DESIGN.md) |
| Evaluation methodology and results | [`EVALUATION.md`](EVALUATION.md) |
| Complete project map | [`docs/PROJECT_BRIEF.md`](docs/PROJECT_BRIEF.md) |

---

# Project status

This is an experimental AI engineering project focused on **source-grounded generation, verification, abstention, and evaluation**.

It should not be interpreted as an authoritative UPSC preparation source.

The corpus is limited, the evaluation datasets are small, and the system has not yet been evaluated against a sufficiently large expert-labelled benchmark.

The most important future improvement is therefore not another generation feature.

It is:

> **Build a strong expert-labelled benchmark for measuring the actual correctness and quality of generated MCQs.**

That benchmark would allow the project to move from:

```text
"the system has strong verification machinery"
```

to a much stronger empirical claim:

```text
"the verification machinery measurably improves the correctness
and quality of generated questions."
```

---

## License

Add the project's license here when one is selected.