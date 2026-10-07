# Architecture Diagrams

This document is the index for every architecture diagram in the project.

It explains:

- which diagram to read first;
- what each diagram represents;
- how to interpret colours and arrows;
- where the editable source files live; and
- how to regenerate every diagram with one command.

The diagrams describe the implementation as of **7 October 2026**.

---

# 1. Start Here

If you are new to the project, read the diagrams in this order.

Each one answers a different question.

| Order | Diagram | Question it answers |
|---:|---|---|
| 1 | [Overall architecture](diagrams/practice_architecture.png) | What are the major components and how do they connect? |
| 2 | [Learner practice](diagrams/practice_flow.png) | What happens when a learner uses the application? |
| 3 | [Retrieval](diagrams/retrieval_flow.png) | How does a query become five source passages? |
| 4 | [Generation](diagrams/generation_flow.png) | How does a candidate MCQ become published — or get withheld? |
| 5 | [Verification](diagrams/verification_flow.png) | How is a generated claim actually verified? |

The remaining diagrams are reference material:

6. **Literal absence** — how exhaustive quoted-term checks work.
7. **Data model** — what the application persists.
8. **Evaluation** — how evaluation runs and is graded.
9. **Baseline** — how the vanilla-RAG comparison works.

---

# 2. Diagram Gallery

| Flow | View | Editable source | What it shows |
|---|---|---|---|
| **Overall architecture** | [PNG](diagrams/practice_architecture.png) | [MMD](diagrams/practice_architecture.mmd) | UI, source corpus, retrieval, workflow, persistence, and diagnostics |
| **Learner practice** | [PNG](diagrams/practice_flow.png) | [MMD](diagrams/practice_flow.mmd) | Topic selection, question reuse, quiz preparation, answering, submission, and review |
| **Retrieval** | [PNG](diagrams/retrieval_flow.png) | [MMD](diagrams/retrieval_flow.mmd) | Page-aware extraction, semantic retrieval, top-20 candidates, reranking to five, context expansion, and evidence |
| **Generation** | [PNG](diagrams/generation_flow.png) | [MMD](diagrams/generation_flow.mmd) | The eight workflow stages, five publication gates, bounded revisions, and final decision |
| **Verification** | [PNG](diagrams/verification_flow.png) | [MMD](diagrams/verification_flow.mmd) | Statement-question verification and direct-option verification |
| **Literal absence** | [PNG](diagrams/absence_flow.png) | [MMD](diagrams/absence_flow.mmd) | Exhaustive quoted-term scanning across the complete corpus |
| **Data model** | [PNG](diagrams/data_model.png) | [MMD](diagrams/data_model.mmd) | `questions`, `runs`, session-only learner state, and evaluation files |
| **Evaluation** | [PNG](diagrams/evaluation_flow.png) | [MMD](diagrams/evaluation_flow.mmd) | Answering, post-hoc Python grading, offline validation, and evaluation reporting |
| **Baseline** | [PNG](diagrams/baseline_flow.png) | [MMD](diagrams/baseline_flow.mmd) | The vanilla-RAG comparison protocol |

---

# 3. How to Read the Diagrams

The diagrams use a common visual language.

Understanding it makes the architecture much easier to follow.

## Colours

| Colour | Represents | Interpretation |
|---|---|---|
| **Green** | Source or evidence data | Text from the corpus: chunks, pages, quotations, or retrieved evidence |
| **Blue** | Python or UI control | Deterministic application logic; no model call |
| **Purple** | Model work | A call to `gpt-4o-mini`; output is untrusted until verified |
| **Amber** | Decision | A branch such as accept, revise, reject, or abstain |
| **Grey** | Local storage | SQLite, JSONL, or saved evaluation reports |

### The most important distinction

Pay particular attention to:

```text
Green → Purple → Blue
```

This represents:

```text
Source evidence
      ↓
Model interpretation
      ↓
Deterministic checking / decision
```

The purple output is not trusted simply because the model produced it.

The blue stage is where the application evaluates what the model produced.

---

# 4. Arrows

| Arrow | Meaning |
|---|---|
| **Solid** | Normal data or control flow |
| **Dashed** | Model-mediated relationship, revision relationship, or workflow loop |

A solid arrow generally means:

> This happens next.

A dashed arrow generally indicates:

> This relationship involves model output, revision, or a non-linear workflow connection.

---

# 5. The Rule Encoded by the Diagrams

All diagrams follow one architectural rule:

> **A source passage is evidence to inspect, never a verdict by itself.**

A green evidence node flowing into another component means that component has source material available.

It does **not** mean the associated claim has already been established.

For example:

```text
Source passage
      ↓
Model reviewer
      ↓
Python validation
      ↓
Decision
```

The model may interpret the source.

Python determines whether the resulting verification state satisfies the system's publication contract.

This distinction is central to the architecture and is discussed in more detail in [`../DESIGN.md`](../DESIGN.md).

---

# 6. Overall Architecture

**File:**

```text
diagrams/practice_architecture.png
```

This is the best starting point for understanding the entire application.

It shows the major system boundaries:

```text
Learner
   ↓
Streamlit UI
   ↓
Practice service
   ↓
LangGraph workflow
   ↓
Retrieval + verification
   ↓
Validated question
   ↓
SQLite / session state
```

It also shows the offline evaluation path and the source-preparation pipeline.

Use this diagram when you want to answer:

> **"What are all the major components of the system?"**

---

# 7. Learner Practice Flow

**File:**

```text
diagrams/practice_flow.png
```

This diagram focuses on the learner's journey rather than internal implementation.

It covers:

1. selecting a topic;
2. selecting question style;
3. selecting difficulty;
4. requesting a question set;
5. reusing an already verified question where eligible;
6. generating missing questions;
7. attempting the quiz;
8. submitting answers;
9. revealing explanations;
10. updating session-only practice state.

Use this diagram when the question is:

> **"What happens when someone actually uses the application?"**

---

# 8. Retrieval Flow

**File:**

```text
diagrams/retrieval_flow.png
```

This diagram explains how source evidence reaches the model.

The main path is:

```text
Corpus
  ↓
Page-aware chunks
  ↓
Embedding retrieval
  ↓
Top 20
  ↓
Cross-encoder reranking
  ↓
Top 5
  ↓
Page/context expansion
  ↓
Evidence packet
```

It also illustrates why the system keeps page information.

A chunk alone may omit:

- a proviso;
- an exception;
- continuation text; or
- a footnote.

Page-aware context restoration can restore that surrounding information.

Use this diagram when you want to understand:

> **"How does the system decide which source material the model gets?"**

---

# 9. Generation Flow

**File:**

```text
diagrams/generation_flow.png
```

This diagram shows the complete eight-stage generation workflow:

```text
retrieve_sources
       ↓
generate_mcq
       ↓
extract_claims
       ↓
verify_claims
       ↓
verify_answer_key
       ↓
audit_quality
       ↓
explain_question
       ↓
decide
```

The final decision has three possible paths:

```text
                 ┌── ACCEPT
                 │
decide ──────────┼── REJECT
                 │
                 └── REVISE
                       ↓
                  generate_mcq
```

The revision path is bounded.

The diagram therefore makes two things visible:

1. verification is part of generation, not an optional post-processing step;
2. unresolved candidates eventually terminate in abstention rather than retrying forever.

Use this diagram when you want to understand:

> **"How does a model-generated candidate become a learner-visible question?"**

---

# 10. Verification Flow

**File:**

```text
diagrams/verification_flow.png
```

This diagram shows the two main verification paths side by side.

## Statement questions

```text
Question
   ↓
Extract statements
   ↓
Build standalone claims
   ↓
Retrieve per claim
   ↓
SUPPORTED / CONTRADICTED / INSUFFICIENT
   ↓
Validate quotations
   ↓
Blind review
   ↓
Python derives answer
```

## Direct questions

```text
Question
   ↓
Evaluate all four options
   ↓
Retrieve evidence per option
   ↓
Support / rule out
   ↓
Exactly one supported option
   ↓
Blind review
   ↓
Answer
```

The direct path has a particularly important rule:

> **Missing evidence cannot prove an option false.**

An option must be positively ruled out rather than merely lacking retrieved evidence.

Use this diagram when you want to understand:

> **"How does the system determine whether the generated answer is actually defensible?"**

---

# 11. Literal Absence Flow

**File:**

```text
diagrams/absence_flow.png
```

Some claims depend on the literal absence of a term from the corpus.

Retrieval cannot establish absence because:

```text
Top 5 chunks
      ≠
Entire corpus
```

The absence path therefore performs an exhaustive scan:

```text
Quoted term
     ↓
All loaded chunks
     ↓
Literal match?
     │
 ┌───┴────┐
 │        │
YES      NO
 │        │
 ↓        ↓
Claim    Literal
contradicted absence supported
```

The result is limited to literal string absence.

It does not establish conceptual absence.

Use this diagram when evaluating claims such as:

> "This term does not appear in the Constitution."

---

# 12. Data Model

**File:**

```text
diagrams/data_model.png
```

This diagram shows what the application stores.

The primary persistent tables are:

```text
questions
runs
```

The diagram also distinguishes durable storage from session-only learner state.

### Persistent

```text
SQLite
 ├── questions
 └── runs
```

### Session-only

```text
Streamlit session state
 ├── learner answers
 ├── scores
 └── My Practice state
```

### Evaluation artifacts

```text
data/evaluation/
 ├── frozen datasets
 └── saved reports
```

Use this diagram when you want to understand:

> **"What survives after the application process ends?"**

---

# 13. Evaluation Flow

**File:**

```text
diagrams/evaluation_flow.png
```

The evaluation architecture deliberately prevents answer-key leakage.

The basic flow is:

```text
Frozen dataset
      ↓
Question + options
      ↓
Answering system
      ↓
Prediction
      ↓
Python grading
      ↓
Evaluation report
```

The official answer key is not provided to the answering system.

Python performs the grading afterward.

The diagram also shows the offline validation path used by:

```bash
.venv/bin/python -m src.evaluation.evaluate_benchmark --check
.venv/bin/python -m src.evaluation.run_all --check
```

Use this diagram when you want to understand:

> **"How do we measure the system without allowing the evaluation to leak the answers?"**

---

# 14. Baseline Flow

**File:**

```text
diagrams/baseline_flow.png
```

The baseline diagram describes the vanilla-RAG comparison.

The baseline intentionally removes the additional verification machinery so that the project can compare:

```text
Vanilla RAG
```

against:

```text
RAG + verification workflow
```

The purpose is not to claim that the baseline represents every possible RAG implementation.

It provides a controlled comparison against a simpler architecture.

Use this diagram when you want to understand:

> **"What does the system look like without the verification machinery?"**

---

# 15. File Formats

Every diagram is generated in four formats.

| Format | Intended use |
|---|---|
| `.png` | GitHub, Markdown, slides, general viewing |
| `.svg` | High-resolution web display and posters |
| `.pdf` | Printing and report appendices |
| `.mmd` | Editable Mermaid source |

The `.mmd` files are the preferred files for understanding and modifying the diagrams.

They are:

- plain text;
- easy to diff in Git;
- editable in Mermaid-compatible tools;
- renderable directly by GitHub Markdown.

---

# 16. Editing Diagrams

The editable source is the Mermaid `.mmd` file associated with each diagram.

For example:

```text
practice_architecture.mmd
```

is the source for:

```text
practice_architecture.png
practice_architecture.svg
practice_architecture.pdf
```

Do not manually edit an exported PNG, SVG, or PDF.

Those files are generated artifacts and will be overwritten by the next render.

For interactive Mermaid editing, the source can also be opened in the [Mermaid Live Editor](https://mermaid.live).

---

# 17. Rebuilding All Diagrams

The project has one diagram-rendering entry point:

```text
docs/diagrams/render_diagrams.py
```

Run:

```bash
.venv/bin/python docs/diagrams/render_diagrams.py
```

On Windows:

```powershell
.venv\Scripts\python docs\diagrams\render_diagrams.py
```

The command:

- runs offline;
- makes no model calls;
- regenerates all nine diagram flows;
- produces `.mmd`, `.png`, `.svg`, and `.pdf` outputs.

The renderer is therefore the single source of truth for the generated diagram artifacts.

---

# 18. Compatibility Files

Some older filenames remain in the repository so existing links continue to work.

They are compatibility files, not additional diagrams.

| File | Actual content |
|---|---|
| `system-design.png` | Copy of the overall architecture diagram |
| `system-design.svg` | Copy of the overall architecture diagram |
| `system-design.pdf` | Copy of the overall architecture diagram |
| `overall-system.mmd` | Copy of the overall architecture Mermaid source |

For new references, prefer:

```text
practice_architecture.*
```

rather than the compatibility names.

---

# 19. Superseded Render Scripts

The following scripts are no longer the primary rendering entry points:

```text
render_practice_architecture.py
render_system_design.py
```

They were replaced by:

```text
render_diagrams.py
```

Use only:

```bash
.venv/bin/python docs/diagrams/render_diagrams.py
```

for rebuilding the complete diagram set.

---

# 20. Diagram-to-Documentation Map

The diagrams correspond directly to the technical documentation.

| Diagram | Primary documentation |
|---|---|
| Overall architecture | [`SYSTEM_DESIGN.md`](SYSTEM_DESIGN.md) |
| Learner practice | [`PRACTICE_DESIGN.md`](PRACTICE_DESIGN.md) |
| Retrieval | [`SYSTEM_DESIGN.md`](SYSTEM_DESIGN.md) — Retrieval Contract |
| Generation | [`SYSTEM_DESIGN.md`](SYSTEM_DESIGN.md) — Generation Contract |
| Verification | [`SYSTEM_DESIGN.md`](SYSTEM_DESIGN.md) — Verification Contract |
| Literal absence | [`SYSTEM_DESIGN.md`](SYSTEM_DESIGN.md) — Literal Absence Verification |
| Data model | [`SYSTEM_DESIGN.md`](SYSTEM_DESIGN.md) — Persistence Model |
| Evaluation | [`EVALUATION.md`](../EVALUATION.md) |
| Baseline | [`rag_baseline.md`](rag_baseline.md) |

The diagrams are visual representations of those contracts; they are not separate sources of truth about application behaviour.

---

# 21. Related Documents

| Document | Covers |
|---|---|
| [`../DESIGN.md`](../DESIGN.md) | Why the system is shaped this way, trade-offs, and rejected alternatives |
| [`SYSTEM_DESIGN.md`](SYSTEM_DESIGN.md) | Technical contracts, components, interfaces, workflow, and failure handling |
| [`../README.md`](../README.md) | Installation, project overview, and plain-language walkthrough |
| [`PRACTICE_DESIGN.md`](PRACTICE_DESIGN.md) | Learner-facing product contract |
| [`../EVALUATION.md`](../EVALUATION.md) | Measurements, benchmarks, and limitations |
| [`retrieval_baseline.md`](retrieval_baseline.md) | Retrieval benchmark and retriever comparison |
| [`rag_baseline.md`](rag_baseline.md) | Vanilla-RAG comparison protocol |

---

# 22. The One Diagramming Principle

All nine diagrams ultimately communicate the same architectural boundary:

```text
Source evidence
      ↓
Model interpretation
      ↓
Deterministic verification
      ↓
Publication decision
```

The model can propose.

The evidence can inform.

But the application decides whether the result is publishable.

That boundary is the visual theme shared by the entire architecture.