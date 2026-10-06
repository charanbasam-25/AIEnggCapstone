# System Design — Source-Verified UPSC Polity MCQ Verifier

The learner-first generation flow is documented in
[Practice system design](PRACTICE_DESIGN.md), including source-first generation,
bounded revisions, answer and explanation checks, storage and the learner UI.
The [project brief](PROJECT_BRIEF.md) covers the current problem, scope, privacy,
architecture/tradeoffs, evaluation tasks, error handling, cost and latency, with
an updated [practice architecture image](diagrams/practice_architecture.png).
The generation sections below describe the earlier workflow and experiments.

For the preceding overall diagram, including direct-MCQ verification, the
75-question benchmark and Evaluation Studio, see
[Architecture diagrams](ARCHITECTURE_DIAGRAMS.md). The record below retains the
earlier module-level design and evaluation discussion.

**Author:** Charan Kumar Basam · **Cohort:** AI Engineering · **Date:** 2026-09-27
**Related documents**
| Document | Role |
|---|---|
| [`DESIGN.md`](../DESIGN.md) | The capstone design doc: problem, data surface, architecture rationale, comparative evaluation, failure analysis |
| [`EVALUATION.md`](../EVALUATION.md) | The measurement record: every arm, every null result, the noise band |
| [`docs/technical_documentation.md`](technical_documentation.md) | Earlier narrative document, written **before** the re-measurement pass. Its stated research question is about *generated* MCQs, which is not what the shipped evaluation measures. See §2.3. |
| [`docs/retrieval_baseline.md`](retrieval_baseline.md), [`docs/rag_baseline.md`](rag_baseline.md) | Baseline notes from the retrieval bake-off |
| [`docs/upsc_polity_capstone_development_log.txt`](upsc_polity_capstone_development_log.txt) | Chronological development log, 15 entries |
This document describes **what the system is and how it is built** — components,
contracts, control flow, and the design decisions that constrain them. It does
not argue for the architecture; `DESIGN.md` does that, and `EVALUATION.md`
supplies the numbers.
---
## 1. Scope and status
**Direct-MCQ extension.** The UI and CLI now route questions without numbered
items through `src/verification/direct_mcq_verifier.py`. Question-level and
option-focused semantic + cross-encoder retrieval supplies a deduplicated
evidence set. One structured comparison assesses A–D as SUPPORTED as the answer,
RULED_OUT as the answer, or INSUFFICIENT. For best-answer questions, answer fit
is distinct from isolated factual truth. Python requires one supported option
and three ruled-out alternatives, with matching evidence quotations, before
selecting a letter. The official key is withheld. This extension has offline
behavioral tests but is not covered by the saved end-to-end measurements;
the numbered-statement evaluation code is unchanged.

**In scope.** Indian Polity questions from the UPSC Civil Services Preliminary
Examination, decided against the text of the Constitution of India and NCERT
Polity.
**Out of scope, by design.** Current affairs, subjects other than Polity,
tutoring dialogue, and any question whose answer is not decidable from the
primary text.
**Status.** Two modes exist in code. One is measured.
| Mode | Code | Evaluated? |
|---|---|---|
| **Verify-an-existing-question** — decide each numbered statement of a real PYQ, then map verdicts to an option or abstain | `src/evaluation/evaluate_verified_pyqs.py`, `src/verification/*` | **Yes** — 13 questions, 37 statements, full metric suite |
| **Generate-and-gate** — generate a candidate MCQ, verify it, then ACCEPT / REVISE / REJECT | `src/generation/`, `src/orchestration/` | **No** — implemented, zero metrics |
That asymmetry is deliberate in retrospect but was not deliberate at the time;
§16 records it as an open gap rather than hiding it.
---
## 2. System context
### 2.1 The reliability problem
UPSC Prelims Polity questions are predominantly multi-statement: a stem, two to
four numbered statements, and options naming statement subsets (`"I and III
only"`) or counts (`"Only two"`, `"None"`). A student's real failure mode is not
*"I don't know the answer"* but *"I can't tell which statement I got wrong."*
An LLM tutor makes this worse: it emits a fluent paragraph naming Articles it
never checked. The system's job is therefore not fluency but **per-statement
adjudication with provenance, and refusal when the corpus does not settle it.**
### 2.2 The governing design principle
> **Generation and verification are separate responsibilities, and the LLM never
> selects the answer.**
Operationally this means an LLM is asked exactly one kind of question — *does
this single claim hold against this single evidence set?* — and Python does
everything else: parsing, claim construction, aggregation, option mapping, and
the decision to abstain. This is what turns abstention into a **policy** rather
than a mood, and it is what makes the evaluation in `EVALUATION.md` possible at
all.
### 2.3 Framing note
`docs/technical_documentation.md` states the research question as *"Does an
independent verification pipeline reduce factual and answer-key errors in
LLM-generated UPSC Polity MCQs?"* The shipped evaluation does not answer that
question — it measures the verifier **answering existing PYQs**, not gating
generated ones. The project's centre of gravity moved from generation to
verification during the build. `DESIGN.md:1` and `DESIGN.md:33` reflect the
current framing; `technical_documentation.md` reflects the original one and is
retained as a historical record.
---
## 3. Architecture overview
Primary flow — verify an existing question:
```
             UPSC question text (stem + numbered statements + options)
                                    │
                                    ▼
                   ┌────────────────────────────────┐
                   │  question_parser (no LLM)      │  classify: STATEMENTS /
                   │  lead-in · items · closing ·   │  STEM_DISTRIBUTED / PAIRS
                   │  options                       │
                   └────────────────┬───────────────┘
                                    ▼
                   ┌────────────────────────────────┐
                   │  claim_builder (no LLM)        │  one propositional claim
                   │  template binding + 2 guards   │  per numbered statement
                   └────────────────┬───────────────┘
                                    │
            ┌───────────────────────┴───────────────────────┐
            │ ordinary claim                                │ absence claim
            ▼                                               ▼
  ┌──────────────────────────┐                  ┌──────────────────────────┐
  │ ClaimRetriever           │                  │ NegativeClaimChecker     │
  │ semantic (bge-small)     │                  │ full scan, all 1,149     │
  │  → cross-encoder rerank  │                  │ chunks, no LLM, exact    │
  │ candidate_k=20 → top_k=5 │                  │ phrase normalisation     │
  └────────────┬─────────────┘                  └────────────┬─────────────┘
               ▼                                             │
  ┌──────────────────────────┐                               │
  │ FactVerifier  (LLM)      │                               │
  │ SUPPORTED /              │◄──────────────────────────────┘
  │ CONTRADICTED /           │
  │ INSUFFICIENT + pages     │
  └────────────┬─────────────┘
               ▼
  ┌──────────────────────────────────────────────┐
  │ aggregate per statement (no LLM)             │
  │ any CONTRADICTED → CONTRADICTED              │
  │ all SUPPORTED    → SUPPORTED                 │
  │ otherwise        → INSUFFICIENT              │
  └────────────┬─────────────────────────────────┘
               ▼
  ┌──────────────────────────────────────────────┐
  │ answer_mapping (no LLM)                      │
  │ policy ∈ {strict, closed_world, elimination} │
  └────────────┬─────────────────────────────────┘
               ▼
       option letter   OR   abstain + typed reason
```
Secondary flow — generate and gate (`src/orchestration/graph.py`):
```
START ─► generate_mcq ─► extract_claims ─► verify_claims
                                                │
                              ┌─────────────────┘
                              ▼
                     verify_answer_key ─► audit_quality ─► decide
                                                              │
                        ┌─────────────────────────────────────┤
                        │ REVISE (retry_count < max_retries)  │ ACCEPT / REJECT
                        ▼                                     ▼
                   generate_mcq                              END
                   (failure_reasons injected into prompt)
```
Three independent gates feed `decide`: per-claim fact verification, whole-question
answer-key verification, and a quality audit. Any gate failing produces a
`failure_reasons` entry, and the union of reasons is fed back into the next
generation prompt.
---
## 4. Component inventory
### 4.1 Ingestion — `src/ingestion/`
| File | Responsibility |
|---|---|
| `extract_text.py` | Page-aware PDF → text for `constitution.pdf`, `NCERTPolity.pdf` |
| `prepare_documents.py` | `clean_text`, `extract_pages` → per-page records carrying `source` and `document` |
| `chunk_documents.py` | `CHUNK_SIZE = 1000`, `CHUNK_OVERLAP = 150`; `split_text`, `create_chunks`, `save_chunks` |
| `save_chunks_jsonl.py` | Emit `data/processed/chunks.jsonl` |
### 4.2 Retrieval — `src/retrieval/`
Six retrievers exist. **One is production**; the rest are benchmark arms whose
numbers appear in `EVALUATION.md §1`.
| File | Model / method | Recall@5 (16 queries) | Role |
|---|---|---|---|
| `semantic_reranker.py` | `bge-small-en-v1.5` → `cross-encoder/ms-marco-MiniLM-L-6-v2` | **93.75%** | **Production** |
| `semantic_retriever.py` | `BAAI/bge-small-en-v1.5` | 87.50% | Benchmark arm; candidate generator for the above |
| `parent_child_retriever.py` | child chunks → parent expansion → rerank | 87.50% | Benchmark arm |
| `hybrid_retriever.py` | BM25 + semantic via `reciprocal_rank_fusion` | 81.25% | Benchmark arm |
| `bm25_retriever.py` | `rank_bm25.BM25Okapi` | 75.00% | Benchmark arm |
| `lexical_index.py` | stdlib reimplementation of BM25Okapi | 75.00% | Benchmark arm; also supplies `load_chunks` to several evaluation modules and keeps the deterministic path dependency-free |
`lexical_index.py` reproduces BM25Okapi exactly — including the IDF floor
`epsilon * mean_idf` for terms appearing in more than ~half the corpus — and
asserts that property in `self_check` (`lexical_index.py:254`), because omitting
the floor is the standard way a hand-rolled BM25 silently diverges from the
library it stands in for.
### 4.3 Claim construction — `src/verification/`
| File | LLM? | Responsibility |
|---|---|---|
| `question_parser.py` | No | Split into lead-in / numbered items / closing / options. Classifies `STATEMENTS`, `STEM_DISTRIBUTED`, `PAIRS`. Character folding for Unicode variants. |
| `claim_builder.py` | No | Bind each item into a standalone proposition. Binding modes: `none`, `trailing_predicate`, `leading_predicate`, `leading_condition`, `pair`, `failed`. Exposes `BuiltClaim.is_propositional`. |
| `claim_extractor.py` | Both | Two paths: `extract_deterministic` (`SOURCE_DETERMINISTIC`) and `extract_with_llm` (`SOURCE_LLM`). The deterministic path is preferred; the LLM path is the fallback. |
**Why claims are built deterministically.** A numbered statement is frequently a
bare noun phrase (`"Police"`) whose predicate lives in the stem. A template
substitution binds them — *"Police is included in the Seventh Schedule…"* —
guarded by two checks: no token may appear that was not present in the source
question, and split pairs must rejoin. An LLM asked to do this paraphrases, and
a paraphrase is a new claim that the corpus was never asked about.
Measured effect (`EVALUATION.md §4.4`): propositional rate **70.27% → 100%** and
false `SUPPORTED` verdicts **7–8 → 0** across 37 claims. This is the largest
effect in the project that clears the noise band.
### 4.4 Verification — `src/verification/`
| File | LLM? | Contract |
|---|---|---|
| `claim_retriever.py` | No | Wraps `SemanticReranker(candidate_k=20)`. `retrieve(claim, top_k=5) -> list[dict]`. Deliberately does **not** reuse evidence from generation. |
| `fact_verifier.py` | Yes | `verify(claim, evidence) -> FactVerificationResult`. Three-way verdict + reasoning + `supporting_pages`. Constructs a `NegativeClaimChecker` **only when given chunks**. |
| `negative_claim_checker.py` | No | `extract_target_term`, `normalise`, `find_occurrences`, `check`. Full-corpus scan for absence claims. |
| `answer_key_verifier.py` | Yes | `analyze`, `verify(mcq, evidence) -> AnswerKeyVerificationResult` with `declared_answer`, `supported_options`, `exactly_one_correct`, verdict `VALID`/otherwise |
| `mcq_quality_auditor.py` | Yes | `audit(mcq) -> QualityAuditResult`: `unambiguous`, `single_best_answer`, `plausible_distractors`, `appropriate_wording`, `topic_relevant`, `issues`, `overall_quality ∈ {PASS, FAIL}` |
**Absence claims bypass retrieval entirely.** A claim like *"The Constitution
does not mention 'political party'"* quantifies over the whole corpus, so ranking
is the wrong instrument — a top-5 result set cannot establish absence. A full
scan of all 1,149 chunks answers it exactly, with no LLM involved.
This is also the site of a recurring defect. `FactVerifier()` constructed without
chunks silently disables the checker, so absence claims fall through to the LLM
path — where *"the evidence does not mention X"* is precisely the inference that
rules 13–17 of that prompt exist to forbid. The bug appeared in
`orchestration/nodes.py` (now fixed, see `nodes.py:56`) and then **recurred
independently** in `verdict_stability.py`. See §15.
### 4.5 Generation and orchestration
| File | Responsibility |
|---|---|
| `src/generation/mcq_generator.py` | `MCQGenerator.generate(topic, difficulty, failure_reasons) -> MCQ`. 11 numbered requirements; `failure_reasons` injected as a revision preamble. |
| `src/orchestration/state.py` | `MCQVerificationState(TypedDict, total=False)` — the single shared state object |
| `src/orchestration/nodes.py` | Six node functions + three `lru_cache(maxsize=1)` factories |
| `src/orchestration/graph.py` | `StateGraph` wiring, `route_after_decision` conditional edge |
| `src/orchestration/run_workflow.py` | Single-topic entry point; prints, does not persist |
### 4.6 Baseline
`src/rag/vanilla_rag.py` — `VanillaRAG.answer()`, returning a `RAGAnswer`. This
is **System A**: retrieve, then ask one model to answer. No claim decomposition,
no abstention. It is the comparison the rubric requires.
---
## 5. Data design
### 5.1 Corpus
| Source | Pages | Chunks |
|---|---|---|
| Constitution of India (full text) | 402 | 1,107 |
| NCERT Polity | 20 | 42 |
| **Total** | **422** | **1,149** |
### 5.2 Chunk schema — `data/processed/chunks.jsonl`
```json
{
  "id": 0,                    // int, stable, used for tie-breaking
  "text": "…",                // str, ~1000 chars
  "source": "constitution",   // str
  "document": "…",            // str
  "page": 1,                  // int — carried into every verdict
  "chunk_index": 0            // int
}
```
**Page provenance is load-bearing, not decoration.** Every verdict cites the
pages it used, and `DESIGN.md §7`'s most important finding was discovered by
auditing those citations rather than the verdicts.
### 5.3 Evaluation dataset
`data/evaluation/pyq_2025_polity.json` — **13 questions, 37 numbered
statements**, UPSC 2025 Prelims Polity Q54–Q66. Fields: `q_number`,
`question_text`, `options`, `official_answer`. Held out from all prompt
development.
Built by `src/evaluation/extract_pyqs.py` from
`data/raw/upsc/prelims2025Questions.pdf` → `pyq_2025_raw.json`.
**Unused raw material.** `data/raw/upsc/prelims2026Questions.pdf` (60.8 MiB, Git
LFS) and `data/raw/answer_keys/prelims2026Key.pdf` are referenced by **zero**
files in `src/`. They are the obvious extension — a second evaluation set would
roughly double a benchmark that is currently smaller than its own measurement
error (§16).
### 5.4 Label sets
Two answer-key label sets are reported throughout, never one:
- **stored** — `official_answer` as extracted
- **audited** — after `src/evaluation/label_audit.py` corrected extraction errors
Every headline metric is quoted under both, because a conclusion that holds under
only one label set is not a conclusion.
---
## 6. Interface contracts
All LLM-facing models are Pydantic and all are parsed via
`client.responses.parse(..., text_format=Model)`, so a malformed response is a
validation error rather than a silent misparse.
```python
class MCQ(BaseModel):                       # mcq_generator.py:12
    question: str
    option_a: str; option_b: str
    option_c: str; option_d: str
    correct_answer: str = Field(pattern="^[ABCD]$")
    explanation: str
class FactVerificationResult(BaseModel):    # fact_verifier.py:17
    verdict: str        # SUPPORTED | CONTRADICTED | INSUFFICIENT
    reasoning: str
    supporting_pages: list[int]
class QualityAuditResult(BaseModel):        # mcq_quality_auditor.py:14
    unambiguous: bool; single_best_answer: bool
    plausible_distractors: bool; appropriate_wording: bool
    topic_relevant: bool
    issues: list[str]
    overall_quality: str = Field(pattern="^(PASS|FAIL)$")
```
The `correct_answer` and `overall_quality` regex constraints matter: they make
the *shape* of the output non-negotiable at the boundary, so downstream Python
never branches on a free-text field.
---
## 7. Control flow — the orchestrated loop
State is a single `TypedDict` (`state.py:10`) threaded through six nodes. Each
node returns a **partial** dict; LangGraph merges it.
| Node | Reads | Writes |
|---|---|---|
| `generate_mcq` | `topic`, `difficulty`, `failure_reasons` | `mcq`, `retry_count += 1`, `failure_reasons = []` |
| `extract_claims` | `mcq` | `claims` |
| `verify_claims` | `claims` | `claim_evidence`, `fact_verifications` |
| `verify_answer_key` | `mcq` | `answer_evidence`, `answer_verification` |
| `audit_quality` | `mcq` | `quality_audit` |
| `decide` | all three gate results, `retry_count`, `max_retries` | `decision`, `failure_reasons` |
**Routing** (`graph.py:14`): `REVISE → generate_mcq`; `ACCEPT | REJECT → END`;
anything else raises. Raising on an unknown decision is intentional — a typo in a
verdict string should stop the graph, not silently route to termination.
**Termination** (`nodes.py:221`): no failure reasons → `ACCEPT`; otherwise
`REVISE` while `retry_count <= max_retries` (default 2); else `REJECT`. Worst case
is 3 generation cycles × 6 nodes = 18 steps, comfortably inside LangGraph's
default recursion limit.
**The bound is `<=`, and `<` was an off-by-one.** `retry_count` is incremented in
`generate_mcq` *before* the gates run (`nodes.py:87`), so when `decide` reads it,
it counts **attempts made, not retries taken** — it is 1 on the first pass, never
0. Under `<`, `max_retries=2` granted exactly one revision and a third attempt
was unreachable. This was found by measurement, not by reading: across 15 topics
every `REJECT` terminated at attempt 2 and attempt 3 never occurred. The counter's
name is the trap — `retry_count` counts attempts, `max_retries` counts retries,
and `attempts = retries + 1`.
**Per-attempt history is not recoverable from the return value.** `generate_mcq`
clears `failure_reasons` on every attempt and the graph returns only terminal
state, so an evaluation of this loop must consume
`workflow.stream(..., stream_mode="updates")` to observe each `decide` output.
`src/evaluation/evaluate_generation_loop.py` does exactly that.
---
## 8. Retrieval depth as a design parameter
Both depths now come from one place — `RETRIEVAL_TOP_K` in
`src/retrieval/retrieval_config.py` — which the five retrieval benchmarks and the
verification pipeline import.
They were `3` in the pipeline while the retrieval benchmark that selected the
retriever was tuned on **Recall@5** — so the pipeline ran one setting and was
argued for with another, and the 93.75% quoted for semantic+cross-encoder
described a configuration nothing ran at. That is the defect, and it is real
independent of what correcting it bought.
**What correcting it bought is not measurable, and this section used to claim
otherwise.** Isolated by the `C0` vs `C1d` arms (stored claims and base prompt
held fixed, `k` the only moving part; `data/evaluation/system_c_results.json`),
the direction of the change depends on which label set is scored:
| labels | `C0` (k=3) | `C1d` (k=5) | direction |
|---|---|---|---|
| stored, STRICT | 30.77% cov · 50.00% prec · 15.38% acc · 15.38% err | **38.46%** · **80.00%** · **30.77%** · **7.69%** | depth **helps** |
| audited, STRICT | 23.08% acc · 75.00% prec · 7.69% err | 23.08% acc · 60.00% prec · 15.38% err | depth **hurts** |
A change whose sign flips with the label set is not an effect. The whole
difference runs through **one statement** — `Q58 I`, `CONTRADICTED` at k=3 →
`SUPPORTED` at k=5 — and that claim sits on the verifier's decision boundary:
holding claim and evidence fixed at temperature 0, its verdict is a function of
three characters of whitespace and the choice of endpoint (§4.2 of
`EVALUATION.md` has the 4-cell table). At k=3 the pipeline answers Q58 = D
(audited-correct); at k=5 it answers A. Across the 37 claims `INSUFFICIENT` fell
18 → 16 and `SUPPORTED` rose 14 → 17, but with n=13 and a measured 15.38-point
accuracy spread across repeats (§13), every arm-to-arm gap here is one question
or zero.
So the lesson is the weaker, cheaper one: **a configuration was running at
settings it was never validated at.** Worth fixing on principle; not worth an
accuracy claim. The project's evidence against complexity bias is the retrieval
ablation instead — hybrid RRF (81.25%) and parent–child (87.50%) were both built
and both **lost** to the simpler pipeline, and were rejected on measurement.
> **Retracted.** Earlier revisions of this section called the k=3→5 correction
> "the single largest accuracy change of any intervention in the project" and
> said the one-integer fix "beat both of the designed interventions." Both
> statements came from reading the stored-label column alone, on a BM25-era run,
> before the boundary-case behaviour of `Q58 I` was known. They are retracted
> rather than deleted, because the error — attaching a confident number to a
> setting it was not cleanly measured at — is the same error this section is
> about, one level up.
> **Corrected.** Earlier revisions of this section, of `nodes.py`, and of the
> `ClaimRetriever.retrieve` docstring quoted coverage **38.46% → 53.85%** and
> accuracy **30.77% → 38.46%** here, and said four `INSUFFICIENT` claims
> resolved. Those are the stored-label row misread one column across — 38.46% is
> where coverage *ended* — and 53.85% appears in no arm of any stored run. The
> correction is recorded rather than silently applied because it is the same
> error as the defect the section is about: a figure sitting next to a setting it
> was never measured at.
---
## 9. Answer mapping and abstention policy
`src/evaluation/answer_mapping.py:74` defines three policies:
| Policy | Reading of an unresolved statement | Behaviour |
|---|---|---|
| `strict` | unknown | Abstain if **any** statement is `INSUFFICIENT` |
| `closed_world` | false | Answer anyway, treating unresolved as not-true |
| `elimination` | unknown, but options may still be eliminated | Answer if exactly one option survives elimination |
`map_answer(..., policy=STRICT)` is production. Abstention reasons are **typed**,
not free text: unresolved statement, no option matches, several options match.
Two measured results shape this design:
- **`elimination` ≡ `strict`** on this dataset — an exact null. With four options
  and this verdict distribution, elimination never leaves exactly one survivor
  when strict would have abstained.
- **`closed_world` is the highest-accuracy configuration in the project**
  (53.85% audited, beating vanilla RAG) and is **rejected anyway**, because error
  rate goes 0% → 30.77%. Accuracy alone would have chosen it.
That second point is why the metric suite is reported as a quadruple, never a
single number.
---
## 10. Metrics design
Four numbers, always together (`src/evaluation/selective_metrics.py`):
| Metric | Definition |
|---|---|
| **coverage** | fraction of questions answered rather than abstained |
| **precision-when-answered** | accuracy among answered questions only |
| **accuracy-overall** | correct / all questions |
| **error-rate-overall** | confidently wrong / all questions |
With the identity `accuracy + error rate + abstention rate = 1`.
**Why all four.** Any single one is gameable. Abstain on everything: accuracy 0%,
error rate 0%, precision undefined — looks safe, is useless. Answer everything:
coverage 100%, error rate maximal. Only the quadruple describes a selective
predictor honestly.
Headline result (`EVALUATION.md §2`), identical under stored and audited labels:
| System | Coverage | Precision | Accuracy | **Error rate** |
|---|---|---|---|---|
| A — vanilla RAG | 100% | 46.15% | 46.15% | **53.85%** |
| B — verified | 30.77% | **100%** | 30.77% | **0%** |
A states a false answer on 7 of 13 audited questions; B on **0 of 13**. That
~46–54 point error-rate gap is far outside the ±2-question noise band, and it is
the one claim in the project that no subsequent retraction touched.
---
## 11. Determinism and reproducibility
**Temperature pinning is a precondition, not a detail.** `responses.parse`
defaults to **temperature 1.0**. Every LLM call site therefore pins
`temperature=0` explicitly — `mcq_generator.py:85`,
`mcq_quality_auditor.py:96`, `answer_key_verifier.py`.
`FactVerifier.verify` is the one exception, and deliberately so:
`temperature: float | None = 0` (`fact_verifier.py:50`). The default pins it;
the parameter exists *only* so `verdict_stability.py` can pass `None` and
measure the unpinned arm. Passing `None` omits the key from the request
entirely (`fact_verifier.py:284`) rather than substituting a value — so the
unpinned arm reproduces the original defect exactly instead of approximating
it. Without that seam, the pinned-vs-unpinned comparison below could not be
made with one code path.
Note also the deliberate trade in `mcq_generator.py:78-81`: generation is the
one place sampling diversity would arguably be *wanted*, and it is pinned
anyway, because generated questions feed the evaluation set and a fixed corpus
matters more than variety.
**Temperature 0 is not determinism.** Measured over repeated runs
(`EVALUATION.md §0.1`):
| Configuration | Stable claims (of 37) | Unstable |
|---|---|---|
| Unpinned | 28 (75.68%) | 9 |
| Pinned | 35 (94.59%) | 2 |
Both counts are **lower bounds** at `REPEATS = 5`. One question (Q58 statement I)
emits all three verdicts across runs. Consequently `EVALUATION.md` reports a
metric **range** from resampled runs, and every arm-to-arm difference in the
six-arm ablation (≤7.69 points) sits **inside that band** — so that table is
reported as six null results, not six findings.
**Decision-boundary fragility.** A 2×2 boundary experiment showed that three
whitespace characters, or an endpoint change, can flip a verdict. Conclusions are
therefore only drawn from effects that survive the band.
---
## 12. Caching and performance
| Mechanism | Location | Reason |
|---|---|---|
| `lru_cache(maxsize=1)` on `get_chunks`, `get_retriever`, `get_fact_verifier` | `nodes.py:33-70` | Both verification nodes previously rebuilt the retriever per invocation, so one revision loop parsed and re-embedded the corpus **four times** |
| Verdict cache | `data/evaluation/system_c_verdict_cache*.json` | Ablation arms share most claims; caching makes a six-arm sweep affordable |
| `candidate_k=20 → top_k=5` | `claim_retriever.py:18` | Cross-encoders are quadratic in pair count; rerank 20 candidates, not 1,149 chunks |
`get_chunks` returns a `tuple` rather than a list purely because `lru_cache`
requires the retriever factory's argument to be hashable.
---
## 13. Evaluation harness architecture
The expanded 75-question benchmark uses a separate runner,
`src/evaluation/evaluate_benchmark.py`, and fixed 33-development/42-test splits.
`--check` and `--dry-run` validate/preview offline; `--run` compares the current
statement/direct verifier with vanilla RAG and checkpoints after every answer.
Dataset, code and corpus hashes prevent mixing resumed runs from different
settings. Answer keys stay outside answering inputs, and API errors are counted
separately from abstentions. Outputs live under `data/evaluation/benchmark/`;
the new dataset has not yet produced model results. Details and source links
are in [BENCHMARK.md](BENCHMARK.md).

`src/evaluation/run_all.py` is a small DAG runner with **mtime-based freshness
checking**. Each stage declares `reads`, `writes`, and a `note`:
```
evaluate_verified_pyqs
        └─► replay_answer_mapping
                └─► direct_option_verifier
                        └─► system_b_metrics
                                └─► selective_metrics
                                        └─► verdict_stability
                                                └─► judge_llm
```
`--check` verifies every report is at least as new as its inputs; `--only`
restricts execution to one stage.
**Why this exists.** Two real incidents, both recorded in the docstring: a report
was regenerated from stale inputs and quoted in a document, and stages were run
out of order so a downstream metric described a pipeline that no longer existed.
The DAG makes "the numbers in `data/evaluation/` are mutually consistent" a
checkable assertion rather than a hope.
`system_c_pipeline` and `vanilla_rag` are deliberately **excluded** — they are
ablation and baseline arms, not part of the production measurement chain.
**LLM-as-judge.** `judge_llm.py` scores A-vs-B head to head on the same
questions. Result: **12–1–0** for B, hallucinated-claim counts **19 → 1**, and B
answered `"None"` on 9 questions and was preferred on 8 of them. Q59 is A's only
win.
---
## 14. Environment and dependencies
| Concern | Resolution |
|---|---|
| Corporate TLS interception | `src/tls_trust.py` — `enable_os_trust_store()` calls `truststore.inject_into_ssl()`. `huggingface_hub` misreported the interception as a **network outage**, which cost a wrong diagnosis and a stdlib BM25 reimplementation. |
| Model | `gpt-4o-mini` at every LLM call site (`MODEL_NAME`) |
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Secrets | `OPENAI_API_KEY` in `.env`, gitignored at `.gitignore:5`. Never committed; absent from the repo and from any `git bundle`. |
| Orchestration | `langgraph` `StateGraph` |
---
## 15. Failure modes and defences
| Failure mode | Defence |
|---|---|
| LLM asserts a fact it never retrieved | Verdicts must cite `supporting_pages`; citations were audited, and that audit produced the project's most important finding |
| LLM paraphrases a statement into a different claim | Claims built by template substitution with a no-new-token guard |
| Absence claim answered from a top-5 slice | `NegativeClaimChecker` full-corpus scan, no LLM |
| `FactVerifier()` built without chunks, silently disabling that checker | Fixed at `nodes.py:56`; **recurred** in `verdict_stability.py`, which is why it is documented here as a class of defect rather than one bug |
| Retriever benchmarked at k=5, queried at k=3 | `CLAIM_TOP_K` pinned and the `ClaimRetriever` default realigned |
| Metric quoted from a stale report | `run_all.py` mtime DAG + `--check` |
| Single run mistaken for a measurement | Resampled runs → noise band; differences inside the band reported as null |
| Temperature drift via `responses.parse` default of 1.0 | `temperature=0` pinned at every call site |
| Accuracy-driven config selection | Four-metric suite; `closed_world` rejected despite winning on accuracy |
---
## 16. Known gaps
1. ~~**The generation loop is unmeasured.**~~ **Closed.** Measured across two
   question-format arms, 15 topics each, by
   `src/evaluation/evaluate_generation_loop.py`. It required exactly what was
   predicted here — per-attempt state via `stream_mode="updates"` (§7) — and
   yielded the gate-attribution, repair-rate and unverified-baseline tables now
   in `EVALUATION.md §2.5`. Two things the measurement changed:
   - The `simple` arm accepts 53.33% of topics; the `statements` arm accepts
     **0 of 15**. The gate is format-sensitive to a degree nothing in the design
     anticipated, and the honest reading is that the quality bar is
     mis-calibrated for multi-statement output, not that the generator is
     uniformly bad.
   - The attempt counter was off by one: `attempts = retries + 1`, so
     `max_retries=2` permits three generations. Every per-attempt rate computed
     before that fix was wrong in the denominator.
   Remaining gap: the loop is measured on **one draw per topic**, so its rates
   carry an unquantified noise band. §16.3 applies here with more force than to
   the verifier, because a topic's outcome is the product of up to three
   sequential LLM calls.
2. **n = 13 is smaller than the measurement error.** The 2026 paper and its
   official key are already in the repo, unused (§5.3).
3. **`REPEATS = 5` under-powers the noise estimate.** Three harness runs
   disagreed (27/37, 29/37, 28/37 unpinned), so instability counts are lower
   bounds.
4. **Retrieval benchmarks were not re-run** in the final pass. `EVALUATION.md §1`
   is the only table not re-verified against current code.
5. **6 of 9 abstentions are pipeline defects**, not genuine unanswerability — on
   questions the corpus *can* settle. This is the only place coverage rises
   without buying errors, and fixing it would remove the one respect in which
   vanilla RAG still beats the verified system.
6. **The fact-verifier prompt has accreted to 26 rules.** It should be decomposed
   rather than extended further.
7. **Git LFS dependency on an unused file.** `.gitattributes` marks
   `prelims2026Questions.pdf` as LFS; nothing reads it, and it breaks `git
   bundle` transport because bundles carry git objects but not LFS payloads.
---
## Appendix A — file map
```
src/
  ingestion/        extract_text · prepare_documents · chunk_documents · save_chunks_jsonl
  retrieval/        semantic_reranker (production) · semantic_retriever · bm25_retriever
                    hybrid_retriever · parent_child_retriever · lexical_index
  verification/     question_parser · claim_builder · claim_extractor · claim_retriever
                    fact_verifier · negative_claim_checker · answer_key_verifier
                    mcq_quality_auditor
  generation/       mcq_generator
  orchestration/    state · nodes · graph · run_workflow
  rag/              vanilla_rag                      (System A baseline)
  evaluation/       run_all (DAG) · evaluate_verified_pyqs · evaluate_vanilla_rag
                    answer_mapping · replay_answer_mapping · direct_option_verifier
                    selective_metrics · system_b_metrics · system_c_pipeline
                    verdict_stability · judge_llm · label_audit · extract_pyqs
                    evaluate_generation_loop        (two-arm gate measurement)
  verify_cli.py     CLI front door; composition root, imports across packages
  tls_trust.py
data/
  raw/              constitution · ncert · upsc · answer_keys
  processed/        chunks.jsonl (1,149) · extracted text
  evaluation/       datasets · results · caches · logs
docs/
  SYSTEM_DESIGN.md (this file) · technical_documentation.md
  retrieval_baseline.md · rag_baseline.md
  upsc_polity_capstone_development_log.txt
app.py            Streamlit demo; the other front door
README.md · DESIGN.md · EVALUATION.md
```
### Entry points
Until late in the project the system had **no front door**: the only way to
submit a question was to edit `data/evaluation/pyq_2025_polity.json` and re-run
the harness. `run_workflow.py` had its topic hardcoded. Both current entry
points are deliberately thin — they parse input and print results, and
reimplement nothing:
| Entry point | For | Reuses |
|---|---|---|
| `src/verify_cli.py` | one question, scriptable, `--save` for JSON | `build_pyq_claims`, `determine_answer_from_claims`, `pyq_to_mcq` |
| `app.py` | demo; policy switching without re-verifying | the same three, plus `AnswerKeyVerifier` for the rejected-architecture comparison |
`verify_cli.py` sits at `src/` root rather than in `src/verification/` because it
is a composition root: it imports from `evaluation`, and placing it inside
`verification` would create a `verification → evaluation` dependency pointing the
wrong way.
`app.py` caches verdicts in `st.session_state`, so changing the abstention policy
re-runs only `determine_statement_statuses` + `map_answer` — pure Python, **0 API
calls, ~0.05 s**. The coverage-versus-error trade-off in `EVALUATION.md §5` stops
being a table and becomes a radio button.
## Appendix B — commands
```bash
# Corpus
python -m src.ingestion.prepare_documents
python -m src.ingestion.chunk_documents
# Production measurement chain, with freshness enforcement
python -m src.evaluation.run_all
python -m src.evaluation.run_all --check
python -m src.evaluation.run_all --only selective_metrics
# Baseline and ablations (excluded from the DAG by design)
python -m src.evaluation.evaluate_vanilla_rag
python -m src.evaluation.system_c_pipeline
# Generate-and-gate loop (single topic, prints only)
python -m src.orchestration.run_workflow
# Generate-and-gate loop, measured (~250 API calls per arm)
python -m src.evaluation.evaluate_generation_loop --format simple
python -m src.evaluation.evaluate_generation_loop --format statements
python -m src.evaluation.evaluate_generation_loop --summarise-only
# Answer one question (the front doors)
streamlit run app.py
python -m src.verify_cli --pyq 54
python -m src.verify_cli --file myquestion.txt --show-evidence
python -m src.verify_cli --pyq 54 --policy closed_world
echo "..." | python -m src.verify_cli --save result.json
# Dependency-free self-check of the stdlib BM25 index
python -m src.retrieval.lexical_index
```
