# System Design — Source-Verified UPSC Polity MCQ Verifier

**Author:** Charan Kumar Basam  
**Cohort:** AI Engineering  
**Date:** 2026-09-27

## Related Documents

| Document | Role |
|---|---|
| `DESIGN.md` | Capstone design document covering the problem, data surface, architecture rationale, comparative evaluation, and failure analysis |
| `EVALUATION.md` | Measurement record containing evaluation arms, null results, and the measured noise band |
| `docs/technical_documentation.md` | Earlier narrative document written before the re-measurement pass |
| `docs/retrieval_baseline.md` | Retrieval benchmark and baseline notes |
| `docs/rag_baseline.md` | Vanilla RAG baseline notes |
| `docs/upsc_polity_capstone_development_log.txt` | Chronological development log |

This document describes **what the system is and how it is built**: its components, interfaces, control flow, and design constraints.

It is intentionally separate from the architectural argument in `DESIGN.md` and the quantitative measurements in `EVALUATION.md`.

---

# 1. Scope and System Status

## 1.1 Direct-MCQ Verification

The system now supports questions that do not contain numbered statements through:

```text
src/verification/direct_mcq_verifier.py
```

The direct-MCQ path performs question-level and option-focused semantic retrieval followed by cross-encoder reranking. Retrieved evidence is deduplicated before being passed to the verifier.

Each option A–D is evaluated as one of:

- `SUPPORTED`
- `RULED_OUT`
- `INSUFFICIENT`

For best-answer questions, **answer fit is evaluated separately from isolated factual correctness**.

The Python layer selects an answer only when:

1. exactly one option is supported,
2. the other three options are ruled out,
3. supporting evidence is available for the selected option.

The official answer key is withheld during verification.

This extension has offline behavioral tests but is **not included in the saved end-to-end measurements**. The existing numbered-statement evaluation pipeline remains unchanged.

## 1.2 In Scope

The system covers:

- Indian Polity questions
- UPSC Civil Services Preliminary Examination questions
- Verification against the Constitution of India
- Verification against NCERT Polity material

## 1.3 Out of Scope

The following are intentionally excluded:

- Current affairs
- Subjects other than Polity
- Open-ended tutoring dialogue
- Questions whose answers cannot be determined from the available primary text

## 1.4 Current Modes

Two modes currently exist in the implementation.

| Mode | Implementation | Evaluated? |
|---|---|---|
| **Verify Existing Question** | `src/evaluation/evaluate_verified_pyqs.py`, `src/verification/*` | **Yes** — 13 questions, 37 statements, full metric suite |
| **Generate and Gate** | `src/generation/`, `src/orchestration/` | **Yes, generation-loop measurement exists; the original end-to-end metrics remain distinct** |

The two modes have different measurement histories. The verification path is the primary measured system, while the generation path has its own evaluation workflow and should not be conflated with the PYQ verification benchmark.

---

# 2. System Context

## 2.1 Reliability Problem

UPSC Prelims Polity questions commonly contain:

- a question stem,
- two to four numbered statements,
- answer options describing combinations of those statements.

Examples include:

```text
I and III only
II and IV only
Only two
None
```

The practical failure mode for a learner is often not:

> “I don't know the answer.”

It is:

> “I don't know which statement I got wrong.”

A conventional LLM tutor can make this problem worse by producing fluent explanations and citing constitutional Articles that it has not actually verified.

The system therefore prioritizes:

**per-statement adjudication + evidence provenance + controlled abstention**

rather than fluent answer generation.

## 2.2 Governing Design Principle

> **Generation and verification are separate responsibilities, and the LLM never selects the final answer.**

The LLM is responsible for answering one narrowly defined question:

> Does this individual claim hold against this evidence set?

Everything surrounding that decision is deterministic Python logic:

- question parsing,
- claim construction,
- evidence retrieval,
- verdict aggregation,
- option mapping,
- abstention decisions.

This separation makes abstention a defined policy rather than an informal model behavior.

## 2.3 Architecture Framing

The project originally focused more heavily on generated MCQs. During implementation, the primary measured system shifted toward **verification of existing UPSC PYQs**.

The current architecture therefore contains both:

1. an existing-question verification pipeline, and
2. a generate-and-gate pipeline.

These should be treated as separate capabilities rather than as a single evaluation target.

---

# 3. Architecture Overview

## 3.1 Existing-Question Verification Flow

```text
UPSC Question
     │
     ▼
┌───────────────────────────────┐
│ Question Parser               │
│ No LLM                        │
│                               │
│ • lead-in                     │
│ • numbered items              │
│ • closing                     │
│ • options                     │
│ • question format            │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Claim Builder                 │
│ No LLM                        │
│                               │
│ Build one propositional       │
│ claim per statement           │
└───────────────┬───────────────┘
                │
        ┌───────┴────────┐
        │                │
        ▼                ▼
 Ordinary Claim     Absence Claim
        │                │
        ▼                ▼
┌────────────────┐ ┌──────────────────┐
│ Claim Retriever│ │ Negative Claim   │
│                │ │ Checker          │
│ Semantic       │ │                  │
│ Retrieval      │ │ Full-corpus scan │
│       ↓        │ │                  │
│ Cross Encoder  │ │ All 1,149 chunks │
│ Reranking      │ │ No LLM           │
│                │ │                  │
│ candidate=20   │ │                  │
│ top_k=5        │ │                  │
└───────┬────────┘ └────────┬─────────┘
        │                   │
        └─────────┬─────────┘
                  ▼
        ┌─────────────────────┐
        │ Fact Verifier       │
        │ LLM                 │
        │                     │
        │ SUPPORTED           │
        │ CONTRADICTED        │
        │ INSUFFICIENT        │
        │                     │
        │ + supporting pages │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ Statement Aggregator│
        │ No LLM              │
        │                     │
        │ CONTRADICTED → C    │
        │ all SUPPORTED → S   │
        │ otherwise → I       │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ Answer Mapping      │
        │ No LLM              │
        │                     │
        │ strict              │
        │ closed_world        │
        │ elimination         │
        └──────────┬──────────┘
                   │
                   ▼
          Option Letter
               OR
       Abstain + Typed Reason
```

## 3.2 Generate-and-Gate Flow

The generation workflow is implemented through LangGraph:

```text
START
  │
  ▼
generate_mcq
  │
  ▼
extract_claims
  │
  ▼
verify_claims
  │
  ▼
verify_answer_key
  │
  ▼
audit_quality
  │
  ▼
decide
  │
  ├──── REVISE ────► generate_mcq
  │
  └──── ACCEPT / REJECT ────► END
```

Three independent gates contribute to the final decision:

1. per-claim factual verification,
2. answer-key verification,
3. question-quality auditing.

When a gate fails, its reason is added to `failure_reasons`. Those reasons are supplied to the next generation attempt.

---

# 4. Component Architecture

## 4.1 Ingestion

Located under:

```text
src/ingestion/
```

| Component | Responsibility |
|---|---|
| `extract_text.py` | Extract page-aware text from Constitution and NCERT PDFs |
| `prepare_documents.py` | Clean text and create page-level records |
| `chunk_documents.py` | Create 1,000-character chunks with 150-character overlap |
| `save_chunks_jsonl.py` | Persist chunks to `data/processed/chunks.jsonl` |

Each chunk retains source and page metadata so that evidence can be traced back to the original document.

## 4.2 Retrieval

Six retrieval implementations exist.

| Component | Method | Recall@5 | Role |
|---|---|---:|---|
| `semantic_reranker.py` | BGE-small → cross-encoder | **93.75%** | **Production retriever** |
| `semantic_retriever.py` | BGE-small semantic retrieval | 87.50% | Candidate generation |
| `parent_child_retriever.py` | Parent expansion + reranking | 87.50% | Benchmark |
| `hybrid_retriever.py` | BM25 + semantic RRF | 81.25% | Benchmark |
| `bm25_retriever.py` | BM25Okapi | 75.00% | Benchmark |
| `lexical_index.py` | Standard-library BM25 implementation | 75.00% | Benchmark / dependency-free path |

The production retriever uses:

```text
BAAI/bge-small-en-v1.5
        ↓
candidate_k = 20
        ↓
cross-encoder/ms-marco-MiniLM-L-6-v2
        ↓
top_k = 5
```

`lexical_index.py` intentionally reproduces BM25Okapi behavior, including the IDF floor, and contains a self-check to ensure the implementation does not silently diverge from the reference library.

## 4.3 Question Parsing and Claim Construction

Located under:

```text
src/verification/
```

### Question Parser

`question_parser.py` performs deterministic parsing into:

- lead-in,
- numbered items,
- closing text,
- options.

It also classifies question structures such as:

- `STATEMENTS`
- `STEM_DISTRIBUTED`
- `PAIRS`

Unicode character normalization is performed during parsing.

### Claim Builder

`claim_builder.py` converts each numbered item into a standalone proposition.

Supported binding modes include:

```text
none
trailing_predicate
leading_predicate
leading_condition
pair
failed
```

The builder also exposes:

```text
BuiltClaim.is_propositional
```

### Why Deterministic Claim Construction?

A numbered statement may be only a noun phrase:

```text
Police
```

while the actual predicate exists in the question stem.

The system binds the pieces deterministically, producing something like:

```text
Police is included in the Seventh Schedule.
```

Two safeguards are applied:

1. no token may be introduced that was not present in the source question,
2. split pairs must be reconstructed correctly.

The rationale is that an LLM-generated paraphrase is technically a new claim. The corpus was not retrieved for that new claim.

The measured claim-level improvement was significant:

- propositional claims: **70.27% → 100%**
- false `SUPPORTED` verdicts: **7–8 → 0**

This was one of the strongest measured effects in the project.

## 4.4 Verification

| Component | LLM? | Responsibility |
|---|---|---|
| `claim_retriever.py` | No | Retrieves evidence for a claim |
| `fact_verifier.py` | Yes | Produces three-way fact verdict |
| `negative_claim_checker.py` | No | Performs full-corpus absence checks |
| `answer_key_verifier.py` | Yes | Verifies generated answer keys |
| `mcq_quality_auditor.py` | Yes | Evaluates generated question quality |

### Fact Verifier

The contract is:

```text
verify(claim, evidence)
    →
FactVerificationResult
```

The result contains:

```text
SUPPORTED
CONTRADICTED
INSUFFICIENT
```

along with reasoning and supporting pages.

### Absence Claims

Absence claims require special treatment.

For example:

> The Constitution does not mention “political party”.

This is not a normal retrieval problem. A top-5 retrieval result cannot establish that a term is absent from a corpus.

Therefore:

```text
Absence claim
      ↓
Full scan
      ↓
All 1,149 chunks
      ↓
Deterministic result
```

No LLM is used for this decision.

A recurring implementation defect was constructing `FactVerifier()` without the corpus chunks. That silently disabled the negative-claim checker and allowed absence claims to fall through to the LLM path.

The issue was fixed in `orchestration/nodes.py` and subsequently discovered again in the stability harness, making it an important class of architectural failure rather than a one-off bug.

## 4.5 Generation and Orchestration

| Component | Responsibility |
|---|---|
| `mcq_generator.py` | Generate candidate MCQs |
| `state.py` | Shared `MCQVerificationState` |
| `nodes.py` | LangGraph node implementations |
| `graph.py` | StateGraph and routing |
| `run_workflow.py` | Single-topic execution entry point |

The generator accepts:

```text
topic
difficulty
failure_reasons
```

The `failure_reasons` from a failed attempt are injected into the next generation prompt.

## 4.6 Vanilla RAG Baseline

```text
src/rag/vanilla_rag.py
```

This is **System A**.

Its architecture is deliberately simple:

```text
Question
   ↓
Retrieve
   ↓
LLM
   ↓
Answer
```

It does not perform:

- claim decomposition,
- explicit abstention,
- deterministic option mapping.

It serves as the baseline against which the verified architecture is compared.

---

# 5. Data Design

## 5.1 Corpus

| Source | Pages | Chunks |
|---|---:|---:|
| Constitution of India | 402 | 1,107 |
| NCERT Polity | 20 | 42 |
| **Total** | **422** | **1,149** |

## 5.2 Chunk Schema

`data/processed/chunks.jsonl` contains records of the following form:

```json
{
  "id": 0,
  "text": "…",
  "source": "constitution",
  "document": "…",
  "page": 1,
  "chunk_index": 0
}
```

The page number is carried through the complete verification pipeline.

**Page provenance is a core architectural requirement.**

Every verdict identifies the pages used as evidence. Auditing these citations was also responsible for uncovering one of the project's most important findings.

## 5.3 Evaluation Dataset

The original evaluation dataset is:

```text
data/evaluation/pyq_2025_polity.json
```

It contains:

- 13 UPSC 2025 Polity questions,
- 37 numbered statements,
- questions Q54–Q66,
- extracted options,
- official answers.

The dataset was held out from prompt development.

It was created using:

```text
src/evaluation/extract_pyqs.py
```

from the UPSC question PDF.

The 2026 UPSC question and answer-key files exist in the repository but are currently unused by the source code. They represent an obvious opportunity for expanding the evaluation dataset.

## 5.4 Answer Labels

Two answer-key label sets are maintained:

### Stored

The answer extracted directly into:

```text
official_answer
```

### Audited

The answer after corrections performed by:

```text
src/evaluation/label_audit.py
```

Both are reported because conclusions should not depend on a single potentially incorrect extraction.

---

# 6. Interface Contracts

All LLM-facing structures use Pydantic models and are parsed through:

```python
client.responses.parse(..., text_format=Model)
```

This ensures malformed responses become validation failures instead of silently entering downstream logic.

### MCQ

```python
class MCQ(BaseModel):
    question: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    correct_answer: str = Field(pattern="^[ABCD]$")
    explanation: str
```

### Fact Verification

```python
class FactVerificationResult(BaseModel):
    verdict: str
    reasoning: str
    supporting_pages: list[int]
```

where:

```text
verdict ∈ {
    SUPPORTED,
    CONTRADICTED,
    INSUFFICIENT
}
```

### Quality Audit

```python
class QualityAuditResult(BaseModel):
    unambiguous: bool
    single_best_answer: bool
    plausible_distractors: bool
    appropriate_wording: bool
    topic_relevant: bool
    issues: list[str]
    overall_quality: str = Field(pattern="^(PASS|FAIL)$")
```

The regex constraints on fields such as `correct_answer` and `overall_quality` ensure that downstream Python logic never has to branch on arbitrary free-form text.

---

# 7. Orchestrated Control Flow

The generation workflow uses one shared `TypedDict`:

```text
MCQVerificationState
```

Six nodes operate on this state.

| Node | Reads | Writes |
|---|---|---|
| `generate_mcq` | topic, difficulty, failure reasons | MCQ, retry count, reset failure reasons |
| `extract_claims` | MCQ | claims |
| `verify_claims` | claims | evidence and fact verifications |
| `verify_answer_key` | MCQ | answer evidence and verification |
| `audit_quality` | MCQ | quality audit |
| `decide` | all gate results and retry state | decision and failure reasons |

Each node returns a partial dictionary which LangGraph merges into the shared state.

## Routing

```text
REVISE
   ↓
generate_mcq

ACCEPT / REJECT
   ↓
END
```

Unknown decisions raise an exception rather than silently terminating the graph.

## Retry Semantics

The implementation uses:

```text
attempts = retries + 1
```

The counter is incremented before verification begins, meaning that `retry_count` represents attempts already made rather than retries consumed.

With:

```text
max_retries = 2
```

the workflow can perform up to three generation attempts.

The maximum path is therefore:

```text
3 attempts × 6 nodes = 18 steps
```

which remains within the default LangGraph recursion limit.

## Attempt History

The returned final state does not preserve every intermediate attempt.

For evaluation, the system therefore consumes:

```python
workflow.stream(..., stream_mode="updates")
```

This allows the evaluation harness to observe every `decide` output and reconstruct the behavior of the loop.

---

# 8. Retrieval Configuration

Retrieval depth is centrally defined in:

```text
src/retrieval/retrieval_config.py
```

The production pipeline and retrieval benchmarks now share the same configuration.

The earlier implementation had a mismatch:

```text
retrieval benchmark → k=5
verification pipeline → k=3
```

This meant the reported retrieval result described a configuration different from the one actually used by the verification system.

That mismatch has been corrected.

However, the correction is **not treated as an accuracy finding**.

The k=3 versus k=5 comparison produced different results depending on the label set, and the difference was driven by a single unstable claim.

Therefore:

> **The evidence supports fixing the configuration mismatch, but does not support claiming that increasing retrieval depth improves accuracy.**

The retrieval ablation provides stronger evidence against unnecessary complexity:

- hybrid RRF: 81.25%
- parent-child: 87.50%
- semantic + reranker: 93.75%

The simpler production retrieval pipeline therefore remains preferred.

---

# 9. Answer Mapping and Abstention

The answer-mapping layer supports three policies.

| Policy | Interpretation of unresolved statements | Behavior |
|---|---|---|
| `strict` | Unknown | Abstain if any statement is unresolved |
| `closed_world` | False | Treat unresolved statements as false |
| `elimination` | Unknown, but eliminate incompatible options | Answer if exactly one option survives |

The production policy is:

```text
STRICT
```

Abstention reasons are typed:

- unresolved statement,
- no option matches,
- multiple options match.

## Measured Policy Behavior

`elimination` was identical to `strict` on the current dataset. It never produced an answer where strict would have abstained.

`closed_world` achieved the highest overall accuracy:

```text
53.85%
```

but was rejected because its error rate increased from:

```text
0%
   →
30.77%
```

The system therefore prefers **calibrated abstention over additional incorrect assertions**.

---

# 10. Metrics

The evaluation always reports four metrics together.

| Metric | Definition |
|---|---|
| **Coverage** | Fraction of questions answered |
| **Precision when answered** | Accuracy among answered questions |
| **Overall accuracy** | Correct answers / all questions |
| **Overall error rate** | Wrong confident answers / all questions |

These satisfy:

```text
accuracy + error rate + abstention rate = 1
```

A single metric is insufficient.

For example:

- abstain on every question → zero errors but zero usefulness,
- answer every question → maximum coverage but potentially high error.

The four-metric view exposes the actual trade-off.

## Headline Comparison

| System | Coverage | Precision | Accuracy | Error Rate |
|---|---:|---:|---:|---:|
| **A — Vanilla RAG** | 100% | 46.15% | 46.15% | **53.85%** |
| **B — Verified System** | 30.77% | **100%** | 30.77% | **0%** |

System A produced false answers on 7 of 13 audited questions.

System B produced false answers on:

```text
0 of 13
```

The verified system therefore trades coverage for a substantially lower error rate.

---

# 11. Determinism and Reproducibility

## 11.1 Temperature Configuration

The Responses API defaults to:

```text
temperature = 1.0
```

All LLM call sites explicitly configure:

```text
temperature = 0
```

The `FactVerifier` retains an optional temperature parameter solely so the stability harness can reproduce the original unpinned condition.

This creates one shared implementation for both:

- pinned measurements,
- unpinned measurements.

## 11.2 Temperature 0 Is Not Determinism

Five-repeat measurements over 37 claims produced:

| Configuration | Stable | Unstable |
|---|---:|---:|
| Unpinned | 28 / 37 | 9 |
| Pinned | 35 / 37 | 2 |

Therefore:

> **Temperature 0 improves stability but does not guarantee deterministic outputs.**

One claim produced all three possible verdicts across repeated runs.

## 11.3 Decision-Boundary Fragility

A controlled 2×2 experiment demonstrated that even:

- three whitespace characters, or
- changing the endpoint

could alter a verdict.

Consequently, small accuracy differences that fall within the measured noise band are not treated as meaningful improvements.

---

# 12. Caching and Performance

| Mechanism | Location | Purpose |
|---|---|---|
| `lru_cache(maxsize=1)` | `nodes.py` | Prevent repeated reconstruction of chunks, retriever, and verifier |
| Verdict cache | `data/evaluation/` | Reuse repeated claim results across ablation arms |
| `candidate_k=20 → top_k=5` | `claim_retriever.py` | Limit cross-encoder computation |

Without caching, one revision loop could repeatedly parse and embed the corpus.

The cached factories therefore reduce unnecessary computation across repeated workflow steps.

`get_chunks()` returns a tuple because the cached factory requires hashable arguments.

---

# 13. Evaluation Harness

The expanded benchmark uses:

```text
src/evaluation/evaluate_benchmark.py
```

with:

```text
33 development questions
42 test questions
```

The benchmark supports:

- `--check`
- `--dry-run`
- `--run`

Dataset, code, and corpus hashes prevent incompatible resumed runs from being mixed.

Answer keys are kept outside the answering input.

API failures are counted separately from abstentions.

The benchmark output is stored under:

```text
data/evaluation/benchmark/
```

The expanded dataset currently has not produced final model results.

## 13.1 Evaluation DAG

The production measurement chain is orchestrated by:

```text
src/evaluation/run_all.py
```

Conceptually:

```text
evaluate_verified_pyqs
        │
        ▼
replay_answer_mapping
        │
        ▼
direct_option_verifier
        │
        ▼
system_b_metrics
        │
        ▼
selective_metrics
        │
        ▼
verdict_stability
        │
        ▼
judge_llm
```

Each stage declares:

- inputs,
- outputs,
- dependencies.

The runner uses modification times to ensure downstream reports are not generated from stale inputs.

This was introduced after two real incidents:

1. a report was generated from stale inputs,
2. evaluation stages were executed in the wrong order.

The DAG therefore makes consistency between evaluation artifacts an explicit, checkable condition.

## 13.2 LLM-as-Judge

The judge evaluates System A and System B head-to-head.

Measured results include:

```text
Head-to-head preference: 12–1–0
Hallucinated claims:     19 → 1
```

The judge preferred the verified system on most questions, including several where it abstained.

However, this metric is treated as supporting evidence rather than the primary proof of correctness because the judge may reward calibrated abstention.

The reduction in hallucinated claims is the stronger non-circular observation.

---

# 14. Environment and Dependencies

| Concern | Resolution |
|---|---|
| Corporate TLS interception | `src/tls_trust.py` uses the operating-system trust store |
| LLM | `gpt-4o-mini` |
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Secrets | `OPENAI_API_KEY` in `.env`, excluded by `.gitignore` |
| Orchestration | LangGraph `StateGraph` |

A corporate TLS interception issue initially appeared as a Hugging Face network failure. The actual cause was a missing corporate root certificate in `certifi`.

The project therefore introduced:

```text
src/tls_trust.py
```

using:

```python
truststore.inject_into_ssl()
```

This corrected the environment problem and prevented the infrastructure failure from being misinterpreted as a retrieval-system limitation.

---

# 15. Failure Modes and Defences

| Failure Mode | Defence |
|---|---|
| LLM asserts a fact without evidence | Require supporting pages and audit citations |
| LLM paraphrases a statement | Deterministic claim construction with no-new-token guard |
| Absence claim evaluated using top-k retrieval | Full-corpus `NegativeClaimChecker` |
| Fact verifier constructed without corpus | Explicit corpus dependency |
| Retrieval benchmark and production k differ | Centralized retrieval configuration |
| Metrics generated from stale reports | DAG freshness checks |
| Single run treated as reliable measurement | Repeated runs and noise-band analysis |
| Temperature defaults drift | Explicit temperature configuration |
| Accuracy-only configuration selection | Coverage, precision, accuracy, and error-rate suite |

The recurring absence-checker defect is particularly important because it illustrates a class of failures where a missing dependency silently disables a safety mechanism.

---

# 16. Known Gaps and Next Steps

## 16.1 Generation-Loop Measurement

The generation workflow has now been measured across:

- two question formats,
- 15 topics per format.

The measurement showed:

- simple format acceptance: **53.33%**
- multi-statement format acceptance: **0/15**

This indicates that the quality gate is highly sensitive to question format.

The appropriate interpretation is not that the generator is universally poor, but that the current quality threshold is not calibrated equally for different MCQ structures.

The remaining limitation is that each topic currently receives one generation draw, so the generation-loop noise band remains unquantified.

## 16.2 Evaluation Dataset Size

The existing evaluation set contains only:

```text
13 questions
37 statements
```

One question therefore represents approximately:

```text
7.7 percentage points
```

of accuracy.

Because the measured noise can span roughly two questions, the dataset is too small for many fine-grained architectural comparisons.

The unused 2026 UPSC dataset provides a natural next evaluation set.

## 16.3 Repetition Count

The current stability experiment uses:

```text
REPEATS = 5
```

This is insufficient for a strong estimate of the true instability distribution.

Repeated harness runs have already produced different instability counts, demonstrating that the measured counts should be interpreted as lower bounds.

## 16.4 Retrieval Benchmark Re-run

The retrieval benchmark table has not yet been fully re-run against the final code state.

Therefore, its results should be treated as the current recorded benchmark rather than a freshly reproduced final measurement.

## 16.5 Abstention Quality

Nine abstentions were identified in the evaluation.

Of these:

```text
6 → pipeline defects
2 → corpus limitations
1 → partial case
```

This is an important gap because improving those six pipeline failures could increase coverage without necessarily increasing error.

## 16.6 Prompt Complexity

The fact-verifier prompt has grown to 26 rules.

The current direction is to **decompose the verifier logic rather than continue adding prompt rules**.

## 16.7 Git LFS Dependency

The repository marks:

```text
prelims2026Questions.pdf
```

as a Git LFS file even though the implementation does not currently use it.

This creates unnecessary dependency on LFS and can interfere with Git bundle transport.

---

# Appendix A — Project File Map

```text
src/
├── ingestion/
│   ├── extract_text
│   ├── prepare_documents
│   ├── chunk_documents
│   └── save_chunks_jsonl
│
├── retrieval/
│   ├── semantic_reranker        # production
│   ├── semantic_retriever
│   ├── bm25_retriever
│   ├── hybrid_retriever
│   ├── parent_child_retriever
│   └── lexical_index
│
├── verification/
│   ├── question_parser
│   ├── claim_builder
│   ├── claim_extractor
│   ├── claim_retriever
│   ├── fact_verifier
│   ├── negative_claim_checker
│   ├── answer_key_verifier
│   └── mcq_quality_auditor
│
├── generation/
│   └── mcq_generator
│
├── orchestration/
│   ├── state
│   ├── nodes
│   ├── graph
│   └── run_workflow
│
├── rag/
│   └── vanilla_rag
│
├── evaluation/
│   ├── run_all
│   ├── evaluate_verified_pyqs
│   ├── evaluate_vanilla_rag
│   ├── answer_mapping
│   ├── replay_answer_mapping
│   ├── direct_option_verifier
│   ├── selective_metrics
│   ├── system_b_metrics
│   ├── system_c_pipeline
│   ├── verdict_stability
│   ├── judge_llm
│   ├── label_audit
│   ├── extract_pyqs
│   └── evaluate_generation_loop
│
├── verify_cli.py
└── tls_trust.py

data/
├── raw/
│   ├── constitution
│   ├── ncert
│   ├── upsc
│   └── answer_keys
│
├── processed/
│   ├── chunks.jsonl
│   └── extracted text
│
└── evaluation/
    ├── datasets
    ├── results
    ├── caches
    └── logs

docs/
├── SYSTEM_DESIGN.md
├── technical_documentation.md
├── retrieval_baseline.md
└── rag_baseline.md

app.py
README.md
DESIGN.md
EVALUATION.md
```

---

# Appendix B — Entry Points

The project now exposes two primary entry points.

| Entry Point | Purpose |
|---|---|
| `src/verify_cli.py` | Scriptable verification of an individual question |
| `app.py` | Streamlit demonstration interface |

The CLI reuses the existing verification pipeline rather than duplicating logic.

The Streamlit application also reuses the same claim-building and answer-mapping components.

Changing the abstention policy does not require another LLM call because verdicts are cached. The application recomputes only deterministic Python stages:

```text
determine_statement_statuses
        ↓
map_answer
```

This makes policy comparison effectively instantaneous and turns the coverage/error trade-off into an interactive UI control.

---

# Appendix C — Commands

## Prepare Corpus

```bash
python -m src.ingestion.prepare_documents
python -m src.ingestion.chunk_documents
```

## Production Evaluation

```bash
python -m src.evaluation.run_all
python -m src.evaluation.run_all --check
python -m src.evaluation.run_all --only selective_metrics
```

## Baseline and Ablation

```bash
python -m src.evaluation.evaluate_vanilla_rag
python -m src.evaluation.system_c_pipeline
```

## Generate-and-Gate Workflow

```bash
python -m src.orchestration.run_workflow
```

## Evaluate Generation Loop

```bash
python -m src.evaluation.evaluate_generation_loop --format simple
python -m src.evaluation.evaluate_generation_loop --format statements
python -m src.evaluation.evaluate_generation_loop --summarise-only
```

## Verify a Question

```bash
streamlit run app.py

python -m src.verify_cli --pyq 54

python -m src.verify_cli \
    --file myquestion.txt \
    --show-evidence

python -m src.verify_cli \
    --pyq 54 \
    --policy closed_world

echo "..." | python -m src.verify_cli --save result.json
```

## BM25 Self-Check

```bash
python -m src.retrieval.lexical_index
```

---

# Final Architecture Summary

The system is intentionally designed around a strict separation of responsibilities:

```text
                ┌─────────────────────┐
                │    Question Input   │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Deterministic Parse │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Deterministic Claim │
                │      Building       │
                └──────────┬──────────┘
                           │
                    ┌──────┴──────┐
                    │             │
                    ▼             ▼
              Retrieval      Absence Scan
                    │             │
                    └──────┬──────┘
                           ▼
                ┌─────────────────────┐
                │    LLM Verifier     │
                │                     │
                │ Supported           │
                │ Contradicted        │
                │ Insufficient        │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Deterministic       │
                │ Aggregation +       │
                │ Answer Mapping      │
                └──────────┬──────────┘
                           │
                    ┌──────┴──────┐
                    │             │
                    ▼             ▼
                 Answer        Abstain
```

The central architectural principle is simple:

> **Use the LLM for bounded factual judgment, not for deciding what the question means or what the final answer should be.**

The system consequently favors provenance, deterministic control, explicit uncertainty, and measurable failure modes over maximum answer coverage.