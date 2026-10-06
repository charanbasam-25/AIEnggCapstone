# Design Document — Source-Verified UPSC Polity MCQ Verifier

**Author:** Charan Kumar Basam  
**Cohort:** AI Engineering  
**Date:** 27 September 2026  

**Companion document:** [`EVALUATION.md`](EVALUATION.md) — complete measurement record, experimental arms, negative results, and detailed failure analysis.

---

# 1. Problem Statement

UPSC Prelims Polity questions are frequently structured as **multi-statement questions**:

- A question stem
- Two to four numbered statements
- Multiple-choice options such as:
  - "I and III only"
  - "II and IV only"
  - "Only two"
  - "None"

The challenge is therefore not simply determining whether an answer is correct.

A student often needs to know:

> **Which individual statement is wrong, and why?**

This becomes particularly problematic with LLM-based tutors. An LLM may produce a fluent explanation containing constitutional Articles or provisions that were never actually verified against the source material.

## 1.1 Narrow Problem

This project addresses a deliberately constrained problem:

> **Given a UPSC Polity question, independently determine whether each numbered statement is supported by the Constitution or other approved source material, provide page-level evidence, and abstain when the available corpus cannot establish the answer.**

The system is therefore designed around **source verification rather than answer generation**.

It is acceptable for the system to answer fewer questions than a guessing system if the questions it answers are better supported.

The primary objective is:

> **Do not assert unsupported constitutional facts.**

This objective was defined before the evaluation and determines how the system treats coverage, accuracy, and abstention.

---

# 1.2 Non-Goals

The current system intentionally does not attempt to solve:

- Current affairs
- Non-Polity UPSC subjects
- General UPSC tutoring conversations
- Open-ended tutoring dialogue
- Questions whose answers cannot be established from the approved corpus

The project is intentionally narrow so that the verification problem can be measured independently.

---

# 2. Data Surface

The system operates over a fixed, page-aware document corpus.

| Source | Pages | Chunks |
|---|---:|---:|
| Constitution of India | 402 | 1,107 |
| NCERT Polity | 20 | 42 |
| **Total** | **422** | **1,149** |

The PDFs are processed using page-aware extraction and then divided into approximately:

- 1,000-character chunks
- 150-character overlap

Each chunk retains provenance metadata:

```text
source
document
page
chunk_index
```

Page provenance is a core requirement rather than optional metadata.

Every verification result must identify the pages used to support its conclusion.

This also makes the evidence independently auditable.

---

# 2.1 Evaluation Dataset

The initial evaluation set consists of:

> **UPSC 2025 Prelims Polity Q54–Q66**

This contains:

- 13 questions
- 37 numbered statements

The questions were held out from prompt development.

An expanded benchmark was subsequently introduced separately for regression testing.

---

# 3. System Architecture

The system follows a **RAG architecture with verification separated from answer generation**.

The central design principle is:

> **One model should not both produce an answer and verify its own answer.**

The pipeline is:

```text
                    ┌──────────────────────────────┐
                    │         UPSC Question        │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                         Deterministic Parser
                                   │
                                   ▼
                    Standalone Propositional Claims
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
            Claim-Level Retrieval          Absence-Claim
                    │                         Full Scan
                    ▼                             │
        Semantic Retrieval + Reranker             │
                    │                             │
                    └──────────────┬──────────────┘
                                   ▼
                           Fact Verifier
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
                SUPPORTED     CONTRADICTED   INSUFFICIENT
                    │              │              │
                    └──────────────┴──────────────┘
                                   │
                                   ▼
                       Deterministic Option Mapping
                                   │
                         ┌─────────┴─────────┐
                         ▼                   ▼
                    Option Letter         Abstain
                                           + Reason
```

The generated-question workflow additionally uses a LangGraph `StateGraph`:

```text
Generate
   ↓
Extract Claims
   ↓
Verify Claims
   ↓
Verify Answer Key
   ↓
Audit Quality
   ↓
ACCEPT / REVISE / REJECT
   ↺
```

When the output fails verification or quality checks, the failure reasons are passed back into the generation stage.

---

# 3.1 Design Principle 1 — Deterministic Claim Construction

Claims are constructed by deterministic code rather than asking an LLM to rewrite the statements.

This is necessary because UPSC statements are often incomplete fragments.

For example:

```text
Question stem:
Which of the following are included in the Seventh Schedule?

Statement:
"Police"
```

The statement itself is not a complete proposition.

Its predicate is supplied by the question stem.

The system therefore binds the statement into a standalone proposition before verification.

The binding process is protected by two checks:

1. **Unsupported-token check**  
   The constructed claim cannot introduce tokens that were not present in the source question.

2. **Reconstruction check**  
   Split statement/predicate pairs must reconstruct correctly.

This makes the verifier operate on actual propositions rather than isolated fragments.

---

# 3.2 Design Principle 2 — Three-Way Verification

The LLM does not select the final answer option.

Instead, it evaluates one claim against its evidence and returns one of:

```text
SUPPORTED
CONTRADICTED
INSUFFICIENT
```

The final option is selected deterministically by Python.

This separation is important because it turns abstention into an explicit **decision policy** rather than an implicit model behavior.

The system can therefore distinguish between:

- "The claim is false."
- "The claim is true."
- "There is not enough evidence to decide."

---

# 3.3 Design Principle 3 — Full-Corpus Handling of Absence Claims

Some claims contain a universal negative such as:

> "The Constitution does not mention political parties."

A ranked retriever is not an appropriate instrument for this type of statement.

Retrieving the top five chunks and failing to find a phrase does not prove that the phrase does not exist elsewhere.

The system therefore routes absence claims through a **full-corpus scan**.

The scan covers all:

> **1,149 chunks**

and does not use an LLM to determine whether the phrase exists.

This makes absence detection fundamentally different from normal semantic retrieval.

---

# 4. Core Processing Flow

The end-to-end pipeline follows six stages.

## Step 1 — Parse the Question

The deterministic parser separates:

- Question lead-in
- Numbered statements
- Closing text
- Answer options

---

## Step 2 — Construct Standalone Claims

Each numbered statement is converted into an independently verifiable proposition.

This ensures the verifier has enough context to determine whether the statement is actually true or false.

---

## Step 3 — Retrieve Evidence

For each claim:

```text
Semantic Retrieval
       ↓
Candidate passages
       ↓
Cross-Encoder Reranking
       ↓
Top-5 evidence
```

Absence claims bypass this process and use the full-corpus scanner.

---

## Step 4 — Verify Each Claim

The verifier receives:

- One claim
- Its retrieved evidence
- Verification rules

It produces:

- Verdict
- Reasoning
- Supporting page references

The verifier is explicitly instructed to reason only from the supplied evidence.

---

## Step 5 — Aggregate Statement Verdicts

Individual evidence results are combined into one verdict per statement.

The aggregation rule is:

```text
Any CONTRADICTED
        ↓
CONTRADICTED

All SUPPORTED
        ↓
SUPPORTED

Otherwise
        ↓
INSUFFICIENT
```

This keeps claim-level reasoning separate from final answer selection.

---

## Step 6 — Map Statements to an Option

The deterministic mapper evaluates the statement verdicts against the answer options.

The current production policy is **STRICT**.

The system abstains when:

- At least one statement is unresolved.
- No option matches.
- Multiple options remain possible.

The output is therefore either:

```text
Option letter + evidence
```

or:

```text
ABSTAIN + typed reason
```

---

# 5. Retrieval Design

Five retrieval approaches were evaluated using a 16-query benchmark.

Metric:

> **Recall@5**

| Approach | Recall@5 |
|---|---:|
| BM25 | 75.00% |
| Hybrid BM25 + Semantic RRF | 81.25% |
| Semantic — `BAAI/bge-small-en-v1.5` | 87.50% |
| Parent-child + reranker | 87.50% |
| **Semantic + Cross-Encoder Reranker** | **93.75%** |

The selected production configuration is:

```text
BAAI/bge-small-en-v1.5
          ↓
Semantic retrieval
          ↓
20 candidates
          ↓
MS-MARCO MiniLM cross-encoder
          ↓
Top evidence
```

## 5.1 Why Hybrid RRF Was Not Selected

Hybrid retrieval was expected to outperform semantic retrieval because it combines:

- Lexical matching
- Semantic similarity

However, the measured result was the opposite:

```text
Hybrid RRF: 81.25%
Semantic:   87.50%
```

The corpus contains conceptual constitutional language where semantic similarity was more useful than lexical fusion.

This is an important negative result:

> **More retrieval sophistication did not automatically produce better retrieval.**

---

# 5.2 Retrieval Benchmark Limitation

The retrieval benchmark contains only 16 queries.

One query therefore represents:

> **6.25 percentage points**

The 93.75% result establishes the best observed configuration, but does not prove that the reranker has a statistically significant advantage over semantic retrieval.

The benchmark should therefore be treated as a configuration-selection experiment rather than a final generalization claim.

---

# 6. Evaluation Summary

The full measurement record is maintained in [`EVALUATION.md`](EVALUATION.md).

A critical issue discovered during evaluation was that the verifier was initially stochastic.

`FactVerifier` did not explicitly set temperature, while the vanilla RAG baseline was already pinned at temperature 0.

After pinning all relevant call sites, 37 claims were reverified five times.

| Condition | Stable claims | Unstable claims |
|---|---:|---:|
| Unpinned | 28/37 | 9 |
| **Pinned, temperature = 0** | **35/37** | **2** |

Temperature 0 therefore improved stability but did not make the verifier deterministic.

Across the five resampled pipeline runs, the observed accuracy spread was:

> **15.38 percentage points — equivalent to two questions.**

This establishes the practical noise threshold for the historical 13-question benchmark.

Therefore:

> **Differences of one question should not be interpreted as architectural improvements.**

---

# 6.1 End-to-End Comparison

The primary comparison uses:

- 13 questions
- Audited labels
- Strict policy
- Shipped semantic + reranker retriever

| System | Coverage | Precision | Accuracy | Error |
|---|---:|---:|---:|---:|
| **A — Vanilla RAG** | **100%** | 46.15% | **46.15%** | **53.85%** |
| **B — Verified system** | 30.77% | **100%** | 30.77% | **0%** |

Vanilla RAG therefore achieves higher accuracy because it answers every question.

The verified system achieves lower coverage but eliminates false final answers in this experiment.

The result should therefore be interpreted as:

> **Verification improves error control, not overall accuracy.**

---

# 6.2 What the A/B Result Establishes

Vanilla RAG produces:

> **7 wrong answers out of 13 under audited labels.**

The verified system produces:

> **0 wrong answers out of 13.**

Under the stored labels, vanilla RAG produces 8 wrong answers.

The resulting error-rate difference is approximately:

> **46–54 percentage points**

or approximately six questions.

This is the strongest end-to-end result because it lies well outside the two-question noise band.

The trade-off is coverage:

```text
Vanilla RAG:
13/13 answered

Verified system:
4/13 answered
```

Therefore, whether the architecture is preferable depends on the objective.

If the objective is:

> "Answer as many questions as possible."

Vanilla RAG performs better on this dataset.

If the objective is:

> "Do not assert unsupported constitutional facts."

The verified architecture performs substantially better.

---

# 6.3 Direct Option Verification

A second architecture was evaluated:

> Ask the LLM to verify an entire answer option rather than individual statements.

This approach avoids:

- Claim construction
- Per-statement verification
- Deterministic mapping

However, an option typically contains two to four propositions.

The model therefore has to verify the entire conjunction simultaneously.

The result:

> **2 of 13 questions answered**

Both were correct.

```text
Coverage: 15.38%
Precision: 100%
Error: 0%
```

When additional evidence was provided by combining question-level and statement-level passages, coverage fell to:

> **1 of 13**

This supports the decomposition strategy.

More evidence does not necessarily make conjunctive verification easier because every additional proposition creates another opportunity for the complete option to become unsupported.

---

# 6.4 LLM-as-a-Judge

A blind LLM-as-a-judge experiment compared vanilla RAG and the verified system.

| Dimension | Vanilla RAG | Verified |
|---|---:|---:|
| Factual correctness | 2.77 | 3.69 |
| Evidence faithfulness | 2.46 | 4.69 |
| Reasoning validity | 2.69 | 4.54 |
| Epistemic honesty | 2.38 | **5.00** |

Head-to-head preference:

```text
Verified: 12
Vanilla:   1
Tie:       0
```

Hallucinated claims:

```text
Vanilla RAG: 19
Verified:     1
```

Questions containing hallucinations:

```text
Vanilla RAG: 12
Verified:     1
```

These results are encouraging but require an important qualification.

The judge explicitly rewards calibrated uncertainty, while the verified system is designed to abstain.

The judge therefore partially rewards the system for behavior built into its architecture.

The strongest independent signal is the hallucination count:

> **19 → 1**

because hallucinated claims are identified against the cited evidence rather than purely by subjective judge preference.

---

# 6.5 Claim Construction

The end-to-end metrics cannot distinguish bound claims from verbatim claims on this dataset.

However, claim-level analysis clearly can.

| Claim representation | Propositional | False `SUPPORTED` |
|---|---:|---:|
| Verbatim statements | 70.27% | 8 |
| **Bound claims** | **100%** | **0** |

The reason is structural.

A statement such as:

```text
"Police"
```

is not itself a proposition.

It has no truth value until the predicate supplied by the question is attached.

Without binding, the verifier may match the noun phrase against retrieved evidence and return:

```text
SUPPORTED
```

even though it has not actually verified the complete statement.

The deterministic binding stage therefore prevents the verifier from evaluating malformed claims.

This is the strongest claim-level result in the project.

---

# 6.6 Abstention Policy

Three policies were implemented:

### STRICT

Any unresolved statement causes abstention.

### ELIMINATION

Eliminate options that conflict with known verdicts and answer if exactly one option remains.

### CLOSED_WORLD

Interpret `INSUFFICIENT` as false.

STRICT and ELIMINATION produced identical results on the 13-question dataset.

This is a valid null result.

ELIMINATION is theoretically more general, but the current questions do not provide enough partial information to reduce the candidate options to exactly one.

---

## 6.7 CLOSED_WORLD

CLOSED_WORLD produced the highest observed accuracy:

> **53.85% under audited labels**

However, it also increased the error rate from:

```text
STRICT:
0%

CLOSED_WORLD:
30.77%
```

The policy effectively trades abstentions for unsupported assertions.

Because the corpus demonstrably does not contain enough information to answer every question, the assumption:

> "Absence of evidence means false"

is invalid for this system.

Therefore, STRICT remains the production policy.

---

# 6.8 Corpus Answerability

A full lexical scan over the 1,149 chunks established that two questions cannot be answered from the current corpus:

- Q55
- Q66

Therefore, the theoretical ceiling for a system that never guesses is:

> **11/13 = 84.62%**

This is an important distinction.

The system should not be evaluated against an assumption of 100% answerability when the source corpus itself is incomplete.

---

# 6.9 Abstention Quality

The nine historical abstentions were classified as:

| Cause | Count |
|---|---:|
| Pipeline defect | **6** |
| Corpus limitation | 2 |
| Partial corpus limitation | 1 |

Only:

> **22.22%**

of abstentions were clearly justified by corpus limitations.

This is the most important weakness identified by the evaluation.

The system's high abstention rate should therefore **not** be interpreted as pure epistemic caution.

Most abstentions represent pipeline failures where the required evidence was already available.

---

# 7. Key Failure Analysis

The evaluation exposed several important failure modes.

## 7.1 Measurement Noise

The verifier was initially evaluated without explicitly pinning temperature.

This made the measurement instrument stochastic.

Even after pinning, two of 37 claims remained unstable.

The lesson is:

> **Measure the noise floor before interpreting small experimental differences.**

---

## 7.2 Incorrect Gold Label

Q58's stored answer was incorrect.

The stored answer was:

```text
A — I only
```

An exhaustive corpus scan showed that both statements were false.

The audited answer is:

```text
D
```

This demonstrated that a benchmark's answer key cannot automatically be treated as infallible ground truth.

The evaluation now reports both stored and audited labels.

---

## 7.3 Answer Leakage

The answer-key verifier was originally given the declared answer in its prompt.

Although the model was instructed to ignore it, the information was still present in context.

This meant the verifier was not genuinely independent.

The component was therefore removed from the reported verification measurements.

The design principle is:

> **If a piece of information must not influence a decision, it should not be included in the model context.**

---

# 7.4 Verifier Failure Modes

Three distinct failure modes were identified.

### Closed-list membership

The verifier may establish that a constitutional rule exists but incorrectly assume that the specific subject in the claim belongs to that rule.

### Cross-scope merging

Evidence about one constitutional entity can incorrectly be applied to another entity with similar language.

### Strength inflation

Evidence establishing a weaker statement can be incorrectly interpreted as supporting a stronger statement.

These failures demonstrate why claim-level traces are necessary.

---

# 7.5 Prompt Accretion

The initial implementation accumulated verification rules inside one increasingly large prompt.

An earlier analysis incorrectly attributed changes to the addition of numeric-conflict rules.

Later experiments showed:

- Temperature had not been pinned.
- The numeric rules produced no changes on the shipped retriever.
- Q58's instability was caused by other factors.

The surviving lesson is:

> **Adding more rules to an unstable monolithic classifier is not necessarily the right solution.**

The preferred direction is decomposition into smaller, independently testable checks.

---

# 7.6 Evaluation Harness Problems

Several evaluation bugs were identified:

- Retrieval benchmark used k=5 while production verification used k=3.
- Duplicate answer-mapping logic existed.
- Negated options were mishandled.
- `"None"` could not be parsed correctly.
- Abstentions were initially scored as incorrect answers.
- Baseline accuracy was hard-coded.
- Evaluation stages could run out of order.
- The orchestrator and evaluation harness used different verifier configurations.

These issues have now been addressed.

The evaluation pipeline is centralized through:

```text
src/evaluation/run_all.py
```

The pipeline now validates:

- Dependency order
- Output freshness
- Retrieval-depth consistency
- Stored evaluation configuration

A `--check` mode performs these validations without making API calls.

---

# 7.7 Orchestrator Consistency

A particularly serious issue was discovered in the LangGraph orchestration.

The production node constructed `FactVerifier()` without the corpus.

This silently disabled the full-corpus absence scanner.

The evaluation harness therefore measured a system stronger than the actual orchestrated implementation.

The same omission later appeared in the stability harness.

This led to an important engineering principle:

> **The system being evaluated must be the same system being shipped.**

---

# 7.8 Retrieval Installation Diagnosis

The semantic retrieval pipeline was temporarily replaced with BM25 because `sentence-transformers` appeared to be unavailable.

Further investigation showed that the issue was not network connectivity but TLS certificate handling through the corporate proxy.

The problem was resolved using:

```python
truststore.inject_into_ssl()
```

The lesson is broader than the specific TLS issue:

> **An error message is evidence about what a library believes happened, not necessarily evidence about what actually happened.**

---

# 7.9 Why the Architecture Changed

The initial architecture used a single LLM call:

```text
Question
   ↓
Evidence
   ↓
LLM
   ↓
Verdict
```

The model effectively had to:

1. Understand the question.
2. Interpret the statements.
3. Retrieve or use evidence.
4. Determine truth.
5. Select an answer.

This made failures difficult to localize.

The final architecture separates:

```text
Claim Construction
       ↓
Retrieval
       ↓
Claim Verification
       ↓
Deterministic Mapping
```

This decomposition made the following problems observable:

- Incorrect labels
- Unsupported claims
- Scope errors
- Strength inflation
- Retrieval mismatches
- Claim-construction errors
- Verification instability
- Mapping bugs

The most important architectural benefit is therefore:

> **The system's failures have identifiable locations.**

---

# 8. Limitations

The current evaluation has several important limitations.

### Dataset size

The historical benchmark contains only:

- 13 questions
- 37 claims

One question represents approximately 7.7 percentage points of accuracy.

### Measurement noise

The measured noise is approximately two questions.

Therefore, differences smaller than approximately 15.4 percentage points should not be treated as reliable improvements on this benchmark.

### Stability estimate

Only five repeated verifier runs were performed.

The observed unstable-claim counts are therefore lower-bound estimates rather than precise probability estimates.

### Retrieval benchmark

The retrieval benchmark contains only 16 queries.

The gold labels are manually constructed and the five retrieval configurations were not all rerun during the final evaluation pass.

### Corpus coverage

Two historical questions are outside the available corpus.

Therefore, 100% answerability is impossible without expanding the source collection.

### Label audit

Only questions contradicted by at least one system were audited.

Questions on which every system abstained have not yet received the same level of label verification.

### Judge circularity

The LLM judge rewards epistemic caution, which is an explicit property of the verified system.

### Abstention quality

Six of nine abstentions were caused by pipeline defects rather than genuine evidence limitations.

Therefore, the current abstention rate should not be interpreted as pure epistemic reliability.

---

# 9. Design Decisions

The final architecture makes the following decisions.

| Decision | Choice | Reason |
|---|---|---|
| Retrieval | Semantic + cross-encoder | Best observed Recall@5 |
| Claim construction | Deterministic | Prevent malformed propositions |
| Verification | Per-claim | Avoid conjunctive verification failure |
| Verdicts | 3-way | Explicit uncertainty |
| Option selection | Deterministic Python | Separate verification from decision |
| Absence claims | Full-corpus scan | Ranked retrieval cannot prove universal negatives |
| Abstention | STRICT | Avoid unsupported assertions |
| Orchestration | LangGraph | Supports generation/verification/revision loop |
| Evidence | Page-level provenance | Enables source auditing |
| Temperature | Explicitly pinned | Reduce measurement variance |
| Evaluation | Dependency-aware pipeline | Prevent stale/inconsistent measurements |

---

# 10. What the System Is Designed to Optimize

The system is **not** optimized for maximum answer rate.

Its optimization target is:

> **Minimize unsupported constitutional assertions while providing auditable evidence for answers that are given.**

This leads naturally to the following behavior:

```text
Strong evidence
      ↓
Answer

Conflicting evidence
      ↓
Contradicted

Insufficient evidence
      ↓
Abstain
```

This is intentionally different from a conventional RAG chatbot, whose default behavior is to generate an answer whenever possible.

---

# 11. Current Status

The current implementation demonstrates:

- Page-aware document ingestion
- Semantic retrieval
- Cross-encoder reranking
- Deterministic claim construction
- Claim-level verification
- Full-corpus negative-claim checking
- Deterministic answer mapping
- Explicit abstention
- LangGraph generation/verification workflow
- Evaluation replay
- Stability testing
- Label auditing
- Failure attribution

The system is therefore functioning as a complete **source-verification pipeline**, rather than simply a question-answering chatbot.

However, the current benchmark does **not** justify claiming broad accuracy or production readiness.

The main remaining challenge is improving coverage without sacrificing the zero-error objective.

---

# 12. Next Steps

The next development priorities are:

### 1. Increase verifier resampling

Move from five repeats toward approximately 20 to obtain a better estimate of variance.

### 2. Expand the benchmark

Increase the number of evaluation questions so architectural differences become measurable.

### 3. Decompose the verifier

Replace the monolithic prompt with independently testable checks for:

- Scope
- Membership
- Strength
- Numeric consistency

### 4. Fix pipeline-defect abstentions

The six corpus-answerable abstentions are the highest-value coverage improvements.

### 5. Complete the label audit

Audit the remaining benchmark questions, including those where every system abstained.

### 6. Re-run retrieval experiments

Re-run all retrieval configurations using the final retrieval configuration and evaluation pipeline.

---

# Conclusion

The project began as a conventional RAG question-answering problem but evolved into a **source-verification system**.

The key architectural insight is:

> **Do not ask one LLM to answer a multi-statement constitutional question and then trust another generation from the same model to verify it.**

Instead:

```text
Question
   ↓
Deterministic claim construction
   ↓
Independent evidence retrieval
   ↓
Per-claim verification
   ↓
Deterministic option mapping
   ↓
Answer or abstain
```

The historical evaluation shows a clear trade-off:

- Vanilla RAG answers every question but produces many false assertions.
- The verified system answers fewer questions but produced zero false final answers in the measured 13-question experiment.
- Predicate binding produced the strongest claim-level improvement.
- The direct option-verification approach failed primarily because options are conjunctive.
- The current abstention rate is too high because most historical abstentions were caused by pipeline defects rather than missing evidence.

The project therefore does **not** conclude:

> "Verification makes the model more accurate."

The defensible conclusion is narrower:

> **Separating claim verification from answer selection provides a measurable mechanism for reducing unsupported constitutional assertions, while making individual failures observable and diagnosable.**

The next stage is to increase coverage while preserving that property.