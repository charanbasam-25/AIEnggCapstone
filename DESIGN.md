# Design: Why This System Is Shaped the Way It Is

This document explains the reasoning behind UPSC Practice.

It focuses on:

- the problem being solved;
- the central design principle;
- the decisions that follow from that principle;
- the alternatives that were considered and rejected;
- the costs and trade-offs of the design; and
- what the current evaluation actually establishes.

For the implementation details, see [`docs/SYSTEM_DESIGN.md`](docs/SYSTEM_DESIGN.md).

For setup and usage, see [`README.md`](README.md).

---

# 1. The Problem

A learner preparing for UPSC Prelims wants practice questions on Indian Polity.

A language model can generate those questions instantly.

The problem is that a generated question can look completely correct while containing a subtle factual or logical error.

Four failure modes matter most:

| Failure | What the learner sees | Why it is dangerous |
|---|---|---|
| **Invented rule** | A stem asserting a constitutional provision that does not exist | The learner may memorise something false |
| **Wrong key** | A valid question with the wrong option marked correct | The learner learns the wrong answer |
| **Ambiguity** | Two options that are defensible | The learner can be marked wrong despite having a reasonable answer |
| **Decorative citation** | A real Article quoted as evidence for a claim it does not actually establish | The citation creates false confidence |

The important observation is that these failures are different.

Checking only whether the stem "looks correct" is insufficient.

A question can have:

```text
Correct stem
    +
Wrong answer
```

or:

```text
Correct stem
    +
Correct answer
    +
Ambiguous distractor
```

or:

```text
Correct claim
    +
Genuine citation
    +
Citation does not actually prove the claim
```

Each failure therefore requires a different check.

---

# 2. The Second Problem: The Corpus Has Limits

The application does not have access to every fact about Indian Polity.

Its source corpus is finite.

A generated question may therefore be reasonable, but still impossible to establish from the available sources.

That creates a fundamental choice:

```text
Evidence insufficient
        │
        ├── Guess
        │
        └── Abstain
```

The project chooses **abstention**.

This is important because a system that has no explicit representation for "I cannot establish this" eventually turns missing evidence into a guess.

That is exactly the behaviour this project is designed to avoid.

---

# 3. Scope Is a Design Boundary

The system is intentionally scoped to:

> **Static Indian Polity as represented by the Constitution of India and one selected NCERT Grade 7 chapter.**

This constraint is not merely a limitation of the current implementation.

It makes the verification model possible.

Every published claim is expected to be traceable to a known document in a known corpus.

Introducing unrestricted current affairs or web search would weaken that guarantee.

The application would no longer have a closed, enumerable evidence boundary.

Therefore the system deliberately does **not** attempt to be:

- a current-affairs tutor;
- a web-search agent;
- a general UPSC answer engine;
- or a general-purpose chatbot.

When the loaded corpus cannot establish an answer, the system can withhold the question or return fewer questions than requested.

That behaviour is intentional.

---

# 4. The Central Design Principle

Almost every architectural decision follows from one commitment:

> **Retrieval is mandatory, and it happens before generation. A generated MCQ is an untrusted candidate until it passes all verification gates.**

The order matters.

A conventional RAG workflow can look like:

```text
Question
   ↓
LLM generates answer
   ↓
Retrieve supporting evidence
   ↓
Answer
```

This creates a subtle problem.

The model has already reached a conclusion before retrieval happens.

Retrieval can then become a search for evidence that supports an existing answer.

UPSC Practice reverses the order:

```text
Retrieve evidence
       ↓
Generate candidate
       ↓
Verify candidate
       ↓
Publish / abstain
```

The generation stage cannot start without retrieved evidence.

This is enforced in code rather than being left as a prompt instruction.

---

# 5. Retrieval Is Evidence, Not a Verdict

A retrieved passage does not automatically establish that a claim is true.

Retrieval answers:

> "Which passages appear relevant?"

Verification answers:

> "Does this evidence actually support or contradict the claim?"

These are different questions.

For example:

```text
Retriever:
"This passage is relevant."

        ≠

Verifier:
"This passage establishes the claim."
```

Keeping those concepts separate is one of the most important architectural boundaries in the system.

A high retrieval score is therefore never treated as proof.

---

# 6. Claim-Level Verification

A complete UPSC-style question can contain multiple independent propositions.

Consider:

> Which of the following statements are correct?

The statements need to be checked individually.

A sentence such as:

> "It is appointed by the President."

cannot be reliably verified if "it" has not been resolved to its actual subject.

The system therefore converts the question into standalone claims before verification.

Conceptually:

```text
Generated question
       ↓
Claim extraction
       ↓
Standalone propositions
       ↓
Evidence retrieval
       ↓
Claim verdicts
```

Each claim can then be independently evaluated.

This provides a much stronger verification boundary than asking a model:

> "Is this question correct?"

The latter produces a judgement.

The former produces a set of explicit claims that can be independently inspected.

---

# 7. Three Verdicts, Not Two

Every claim can resolve to one of three states:

```text
SUPPORTED
CONTRADICTED
INSUFFICIENT
```

The third state is essential.

### SUPPORTED

The source evidence supports the claim.

### CONTRADICTED

The source evidence establishes the opposite.

This is a valid and useful result because a well-formed MCQ needs false statements as distractors.

### INSUFFICIENT

The available evidence does not establish the claim either way.

This is fundamentally different from contradiction.

```text
CONTRADICTED
    =
Evidence says otherwise.

INSUFFICIENT
    =
Evidence does not establish either side.
```

Treating `INSUFFICIENT` as `CONTRADICTED` would fabricate a negative conclusion.

Treating it as `SUPPORTED` would fabricate a positive conclusion.

Keeping it as a third state gives the system a principled way to abstain.

---

# 8. Python Derives the Answer

The LLM may propose an answer during generation.

That proposed answer is not authoritative.

Once the claims have been verified, Python derives the answer from their truth values.

For example:

```text
Statement 1 → SUPPORTED
Statement 2 → CONTRADICTED
Statement 3 → SUPPORTED
```

becomes:

```text
1 and 3 only
```

through deterministic application logic.

The model therefore does not get to both:

1. generate the claims; and
2. decide whether its own claims are true.

That would undermine the purpose of verification.

The generator's answer is treated as input to the verification process, not as the final answer key.

---

# 9. Direct Questions Need a Different Verification Path

Not every question is a statement-mapping question.

Consider:

> Which of the following is correct?

There are no independent statements whose truth pattern can simply be mapped to an answer.

Direct questions therefore use a stricter option-comparison path.

The system requires:

1. positive support for exactly one option;
2. affirmative evidence against the alternatives; and
3. agreement from an independent review.

The important rule is:

> **Missing evidence cannot prove an option false.**

For example:

```text
Option A → supported
Option B → no evidence found
Option C → no evidence found
Option D → no evidence found
```

does not establish that A is the only correct answer.

Silence is not contradiction.

Therefore the alternatives must be positively ruled out.

---

# 10. Independent Roles

Different stages have different responsibilities.

The system separates:

- candidate generation;
- claim verification;
- quality review;
- explanation generation;
- explanation verification.

In particular, the explanation writer does not receive the earlier verification reasoning.

It works from the source material instead.

The reason is simple:

> A reviewer should evaluate the evidence, not merely agree with another model's reasoning.

If the explanation writer saw the earlier reasoning, the subsequent reviewer could end up evaluating whether the explanation is consistent with the previous reasoning rather than whether it is actually supported by the source.

The cost is additional model calls and latency.

The benefit is stronger separation between reasoning stages.

---

# 11. Quotation Binding

A verifier must return evidence together with its verdict.

The system then performs an additional deterministic check:

> Does the quotation actually occur in the cited passage?

Conceptually:

```text
Verifier
   ↓
Verdict + quotation
   ↓
Python substring check
   ↓
Accept / reject evidence
```

A verifier cannot pass simply by producing a plausible-looking citation.

If the quotation cannot be located in the cited passage, the verification result is rejected.

This is intentionally a cheap check.

It does not prove that the quotation entails the claim.

It does establish that:

- the quotation is present;
- the citation is not fabricated at the string level; and
- the evidence returned by the verifier corresponds to the retrieved material.

---

# 12. Provenance Is Not Entailment

This distinction is important enough to state explicitly.

Quotation binding establishes:

> **The evidence is genuinely present in the cited source.**

It does not establish:

> **The evidence logically entails the generated claim.**

For example, a constitutional passage may genuinely mention a concept without establishing the broader conclusion that the model attaches to it.

Therefore provenance is treated as one layer of verification, not the final definition of truth.

This is also why the system retains independent claim review.

---

# 13. Bounded Retries

The system does not retry indefinitely.

A candidate receives:

- one initial generation;
- at most two complete revisions.

Explanation generation receives one local repair.

An unlimited retry loop would create two problems.

### Cost

An unanswerable question could generate an unbounded number of model calls.

### Selection pressure

Repeatedly rewriting the same candidate could eventually produce wording that happens to pass the gates without making the underlying proposition more defensible.

The system therefore prefers:

```text
Try
 ↓
Verify
 ↓
Repair if appropriate
 ↓
Verify again
 ↓
Abstain if still unresolved
```

rather than:

```text
Retry forever until something passes
```

---

# 14. Revisions Reset the State

Every revision clears all previous gate results.

This is a subtle but important correctness property.

Without a full reset, a question could accumulate successful checks across multiple versions:

```text
Draft 1
 ├── source gate ✓
 ├── answer gate ✓
 └── quality gate ✗

Draft 2
 ├── revised question
 └── quality gate ✓

Final result
 ├── old source result
 ├── old answer result
 └── new quality result
```

Those results may never have applied simultaneously to the same question.

The system therefore treats every revision as a new candidate:

```text
Old candidate
    ↓
Discard gate state
    ↓
New candidate
    ↓
Run every required gate again
```

This keeps publication decisions internally consistent.

---

# 15. Literal Absence Requires Exhaustive Search

Some questions depend on the absence of a term from the source corpus.

A dangerous shortcut would be:

```text
Term not found in top 5 chunks
        ↓
Term does not exist in corpus
```

That inference is invalid.

The corpus contains 1,149 chunks.

The top five tell us nothing definitive about the remaining 1,144.

Therefore recognised literal-absence claims trigger an exhaustive lexical scan across the loaded corpus.

The result is intentionally narrow:

> The literal string is absent from the loaded corpus under the assumptions of corpus completeness and PDF extraction quality.

It does **not** establish:

> The underlying concept is absent from the corpus.

String matching cannot establish conceptual absence.

---

# 16. One Source of Truth for Retrieval Depth

Retrieval depth is defined centrally in:

```text
src/retrieval/retrieval_config.py
```

Both the production pipeline and benchmark code use that configuration.

This exists because the project previously encountered a real configuration mismatch.

The retriever was evaluated using `Recall@5`, while the production pipeline was querying at `k=3`.

That meant:

```text
Benchmark:
k = 5

Production:
k = 3
```

The benchmark therefore described a configuration different from the one actually running.

The lesson is broader than the specific bug:

> **A benchmark metric must measure the configuration consumed by production.**

A `Recall@k` result is meaningful only when `k` matches the actual retrieval depth used by the system.

---

# 17. Alternatives Considered

Several approaches were implemented or evaluated but deliberately not selected as the default.

| Alternative | Status | Reason |
|---|---|---|
| **BM25 keyword retrieval** | Not default | Lexical overlap often favoured plainly worded NCERT content over the constitutional provision that actually answered the query |
| **Hybrid BM25 + semantic retrieval using RRF** | Not default | Reciprocal-rank fusion displaced some exact constitutional provisions by rewarding documents that ranked moderately in both systems |
| **Parent-child retrieval** | Not default | Matched bare semantic retrieval without enough measured benefit to justify the additional machinery |
| **`closed_world` answer mapping** | Not default | Treats absence of support as falsity, effectively turning silence into evidence |
| **`elimination` answer mapping** | Not default | Selects the least-contradicted option without requiring a positively supported answer |
| **Autonomous agent** | Rejected | Could change the workflow, skip retrieval, or introduce new sources, weakening determinism and testability |
| **Qdrant / external vector service** | Not introduced | A normalised NumPy index is sufficient for 1,149 chunks and is easier to inspect and reproduce |

The rejected retrieval implementations remain useful as benchmark arms.

Their historical results are documented separately in:

[`docs/retrieval_baseline.md`](docs/retrieval_baseline.md)

---

# 18. Why a Fixed Graph Instead of an Agent?

The application uses LangGraph, but deliberately does **not** use an autonomous agent architecture.

The workflow is fixed:

```text
Retrieve
   ↓
Generate
   ↓
Extract claims
   ↓
Verify
   ↓
Derive answer
   ↓
Audit
   ↓
Explain
   ↓
Publish / abstain
```

A fully autonomous agent could theoretically decide:

```text
"I don't have enough evidence.
I'll search the web."
```

That would violate the corpus boundary.

Or:

```text
"I don't need verification for this question."
```

That would violate the publication policy.

The fixed graph makes those paths impossible.

This is an intentional trade:

> **Less adaptability in exchange for more determinism, observability, and testability.**

For this capstone, that trade is preferable.

---

# 19. Why Not Just Use a Better Model?

A larger or more capable model could improve generation quality.

It does not eliminate the fundamental problem.

The model is still capable of:

- hallucinating a constitutional rule;
- selecting the wrong answer;
- creating ambiguous distractors;
- producing a plausible but irrelevant citation.

The project therefore focuses on **system design around the model**, rather than assuming that model capability alone solves verification.

The LLM is one component.

It is not the authority.

---

# 20. What the Design Costs

The architecture intentionally pays several costs.

## More model calls

Generation, claim verification, answer verification, quality review, explanation generation, and explanation review are separate stages.

A simple:

```text
prompt → LLM → answer
```

system would be substantially cheaper and faster.

The additional calls buy stronger checks against unsupported publication.

---

## Fewer questions

A learner asking for five questions may receive three.

Abstention is therefore a visible product cost.

The system prefers:

> three defensible questions

over:

> five plausible questions with two unsupported ones.

---

## Local NumPy retrieval does not scale indefinitely

The NumPy index is:

- easy to inspect;
- easy to reproduce;
- sufficient for the current corpus.

It is not a production vector-search infrastructure.

It does not provide the scaling, sharding, or concurrency characteristics of a dedicated vector service.

---

## SQLite is intentionally simple

SQLite is appropriate for a single-user capstone.

It is not designed here for:

- multi-user workloads;
- account isolation;
- distributed deployments;
- high write concurrency.

---

## Fixed workflow means limited adaptability

The system cannot dynamically discover new sources or invent new verification steps.

That is a limitation.

It is also what makes the workflow predictable enough to evaluate.

---

## Blind reviewers can share model blind spots

The verification roles use the same model family.

The contexts are separated, but the underlying model is not independently trained.

Therefore:

> **Context independence ≠ model independence.**

A future stronger evaluation could introduce different models or human review.

---

## The reranker currently has no measured benchmark gain

On the current 16-query benchmark:

```text
Semantic retrieval      14/16
Semantic + reranking    14/16
```

The reranker therefore has not demonstrated a measurable Recall@5 improvement on this dataset.

It remains because:

- the cost is relatively low;
- it can change ordering;
- the benchmark is extremely small;
- the current result is insufficient to establish that reranking has no value.

This is explicitly an engineering choice, not an empirical claim of improvement.

---

# 21. What Is Currently Established

The current evidence supports the following statements.

### Retrieval

On 16 annotated queries:

```text
Semantic retrieval:
14 / 16 = 87.5%

Semantic + reranking:
14 / 16 = 87.5%
```

Both approaches miss the same two queries:

- q06 — Parliament, p.67
- q09 — High Courts, p.130

Therefore:

> **No measured Recall@5 improvement from reranking has been observed on this benchmark.**

---

### Benchmark granularity

With only 16 queries:

```text
1 query = 6.25 percentage points
```

Small differences should therefore be treated cautiously.

---

### Selective answering

The preserved verifier snapshots show a strong preference for abstention.

This demonstrates the intended behaviour of the system:

> When evidence is insufficient, publication is blocked.

It should not be interpreted as a high-accuracy benchmark because the answered sample is very small.

---

### Regression dataset

The 75-question mixed set is maintained as a regression diagnostic.

It is not an accuracy claim.

---

# 22. What Has Not Been Established

The project does **not** currently have an expert-labelled benchmark large enough to establish the actual correctness of generated MCQs.

This is the largest remaining evaluation gap.

A gate passing means:

```text
"The system's verification conditions were satisfied."
```

It does not mean:

```text
"An expert examiner would definitely accept this question."
```

Those are different claims.

Other unresolved limitations include:

### No guarantee of zero factual errors

The system reduces unsupported publication.

It does not mathematically guarantee factual correctness.

### No proof of perfect entailment

Quotation binding establishes provenance, not logical entailment.

### No independent model family

Blind reviewers can share model-specific weaknesses.

### Corpus dependence

All verification depends on:

- corpus completeness;
- source quality;
- PDF extraction;
- retrieval quality.

### Small evaluation datasets

The current retrieval benchmark is too small for strong statistical conclusions.

---

# 23. The Most Important Future Evaluation

The next major improvement should not simply be another generation feature.

It should be an **expert-labelled MCQ benchmark**.

A useful evaluation set would contain questions labelled by human reviewers for:

- factual correctness;
- answer-key correctness;
- uniqueness of the correct answer;
- distractor quality;
- source support;
- explanation correctness;
- UPSC-style quality;
- appropriate difficulty.

Then the project could directly compare:

```text
Vanilla RAG
      vs.
RAG + verification
```

using outcomes that matter to the learner.

That would allow the project to move from:

> "The system contains extensive verification machinery."

to the much stronger claim:

> **"The verification machinery measurably improves the correctness and quality of generated MCQs."**

---

# 24. The Design in One Sentence

The entire architecture can be reduced to one rule:

> **Generate freely enough to produce useful candidates, but publish only what the system can independently establish from its known evidence.**

Everything else follows from that constraint:

```text
Finite corpus
     ↓
Mandatory retrieval
     ↓
Untrusted generation
     ↓
Atomic claims
     ↓
Three-way verification
     ↓
Deterministic answer derivation
     ↓
Independent quality review
     ↓
Evidence-bound explanations
     ↓
Bounded revision
     ↓
Publish or abstain
```

The goal is not to build an LLM that never makes mistakes.

The goal is to build a system in which **unsupported confidence has difficulty reaching the learner**.