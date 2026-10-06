# Evaluation Report — Source-Verified UPSC Polity MCQ Verifier

**Author:** Charan Kumar Basam  
**Cohort:** AI Engineering  
**Date:** 27 September 2026  
**Companion document:** [`DESIGN.md`](DESIGN.md)

---

## Executive Summary

This report evaluates a source-grounded UPSC Polity MCQ verification system whose primary objective is **not to answer every question, but to avoid unsupported answers**.

The system retrieves evidence from a fixed Indian Polity corpus, decomposes questions into verifiable claims, evaluates those claims independently, and uses deterministic Python logic to map verified claims to answer options.

The central architectural decision is:

> **The language model does not select the final answer option.**

Instead, the model returns evidence-backed `SUPPORTED`, `CONTRADICTED`, or `INSUFFICIENT` verdicts for individual claims. Deterministic code then decides whether those verdicts justify an answer. If they do not, the system abstains.

The evaluation produced five important findings:

1. **Verification eliminates false assertions in the historical 13-question experiment.**  
   Vanilla RAG produced 8 wrong answers, while the verified system produced 0 wrong answers, at the cost of answering fewer questions.

2. **The system is not simply "more accurate."**  
   On the original 13 questions, vanilla RAG had higher overall accuracy because it answered everything. The verified system's advantage is its **0% overall error rate**, not higher coverage.

3. **Predicate binding is the strongest measured architectural improvement.**  
   It increased propositional claims from 70.27% to 100% and reduced false `SUPPORTED` verdicts from 7–8 to 0 across 37 claims.

4. **Several earlier experimental conclusions were invalidated.**  
   Retrieval depth, numeric rules, and some prompt changes initially appeared to affect accuracy, but later analysis showed that the observed differences were driven by an unstable boundary claim and evaluation artifacts.

5. **The largest remaining problem is coverage.**  
   Only 4 of 13 historical questions were answered, and 6 of the 9 abstentions were caused by pipeline defects rather than missing evidence.

The latest 75-question benchmark reinforces the need for more evaluation: the development split produced **2 correct, 0 wrong, 31 abstentions**, while the test split produced **0 correct, 0 wrong, 42 abstentions**. Only two questions were answered, so these results do not establish broad accuracy.

---

# 0. Evaluation Scope and How to Read the Results

The historical measurements in this report primarily evaluate the earlier **13-question numbered-statement pipeline** using UPSC 2025 Prelims Polity questions Q54–Q66.

An expanded benchmark now contains **75 additional official UPSC Polity/governance PYQs from 2019–2023**:

- 33 development questions
- 42 test questions
- Booklet-matched official answer keys
- PDF-page provenance
- Frozen train/test splits
- Statement and direct/best-answer MCQs

The expanded benchmark evaluates both verification paths and vanilla RAG. Its results are maintained separately in [`docs/BENCHMARK.md`](docs/BENCHMARK.md).

### Latest expanded benchmark

The latest source-only runs, completed on **3 October 2026**, produced:

| Split | Correct | Wrong | Abstentions |
|---|---:|---:|---:|
| Development | 2 | 0 | 31 |
| Test | 0 | 0 | 42 |

All runs completed without service errors.

Only two questions were answered. Therefore:

- Test precision is undefined.
- Zero observed errors does not establish broad reliability.
- Fresh untouched data is still required for an independent accuracy estimate.
- Corpus answerability has not yet been fully audited for this benchmark.

A previous reviewed test run accepted one incorrect Assertion/Reason answer. That failure informed the current format guard.

The historical grounding and judge experiments below have **not** been rerun on the expanded benchmark.

---

## 0.1 Dataset Size and Label Quality

The historical benchmark contains:

- 13 questions
- 37 numbered statements

One question therefore represents **7.69 percentage points** of accuracy.

Two answer-label sets are reported:

### Stored labels

The published answer key as originally recorded.

### Audited labels

The labels after manually investigating contradictions.

The audit found that **Q58's stored answer was incorrect**.

Both label sets are retained in the report so the correction is visible rather than silently incorporated into the results.

---

# 0.2 Metrics

Because the system is allowed to abstain, accuracy alone is insufficient.

| Metric | Definition |
|---|---|
| Coverage | Answered / total |
| Precision when answered | Correct / answered |
| Accuracy | Correct / total |
| **Error rate** | **Wrong / total** |

These satisfy:

```text
accuracy + error rate + abstention rate = 1
```

This distinction is central to the project.

A system that answers every question but is frequently wrong should not be treated as equivalent to a system that abstains when evidence is insufficient.

---

# 0.3 Measurement Noise

One of the most important findings of the evaluation was that the verifier was initially **stochastic when the evaluation assumed it was stable**.

`FactVerifier.verify` did not explicitly set `temperature`. The Responses API therefore used its default behavior, while the vanilla RAG baseline had already been pinned at temperature 0.

This meant the original comparison unintentionally compared:

```text
stochastic verifier
        vs
deterministic baseline
```

All five relevant LLM call sites are now pinned:

- `fact_verifier`
- `claim_extractor`
- `answer_key_verifier`
- `mcq_quality_auditor`
- `mcq_generator`

A stability experiment re-verified all 37 claims five times.

| Condition | Stable | Unstable |
|---|---:|---:|
| Unpinned | 28/37 (75.68%) | 9 |
| **Pinned, temperature = 0** | **35/37 (94.59%)** | **2** |

Pinning improves stability, but it does **not** produce complete determinism.

Q58 I remained particularly unstable and produced all three possible verdicts across five identical calls.

### Resampled pipeline results

| Condition | Coverage | Precision | Accuracy | Error rate |
|---|---:|---:|---:|---:|
| Unpinned | 23.08–46.15% | 75–100% | 23.08–38.46% | 0–7.69% |
| Pinned | 23.08–38.46% | 75–100% | 23.08–38.46% | 0–7.69% |

The observed accuracy spread is **15.38 percentage points**, equivalent to two questions.

Therefore:

> **A one-question or two-question difference is not considered a reliable result on this dataset.**

This applies particularly to the System C ablation.

Claim-level metrics are more useful because they provide 37 observations instead of 13 and some require no gold labels.

---

# 1. Retrieval Evaluation

The retrieval system was evaluated on a 16-query gold benchmark over the same 1,149-chunk corpus.

Metric: **Recall@5**.

| Retriever | Hits | Recall@5 |
|---|---:|---:|
| BM25 | 12/16 | 75.00% |
| Hybrid BM25 + Semantic RRF | 13/16 | 81.25% |
| Semantic — `BAAI/bge-small-en-v1.5` | 14/16 | 87.50% |
| Parent-child + reranker | 14/16 | 87.50% |
| **Semantic + MS-MARCO MiniLM cross-encoder** | **15/16** | **93.75%** |

The shipped configuration is therefore:

```text
Semantic retrieval
        ↓
20 candidates
        ↓
Cross-encoder reranking
```

## Retrieval Findings

### Hybrid RRF did not help

Hybrid BM25 + semantic RRF performed worse than semantic retrieval alone:

```text
Hybrid RRF       81.25%
Semantic         87.50%
```

The lexical component sometimes pushed semantically relevant passages lower in the ranking.

### Parent-child retrieval did not improve recall

Parent-child retrieval also achieved 14/16, adding complexity without improving the benchmark result.

### Cross-encoder reranking produced the best observed result

Semantic retrieval followed by cross-encoder reranking achieved 15/16.

However, the benchmark contains only 16 queries. One query represents 6.25 percentage points.

Therefore:

> 93.75% demonstrates the best observed configuration, but does not establish a statistically reliable 6.25-point advantage over semantic retrieval alone.

### Retrieval benchmark provenance

The original five retrieval configurations were not all rerun during the final measurement pass.

BM25 was rerun as a regression check and reproduced 75.00%.

The other four rows remain historical measurements.

---

# 1.1 Retrieval Configuration Consistency

An important evaluation defect was discovered:

The retriever was selected using **Recall@5**, but the production verification pipeline queried at `top_k=3`.

The benchmark and production pipeline were therefore evaluating different evidence depths.

This has now been corrected by centralizing:

```text
RETRIEVAL_TOP_K
RERANK_CANDIDATE_K
```

in:

```text
src/retrieval/retrieval_config.py
```

The evaluation pipeline also records the retrieval depth used to generate each report.

`run_all --check` verifies that stored evaluation results were generated with the current retrieval configuration.

This prevents an evaluation report from silently becoming stale after changing a configuration constant.

---

# 2. Verification Architecture Comparison

Three inference strategies were evaluated using the same retrieval configuration.

| System | LLM decision | Coverage | Precision | Accuracy | Error |
|---|---|---:|---:|---:|---:|
| **A — Vanilla RAG** | Reads question + evidence and writes answer | 100% | 38.46% | 38.46% | **61.54%** |
| **B — Statement verification** | Verifies each claim independently; Python maps answer | 30.77% | **100%** | 30.77% | **0%** |
| **B2 — Direct option verification** | Verifies whether each option is established | 15.38% | **100%** | 15.38% | **0%** |

Under audited labels, vanilla RAG becomes:

```text
Accuracy: 46.15%
Error:    53.85%
```

System B remains:

```text
Accuracy: 30.77%
Error:    0%
```

## What this does — and does not — prove

The result does **not** show that verification improves overall accuracy.

Vanilla RAG has higher accuracy because it answers every question.

The stronger result is:

```text
Vanilla RAG:
8/13 wrong under stored labels
7/13 wrong under audited labels

Verified system:
0/13 wrong
```

The verified system eliminates false assertions by abstaining on questions it cannot establish.

This is the intended trade-off of the architecture.

---

# 2.1 Direct Option Verification

B2 verifies complete options instead of decomposing them into individual statements.

It achieved:

```text
Coverage: 15.38%
Precision: 100%
Error: 0%
```

Two evidence configurations were tested:

| Evidence | Answered | Correct | Coverage |
|---|---:|---:|---:|
| Question-level top-5 | 2 | 2 | 15.38% |
| Question + per-statement evidence | 1 | 1 | 7.69% |

Adding more evidence actually reduced coverage.

The reason is structural: an option may contain two to four propositions. Asking the model to establish the entire conjunction in one step makes it easier for one unsupported component to block the whole option.

This supports the shipped design:

> **Decompose the claims, verify them independently, and recombine them deterministically.**

---

# 2.2 Paired Comparison

On the four questions answered by System B:

- System A: **2/4**
- System B: **4/4**

This is directionally favorable to verification, but the difference is still only two questions and therefore falls inside the measured noise band.

The robust conclusion remains the **0% error rate**, not the paired accuracy difference.

---

# 2.5 Generate-and-Verify Evaluation

The same verifier was also tested as a quality gate for questions generated by the system.

The LangGraph pipeline follows:

```text
Generate
   ↓
Extract claims
   ↓
Verify claims
   ↓
Verify answer key
   ↓
Audit question quality
   ↓
ACCEPT / REVISE / REJECT
```

The experiment used:

- 15 Polity topics per format
- `max_retries=2`
- Two formats:
  - `simple`
  - `statements`

### Results

| Metric | Simple | Statements |
|---|---:|---:|
| Topics completed | 15 | 15 |
| ACCEPT | **8 (53.33%)** | **0 (0%)** |
| REJECT | 7 | 15 |
| Total attempts | 34 | 45 |
| Blocked attempts | 26 | 45 |
| Attempt-1 defect rate | 73.33% | **100%** |
| Repair rate | 36.36% | **0%** |
| Claims verified | 35 | 149 |
| Mean claims/attempt | 1.03 | **3.31** |

The statements format produced:

- 105 `SUPPORTED`
- 22 `CONTRADICTED`
- 22 `INSUFFICIENT`

The simple format produced:

- 26 `SUPPORTED`
- 9 `INSUFFICIENT`

---

## 2.5.1 Ungated Generator Baseline

Attempt 1 provides a free baseline for what would have been released without verification.

| | Simple | Statements |
|---|---:|---:|
| Questions | 15 | 15 |
| Clean | 4 | **0** |
| Defective | 11 (73.33%) | **15 (100%)** |

This is strong evidence that the verification gate detects real problems in generated content.

However, the statements gate currently rejects everything.

Therefore:

> **A gate that accepts 0/15 generated statement questions is not production-ready, even if its individual rejections are justified.**

The gate and generator need calibration.

---

## 2.5.2 Gate Attribution

| Gate | Simple: blocked / sole | Statements: blocked / sole |
|---|---:|---:|
| Fact | 9 / **6** | 38 / 1 |
| Answer key | 10 / 2 | 41 / 1 |
| Quality | 17 / **8** | 42 / 0 |

The gates are not independent in the statements format.

Almost every failed attempt triggers all three gates.

Therefore, the 0% acceptance rate should not be interpreted as three independent systems agreeing that every question is bad.

It is more likely that a common underlying generation defect is being detected through multiple gates.

---

# 2.5.3 Generation Findings

Two conclusions are supported:

### Positive

The gate detects a genuine degradation in multi-statement generation quality.

### Negative

The current gate is too restrictive for the statements format.

The statements generator produces contradicted claims frequently enough that the current retry budget cannot recover.

The next iteration should therefore investigate:

- Generator prompt quality
- Statement construction
- Claim difficulty
- Verification threshold
- Retry policy

rather than simply increasing the number of gates.

---

# 3. LLM-as-a-Judge Evaluation

A separate judge evaluation compared the reasoning quality of vanilla RAG and the verified system.

Protocol:

- Judge: `gpt-4o`
- Generator: `gpt-4o-mini`
- 13 questions
- Randomized output slots
- System identity hidden
- Correctness determined by answer key
- Judge evaluates reasoning quality only

| Dimension | Vanilla RAG | Verified | Δ |
|---|---:|---:|---:|
| Factual correctness | 2.77 | 3.69 | +0.92 |
| Evidence faithfulness | 2.46 | 4.69 | +2.23 |
| Reasoning validity | 2.69 | 4.54 | +1.85 |
| Epistemic honesty | 2.38 | **5.00** | **+2.62** |
| Mean | 2.58 | **4.48** | **+1.90** |
| Hallucinated claims | **19** | **1** | **−18** |
| Questions with hallucination | **12** | **1** | **−11** |

Head-to-head preference:

```text
Verified: 12
Vanilla:   1
Tie:       0
```

The randomized slot distribution was:

```text
Slot 1: 5
Slot 2: 8
Tie:    0
```

This does not indicate a meaningful position bias.

---

## 3.1 Judge Circularity

The judge result requires an important qualification.

The verified system is explicitly designed to abstain when evidence is insufficient.

The judge also rewards epistemic caution.

Therefore, the judge is partly measuring the architecture's intended behavior.

For example, System B was preferred on eight questions where it answered nothing.

A system that abstained on every question could potentially score well under such a rubric while being useless.

Therefore:

> **The judge scores should not be interpreted independently of coverage and error rate.**

The strongest part of this evaluation is the hallucination count:

```text
19 → 1 hallucinated claims
12 → 1 questions containing hallucinations
```

These hallucinations were identified extractively by checking whether named Articles, dates or provisions actually appeared in the cited evidence.

This makes the hallucination metric less dependent on subjective judging.

---

# 4. System C Ablation

System C evaluates individual changes while keeping the remaining pipeline fixed.

The current ablation uses the shipped retriever:

```text
BAAI/bge-small-en-v1.5
        ↓
MS-MARCO MiniLM cross-encoder
        ↓
candidate_k = 20
```

All arms use cached verdicts and temperature 0.

---

## 4.1 Ablation Results

| Arm | Change | Coverage | Precision | Accuracy | Error |
|---|---|---:|---:|---:|---:|
| C0 | Base, k=3 | 30.77% | 75% | 23.08% | 7.69% |
| C1 | + predicate binding | 38.46% | 80% | 30.77% | 7.69% |
| C1d | k=3 → 5 | 38.46% | 60% | 23.08% | 15.38% |
| C2 | + binding at k=5 | 38.46% | 60% | 23.08% | 15.38% |
| C3 | + numeric rules | 38.46% | 60% | 23.08% | 15.38% |
| C3d | Numeric rules without binding | 38.46% | 60% | 23.08% | 15.38% |

Every end-to-end difference is at most one question.

Therefore:

> **The System C accuracy table contains no statistically resolvable differences on this dataset.**

---

# 4.2 Retrieval Depth: Retracted Finding

An earlier BM25 experiment claimed that increasing retrieval depth from 3 to 5 produced the largest measured accuracy improvement.

That conclusion is now retracted.

Under stored labels:

```text
k=3 → k=5
15.38% → 30.77% accuracy
```

Under audited labels:

```text
23.08% → 23.08% accuracy
```

The apparent effect is therefore dependent on which answer key is used.

The entire difference comes from one claim:

```text
Q58 I:
CONTRADICTED → SUPPORTED
```

Q58 I was independently identified as unstable.

Further investigation showed that the same claim could change verdict based on:

- Endpoint
- Prompt whitespace
- Sampling

even at temperature 0.

Therefore:

> **The observed depth effect was not evidence that retrieval depth causes an accuracy change. It was an unstable boundary-case claim being resampled.**

The remaining valid lesson is architectural:

The system should not be benchmarked at one retrieval depth and operated at another.

---

# 4.3 Verdict Distribution

Across 37 claims:

| Arm | Supported | Insufficient | Contradicted | Propositional | False Supported |
|---|---:|---:|---:|---:|---:|
| C0 | 14 | 18 | 5 | 70.27% | 7 |
| C1 | 15 | 16 | 6 | **100%** | **0** |
| C1d | 17 | 16 | 4 | 70.27% | 8 |
| C2 | 17 | 15 | 5 | **100%** | **0** |
| C3 | 17 | 15 | 5 | **100%** | **0** |
| C3d | 17 | 16 | 4 | 70.27% | 8 |

Numeric rules produced no verdict changes in the shipped retriever.

Therefore, their earlier apparent benefit was a BM25-specific artifact.

---

# 4.4 Claim Soundness

This is the strongest ablation result because it:

- Uses 37 observations.
- Does not require gold labels.
- Is reproduced at both retrieval depths.

| Claim construction | Propositional | Non-propositional | False `SUPPORTED` |
|---|---:|---:|---:|
| Verbatim statement text | 70.27% | 11 | 7–8 |
| **Deterministic predicate binding** | **100%** | **0** | **0** |

Some extracted claims were not complete propositions.

For example:

```text
"List I — Union List, in the Seventh Schedule"
```

is not independently true or false because its predicate belongs to the question stem.

Sending such fragments directly to the verifier allowed lexical overlap to produce `SUPPORTED` verdicts that looked grounded but were logically meaningless.

Predicate binding reconstructs the missing context before verification.

Across all 37 claims:

```text
Invented tokens: 0
Propositional claims: 37/37
False SUPPORTED verdicts: 0
```

This is the strongest measured architectural improvement in the project.

---

# 4.5 Final Ablation Decision

The shipped system uses **bound claims**.

The decision is not based on a statistically meaningful accuracy improvement.

Instead:

- End-to-end accuracy cannot distinguish the arms.
- Claim-level soundness clearly distinguishes them.
- Bound claims eliminate non-propositional verification.
- False `SUPPORTED` verdicts fall from 7–8 to 0.

Therefore:

> **Predicate binding is retained because it improves the correctness of the verifier's input representation, even when the small end-to-end benchmark cannot resolve an accuracy difference.**

---

# 5. Abstention Policy Evaluation

Three deterministic policies were evaluated on identical verifier outputs.

| Policy | Behavior |
|---|---|
| STRICT | Any unresolved statement → abstain |
| ELIMINATION | Remove incompatible options; answer if exactly one remains |
| CLOSED_WORLD | Treat `INSUFFICIENT` as false |

---

## 5.1 STRICT vs ELIMINATION

STRICT and ELIMINATION produced identical results on the historical dataset.

Both answered:

```text
Q56
Q61
Q62
Q63
```

All four were correct.

This is a meaningful null result.

ELIMINATION is theoretically more general than STRICT, but the current questions never provide enough partial information to reduce the options to exactly one.

For example, knowing one of two statements is true generally leaves two possible options.

A benefit from ELIMINATION would require:

- More statements per question, or
- More constrained answer formats.

---

# 5.2 CLOSED_WORLD

| Policy | Labels | Coverage | Precision | Accuracy | Error |
|---|---|---:|---:|---:|---:|
| STRICT | Stored | 30.77% | 100% | 30.77% | **0%** |
| CLOSED_WORLD | Stored | 84.62% | 54.55% | 46.15% | **38.46%** |
| STRICT | Audited | 30.77% | 100% | 30.77% | **0%** |
| CLOSED_WORLD | Audited | 84.62% | 63.64% | **53.85%** | **30.77%** |

CLOSED_WORLD achieves the highest observed accuracy in the report:

**53.85% under audited labels.**

However, it does so by assuming:

> Absence of evidence = evidence of absence.

That assumption is invalid for an incomplete corpus.

CLOSED_WORLD converts nine abstentions into additional answers, but introduces false assertions.

For this project, the objective is source-grounded reliability rather than maximum answer coverage.

Therefore:

> **STRICT remains the shipped policy.**

---

# 5.3 Answer-Mapping Fix

The original answer mapper failed to parse the option:

```text
"None"
```

for Q58.

This caused a technical `no_option_matched` abstention.

The parser was corrected.

The current verifier abstains on Q58 for a substantive evidence reason rather than because of a parsing failure.

This illustrates why answer-key audits and executable mapping tests are both necessary.

---

# 6. Corpus Answerability

Before judging retrieval quality, the evaluation checks whether the corpus contains the information needed to answer each question.

A lexical scan was performed over **all 1,149 chunks**.

| Corpus status | Questions | Missing concepts |
|---|---|---|
| Out of corpus | **Q55, Q66** | Lokpal; crude oil / petroleum regulator |
| Partially covered | Q64 | National parks |
| Covered | Q54, Q56–Q65 | — |

Therefore:

> **2 of 13 questions cannot be answered from this corpus regardless of retrieval quality.**

The theoretical accuracy ceiling for a system that never guesses is:

```text
11 / 13 = 84.62%
```

---

## 6.1 Abstention Taxonomy

The nine historical abstentions were classified as:

| Cause | Count | Questions |
|---|---:|---|
| Pipeline defect | **6** | Q54, Q57, Q58, Q59, Q60, Q65 |
| Corpus limitation | 2 | Q55, Q66 |
| Partial corpus limitation | 1 | Q64 |

Only:

```text
2 / 9 = 22.22%
```

of abstentions were clearly justified by corpus limitations.

This is one of the most important findings in the report.

A high abstention rate is not automatically evidence of epistemic reliability.

If the evidence exists but the system fails to use it, abstention is simply a pipeline failure.

---

# 7. Failure Analysis

## 7.1 Incorrect Gold Label — Q58

The stored answer for Q58 was:

```text
A — I only
```

However, an exhaustive scan of the Constitution corpus found the phrase **"political party"** 12 times across pages 65, 104, 251, 376–379.

Therefore Statement II is false.

Statement I is also false.

The audited answer is:

```text
D
```

This matters because the wrong gold label could have caused the evaluation to select a verifier configuration that produced a false `SUPPORTED` verdict.

It also demonstrates that:

> A benchmark can contain a wrong label, and a system can appear worse because it disagrees with the incorrect label.

Only labels contradicted by at least one system were audited, so the complete answer key has not yet been independently verified.

---

# 7.2 Prompt Accretion

An earlier report attributed changes in Q58 and Q63 to additional numeric rules.

That conclusion was retracted.

The later investigation showed:

1. Temperature had not been pinned.
2. The numeric rules did not change verdicts on the shipped retriever.
3. Q58 changed because of endpoint and whitespace differences.

The stronger lesson is:

> **When a classifier has an unstable decision boundary, adding another prompt rule is not necessarily the solution.**

The better architectural response is to decompose the verifier into independently testable checks.

---

# 7.3 Three Verifier Failure Modes

Detailed claim-level analysis identified three distinct failure modes.

### 1. Closed-list membership

Evidence that Article 368 requires State ratification for some subjects does not prove that a particular subject belongs to that list.

### 2. Cross-scope merging

Evidence about State Legislatures can be incorrectly applied to Parliament because the provisions have similar wording.

### 3. Strength inflation

Evidence that an authority "may give directions" does not establish that it can "take over total administration."

These failures are difficult to diagnose from final accuracy alone.

The per-claim architecture makes each failure traceable.

---

# 7.4 Answer Leakage

The answer-key verifier originally received the declared answer in its prompt.

Even though the prompt instructed the model not to use it as evidence, the answer remained in context.

Therefore the verifier was not genuinely independent.

The affected measurements were marked:

```text
answer_key_verifier.CONTAMINATED = true
```

and excluded from the reported verification results.

The lesson is general:

> **If information must not influence a model's decision, do not put that information in the model's context.**

---

# 7.5 Universal Negative Claims

Claims such as:

> "The Constitution does not mention X."

cannot be established reliably by searching only the top five ranked chunks.

Such claims quantify over the entire corpus.

The system now performs an exhaustive scan over all 1,149 chunks for these cases.

The scan is:

- Case-normalized
- Whitespace-normalized
- Standard-library based
- Explicitly reported using `chunks_scanned`

Two assumptions remain:

1. The corpus actually covers the required scope.
2. PDF extraction preserved the relevant term.

These assumptions are explicitly reported with the verdict.

---

# 7.6 Evaluation Harness Defects

Several defects were found in the evaluation infrastructure itself.

### Defect 1 — Unpinned verifier

The verifier was stochastic while the baseline was pinned.

### Defect 2 — Retrieval-depth mismatch

The benchmark justified k=5 while production verification used k=3.

### Defect 3 — Duplicate answer-mapping logic

The evaluation harness contained a second mapping implementation with bugs involving:

- Negated options
- `"None"` options

### Defect 4 — Abstentions treated as wrong

The original scoring penalized the exact behavior the architecture was designed to provide.

### Defect 5 — Hard-coded baseline

The baseline was hard-coded using the stored-label accuracy even when results were reported under audited labels.

### Defect 6 — Evaluation ordering

Evaluation modules depended on one another but did not enforce execution order.

The pipeline has now been centralized in:

```text
src/evaluation/run_all.py
```

It:

- Encodes the dependency graph.
- Runs stages in topological order.
- Checks report freshness.
- Checks retrieval-depth consistency.
- Prevents stale numbers from being quoted.

---

# 7.7 Orchestrator vs Evaluation Harness

Another important defect was discovered:

```text
src/orchestration/nodes.py
```

constructed `FactVerifier()` without the corpus.

This disabled the exhaustive absence-claim scanner in the actual orchestration path.

The evaluation harness had the corpus, while the shipped system did not.

Therefore:

> **The system being measured and the system being shipped were temporarily different systems.**

This is one of the most serious classes of evaluation bug.

The issue was fixed, and shared corpus/retriever/verifier components are now cached and reused.

---

# 7.8 Latent Claim-Builder Defects

Synthetic tests found two claim-binding cases that were not exercised by any of the 13 real questions:

- Missing leading-predicate case
- Unguarded leading-condition case

These fixes are deliberately not credited with an improvement in the benchmark.

The lesson is:

> A small dataset cannot validate behavior it never exercises.

Component-level contract tests are therefore required in addition to end-to-end evaluation.

---

# 7.9 Why the Architecture Changed

The initial architecture asked one LLM call to:

```text
Question
   ↓
Evidence
   ↓
Answer
```

This made it difficult to inspect why the model was wrong.

The current architecture separates:

```text
Claim construction
       ↓
Retrieval
       ↓
Per-claim verification
       ↓
Deterministic answer mapping
```

This decomposition made it possible to identify:

- Incorrect labels
- Retrieval-depth mismatches
- Claim-construction failures
- Scope errors
- Strength inflation
- Sampling instability
- Evidence-grounding failures

The key architectural benefit is therefore not merely a better answer.

It is:

> **A system whose failures have identifiable locations.**

---

# 7.10 TLS Diagnosis

The System C ablation originally ran on BM25 because `sentence-transformers` appeared impossible to install.

The actual issue was not network connectivity.

`urllib` successfully reached Hugging Face while `httpx` failed, revealing a TLS trust-store mismatch caused by a corporate proxy.

The fix was implemented using:

```python
truststore.inject_into_ssl()
```

This allowed the shipped semantic + cross-encoder retriever to be used for the final ablation.

The broader lesson:

> **A library's error message is a hypothesis, not proof of the underlying cause.**

The incorrect diagnosis caused several earlier ablation conclusions to be made on the wrong retriever.

---

# 7.11 Generation Retry Counter

The generation loop interpreted:

```text
max_retries = 2
```

as two total generations.

The implementation actually permits:

```text
1 initial generation + 2 retries = 3 generations
```

Therefore earlier attempt-level rates used the wrong denominator.

This defect became visible only after the generation loop was explicitly measured.

It reinforces the importance of measuring orchestration mechanics, not only final outputs.

---

# 7.12 Run-to-Run Instability

Claim-level instability propagates into system-level metrics.

Pinned verifier runs still produced:

```text
Coverage: 23.1–38.5%
Precision: 1.00 in 4/5 runs
Precision: 0.75 in 1/5 runs
```

Therefore the headline:

> "100% precision when answered"

describes one run, not a guaranteed system property.

The more defensible result is:

> **Precision 0.95 ± 0.11 over five resampled runs.**

This is why small architectural differences should not be interpreted from a single 13-question run.

---

# 8. Threats to Validity

The main threats are:

1. **Verifier instability**  
   Temperature 0 does not guarantee deterministic outputs.

2. **Small sample size**  
   Only 13 historical questions and 37 claims.

3. **Underpowered stability estimate**  
   Only five repeats were used.

4. **Small retrieval benchmark**  
   Only 16 queries.

5. **Historical retrieval results not fully rerun**  
   Four of five retrieval configurations remain historical measurements.

6. **Incomplete corpus**  
   Two questions are outside the available corpus.

7. **Unverified answer labels**  
   Only one contradicted label has been audited so far.

8. **Judge circularity**  
   The judge rewards behavior the system was explicitly designed to exhibit.

9. **Judge/model family overlap**  
   `gpt-4o` judges `gpt-4o-mini` outputs.

10. **Pipeline defects**  
    Six of nine historical abstentions were caused by system defects.

11. **Prompt instability**  
    Boundary-case verdicts can change with whitespace or endpoint.

12. **Generated-question evaluation is under-sampled**  
    Only one generation run was performed per topic.

---

# 9. What the Results Actually Establish

The evaluation supports the following conclusions.

### Supported

- Evidence decomposition reduces unsupported final assertions.
- Deterministic answer mapping provides a clear separation between reasoning and option selection.
- Predicate binding eliminates non-propositional claims in the tested 37-claim set.
- The verification system produced zero false final answers in the historical benchmark runs where it committed.
- Vanilla RAG produced substantially more false answers in the same experiment.
- Semantic retrieval + cross-encoder reranking produced the best observed retrieval result.
- The generation gate detects significant defects in generated questions.

### Not established

The evaluation does **not** establish that:

- The verifier is generally accurate.
- Temperature 0 makes the system deterministic.
- The reranker has a statistically significant advantage over semantic retrieval.
- Retrieval depth 5 is better than depth 3.
- Numeric conflict rules improve accuracy.
- The system is ready for student-facing generated questions.
- The 75-question benchmark demonstrates broad reliability.
- The 53.33% generation acceptance rate is a stable production yield.

These distinctions are important because the project prioritizes measurement integrity over favorable headlines.

---

# 10. Next Steps

The next work should be prioritized as follows.

## 1. Measure instability more thoroughly

Increase claim resampling from 5 runs to approximately 20.

This would provide a more useful estimate of the variance band.

---

## 2. Expand the evaluation dataset

The current 13-question benchmark is smaller than the observed measurement noise.

A larger benchmark is necessary before comparing architectural variants.

The 75-question benchmark is the current regression set, but fresh untouched questions are still needed for independent evaluation.

---

## 3. Decompose the verifier

Replace the increasingly large verifier prompt with independently testable checks for:

- Scope
- Strength
- Numeric consistency
- Membership
- Other domain-specific constraints

The Q58 boundary-case analysis strongly supports this direction.

---

## 4. Fix the six pipeline-defect abstentions

These are the most valuable coverage improvements because the required evidence already exists.

Unlike CLOSED_WORLD, fixing these should increase coverage without deliberately increasing false assertions.

---

## 5. Audit the remaining answer labels

Q58 demonstrates that official or stored labels should not automatically be treated as unquestionable ground truth.

The remaining benchmark labels should be independently checked.

---

## 6. Re-run the retrieval benchmark

The five retrieval configurations should be rerun together after the final configuration and evaluation pipeline changes.

This will place every retrieval number on the same measurement footing.

---

# Appendix — Evaluation Artifacts

| Artifact | Purpose |
|---|---|
| `data/evaluation/verdict_stability.json` | Verdict repeatability, pinned/unpinned comparisons and resampled runs |
| `data/evaluation/system_c_results.json` | Six-arm ablation, verdict distributions, claim soundness and policy results |
| `data/evaluation/system_c_results.bm25.json` | Historical BM25 ablation and retracted conclusions |
| `data/evaluation/system_b_metrics.json` | A/B/B2 results, grounding, claim construction and answerability |
| `data/evaluation/judge_results.json` | Judge scores, hallucinations and preferences |
| `data/evaluation/answer_mapping_replay.json` | Abstention-policy evaluation |
| `data/evaluation/direct_option_results.json` | B2 direct-option verification |
| `data/evaluation/selective_metrics.json` | Selective prediction and paired comparison |
| `data/evaluation/verified_pyq_results.json` | Final verifier traces and evidence |
| `data/evaluation/verified_pyq_results.unpinned.json` | Historical unpinned verifier run |
| `data/evaluation/vanilla_rag_scored.json` | Vanilla RAG baseline |
| `docs/retrieval_baseline.md` | Retrieval benchmark |
| `docs/upsc_polity_capstone_development_log.txt` | Chronological development record |

---

# Reproducing the Evaluation

Run the complete evaluation pipeline:

```bash
python -m src.evaluation.run_all
```

Check consistency without making API calls:

```bash
python -m src.evaluation.run_all --check
```

The check validates:

- Dependency ordering
- Report freshness
- Retrieval-depth consistency

Deterministic component checks can be reproduced without an API key:

```bash
python -m src.verification.claim_builder

python -m src.verification.negative_claim_checker

python -m src.evaluation.answer_mapping

python -m src.evaluation.label_audit

python -m src.evaluation.system_c_pipeline
```

---

# Final Conclusion

The central result of this project is not that the verifier is highly accurate.

It is more specific:

> **A source-grounded system can substantially reduce unsupported answers by separating claim verification from answer selection and by allowing the system to abstain when the evidence is insufficient.**

The historical experiment demonstrates the trade clearly:

```text
                    Vanilla RAG       Verified System
------------------------------------------------------
Coverage              100%                30.77%
Accuracy             46.15%*              30.77%
Error rate           53.85%*               0%
```

`*` Audited-label result.

The verified system sacrifices coverage to eliminate false assertions.

However, the evaluation also exposes an important weakness:

> **Most historical abstentions were not justified.**

Six of nine abstentions occurred because the pipeline failed to use evidence that was already present in the corpus.

Therefore, the next objective is not to make the system more willing to guess.

It is to make the system **better at recognizing evidence it already has**.

That is the key engineering problem left by this evaluation.