# Evaluation Report — Source-Verified UPSC Polity MCQ Verifier

**Author:** Charan Kumar Basam · **Cohort:** AI Engineering · **Date:** 2026-09-27
**Companion document:** [`DESIGN.md`](DESIGN.md) (problem, architecture, decisions)

This report is the measurement record. It states what was measured, what the
numbers are, which of them I am willing to stand behind, and which findings
are negative, null, or self-inflicted. Where a number is not trustworthy it is
marked unusable rather than omitted — an omitted arm reads as an arm that was
never run.

---

## 0. How to read these numbers

**The dataset is small.** UPSC 2025 Prelims Polity, Q54–Q66: **13 questions,
37 numbered statements**. One question is 7.69 accuracy points.

**Two label sets, both reported.** The stored labels come from the published
answer key. The audit in §7.1 found one of them wrong (Q58). Every end-to-end
figure is therefore reported under **stored labels** and **audited labels**,
so the size of the correction is visible instead of being a choice I made
silently.

**Four metrics, because a single accuracy number cannot express abstention.**
For a system allowed to decline, scoring an abstention as a wrong answer
conflates *declined to commit* with *asserted something false* — the exact
distinction the architecture exists to make. So:

| metric | definition |
|---|---|
| coverage | answered / total |
| precision when answered | correct / answered |
| accuracy overall | correct / total |
| **error rate overall** | **wrong / total** — the metric the design targets |

`accuracy + error rate + abstention rate = 1`. A system that abstains on
everything scores 0% accuracy and 0% error rate; one that guesses everything
maximises coverage and error rate together. Reporting both ends is what makes
the trade visible.

### 0.1 Measurement noise — what this evaluation can and cannot resolve

**This is the most important thing to know before reading any table below, and
it was discovered late.** `FactVerifier.verify` never set `temperature`. The
OpenAI Responses API defaults to 1.0, so the verifier was a **stochastic
classifier**, and every System B figure ever reported was one draw from a
distribution nobody had looked at. The baseline in `rag/vanilla_rag.py` *was*
pinned at 0 all along, so the headline comparison put a sampling system against
a deterministic one and attributed the whole difference to verification.

All five unpinned call sites are now pinned (`fact_verifier`,
`claim_extractor`, `answer_key_verifier`, `mcq_quality_auditor`,
`mcq_generator`). `src/evaluation/verdict_stability.py` then measured what the
defect was worth by re-verifying all 37 claims 5× against their own stored
evidence, in both conditions:

| condition | stable claims | unstable | questions touched |
|---|---|---|---|
| unpinned (produced every previously reported number) | 28/37 (75.68%) | 9 | Q55, Q57, Q58, Q60, Q63, Q64, Q65, Q66 |
| **pinned, temperature = 0** | **35/37 (94.59%)** | **2** | Q58, Q63 |

**Pinning is not determinism.** Two claims still disagree with themselves at
temperature 0, and one of them (Q58 I) returned **all three verdicts** across
five identical calls. `temperature=0` reduces variance in this model; it does
not remove it.

Because each claim was sampled independently, draw *i* across all 37 claims
forms a valid independent pipeline run. Five such runs, strict policy, audited
labels:

| condition | coverage | precision | accuracy | error rate |
|---|---|---|---|---|
| unpinned | 23.08 – 46.15% | 75.00 – 100% | **23.08 – 38.46%** | 0 – 7.69% |
| pinned | 23.08 – 38.46% | 75.00 – 100% | **23.08 – 38.46%** | 0 – 7.69% |

**The accuracy spread is 15.38 points — two questions — pinned or not.** The
pinned condition has fewer unstable claims but the two survivors (Q58, Q63) are
exactly the ones that swing a predicted option, so the metric spread barely
moved.

Three consequences, applied throughout this report:

1. **No difference of one or two questions is a result.** That covers **every
   arm-to-arm difference in the §4 ablation**, all of which are ≤7.69 points.
   §4 is therefore reported as a set of null results with one exception that is
   not an accuracy number.
2. **Claim-level metrics are the trustworthy ones** — 37 observations instead of
   13, and claim soundness (§4.4) needs no gold label at all.
3. **The A-vs-B headline survives comfortably**, because it is a ~46-point gap
   (§2), roughly six questions, far outside the band.

**Limitation of the noise estimate itself:** 5 repeats is few. A claim whose
minority verdict appears 10% of the time will usually look stable at n=5, so
**9 and 2 unstable claims are lower bounds**, not point estimates. The
direction of the error is known and it is the unflattering one.

Implementations: `src/evaluation/system_b_metrics.py`,
`src/evaluation/answer_mapping.py` (mapping and abstention policies, with an
executable `self_check`), `src/evaluation/label_audit.py`,
`src/evaluation/verdict_stability.py`, and `src/evaluation/run_all.py` (runs
the chain in dependency order and refuses to let stale numbers be quoted —
see §7.6).

---

## 1. Retrieval: five approaches, 16-query gold benchmark

Metric: **Recall@5** against page-level gold labels, same 16 queries and same
1,149-chunk corpus for every approach.

| approach | hits | Recall@5 |
|---|---|---|
| BM25 (lexical) | 12/16 | 75.00% |
| Hybrid BM25 + semantic, Reciprocal Rank Fusion | 13/16 | 81.25% |
| Semantic — `BAAI/bge-small-en-v1.5` | 14/16 | 87.50% |
| Parent–child (page parents) + reranker | 14/16 | 87.50% |
| **Semantic → MS-MARCO MiniLM cross-encoder rerank** | **15/16** | **93.75%** ← selected |

**The negative result is the hybrid.** RRF fusion is the more sophisticated
method and it *lost* to plain semantic retrieval (81.25% vs 87.50%). BM25's
lexical hits pulled genuinely relevant semantic hits down the fused ranking on
this corpus, where the queries are conceptual ("which subjects require State
ratification") and the documents are legal prose. Parent–child retrieval added
nothing over plain semantic either: it cost a page-assembly stage to arrive at
the same 14/16.

What survived was the cheap thing: keep semantic recall, fix precision with a
reranker over 20 candidates. That is the configuration the system ships.

**Caveat carried forward:** 16 queries. 93.75% is 15/16; one query is 6.25
points. This benchmark is strong enough to *reject* hybrid RRF and parent–child
(both lost by 1–2 queries and cost more) and not strong enough to claim the
reranker's margin is 6.25 points rather than noise.

**Provenance note:** this table is from the original retrieval pass. These five
benchmarks were **not re-run** in the final measurement pass, unlike every other
table in this report. They are deterministic given the corpus (embeddings and
cross-encoder scores, no sampling), so re-running should reproduce them, but I
have not verified that and am labelling it rather than implying it.

**The defect this benchmark caused.** The retriever was selected on Recall@5
and then queried at `top_k=3` throughout claim verification. The depth that
justified the choice was never the depth it ran at. Both paths are now pinned to
5. What this cost is *not* cleanly measurable — see §4.2, where the depth
change turns out to run entirely through one unstable claim.

---

## 2. Approach comparison: three inference paths over the same evidence

All three share the retrieval stage (semantic → cross-encoder rerank), so this
comparison is retriever-matched. They differ in what the LLM is asked to decide.

| arm | what the LLM decides | cov | prec | acc | **err** |
|---|---|---|---|---|---|
| **A** — vanilla RAG | reads question + top-5, writes an answer | 100% | 38.46% | 38.46% | **61.54%** |
| **B** — statement verification → deterministic mapping | one claim vs its own evidence, three-way verdict; Python picks the letter | 30.77% | **100%** | 30.77% | **0%** |
| **B2** — direct option verification | whether an *option* is established by the evidence | 15.38% | **100%** | 15.38% | **0%** |

Figures under stored labels. Under **audited labels**: **A** 46.15% acc /
53.85% err · **B** unchanged at 30.77% / 0% (its four answers — Q56, Q61, Q62,
Q63 — are correct under both label sets) · **B2** unchanged.

**A vs B, stated honestly.** B's accuracy is *worse* than A's under both label
sets: 30.77% against 38.46% stored, 46.15% audited. **The claim is not that
verification makes the system more accurate.** It is the error column:

> A asserts a false answer on **8 of 13** (stored) / **7 of 13** (audited).
> B asserts a false answer on **0 of 13** under both.

B buys the elimination of *every* false assertion by declining 9 questions, and
gives up one-to-two questions of accuracy to do it. That is a ~46–54 point
error-rate difference — about six questions — which is the one end-to-end result
in this report comfortably larger than the ±2-question noise band of §0.1.
Whether the trade is worth making depends on the objective, and the objective in
`DESIGN.md` §1 was fixed before any of this was measured.

**B2 is a real architectural alternative and it is reported because it failed
informatively.** Asking the model to verify a whole option is asking it to
verify a conjunction of two to four statements in one step, so it almost never
reaches the confidence threshold. The variant given *more* evidence did worse:

| B2 condition | evidence | answered | correct | coverage |
|---|---|---|---|---|
| answer-evidence | top-5 question-level passages | 2 | 2 | 15.38% |
| union | question-level **+** per-statement passages | **1** | 1 | **7.69%** |

Widening the context to the union of all per-statement evidence **halved
coverage, from 2 questions to 1**. More evidence for a conjunctive claim means
more surface on which some part of it looks unestablished. Both conditions were
right when they answered (100% precision, zero errors), but a system that
answers 1–2 questions in 13 is not a system.

That is the observation that justifies the shipped architecture: **decompose the
conjunction, verify the parts independently, and let deterministic code
recombine them.** B reaches 30.77% coverage on the same evidence B2 gets 15.38%
on, because B never asks one model to hold four propositions at once.

(A 2→1 coverage change is itself one question and inside the noise band. The
*direction* is what the argument rests on, and it is corroborated by the
mechanism being visible in the per-option verdicts, not only by the count.)

**Paired check on the questions B attempted** (`selective_metrics.py`): on the
4 questions B answered — Q56, Q61, Q62, Q63 — **A scored 2/4 and B scored 4/4**.
B is better where both commit, by two questions. Per §0.1 that margin is inside
the noise band and I am not claiming it; the robust statement remains the error
column above.

---

## 2.5 The same verifier as a gate: generate-and-verify, two format arms

Everything above measures the verifier **answering** questions. This section
measures it **gating** questions it generated itself — the same claim
verification, the same corpus, applied to text the system has never seen
scored. It is a second consumer of the verifier, not a second product, and it
is reported here because it stresses the verifier on input that was not
curated by UPSC.

**Protocol.** `src/evaluation/evaluate_generation_loop.py`, 15 Polity topics per
arm, `max_retries=2`. The LangGraph loop is
`generate → extract claims → verify claims → verify answer key → audit quality
→ decide`, with `decide` emitting ACCEPT, REVISE (loop back to generate) or
REJECT. Per-attempt state recovered with `stream_mode="updates"`; `invoke`
returns only the final state and would have made every table below impossible.
One draw per topic — see the noise caveat at the end.

| | `simple` | `statements` |
|---|---|---|
| Topics completed | 15 | 15 |
| ACCEPT | **8 (53.33%)** | **0 (0.0%)** |
| REJECT | 7 (46.67%) | 15 (100.0%) |
| Total attempts | 34 | 45 |
| Blocked attempts | 26 | 45 |
| Attempt-1 defect rate | 73.33% | **100.0%** |
| Repair rate (blocked → eventually accepted) | 36.36% | **0.0%** |
| Claims verified | 35 | 149 |
| Mean claims per attempt | 1.03 | **3.31** |
| Verdicts | SUPP 26 / INSUFF 9 | SUPP 105 / **CONTRA 22** / INSUFF 22 |

### 2.5.1 The unverified-generator baseline, for free

Attempt 1 of every topic is what a RAG generator with **no gate in front of it**
would have shipped. That makes the baseline free — no extra arm, no extra
tokens:

| | `simple` | `statements` |
|---|---|---|
| n | 15 | 15 |
| Clean | 4 | **0** |
| Defective | 11 (**73.33%**) | 15 (**100.0%**) |

**This is the strongest single argument for the gate in the whole report.**
Roughly three of four ungated questions carry a defect the pipeline can detect,
and in the multi-statement format it is four of four. Unlike the §2 comparison,
this does not depend on a 13-question dataset or on my audited labels — the
defects are found by the system's own gates and each one is attributable.

### 2.5.2 Gate attribution

Gates are not mutually exclusive. `blocked_by` counts every gate that fired;
`sole_blocker` counts attempts where exactly one did — which is the column that
says whether a gate is earning its place.

| gate | `simple` blocked / sole | `statements` blocked / sole |
|---|---|---|
| fact | 9 / **6** | 38 / 1 |
| answer_key | 10 / 2 | 41 / 1 |
| quality | 17 / **8** | 42 / 0 |

In the `simple` arm all three gates catch things nothing else catches, so all
three are justified. In the `statements` arm `sole_blocker` collapses to 2 of 45:
almost every attempt trips **all three gates at once**. That is not three
independent checks agreeing; it is one underlying defect visible three ways, and
it is why the 0% accept rate should be read as a calibration problem rather than
as three-fold confirmation that the output was bad.

### 2.5.3 What the 0% accept rate actually shows

The tempting headline — "the gate rejects 100% of multi-statement questions, so
it is strict and therefore good" — is wrong, and the claims column is what
refutes it. The `statements` arm produced **22 CONTRADICTED verdicts**; the
`simple` arm produced zero. Generating three statements per question instead of
roughly one does not just triple the claim count (1.03 → 3.31 per attempt), it
changes the *kind* of error: the generator starts asserting things the corpus
actively denies, not merely things it cannot confirm.

So two findings sit here, and only the first is flattering:

1. The gate detects a real, format-dependent degradation in generation quality
   that an accuracy number alone would have hidden.
2. **A gate that accepts 0 of 15 is not usable**, however correct each
   individual rejection is. With `max_retries=2` there is no path to ACCEPT if
   the generator reliably produces at least one contradicted statement per
   attempt. Either the retry budget or the quality bar is mis-set for this
   format, and §2.5.2's `sole_blocker` collapse says the bar.

### 2.5.4 Two defects this measurement exposed

**The attempt counter was off by one.** `attempts = retries + 1`, so
`max_retries=2` permits three generations, not two. Every per-attempt rate
computed before the fix was wrong in the denominator. The bug was invisible
while the loop was unmeasured, which is the general hazard of §16.1: an
unmeasured component's defects are not *absent*, only *unobserved*.

**A 3-of-15 run was nearly published.** An earlier `statements` run lost three
topics to connection errors and reported ACCEPT 2/12 = 16.67%. That file is
retained as `generation_loop_results.statements.partial12.json` rather than
deleted, because the two files side by side are the evidence for why
`topics_errored` is a reported field and not a log line. The numbers in this
section are the complete 15-topic re-run.

### 2.5.5 Noise

**One draw per topic.** §0.1 applies here with more force than to the verifier:
a topic's outcome is the product of up to three sequential LLM calls, each of
which the stability harness shows is not reproducible even at temperature 0
(35/37 claim verdicts stable when pinned, 28/37 unpinned). Treat 53.33% as "about
half" and 0% as "none of 15" — the second is robust because the floor is not
reachable by chance drift, the first is not. No resampling was run for this arm;
at roughly 250 API calls per arm, five draws was not affordable, and saying so is
more useful than presenting 53.33% as a point estimate.

---

## 3. Judge evaluation: LLM-as-a-judge on reasoning quality

**Protocol.** Judge `gpt-4o`; generator `gpt-4o-mini`. Both systems' outputs
for the same question presented in **randomised slots** with system identity
withheld. The judge scores **reasoning quality only** — 1–5 on four dimensions
— and is explicitly told not to determine correctness. Correctness comes from
the answer key, never the judge. 13 questions judged.

| dimension | A (vanilla RAG) | B (verified) | Δ |
|---|---|---|---|
| factual correctness | 2.77 | 3.69 | +0.92 |
| evidence faithfulness | 2.46 | 4.69 | +2.23 |
| reasoning validity | 2.69 | 4.54 | +1.85 |
| **epistemic honesty** | 2.38 | **5.00** | **+2.62** |
| mean across dimensions | 2.58 | 4.48 | +1.90 |
| hallucinated claims (count) | **19** | **1** | −18 |
| questions containing ≥1 hallucination | **12** | **1** | −11 |

Head-to-head preference: **B 12 · A 1 · TIE 0**.

**Position-bias control.** Preference by *slot* rather than by system: slot 1
= 5, slot 2 = 8, TIE = 0. Since slot assignment was randomised, a judge with no
position bias should split slots roughly evenly, and 5/8 is well within what 13
trials produce by chance. More usefully, the slot split (5/8) is much flatter
than the system split (1/12), meaning B won from both slots — the preference
tracks the system, not the position.

**What the judge result is worth — and the circularity in it.** B scores a
**perfect 5.00 on epistemic honesty**, and its mechanism for winning is *saying
"insufficient evidence" with page citations*. A judge asked to reward calibrated
hedging will reward a system that hedges by construction. The per-question trace
makes this explicit: B is preferred on **Q54, Q55, Q57, Q58, Q60, Q64, Q65,
Q66 — eight questions where B answered nothing at all**. The single question
where A wins is Q59, where A answered correctly and B abstained.

So the judge is partly measuring *the willingness to abstain*, which is the
architecture's premise rather than an independent check on it. **A system that
abstained on all 13 questions would score near-perfectly on this rubric and be
useless.** That is precisely why the quantitative selective-prediction metrics
in §2 are reported alongside it: the judge cannot see that B answers only 4 of
13, and coverage cannot see that A's reasoning is unfaithful. Neither metric is
sufficient alone.

The number least exposed to the circularity is the hallucination count:
**19 → 1**, across **12 → 1** questions. Hallucinated claims are identified
extractively — a named Article, date, or provision absent from the cited
evidence — so it is closer to a check than a judgment. It is also a claim-level
count rather than a 13-question accuracy, which puts it on the right side of
§0.1. This is the single strongest measured result in the project, and it is the
one the whole architecture was built to produce.

---

## 4. Single-variable ablation (System C)

`src/evaluation/system_c_pipeline.py`. Six arms, each changing **exactly one
variable** against a named `compare_to` arm, running from a verdict cache keyed
by (claim, prompt variant, evidence chunk ids). Last run: 57 cache hits, 159
misses. The harness pins `temperature: 0` and always has, so it is unaffected by
the defect in §0.1 — but it is subject to the same residual instability.

**This ablation now runs on the shipped retriever** (semantic
`BAAI/bge-small-en-v1.5` → `cross-encoder/ms-marco-MiniLM-L-6-v2`,
`candidate_k=20`). An earlier version ran on a stdlib BM25 index because
`sentence-transformers` could not be installed; that is fixed (see
`src/tls_trust.py` for why the install was blocked, and §7.10). **C0 is now a
genuine reference point for System B**, and C-to-A comparisons are
retriever-matched. The BM25 run is preserved at
`data/evaluation/system_c_results.bm25.json` for comparison.

### 4.1 Audited labels, strict policy

| arm | single change | vs | cov | prec | acc | **err** |
|---|---|---|---|---|---|---|
| C0 | stored claims, k=3, base prompt | — | 30.77% | 75.00% | 23.08% | 7.69% |
| C1 | + predicate binding | C0 | 38.46% | 80.00% | 30.77% | 7.69% |
| C1d | **k=3 → 5 only** | C0 | 38.46% | 60.00% | 23.08% | 15.38% |
| C2 | + predicate binding at k=5 | C1d | 38.46% | 60.00% | 23.08% | 15.38% |
| C3 | + numeric conflict rules | C2 | 38.46% | 60.00% | 23.08% | 15.38% |
| C3d | numeric rules, **no** binding | C1d | 38.46% | 60.00% | 23.08% | 15.38% |

Stored labels, same arms, strict: C0 15.38% acc / 15.38% err · C1 23.08% /
15.38% · C1d, C2, C3, C3d all **30.77% / 7.69%**. Note that the label sets
disagree about the *direction* of the k=3→5 change — see §4.2.

**Read this table as six null results.** Every arm-to-arm difference is one
question (7.69 points) or zero. Per §0.1 the resolvable difference on this
dataset is two questions. **Nothing in this table is measurable**, and the three
conclusions the earlier BM25 run drew from it did not survive the retriever
swap. They are recorded in §4.2, §4.3 and §4.5 as retracted rather than deleted.

### 4.2 The depth effect: retracted, and why

The earlier BM25-based version of this report claimed C0→C1d (retrieval depth
3→5) was "the largest measured accuracy change in the project" and drew a
general lesson from it. **On the shipped retriever that claim is false in both
magnitude and sign:**

| labels | C0 (k=3) | C1d (k=5) | direction |
|---|---|---|---|
| stored | 15.38% acc / 15.38% err | 30.77% / 7.69% | depth **helps** |
| audited | 23.08% acc / **75.00%** prec | 23.08% / **60.00%** prec, err 7.69% → 15.38% | depth **hurts** |

A change whose sign depends on which label set you read is not an effect.

**Where it actually comes from.** The entire depth difference runs through a
single statement. C0→C1d changed three verdicts, and only one changes an answer:

```
Q58 I: CONTRADICTED (k=3) -> SUPPORTED (k=5)
```

At k=3 the pipeline answers Q58 = D (audited-correct); at k=5 it answers A (the
wrong stored label). And **Q58 I is precisely the claim §0.1 identified as
unstable — the one that returns all three verdicts across five identical calls
at temperature 0.** The "depth effect" is one boundary-case claim being
resampled, not a property of retrieval depth.

I investigated that claim directly, because if the ablation harness and the
shipped verifier disagree on it, the harness is not a faithful proxy. Holding
the claim and its five evidence pages fixed, at temperature 0 throughout:

| prompt | endpoint | verdict (3 runs) |
|---|---|---|
| shipped | `responses.parse` | INSUFFICIENT ×3 |
| shipped | `chat.completions` + JSON schema | SUPPORTED ×3 |
| harness | `responses.parse` | SUPPORTED ×3 |
| harness | `chat.completions` + JSON schema | SUPPORTED ×3 |

The two prompts differ by **three characters of whitespace** (one blank line
before rule 25, two between evidence blocks) and are otherwise byte-identical,
rules included. Only the conjunction of the shipped prompt *and* the
`responses.parse` endpoint yields INSUFFICIENT; changing **either** nuisance
factor flips the verdict. Neither factor alone is sufficient and both are
necessary.

**The honest conclusion is not about depth.** It is that this claim's verdict is
not a function of the claim and its evidence — it is a function of whitespace,
endpoint, and sampling. A verifier with a claim on its decision boundary will
produce different verdicts for reasons that have nothing to do with the
constitutional text, and a 13-question harness will faithfully convert that into
an "accuracy effect" with a plausible story attached.

**What still holds from the original lesson** is the weaker, cheaper version:
the retriever was justified at Recall@5 and queried at k=3, so a configuration
was running at settings it was never validated at. That was worth fixing on
principle. It is *not* worth the accuracy claim I previously attached to it.

### 4.3 Verdict distribution — mechanism, where accuracy is silent

| arm | SUPPORTED | INSUFFICIENT | CONTRADICTED | propositional | false SUPPORTED |
|---|---|---|---|---|---|
| C0 (stored, k=3) | 14 | 18 | 5 | 70.27% | **7** |
| C1 (bound, k=3) | 15 | 16 | 6 | **100%** | **0** |
| C1d (stored, k=5) | 17 | 16 | 4 | 70.27% | **8** |
| C2 (bound, k=5) | 17 | 15 | 5 | **100%** | **0** |
| C3 (bound, k=5, numeric) | 17 | 15 | 5 | **100%** | **0** |
| C3d (stored, k=5, numeric) | 17 | 16 | 4 | 70.27% | **8** |

Total = 37 claims. Depth converts INSUFFICIENT into SUPPORTED (C0→C1d: 18→16
INSUFFICIENT, 14→17 SUPPORTED) — more evidence lets true claims be *seen* to be
true, and also produces one more false SUPPORTED (7→8). Binding is the only
variable that moves the soundness columns, and it moves them completely.

**The numeric-rule arms are exactly null.** C2→C3 and C1d→C3d each report
`no statement changed verdict` — byte-identical verdict counts, byte-identical
metrics. On the BM25 run those rules had resolved Q63 II from INSUFFICIENT to a
correct CONTRADICTED, and that was written up as measurement-driven evidence for
keeping them. **On the shipped retriever they do nothing at all.** The earlier
result was a BM25-specific artifact. Rules 25–26 remain in the prompt (they are
harmless here and defensible on their own terms) but they are no longer claimed
to be justified by measurement.

### 4.4 Claim soundness — the one signal that exceeds the noise

Measured over **37 claims**, requiring no gold label:

| claim construction | propositional | non-propositional | **false SUPPORTED** |
|---|---|---|---|
| stored / verbatim statement text | 70.27% (26/37) | 11 | **7–8** |
| **deterministic predicate binding** | **100% (37/37)** | **0** | **0** |

A non-propositional claim is a fragment with no truth value — the statement
text `"List I-Union List, in the Seventh Schedule"` is a noun phrase, not an
assertion, because its predicate lives in the question stem. A verifier handed a
fragment cannot return a meaningful verdict, and what it actually does is worse
than abstain: it matches the fragment lexically against retrieved text and
returns **SUPPORTED**, which is a confident wrong answer wearing a page
citation.

This is the only ablation signal in §4 that clears §0.1's bar, and it clears it
by a distance: 11 defective claims → 0, and 7–8 false SUPPORTED verdicts → 0,
on 37 observations, reproduced at **both** retrieval depths (C0→C1 and C1d→C2),
with **no dependence on the gold labels** that §7.1 shows to be partly wrong.

Predicate binding is a template substitution guarded by two checks: no token may
appear that was absent from the source question (`unsupported_tokens`), and
split pairs must rejoin (`reconstructs`). Binding distribution over the 37
claims: `none` 22, `trailing_predicate` 7, `pair` 6, `leading_condition` 2;
invented tokens **0**.

### 4.5 The decision, and the evidence against it

The BM25 run had C3d (verbatim claims) tying C3 on accuracy and beating it on
error rate, so the arm I did *not* ship looked better and §4.5 previously
defended shipping C3 anyway. **On the shipped retriever that tension is gone**:
C3 and C3d are identical on every end-to-end metric (38.46 / 60.00 / 23.08 /
15.38 audited), and they differ only where §4.4 measures — propositional rate
100% vs 70.27%, false SUPPORTED 0 vs 8.

So the decision is now easy for the reason it should have been easy all along:
**the end-to-end metrics cannot distinguish the arms, and the claim-level
metrics separate them completely.** I ship bound claims (C3) because they are
the only arm that never asks the verifier to assign a truth value to a noun
phrase.

The reasoning that survives from the earlier version is the general principle:
when a 13-question accuracy column and a 37-claim soundness column disagree,
prefer the one with more observations and no dependence on a label set with a
known error in it. In this run they no longer disagree, which is a weaker test
of the principle than the BM25 run provided.

---

## 5. Abstention policy ablation

Three policies over **identical verdicts** — the verdicts are fixed inputs and
only the deterministic mapping changes, so this isolates the policy exactly and
costs no API calls (`replay_answer_mapping.py`).

| policy | rule |
|---|---|
| STRICT | any unresolved statement → abstain |
| ELIMINATION | eliminate options incompatible with known verdicts; answer iff exactly one survives |
| CLOSED_WORLD | read INSUFFICIENT as false (absence of evidence = evidence of absence) |

### 5.1 A null result worth reporting: STRICT ≡ ELIMINATION

**ELIMINATION measured identically to STRICT on every arm, under both label
sets, on every metric.** On the shipped verdicts both answer the same 4
questions (Q56, Q61, Q62, Q63), all 4 correct. All six C arms: identical to four
decimal places.

This is not a bug and not a wasted experiment. ELIMINATION is provably a strict
generalisation of STRICT — it can never answer *fewer* questions, since any case
STRICT resolves has all statements known and therefore exactly one surviving
option. The null result says something specific about the data: on these 13
questions, **partial verdict information never narrowed the candidate set to
exactly one option.** Two-statement questions offer four options, so knowing one
statement halves the field to two and cannot finish the job; the larger
questions leave more than one survivor. ELIMINATION would need either more
statements per question or a more constrained option format to pay off.

`answer_mapping.py`'s `self_check` asserts both the equivalence on
unresolved-free input *and* a synthetic case where ELIMINATION answers `B` while
STRICT abstains, so the generalisation is verified even though the dataset never
exercises it. The two policies also fail differently, which the debug reasons
record: STRICT reports `policy_insufficient_evidence`, ELIMINATION reports
`multiple_options_possible` on the same nine questions.

### 5.2 CLOSED_WORLD: the price of an assumption

On the shipped verdicts:

| policy | labels | cov | prec | acc | **err** |
|---|---|---|---|---|---|
| STRICT | stored | 30.77% | 100% | 30.77% | **0%** |
| CLOSED_WORLD | stored | 84.62% | 54.55% | **46.15%** | **38.46%** |
| STRICT | audited | 30.77% | 100% | 30.77% | **0%** |
| CLOSED_WORLD | audited | 84.62% | 63.64% | **53.85%** | **30.77%** |

**CLOSED_WORLD is the highest-accuracy configuration in this entire report**
(53.85% audited), and it beats vanilla RAG on *both* metrics simultaneously —
53.85% vs 46.15% accuracy, 30.77% vs 53.85% error rate. If the objective were
accuracy, this is the arm to ship.

It is not the arm I ship, and the reason is in the same row: it converts nine
abstentions into eleven answers at roughly a coin-flip rate, taking the error
rate from **0% to 30.77%** — from zero false assertions to four. For a system
whose stated purpose (`DESIGN.md` §1) is never to assert an unsupported
constitutional claim, buying 3 questions of accuracy with 4 false assertions is
the wrong side of the trade.

It is also *wrong on this data* in a way it cannot detect: the corpus
demonstrably does not cover 2 of 13 questions (§6), so "absence of evidence is
evidence of absence" is false here. CLOSED_WORLD has no way to know that and
STRICT does not need to. It is implemented and reported as the measured price of
the assumption, never as a result.

The accuracy gap between STRICT and CLOSED_WORLD is 3 questions, which is the
one policy comparison in this report that clears the §0.1 noise band.

### 5.3 The mapping fix, and why it needed the label audit

The original mapping abstained on **Q58** with `no_option_matched` because the
option text `"None"` parsed as neither a statement set nor a count. The
corrected mapping (§7.6) parses it. On the current verdicts Q58 resolves to
INSUFFICIENT and is abstained on for a *substantive* reason rather than a
parsing bug, so the fix no longer changes the headline count — but it changes it
for CLOSED_WORLD, where Q58 now maps to `D`, which is the **audited-correct**
answer and the **stored-wrong** one.

That is the concrete reason the audit in §7.1 is part of the evaluation rather
than an aside: under stored labels the parsing fix looks like it introduced an
error, and under audited labels it fixed one. The fix was always right; the
label made it look wrong.

---

## 6. The answerability ceiling

A lexical probe over **all 1,149 chunks** (not a ranked retrieval) for every key
concept in every question, counting a concept as absent only when all its
surface variants score zero. Absence is decisive — no retriever can find what is
not there; presence is only weak evidence of answerability.

| verdict | questions | missing concepts |
|---|---|---|
| out of corpus | **Q55, Q66** | `lokpal institution`; `crude oil`, `petroleum regulator` |
| partially out of corpus | Q64 | `national parks` |
| in corpus | Q54, Q56–Q65 (11) | — |

**2 of 13 questions are not answerable from this corpus at any retrieval
quality.** The bare accuracy ceiling for a system that never guesses is
therefore **11/13 = 84.62%**, and every accuracy figure in this report should be
read against that, not against 100%.

Restricting to the 11 in-corpus questions (`answerability_adjusted`): coverage
36.36%, precision **100%**, accuracy 36.36%, error rate **0%**. Restricting
scope barely moves the numbers — because the abstentions are mostly *not* the
corpus's fault, which the taxonomy below makes explicit.

### Abstention taxonomy — 9 abstentions, classified by cause

| cause | count | questions |
|---|---|---|
| pipeline defect (evidence exists, system failed to use it) | **6** | Q54, Q57, Q58, Q59, Q60, Q65 |
| corpus limit (correct abstention) | 2 | Q55, Q66 |
| partial corpus limit | 1 | Q64 |

**Correct-abstention share: 22.22%.** This is the most self-critical number in
the report and the reason it is here. A high abstention rate is only a virtue
when the abstentions are *justified*; **6 of 9 are not**. The system declines
questions whose answers are sitting in the corpus, which means most of its
apparent epistemic humility is undiagnosed failure.

This also qualifies §3 directly. The judge awarded System B a perfect 5.00 on
epistemic honesty, and 6 of the 9 abstentions it was rewarding for are bugs.
Classifying abstentions by cause is what converts "the system is appropriately
cautious" into a list of six specific things to go fix.

---

## 7. Failure analysis

Ordered by how much each one changed my understanding, not by severity.

### 7.1 A wrong gold label (Q58) — the finding that inverted a gradient

Five of six ablation arms contradicted the stored answer for Q58. When most arms
disagree with the key, the arms are usually wrong. Here they were not.

Stored label: `A` ("I only"). Statement II claims the Constitution never uses
the phrase *political party*. An **exhaustive scan of all 1,149 chunks** finds
it **12 times across pages 65, 104, 251, 376, 377, 378, 379**. Statement II is
false; statement I is also false against the text; the answer is **`D`**.

Two consequences:

1. **The label error inverted the optimisation gradient.** Under the stored
   label, the arm that scored best on Q58 was the arm returning a **false
   SUPPORTED** verdict. I was one decision away from selecting a
   hallucination-producing configuration *because* it agreed with a wrong key.
2. **Pages 65 and 104 never appeared in any top-k retrieval.** A ranked search
   at any k would have missed occurrences the exhaustive scan found. Same
   structural point as §7.5: a claim quantifying over the whole corpus cannot be
   settled by a ranked sample of it.

Q58 is also, independently, the unstable boundary claim of §0.1 and §4.2. The
question with the wrong label is the question the verifier cannot make up its
mind about — which is not a coincidence so much as a sign that genuinely
ambiguous items are hard for both the examiner and the model.

**Limitation, stated plainly:** only labels that *some arm contradicted* could
be audited. Label quality is unverified for the questions every arm abstained
on. The audit is not a clean pass over the key; it is the questions the system
happened to argue about.

### 7.2 Prompt accretion, and a claim I had to retract

The earlier version of this report said that appending rules 25–26 (numeric
conflict detection) fixed Q63 II *and* flipped Q58 I "on byte-identical evidence
**at temperature 0**", and concluded that prompt rules are not additive.

**Three parts of that were wrong**, and finding out why produced §0.1:

1. **It was not at temperature 0.** `temperature` was never set; the Responses
   API default is 1.0. That observation was a single sample from an unpinned
   sampler.
2. **Rules 25–26 were never the cause.** With the retriever fixed and
   temperature pinned, C2→C3 changes **no verdict at all** (§4.3). The Q63 II
   fix was a BM25-specific artifact.
3. **Q58 I flips for reasons unrelated to any rule.** §4.2's 2×2 shows three
   whitespace characters or a change of endpoint flips it, at temperature 0.

**What survives is a stronger version of the original lesson, not a weaker
one.** At 26 rules the prompt is long enough that a claim near the decision
boundary is decided by whatever perturbs it last — a rule, a blank line, an
endpoint, or a sample. So single-variable ablation over prompt text is only
approximately single-variable, and the mechanism is not "rules interact" but
"the classifier has no stable answer for some inputs, and *any* perturbation
selects one". **The fix is decomposition, not another rule.** This is recorded
in `src/verification/fact_verifier.py` next to the prompt, including the
retraction, so the next person to reach for rule 27 reads it first.

### 7.3 Three distinct verifier failure modes, not one

Diagnosing the false verdicts individually rather than counting them turned one
"accuracy problem" into three mechanisms, each with a different fix:

1. **Closed-list membership.** Confirming a rule exists is not confirming that
   *this* subject falls under it. Evidence that Article 368 requires State
   ratification for some subjects was used to support a claim about a subject
   not on that list (Q54).
2. **Cross-scope merging.** Provisions about State Legislatures used as
   contradictory evidence against a claim about Parliament. Same grammatical
   shape, different constitutional entity (rules 20–24 target this).
3. **Strength inflation.** Evidence establishing "may give directions" accepted
   as establishing "can take over total administration" (Q65 II). The claim is
   strictly stronger than the evidence, and the semantic overlap is high enough
   to hide it.

All three are invisible in an accuracy number and all three are visible in a
per-claim verdict trace. This is what the architecture buys beyond the metrics:
the failures are *addressable* because each one localises to one component.

### 7.4 Answer leakage in the answer-key verifier

`src/verification/answer_key_verifier.py` built its prompt with the declared
answer under a `DECLARED ANSWER` heading, plus a rule instructing the model not
to treat it as evidence. The declared answer is the official key. So the
component's independence was **conditional on instruction-following**, which is
not independence.

`system_b_metrics.json` carries an explicit `answer_key_verifier.CONTAMINATED:
true` flag and its figures (INSUFFICIENT 3, INVALID 8, VALID 2, apparent
agreement 15.38%) **must not be reported as verification performance**. They are
excluded from every table above. The block and its guard rule are removed; the
actual comparison was already being done in Python, so nothing was lost. The B2
arm in §2 is the clean replacement — it runs with the declared answer withheld.

The general lesson: an instruction telling a model to ignore information in its
context is a design smell. If the information must not be used, it must not be
in the context.

### 7.5 Absence claims: a ranked top-5 cannot settle a universal quantifier

`NegativeClaimChecker` decided claims of the form "the Constitution does not
mention X" using a **BM25 top-5** search. The claim quantifies over the entire
corpus; the check sampled 5 ranked chunks. On this data it returned the right
answer, by luck.

Two fixes: the scan is now **exhaustive over all 1,149 chunks**, whitespace- and
case-normalised, stdlib-only; and a null result now returns **SUPPORTED** rather
than INSUFFICIENT, because absence from an exhaustive scan is informative where
absence from a ranked sample is not. Two residual assumptions remain — that the
corpus covers the claim's scope, and that PDF extraction preserved the term —
and both are **named in the emitted reasoning**, because neither can be checked
from inside the system. `chunks_scanned` is reported with every verdict so a
reader can confirm the scan was exhaustive rather than trusting that it was.

### 7.6 The evaluation harness was distorting its own measurement

Five defects, all in the instrument rather than the system:

1. **An unpinned sampler measured against a pinned baseline** (§0.1). The
   largest instrument defect in the project, found last.
2. **Querying at k=3 while justified on Recall@5** (§4.2).
3. **A second copy of the option-mapping logic**, and the copy held the bugs: a
   `negated` flag computed and then discarded (so "Neither I nor II" mapped as if
   it were "Both"), and the option text `"None"` parsing as neither a statement
   set nor a count, so it could never match (§5.3).
4. **Abstentions scored as incorrect answers** — which made every abstention the
   verifier added read as a regression, i.e. the harness was penalising the
   architecture's entire purpose.
5. **A hard-coded baseline.** `system_c_pipeline.py` pinned vanilla RAG at
   38.46% — the *stored-label* figure — while reporting every arm under both
   label sets. Under audited labels the baseline is **46.15%**. So the baseline
   was only comparable to half the table, and the half it was **not** comparable
   to was the half being quoted. Found by fact-checking my own design doc against
   the JSON, not by any test.

**A sixth defect was an ordering hazard, and it is now mechanically prevented.**
The evaluation modules read each other's output files, and nothing enforced the
order. Running them in the wrong one does not fail — it produces a
document-ready table built from a previous run's inputs. It happened twice:
`system_b_metrics` before `replay_answer_mapping` (making a consistency check
appear to fail), and `direct_option_verifier` before it (printing System B at
38.46/60.00/23.08 while the results file said 30.77/100/30.77).

`src/evaluation/run_all.py` now declares the dependency graph as data, runs the
stages in topological order, and afterwards asserts that **every report is at
least as new as every input it was derived from**. `--check` verifies the whole
chain without spending a token, and is the thing to run before quoting any
number in a document. It caught both of the above the first time it ran.

Mapping now has **one implementation** (`answer_mapping.py`) with an assertion
per fixed defect in an executable `self_check`, and the baseline is computed from
`vanilla_rag_scored.json` through the same label function every arm uses.

### 7.7 The orchestrated pipeline was weaker than the thing being measured

`src/orchestration/nodes.py` constructed `FactVerifier()` with **no corpus**,
which silently disabled the absence-claim scanner. Absence claims in the
orchestrated path therefore fell through to the LLM route that verifier rules
13–17 exist specifically to forbid. The evaluation harness passed a corpus; the
orchestrator did not. **The measured system and the shipped system were not the
same system** — the worst class of bug in an evaluation-heavy project, and one no
metric would have surfaced, because every metric was computed on the harness
path. Fixed, along with `lru_cache` on the shared chunks/retriever/verifier and
`CLAIM_TOP_K = ANSWER_TOP_K = 5` so the two paths cannot drift on depth again.

**This exact bug recurred in the new stability harness**, which is why it is
worth restating rather than declaring closed. `verdict_stability.py` initially
built `FactVerifier()` with no chunks; its resampled runs then abstained on Q58
in every draw while the pipeline answered it, and the reported metric spread was
the spread of a system nobody ships. Caught by noticing that the resample could
not reproduce the headline. The same one-line omission, in a module written
specifically to audit the system, three weeks after fixing it in the
orchestrator.

### 7.8 Two latent parser defects, found by construction rather than by data

A synthetic test of the claim builder found two binding defects — a missing
leading-predicate case and an unguarded leading-condition case — that **do not
fire on any of the 13 real questions**. I confirmed they don't fire *before*
fixing them, so their fix is credited with nothing in the ablation table and
appears in no metric.

They are recorded because they are the honest counterweight to the rest of this
report: a 13-question dataset cannot find defects its 13 questions don't
exercise, and the correct response is to test the component's contract directly
rather than to conclude from green metrics that the component is right.

### 7.9 The pivot that produced the architecture

The first design had one LLM read the question, the evidence, and produce a
verdict. It agreed with itself — asked to check its own reasoning, it confirmed
it, and there was no seam at which the agreement could be inspected.

Splitting the work into **claim construction → retrieval → per-claim
verification → deterministic mapping** is what made every finding in §7
*findable*. §7.3's three failure modes required per-claim traces. §7.1's label
error required per-arm disagreement. §4.4's claim soundness required claims to be
a separate artifact from verdicts. §0.1's noise measurement required claims and
their evidence to be stored as replayable data. The architecture's real output is
not a better answer; it is a system whose failures have addresses.

### 7.10 A TLS misdiagnosis that gated the main table

The §4 ablation ran on a stdlib BM25 index for most of the project, forcing a
disclaimer onto the main table, because `sentence-transformers` "could not be
installed". The model download failed with `LocalEntryNotFoundError: ... check
your connection` — i.e. `huggingface_hub` reported it as a **network outage**.

It was not a network problem. `urllib` reached huggingface.co with status 200
while `httpx` failed on the same URL, which isolates it: the connection is
TLS-intercepted by a corporate proxy, the private root is in the Windows
certificate store (which `urllib` uses) and absent from `certifi` (which `httpx`
pins). The fix is four lines — `truststore.inject_into_ssl()`, the same mechanism
as pip's `--use-feature=truststore` — recorded in `src/tls_trust.py` along with
the two rejected alternatives and why (`verify=False` disables verification for
the OpenAI calls too; `SSL_CERT_FILE` is ignored because `httpx` passes `cafile`
explicitly, and the corporate bundle fails Python 3.13+'s `VERIFY_X509_STRICT`).

The lesson is about diagnosis, not TLS: **a library's error classification is a
hypothesis, not evidence.** Taking "check your connection" at face value cost
weeks of the ablation running on the wrong retriever, which in turn produced the
three conclusions retracted in §4.2, §4.3 and §4.5.

### 7.11 `attempts = retries + 1` — an off-by-one hidden by not measuring

The generation loop took `max_retries`, and I read it as a cap on total
generations. It is a cap on *re*-generations: `max_retries=2` permits three
calls to the generator. Every per-attempt rate — defect rate, repair rate, mean
claims per attempt — was therefore computed against a denominator one too small
for any topic that exhausted its budget, which in the `statements` arm was all
15 of them.

What makes this worth reporting is not the arithmetic, which is trivial, but
**when** it was found: the moment the loop was measured, and not before. The
orchestration had been running and visibly "working" for days. A three-gate
retry cycle that produces plausible output on inspection can be wrong in its
control flow without anything looking wrong, because the only observable is the
final decision and the final decision is reached either way.

This is the concrete form of §16.1's general claim: an unmeasured component's
defects are not absent, only unobserved. It also argues for measuring the *cheap*
structural properties of an orchestration — attempt counts, loop bounds — before
the expensive semantic ones, since they are the ones an eyeball cannot audit.

### 7.12 Run-to-run instability is topic-level, not just claim-level

§0.1 establishes that individual claim verdicts are unstable: 35 of 37 stable
when pinned to temperature 0, 28 of 37 unpinned. The generation-loop measurement
showed that this does not stay contained at the claim level.

Resampling the verifier five times at temperature 0 moves **coverage between
23.1% and 38.5%**, and precision-when-answered from 1.00 in four draws to 0.75
in the fifth — a wrong answer got through on one draw of five. The headline
"100% precision when answered" in §2 is therefore a property of *one run*, and
the defensible statement is precision 0.95 ± 0.11 over 5 draws against vanilla
RAG's 0.38. The comparison survives; the perfect score does not.

Two consequences I had to accept rather than work around:

1. **ACCEPT/REJECT rates from §2.5 must not be quoted as point estimates.** A
   topic's outcome is the product of up to three sequential unstable LLM calls,
   so the compounding is worse than for a single verdict, and no resampling was
   affordable at ~250 API calls per arm. "About half" and "none of 15" are the
   strongest honest readings of 53.33% and 0%.
2. **A single run cannot establish a small improvement in this system.** Any
   change worth less than roughly 2 questions on a 13-question set is
   indistinguishable from redraw noise. This is why §2's argument rests on the
   error column — 8 wrong versus 0 — rather than on the coverage difference, and
   why §4.2, §4.3 and §4.5 are retractions: each claimed a difference smaller
   than the noise I had not yet measured.

The uncomfortable implication is that temperature pinning is a *reproducibility*
measure, not a determinism guarantee, and I had been treating the two as the same
thing. Pinning improved stability from 75.7% to 94.6% of claims; it did not reach
100%, and the residual is enough to move a headline metric.

---

## 8. Threats to validity

- **Verifier verdicts are not deterministic even at temperature 0** (§0.1). Two
  of 37 claims disagree with themselves across repeats, and the resulting
  accuracy spread is **15.38 points — two questions**. This governs every
  end-to-end number in the report and invalidates every one-question difference
  in it.
- **The noise estimate is itself under-powered.** 5 repeats per claim; the 9 and
  2 unstable-claim counts are lower bounds.
- **n = 13 questions, 37 claims.** One question is 7.69 accuracy points. The
  claim-level metrics (37 observations) and the hallucination counts are the most
  robust figures here; the accuracy column is the least.
- **The entire §4 ablation is within noise.** Every arm-to-arm end-to-end
  difference is ≤1 question. Only the claim-soundness column (§4.4) clears the
  band.
- **The ceiling is 84.62%, not 100%** (§6). 2 of 13 questions are unanswerable
  from this corpus.
- **Retrieval is judged on 16 queries** with page-level gold labels, the gold
  labels are mine, and those five benchmarks were not re-run in the final pass
  (§1).
- **Prompt arms are not isolated** (§7.2) — and the mechanism is worse than
  interaction between rules: a claim on the decision boundary is decided by
  whitespace and endpoint choice, so *any* prompt edit can flip an unrelated
  verdict.
- **The judge is substantially measuring willingness to abstain** (§3). B is
  preferred on 8 questions where it answered nothing, and scores a perfect 5.00
  on epistemic honesty. A system that abstained on everything would score well.
  The hallucination count is the part least exposed to this.
- **The judge shares the generator's family** (`gpt-4o` judging `gpt-4o-mini`).
- **The label audit is not a clean sweep** — only labels some arm contradicted
  were examined (§7.1).
- **6 of 9 abstentions are bugs, not epistemics** (§6). The system's caution is
  substantially undiagnosed failure.

---

## 9. What I would do next, in order

1. **Raise the repeat count and re-measure the noise band properly** (§0.1). 5
   repeats gives a lower bound on instability; 20 would give a usable confidence
   interval. This is first because every other number's interpretation depends on
   it, and it is cheap — the claims and evidence are already stored, so it is
   pure resampling.
2. **Extend the dataset past 13 questions.** Promoted from last to second. The
   noise measurement shows the problem is not that n is merely "small" — it is
   that n is smaller than the measurement error, so no architectural difference
   below two questions can be detected at all. No amount of further architecture
   work is measurable until this changes.
3. **Decompose the fact-verifier prompt** into independently testable checks
   (scope match, strength match, numeric match, membership) instead of appending
   rule 27 (§7.2). The boundary-case finding strengthens this: a decomposed
   verifier lets a single check be tested for stability in isolation, which a
   26-rule monolith does not.
4. **Fix the six pipeline-defect abstentions** (§6). These are questions the
   corpus can answer and the system declined — the only place where coverage can
   rise without buying it with errors.
5. **Audit the remaining labels**, including those no arm contradicted, so the
   ground truth stops being a partially-verified input.
6. **Re-run the five retrieval benchmarks** (§1) so the one table not regenerated
   in the final pass is on the same footing as the rest.

---

## Appendix — artifacts

| file | contents |
|---|---|
| `data/evaluation/verdict_stability.json` | **verdict repeatability, pinned vs unpinned, plus 5 resampled pipeline runs and the metric spread** |
| `data/evaluation/system_c_results.json` | six-arm ablation on the shipped retriever: verdict counts, claim soundness, all three policies × both label sets |
| `data/evaluation/system_c_results.bm25.json` | the superseded BM25 ablation, kept so the retracted conclusions in §4.2–§4.5 can be checked |
| `data/evaluation/system_b_metrics.json` | A/B/B2 arms, verdict distributions, grounding, claim construction flags, answerability, abstention taxonomy, contamination flag |
| `data/evaluation/judge_results.json` | per-question judge scores, slot assignments, hallucinated claims, preferences |
| `data/evaluation/answer_mapping_replay.json` | policy ablation over fixed verdicts |
| `data/evaluation/direct_option_results.json` | B2 direct option verification, both evidence conditions |
| `data/evaluation/selective_metrics.json` | A/B selective metrics + paired comparison on the attempted subset |
| `data/evaluation/verified_pyq_results.json` | the shipped pipeline's per-claim verdicts, evidence, and mapping traces |
| `data/evaluation/verified_pyq_results.unpinned.json` | the same run before temperature was pinned, kept for §0.1 |
| `data/evaluation/vanilla_rag_scored.json` | baseline, scored per question |
| `docs/retrieval_baseline.md` | retrieval experiments, 16-query benchmark |
| `docs/upsc_polity_capstone_development_log.txt` | chronological log |

Run the whole chain in dependency order, or verify consistency without spending
a token:

```bash
python -m src.evaluation.run_all            # run every stage in order, then check
python -m src.evaluation.run_all --check    # verify freshness only, no API calls
```

Reproduce the deterministic parts without an API key:

```bash
python -m src.verification.claim_builder            # 37 claims, 37/37 propositional, 0 invented tokens
python -m src.verification.negative_claim_checker   # exhaustive-scan self-check
python -m src.evaluation.answer_mapping             # mapping + all three policies, one assertion per fixed defect
python -m src.evaluation.label_audit                # Q58 finding
python -m src.evaluation.system_c_pipeline          # full ablation from the verdict cache
```
