# Design Doc — Source-Verified UPSC Polity MCQ Verifier
**Author:** Charan Kumar Basam · **Cohort:** AI Engineering · **Date:** 2026-09-27
**Companion document:** [`EVALUATION.md`](EVALUATION.md) — full measurement
record, every arm, every negative result.
**Structure.** §1–§4 are the design doc proper and are sized to the spec's 1–2
pages. §5–§8 are the evaluation and decision summary the rubric asks for;
`EVALUATION.md` holds the full measurement record behind them.
A note on sequence, since the spec asks for a design doc *before*
implementation: this project did not run that way. The build came first and
this document was written against the system that exists. That ordering is
itself one of the findings in §7 — it is exactly the "architecture-first"
failure the spec warns about, and I paid for it in ways that are recorded
rather than smoothed over.
---
## 1. Problem
UPSC Prelims Polity questions are mostly multi-statement: a stem, two to four
numbered statements, and options that name statement subsets ("I and III
only") or counts ("Only two", "None"). A student's real failure mode is not
"I don't know the answer" but "I can't tell which *statement* I got wrong."
An LLM tutor makes this worse, because it produces a fluent paragraph that
names Articles it never checked.
**The narrow problem I chose:** given one such question, decide the truth of
each numbered statement *independently*, against the text of the Constitution,
with a page citation — and refuse to answer when the corpus does not settle it.
This is deliberately not "answer UPSC questions." The system is allowed to
score worse than a guesser on accuracy, as long as it is right about what it
claims and silent about the rest. Stating the objective that way is what made
the evaluation in §5 measurable.
**Non-goals:** current affairs, subjects other than Polity, tutoring dialogue,
and any question whose answer is not decidable from the primary text.
## 2. Data surface
| Source | Pages | Chunks |
|---|---|---|
| Constitution of India (full text) | 402 | 1,107 |
| NCERT Polity | 20 | 42 |
| **Total** | **422** | **1,149** |
Page-aware PDF extraction → 1000-character chunks, 150 overlap, each carrying
`{source, document, page, chunk_index}`. Page provenance is not decoration:
every verdict the system emits cites the pages it used, and §7's most
important finding was found by auditing those citations.
Evaluation set: **UPSC 2025 Prelims Polity, Q54–Q66 (13 questions, 37
numbered statements)**, held out from all prompt development.
## 3. Architecture
**RAG with verification as a separate responsibility from generation.** The
system never asks one model to both answer and check. Concretely:
```
question ─► deterministic parser ─► one propositional claim per statement
                                              │
                          ┌───────────────────┴──────────────────┐
                          ▼                                      ▼
              per-claim retrieval (k=5)              absence-claim scanner
              semantic + cross-encoder                 (full-corpus, no LLM)
                          │                                      │
                          ▼                                      │
              fact verifier → SUPPORTED /                        │
              CONTRADICTED / INSUFFICIENT ◄─────────────────────┘
                          │
                          ▼
              deterministic option mapping (policy: strict)
                          │
                          ▼
              option letter  or  abstain + reason
```
Orchestration (`src/orchestration/`) is a LangGraph `StateGraph` with an
ACCEPT / REVISE / REJECT loop, used when the system *generates* a question:
generate → extract claims → verify claims → verify answer key → audit quality
→ decide, looping back to generation with the failure reasons on REVISE.
Three design choices carry the weight:
1. **Claims are built deterministically, not by an LLM.** A numbered statement
   is usually a noun phrase ("Police") whose predicate lives in the stem. A
   template substitution binds them ("Police is included in the Seventh
   Schedule…"), guarded by two checks: no token may appear that was not in the
   source question, and split pairs must rejoin. See §5 for why this matters
   more than its accuracy contribution suggests.
2. **Verdicts are three-way, and mapping is deterministic.** The LLM never
   picks an option. It only judges one claim against one evidence set; Python
   turns verdicts into a letter. This is what makes abstention a *policy*
   rather than a mood.
3. **Absence claims bypass retrieval entirely.** "The Constitution does not
   mention 'political party'" quantifies over the whole corpus, so ranking is
   the wrong instrument; a full scan of all 1,149 chunks answers it exactly.
## 4. Core flow
1. Parse the question into lead-in, numbered items, closing, and options.
2. Bind each item into a standalone proposition.
3. For each claim: retrieve top-5 chunks (semantic candidates → cross-encoder
   rerank), or run the full-corpus scan if it is an absence claim.
4. Verify each claim against *only* its own evidence. Return a verdict,
   reasoning, and supporting pages.
5. Aggregate per statement (any CONTRADICTED → CONTRADICTED; all SUPPORTED →
   SUPPORTED; else INSUFFICIENT).
6. Map to an option under the strict policy; abstain with a typed reason if any
   statement is unresolved, if no option matches, or if several do.
## 5. Comparative evaluation
**Retrieval** (16-query gold benchmark, Recall@5): BM25 75.00% · semantic
(bge-small-en-v1.5) 87.50% · hybrid RRF 81.25% · parent-child + reranker
87.50% · **semantic + cross-encoder reranker 93.75%** ← selected. Hybrid RRF
is the honest negative result: the more sophisticated fusion lost to plain
semantic retrieval.
**How much of this is measurable, stated up front.** The verifier is not
deterministic. `FactVerifier` never set `temperature`, so it ran at the
Responses API default of 1.0 while the vanilla-RAG baseline was pinned at 0 — a
sampling system measured against a fixed one. All call sites are now pinned, and
`src/evaluation/verdict_stability.py` re-verified all 37 claims five times to
find out what that was worth: **unpinned, 9 of 37 claims disagreed with
themselves; pinned, still 2** (one of them returns all three verdicts across
five identical calls). Recombining the independent per-claim draws into five
independent pipeline runs gives an **accuracy spread of 15.38 points — two
questions — in both conditions.**
So the resolvable difference on this dataset is two questions. That threshold
governs everything below, and `EVALUATION.md` §0.1 has the full table.
**End-to-end**, 13 questions, audited labels, strict policy, all arms on the
shipped retriever. Each C arm changes exactly one variable against a named
comparison arm. The baseline row is computed from the same label function the
arms use — it had been hard-coded at the stored-label figure (38.46%) while
every arm was reported under both, so it was only comparable to half the table.
Vanilla RAG answered `D` on Q58, so the label audit *raises* it to 46.15%.
| Arm | change | coverage | precision when answered | accuracy | **wrong** |
|---|---|---|---|---|---|
| A — vanilla RAG | retrieve → generate, always answers | 100% | 46.15% | **46.15%** | **53.85%** |
| B — shipped (bound claims, k=5, strict) | verify per statement, map deterministically | 30.77% | **100%** | 30.77% | **0%** |
| C0 | stored claims, k=3 | 30.77% | 75.00% | 23.08% | 7.69% |
| C1 | + binding (vs C0) | 38.46% | 80.00% | 30.77% | 7.69% |
| C1d | k=3→5 only (vs C0) | 38.46% | 60.00% | 23.08% | 15.38% |
| C2 | binding at k=5 (vs C1d) | 38.46% | 60.00% | 23.08% | 15.38% |
| C3 | + numeric rules (vs C2) | 38.46% | 60.00% | 23.08% | 15.38% |
| C3d | numeric rules, no binding (vs C1d) | 38.46% | 60.00% | 23.08% | 15.38% |
**Every C-to-C difference in that table is one question or zero, which is inside
the noise band. Read the six C arms as six null results.** The earlier version of
this document drew three conclusions from this table; all three were artifacts of
an earlier BM25 index and are retracted in `EVALUATION.md` §4.2–§4.5, including
the claim that retrieval depth 3→5 was the project's largest accuracy effect.
Under audited labels that change *lowers* precision, and the whole of it runs
through one claim — the same claim the stability harness shows emitting all three
verdicts.
**A third inference path was built and rejected on measurement.** Arm B2 asks
the LLM to verify a whole *option* instead of one statement — the obvious
simpler design, since it skips claim construction and mapping entirely. It
answered **2 of 13** (both correct, 100% precision, zero errors) because an
option is a conjunction of two to four statements and almost never clears the
threshold as a unit. The variant given *more* evidence did worse: widening the
context to the union of question-level and per-statement passages **halved
coverage, to 1 of 13**. More evidence for a conjunction means more surface on
which some part of it looks unestablished. That result is the empirical case for
decomposing the conjunction, verifying the parts separately, and letting Python
recombine them — which is what the shipped architecture does, and it reaches
30.77% coverage on the same evidence B2 gets 15.38% on.
**LLM-as-a-judge** (gpt-4o, blind slot-randomised, reasoning quality only —
never correctness), A vs B: factual correctness 2.77 → 3.69, evidence
faithfulness 2.46 → 4.69, reasoning validity 2.69 → 4.54, epistemic honesty
2.38 → **5.00**; head-to-head **12–1–0** for B. Hallucinated claims **19 → 1**;
questions containing one **12 → 1**. Position-bias control: preference by slot
was 5/8/0, much flatter than the by-system 1/12/0, so B won from both slots.
Discount most of this. A judge asked to reward calibrated hedging will reward a
system that hedges by construction — B scores a **perfect 5.00** on epistemic
honesty, and it is preferred on **eight questions where it answered nothing at
all**. The single question where A wins (Q59) is one where A answered correctly
and B abstained. The number that is not circular is the hallucination count,
because a hallucinated claim is identified extractively — a named Article, date
or provision absent from the cited evidence — not by the judge's taste. It is
also a claim-level count rather than a 13-question accuracy, which puts it on the
right side of the noise threshold. **19 → 1 across 12 → 1 questions** is the
strongest result in the project.
**What this table does and does not establish.** On accuracy, **vanilla RAG beats
every verified arm under both label sets** — 46.15% audited against B's 30.77%
and the best C arm's 30.77%. I do not claim an accuracy win and never had one.
The finding that survives is the last column: vanilla RAG states a false answer
on **7 of 13** questions (8 of 13 under stored labels); the shipped verified
system on **0 of 13** under both. That is a ~46–54 point error-rate gap, roughly
six questions, and the only end-to-end result in this project comfortably outside
the two-question noise band. It is bought with coverage — B answers 4 of 13
instead of all 13, and gives up one to two questions of accuracy. If the
objective is "be right more often," this architecture is not worth its cost. If
the objective is "do not assert false constitutional facts," it is, and that was
the stated objective in §1 before any of this was measured.
One more thing the table cannot show: **2 of the 13 questions (Q55, Q66) are
not answerable from this corpus at all**, established by a lexical probe over
all 1,149 chunks rather than a ranked search. The ceiling for a system that
never guesses is 11/13 = **84.62%**, not 100%, and every accuracy figure above
should be read against that.
**The decision, and what it actually rests on.** The end-to-end metrics cannot
distinguish bound claims (C3) from verbatim ones (C3d) — they are identical on
all four: 38.46 / 60.00 / 23.08 / 15.38. The claim-level metrics separate them
completely: verbatim claims are **70.27% propositional with 8 false SUPPORTED
verdicts**; bound claims are **100% propositional with 0**. A non-propositional
claim is a bare noun phrase with no truth value, and what the verifier does with
one is worse than abstaining — it matches the fragment lexically and returns
SUPPORTED, a confident wrong answer wearing a page citation.
So I ship bound claims on the 37-observation signal that needs no gold label,
rather than on a 13-question accuracy column that is inside its own measurement
error and depends on a label set with a confirmed defect (§7). Applied
consistently, that same rule is what forced the retractions above: the numeric
conflict rules (C2→C3) change **no verdict at all** on the shipped retriever, so
they are kept as harmless rather than claimed as justified.
## 6. Abstention as a measured property
Scoring an abstention as a wrong answer conflates "declined to commit" with
"asserted something false," which makes the headline number unable to express
what the architecture is for. Reported metrics are therefore coverage,
precision-when-answered, accuracy-overall, and **error-rate-overall**.
Three policies were implemented and compared over **identical verdicts**, so the
policy is isolated exactly and the comparison costs no API calls. STRICT and
ELIMINATION measure **identically** on this data — a null result: ELIMINATION is
a strict generalisation of STRICT and provably cannot answer fewer questions, but
on these 13 questions the extra information it exploits never narrows the
candidates to exactly one. Two-statement questions offer four options, so
resolving one statement halves the field and cannot finish the job.
CLOSED_WORLD reads INSUFFICIENT as false — the "absence of evidence is evidence
of absence" move. **It is the highest-accuracy configuration in the project:
53.85% audited, beating vanilla RAG on accuracy *and* error rate
simultaneously** (46.15% / 53.85%). It is not what I ship, because it converts
nine abstentions into eleven answers at roughly a coin-flip rate, taking the
error rate from **0% to 30.77%** — from zero false assertions to four. Buying 3
questions of accuracy with 4 false assertions is the wrong side of the objective
in §1. It is also *wrong on this data* in a way it cannot detect: the corpus
demonstrably does not cover 2 of 13 questions, so absence of evidence here is not
evidence of absence. Reported as the measured price of that assumption, never as
a result. (This is the one policy comparison large enough to clear the
two-question noise threshold.)
Abstention is also where the system's least flattering number lives. Each of the
**9** abstentions was classified by cause: **6 were pipeline defects** — evidence
present in the corpus that the system failed to use — against 2 genuine corpus
limits and 1 partial. **Correct-abstention share: 22.22%.** A high abstention
rate is only a virtue when the abstentions are justified, and most of mine were
undiagnosed failure wearing the costume of epistemic humility. This also
qualifies the judge result in §5 directly: the perfect 5.00 on epistemic honesty
was awarded for abstentions that are two-thirds bugs. Classifying them is what
turned "the system is appropriately cautious" into six specific things to fix.
## 7. Failure analysis
- **The measuring instrument was stochastic and the baseline was not.**
  `FactVerifier` never set `temperature`, so it sampled at the API default of
  1.0 while `vanilla_rag.py` was pinned at 0. Every System B number ever
  reported was one draw against a fixed reference. Found late, by chasing a
  single claim on which the ablation harness and the shipped pipeline disagreed.
  All five unpinned call sites are pinned now — and **pinning is not
  determinism**: 2 of 37 claims still disagree with themselves at temperature 0,
  and holding one claim's evidence fixed, **three characters of whitespace or a
  change of endpoint flips its verdict**. The lesson is not "pin the
  temperature"; it is that a 13-question harness will faithfully convert a
  boundary-case classifier into an "effect" with a plausible story attached, and
  that the noise floor has to be measured before any difference is reported.
- **A wrong gold label (Q58).** Five of six arms contradicted the stored
  answer; the corpus sided with the arms. The label is `A` ("I only"); both
  statements are false against the text, so the answer is `D`. This inverted
  the gradient — the label error made a *false* SUPPORTED look like the
  project's best result, and I nearly kept it. All accuracy is now reported
  under both stored and audited labels so the correction's size is visible.
- **Answer leakage in the answer-key verifier.** The prompt included the
  declared answer under a `DECLARED ANSWER` heading plus a rule telling the
  model to ignore it as evidence. That made the component's independence
  conditional on instruction-following. Removed; the comparison was already
  done in Python.
- **Three distinct verifier failure modes**, not one: *closed-list membership*
  (confirming a rule exists ≠ this subject falls under it — Q54), *cross-scope
  merging* (State-legislature provisions used against a Parliament claim), and
  *strength inflation* (evidence for "may give directions" accepted for "can
  take over total administration" — Q65 II).
- **Prompt accretion is not additive — and my write-up of it was wrong.** I had
  reported that appending two numeric-conflict rules fixed Q63 II and flipped an
  unrelated verdict "at temperature 0." Temperature was never set, the Q63 fix
  was an artifact of the earlier BM25 index (the rules now change *no* verdict),
  and the flipped claim flips for reasons unrelated to any rule. The surviving
  lesson is stronger than the original: at 26 rules a claim near the decision
  boundary is decided by whatever perturbs it last — a rule, a blank line, an
  endpoint, or a sample. The fix is decomposition, not rule 27. The retraction
  is recorded next to the prompt itself.
- **The evaluation harness was distorting its own measurement**: querying at
  k=3 while justified on Recall@5, carrying a second buggy copy of the mapping
  logic (a `negated` flag computed then discarded; "None" that could never
  match), scoring abstentions as wrong, and hard-coding the baseline at the
  stored-label figure. All fixed; mapping now has one implementation with an
  assertion per defect.
- **Fixing the depth instance would not have fixed the class.** The benchmark's
  `k` and the pipeline's `k` were unrelated literals in unrelated files —
  roughly twenty of them — agreeing only because each had been hand-edited to
  agree, so moving the benchmark to Recall@10 and missing one would reproduce the
  defect silently. Both now come from `src/retrieval/retrieval_config.py`: the
  invariant holds because there is one value, not because it is remembered.
  Re-running `evaluate_bm25` after the change returned 75.00%, unchanged, which
  is how I know the refactor moved no numbers.
- **Nothing enforced the order the evaluation modules run in**, and they read
  each other's output files. Running them out of order does not fail — it prints
  a document-ready table built from a previous run's inputs. It happened twice.
  `src/evaluation/run_all.py` now declares the dependency graph as data, runs the
  stages topologically, and asserts every report is at least as new as its
  inputs; `--check` verifies the whole chain without spending a token, and is the
  thing to run before quoting any number. It also asserts the one invariant file
  times cannot express: changing the retrieval depth invalidates every stored
  number *without modifying any file*, so `--check` reads the `claim_top_k`
  recorded in the results back off disk and fails if the code has moved past it.
- **The orchestrated pipeline was weaker than the thing being measured.** The
  LangGraph node constructed `FactVerifier()` with no corpus, which silently
  disabled the absence-claim scanner, so absence claims fell through to the LLM
  path that rules 13–17 exist to forbid. **The same one-line omission recurred
  weeks later in the new stability harness** — the module written specifically to
  audit the system — where it made the measured metric spread the spread of a
  system nobody ships.
- **A library's error classification is a hypothesis, not evidence.** The
  ablation ran on a stdlib BM25 index for most of the project because
  `sentence-transformers` "could not be installed" — `huggingface_hub` reported
  `LocalEntryNotFoundError: check your connection`. It was not a network problem:
  the connection is TLS-intercepted, and the corporate root is in the Windows
  certificate store but not in `certifi`. The fix is four lines
  (`truststore.inject_into_ssl()`, see `src/tls_trust.py`). Taking the error
  message at face value is what left the main table on the wrong retriever, which
  is what produced the three conclusions retracted in §5.
- **Pivot that produced the core idea:** the first design had one LLM read the
  question and produce a verdict. It agreed with itself. Splitting claim
  construction, retrieval, verification, and option mapping into separately
  measurable components is what made every finding above *findable*.
## 8. Limitations
13 questions and 37 claims — a single question is 7.7 accuracy points. Worse,
the *measurement error is larger than that*: resampling the verifier gives an
accuracy spread of two questions even at temperature 0, so no difference below
15.4 points is detectable at all. Two things clear that bar — the A-vs-B error
rate (~46–54 points) and the claim-soundness figures (37 observations, no label
dependency). Everything else in this document is reported as null or indicative.
The noise estimate is itself under-powered: 5 repeats per claim, so "9 unstable
unpinned, 2 pinned" are lower bounds, and the error runs in the unflattering
direction.
Retrieval is judged on 16 queries with page-level gold labels that are mine, and
those five benchmarks were not re-run in the final measurement pass. The
absence-checker's SUPPORTED verdict assumes the corpus covers the claim's scope
and that PDF extraction preserved the term; both are stated in the emitted
reasoning because neither is checkable from inside the system. Only labels that
some arm contradicted could be audited, so label quality is unverified for the
questions every arm abstained on — the audit is not a clean pass over the key,
it is the questions the system happened to argue about. And 6 of the 9
abstentions are pipeline defects rather than epistemics, so most of the system's
apparent caution is undiagnosed failure.
`EVALUATION.md` §8 lists these in full, and §9 orders what I would do next:
measure the noise band properly, then extend the dataset — because until the
dataset is larger than the measurement error, no further architectural work on
this problem is measurable.
