# UPSC Practice: project brief

Current local implementation, reviewed 7 October 2026.

This capstone is an evidence-first static Polity practice assistant: choose a topic,
receive checked UPSC-style MCQs, attempt them, and study explanations for every
option with source quotations. Automated checks reduce errors; they do not
guarantee a perfectly correct question bank.

## 1. Problem statement

UPSC aspirants need topic-wise practice with answers and explanations they can
trust and inspect. A generated MCQ can contain an incorrect key, multiple
defensible answers, weak distractors or unsupported explanations. Memorising
these errors can teach the wrong concept.

The project addresses this by retrieving evidence before generation and
requiring separate answer, quality and explanation checks before publication.
When evidence cannot resolve a draft, the system revises it within a limit or
withholds it. It can return fewer questions than requested.

The intended user is a student practising Polity. The intended outcome is useful
practice with correct keys, clear alternatives and source-supported option
explanations. Passing the workflow's internal gates is an operational signal;
independent expert labels are needed to measure whether that outcome is achieved.

## 2. Scope and task contract

| Area | Current scope | Boundary |
|---|---|---|
| Subject | Static Polity, with 13 curated topics | Other subjects are locked; current-affairs and scheme implementation questions are outside the initial scope |
| Sources | Constitution of India and a selected NCERT Grade 7 chapter | Limited coverage; no live web search or complete current-affairs library |
| Input | Topic, direct/statement/mixed style, requested level and count | UI requests 1, 5 or 10; service contract allows 1–10 |
| Difficulty | Foundation, Standard, Challenging | Prompt settings, not calibrated learner difficulty |
| Output | Up to the requested count of accepted four-option MCQs | A partial or empty set is a valid preparation outcome |
| Explanation | Reviewed summary and one cited explanation for each of A–D | The generator's draft explanation is replaced before publication |
| Learning flow | Answer first; reveal key and explanations after Submit | No separate learner-facing question-verification screen |
| History | Latest 20 completed sets in session memory | No durable learner account, adaptive learning or cross-device progress |
| Deployment | Local Streamlit app and SQLite store | No production identity isolation, queue or distributed serving |

The agreed objective is useful static practice with source-supported answers
and explanations. Evaluate delivered-question correctness, ambiguity, grounding
and topic/style fit alongside delivery yield, latency and cost. The existing
mixed 75-question set is a broader regression diagnostic; answering every item
correctly is not a capstone acceptance criterion. A scoped static verification
benchmark and independently reviewed generated-question evaluation remain to
be curated. The publication checks still require unresolved drafts to be withheld.

Example task: `Fundamental Rights / Direct questions / Standard / 1`.
The output is one accepted MCQ with a reviewed key and all option notes, or no
accepted question if its evidence checks cannot be satisfied. A generated
question's proposed answer is untrusted until the publication checks pass.

## 3. Handling PII and data

PII is information that can identify a person, such as a name, email, phone
number or identity number. The learner UI does not ask for these fields, take
free-text questions, or accept private uploads. It limits input to curated study
settings and answer choices. This is data minimisation, not an automatic PII
detection system.

| Data | Where it goes |
|---|---|
| Topic/style/level/count | Streamlit session state and saved run settings; topic/style/level guide generation |
| Learner answers, scores and history | Streamlit **server memory**, scoped to the browser session; not browser storage, SQLite learner records or model prompts |
| Accepted MCQs, notes and citations | Shared local `data/practice/practice.sqlite3` question bank |
| Actual retrieval queries, source excerpts, rankings and review outcomes | SQLite run reports; selected source pages and feedback also enter model prompts |
| Model, tokens, cached-input counts, service tier, timings and error class | Run telemetry; SDK prompt bodies and provider error messages are not logged |
| API key | Process environment / local `.env`; used for authentication, never inserted into prompts or run reports |
| Evaluation cases and reports | Local benchmark/report files, separate from learner history |

Public documents can contain personal names. Retrieval diagnostics intentionally
retain source excerpts, so the app does not claim that every stored text is
PII-free. Diagnostics are shared by the app and are not separated by user.

**Gaps:** no general PII scanner/redactor, login or role-based access, application
database encryption, automatic retention/deletion, or account-separated reports.
Before accepting personal text or public hosting, implement checks before both
LLM calls and persistence, minimise required fields, isolate users/diagnostics,
and define retention/deletion controls. Verify these boundaries with synthetic
PII. These are future requirements, not existing protections or a compliance
guarantee. Provider retention is a separate policy question.

Implementation: [practice UI](../src/ui/practice.py),
[store](../src/practice/store.py), [telemetry](../src/orchestration/telemetry.py).

## 4. Guardrails and publication

All five gates are mandatory. Python validates structured outputs and source
bindings; model reviewers assess meaning and quality. A positive model label
alone cannot release a question.

| Gate | Required result | Example failure |
|---|---|---|
| Sources | Source evidence available with page provenance | No useful source material |
| Format | Supported structure and valid alternatives; exact-stem duplicates blocked | Invalid option structure or repeated stem |
| Answer | One resolved key, with checked evidence and agreeing review | Unresolved statement, unsupported key or unresolved direct alternative |
| Quality | All required flags pass: clarity, unique answer, distractors, wording and topic fit | Ambiguous wording or multiple defensible answers |
| Explanations | Reviewed summary and all A–D notes, with valid source references and supporting evidence | A correct key paired with an incorrect or uncited option explanation |

For statement MCQs, `CONTRADICTED` can be a resolved false statement in a valid
question. `INSUFFICIENT` means the truth pattern remains unresolved and blocks
release. For direct MCQs, the verifier needs one `SUPPORTED` option and three
`RULED_OUT` alternatives; lack of evidence is not proof that an option is wrong.

Python binds numbered quotations to original source text and rejects unknown
references. Separate reviews assess whether the actual evidence supports each
judgment or explanation. Quote provenance is not itself semantic entailment.

Additional controls include bounded requests, source/policy-bound bank reuse,
content-derived IDs, validation on record loading, source checks before
publication/display/submission, and answer withholding before submission.
Prompts distinguish source data from instructions, but hardened prompt-injection
evaluation and a general PII filter are not implemented.

Implementation: [graph nodes](../src/orchestration/nodes.py),
[publication models](../src/practice/models.py),
[service boundary](../src/practice/service.py).

## 5. System design and architecture

![Current practice architecture](diagrams/practice_architecture.png)

1. Extract public PDFs into page-aware chunks, retaining document/page metadata.
   The processed corpus file is hashed to bind questions to a snapshot.
2. Collect a bounded learner request. Reuse an eligible checked question when
   topic, style, level, source hash and policy match.
3. For new questions, semantic retrieval finds 20 candidates, a cross-encoder
   selects the top five, and the evidence packet restores full pages and footnotes.
   Named Articles in the focus or constructed claims also locate their provision
   pages in the approved Constitution; source location does not determine truth.
4. A generator proposes a structured MCQ. Python checks its format and exact
   stem duplication.
5. Statement questions retrieve/review their claims, bind selected quotation IDs
   to exact source words and map resolved truth patterns to an answer. Direct questions retrieve from the stem and options,
   compare A–D and require an agreeing blind review.
6. A quality auditor checks the question. Fresh summary/option notes are written,
   source-bound and reviewed individually against their evidence.
7. Python rechecks all five gates: accept, revise up to twice, or reject. Only
   accepted records enter the SQLite bank and learner quiz.
8. Submission runs Python scoring and reveals checked notes/sources. Session
   history supports a mistake notebook. Saved diagnostics feed the technical UI.

LangGraph orchestrates **fixed, bounded model roles**: generator, fact/direct
verifiers, blind reviewer, quality auditor and explanation writer/reviewer.
These are workflow steps, not autonomous web-browsing agents. Runtime practice
calls use `gpt-4o-mini`; agreement between reviewers from the same family is not
independent human adjudication.

Architecture image source: [local renderer](diagrams/render_practice_architecture.py).
For the detailed state/publication contracts, see [Practice design](PRACTICE_DESIGN.md).
The earlier `SYSTEM_DESIGN.md` and its original diagram describe historical
verification experiments; they do not replace this current practice architecture.

## 6. Tradeoffs

| Choice | Benefit | Limitation |
|---|---|---|
| Small approved corpus | Traceability and manageable capstone scope | Lower coverage/freshness; a matching hash does not establish legal currency |
| Strict gates / withholding | Keeps unresolved drafts out of practice | Lower yield and sometimes an empty set; model errors can still pass |
| Semantic top 20 / reranked top 5 | Focused evidence and a bounded candidate window | Relevant passages can be missed |
| Full-page restoration | Preserves qualifications, exceptions and footnotes | More model input tokens and latency |
| Separate model reviews | Can catch unsupported claims and disagreements | Added cost; correlated model mistakes remain possible |
| Checked bank reuse | Avoids new searches/calls for eligible records | Changes invalidate reuse; exact-stem deduplication misses paraphrases |
| Small model / local Streamlit + SQLite | Affordable, inspectable capstone implementation | Model-quality limits and missing production serving/isolation controls |

## 7. Evaluation tasks and metrics

The custom Python harness evaluates separate tasks. Dataset size and unit must
be stated alongside every metric.

| Task | Unit and reference | What the measurement means |
|---|---|---|
| Answer existing PYQs | 75 official questions: development 33, test 42; official key hidden from answering | Predicted letter versus saved gold key; correct/wrong/abstained/errors |
| Retrieve sources | 16 annotated queries with expected page labels | Any expected page in the top five; the legacy `Recall@5` is a query hit rate |
| Grounding / faithfulness | Earlier 13-question regression; citation checks and model-judge rubric | Provenance coverage and judge ratings, not human factual ground truth |
| Generated workflow behavior | Per topic/candidate, bound to source and policy | Gate acceptance, revisions, rejection and blocking reasons; current complete format comparison not run |
| Verdict stability | 37 claims from the earlier 13 questions, five repetitions | Repeated agreement or changes, not correctness |
| Software boundaries | Synthetic pytest cases and offline models | Correct handling of gates, quotations, errors, stale sources and UI state |
| Preparation performance | Per saved run with actual traces | Elapsed seconds, calls, tokens and estimated model cost |

For `N` questions, `C` correct answers and `W` wrong answers:

- Precision when answered = `C / (C + W)`.
- Coverage = `(C + W) / N`.
- Overall accuracy = `C / N`.
- Zero answers means undefined precision, not 100%.
- Run errors are separate from deliberate abstentions and prevent treating an
  incomplete report as a complete benchmark.

Saved benchmark views validate dataset/source hashes, split, question IDs and
metric totals. The 42-question test split has informed debugging and is now a
regression set, not an untouched holdout. Gold evidence/answerability annotations
and manual adjudication remain incomplete. The earlier grounding, judge and
stability runs are not 75-question measurements.

**Generation accuracy gap:** the product does not yet have an independently
expert-labeled, frozen generated-MCQ benchmark. Internal gate acceptance must
not be advertised as student-facing accuracy. A release evaluation should label
key correctness, ambiguity, every option note, evidence support and topic/style
fit, include exceptions and insufficient evidence, and freeze a fresh holdout.
Report expert error rates alongside yield; zero observed errors on a finite set
cannot guarantee zero future errors.

Open **How it works → Evals → Task specification** for the contract.
[Evaluation UI details](EVALS_UI.md) describe saved reports and reproduction.
Viewing these screens makes no new model calls.

## 8. Error handling

| Condition | Current recovery |
|---|---|
| Insufficient or conflicting evidence | Withhold the draft; bounded revision, then reject |
| Invalid output/quotation/publication record | Validation blocks release; reject and continue other slots where possible |
| Model API / timeout / connection failure | Stop the batch; save the error class and return already accepted items when the source still matches |
| Missing key | Eligible bank reuse can work; new generation stops |
| Source change | Retire the whole in-flight/active set and exclude stale bank items |
| Corrupt bank record | Validate on load and skip it |
| Unexpected UI failure | Class-only session diagnostic and generic retry message; durable report may be absent |

Each slot permits one initial draft plus at most two revisions. Explanation
writing permits one local repair. The model client has a 45-second timeout and
one SDK transport retry, independent of question revisions. The graph also has
a recursion limit. These bounds do not create a fixed total run deadline.

If three out of five requested questions pass, the student can use those three.
If the corpus changes before publication, the whole set is retired. Rejected
work still consumes time/tokens. Provider error bodies, credentials and partial
drafts are not rendered to the learner.

## 9. Cost

The UI estimates standard-tier model cost from per-call recorded tokens. It
supports only `gpt-4o-mini` and `gpt-4o-mini-2024-07-18` at the following rates,
verified 7 October 2026: input **$0.15**, cached input **$0.075**, output **$0.60**
per million text tokens. [Official model pricing](https://developers.openai.com/api/docs/models/gpt-4o-mini).

```text
Estimated USD = ((input - cached input) * 0.15
                 + cached input * 0.075
                 + output * 0.60) / 1,000,000
```

Add this over recorded calls. Old traces without cache details use the uncached
rate and disclose that assumption; absent tier metadata assumes standard
pricing. Missing usage, unsupported model/tier or inconsistent call counts leave
the full estimate unavailable. A known subtotal is labeled partial. Bank-only
reuse has zero **new model** cost, not zero infrastructure cost.

Cost per delivered question divides the complete estimate by accepted count,
including revised/rejected work. It is undefined when none are delivered.
Estimates use the stated pricing date rather than reconstructing historical
invoices. Infrastructure, taxes, account-specific pricing and unreported retry
usage are excluded. The API invoice remains the billing reference.

Implementation: [dated estimator](../src/practice/costs.py).

## 10. Latency

The service timer covers source checks, bank lookups, retrieval-model loading,
searches, model calls, validation and revisions after initial request/store setup.
It stops before saving the final run report. Study time and UI rendering are
outside this preparation measurement. Trace stage durations help explain where
time went; SDK-call counts do not count internal transport retries separately.

Cold model loading, new generation, multiple review calls and rejected drafts
increase latency. Checked-bank reuse avoids new retrieval/model calls. The UI
reports observed individual runs, not a promised response time. Repeated,
representative runs are needed for median/p95 latency, cold/warm comparisons and
a defensible service-level target.

## Where to see this in the app

Run `.venv/bin/python -m streamlit run app.py` from the project root and open
**How it works**. Its sections are **About this site**, **Privacy & PII**,
**Guardrails**, **System design**, **Evals**, **Error handling**, and
**Cost & latency**. The practice page stays focused on generating and attempting MCQs.
