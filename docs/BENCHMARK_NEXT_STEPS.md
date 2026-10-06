# Next steps for static Polity practice

The agreed capstone objective is a useful static UPSC Polity practice assistant
with checked answers, source-supported explanations and measured quality.
Achieving a perfect score on the mixed 75-question benchmark is not an
acceptance criterion. Unresolved drafts still require revision or withholding.

## Scope

Focus on the 13 supported topics: Fundamental Rights, Directive Principles,
Fundamental Duties, Preamble and constitutional features, Parliament, President,
Supreme Court, Election Commission, federalism, local government, constitutional
amendment procedure, emergency provisions and citizenship.

Current-affairs announcements, current office-holders and scheme implementation
details are outside the initial scope. Historical amendments, landmark
interpretations and older Acts can concern static Polity; assess their syllabus
fit and source requirements individually. Static questions still need a
declared source version and applicable date.

The approved corpus currently contains the Constitution and a selected NCERT
Grade 7 chapter. Wider static coverage requires reviewed source material before
the system can safely rely on it. Missing evidence is not a reason to guess.

## Immediate work

Start with Fundamental Rights and Directive Principles. Review a small batch of
generated direct and statement questions against the actual cited passages.
Record the complete stem/options, the proposed key, source/page/version, whether
each explanation is correct, whether any other option is defensible, and whether
the item fits its requested topic and style. Passing model reviews is separate
from this independent evidence review.

Use the findings to fix incorrect or ambiguous delivered questions, unsupported
explanations and unnecessary withholding. Source misses, retrieval misses,
claim-construction failures, evidence-interpretation errors and answer-mapping
errors need different fixes. Expand source coverage for in-scope topics as the
review shows a need.

Prepare a supporting verification benchmark of about 40–50 reviewed static
questions across the supported topics and formats. Select cases by the agreed
scope before running the system. Include difficult statements, exceptions and
footnotes. Annotate the passages required to settle each case; a literal term
match alone does not establish the proposition. Record expected withholding
cases separately from ordinary answer-key scoring.

Freeze independent questions for evaluation after development. Cases already
used to debug the system remain regression cases. Keep official answer keys,
model predictions and diagnostic worksheets out of the retrieval corpus and
answering prompts. Answer keys are grading inputs after prediction.

## Evidence to present

| Evaluation task | What to report |
|---|---|
| Generated-question correctness | Independently reviewed key and option-explanation error counts/rates, with sample size and topic/style coverage |
| Question usefulness | Ambiguous options, weak distractors and topic/style mismatch found during review |
| Grounding | Whether the exact cited passages establish each complete proposition, including qualifications |
| Delivery | Accepted questions per requested set, withheld drafts, revisions and blocking reasons |
| Static verification | Correct, wrong, abstained and service-error counts; precision alongside coverage |
| Preparation | Elapsed time, model calls, recorded tokens and cost per accepted question, including rejected work |
| Software behaviour | Offline gate, quote-binding, reuse, UI and error-handling checks, separate from factual-accuracy claims |

The learner demo should show topic selection, practice, submission and reviewed
explanations, plus a case that is withheld when its evidence cannot settle an
answer. Project metrics remain in **How it works**, separate from the practice
flow.

The generated-question evaluation and the scoped static verification benchmark
are planned work, not completed measurements. Set numerical improvement targets
after recording their reviewed baselines.

## Historical diagnostics

The [75-question evidence sheet](../data/evaluation/evidence_review_75.csv)
remains an optional diagnostic record. It is not the primary static evaluation
set, and its human source annotations remain pending. It should not drive
corpus expansion into schemes outside the agreed scope.

The saved full mixed-set results contain 2 correct answers, 0 wrong answers and
73 abstentions. The 7 October first-five development trial selected one correct
answer and abstained on four cases. Inspection found unrelated quotations behind
its correct mission answer; the report records that grounding defect separately
from answer-key scoring.

A subsequent direct-citation screen rejected those recorded unrelated quotations
in an offline replay, and a fresh one-question live follow-up withheld the mission
answer with no service error. Term overlap remains a coarse rejection rule, not
proof of entailment. All 244 offline tests passed during that change. These
historical checks do not measure correctness of a new generated static question set.

Open **How it works → Evals → Reports** to inspect the saved experiments and
their scope and grounding notes.
