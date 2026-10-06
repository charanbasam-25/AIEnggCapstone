# Verifier reliability: observed errors and acceptance checks

The original 33-question development run answered four questions: **three
correct, one wrong, and 29 abstentions**. Its precision among answered questions
was 75%; its overall accuracy was 3/33. The original result is preserved in
[development_original_2026_10_02.json](../data/evaluation/benchmark/development_original_2026_10_02.json).

## The wrong case: UPSC 2019, booklet A, question 81

The question asks which of these statements is correct:

1. No High Court has jurisdiction to declare a central law constitutionally invalid.
2. A constitutional amendment cannot be called into question by the Supreme Court.

| Item | Original verifier judgment | Effect |
|---|---|---|
| Statement I | CONTRADICTED | Considered false by the option mapper |
| Statement II | SUPPORTED | Considered true by the option mapper |
| Selected answer | B: II only | Wrong; the recorded official key is D: neither |

Python mapped the supplied verdicts consistently. The failure was in the
evidence context and the model's interpretation of that evidence.

## What went wrong

The main supporting passage for statement II was **chunk 733, PDF page 260**.
It contains the wording of Article 368(4) preventing amendments from being
questioned in court. The model treated that wording as an operative rule.

**Chunk 734 on the same page** contains the qualification: the section inserting
that provision was declared invalid by the Supreme Court in *Minerva Mills*.
The first chunk ended before that footnote, and the next chunk was outside the
top-five evidence set. The relevant qualification was already in the corpus.

Other retrieved passages also described an amendment struck down by the Supreme
Court. The original model still supported the universal no-review statement.
This exposes a counterevidence interpretation failure as well as the split
footnote problem.

The statement I result had another defect: it cited **page 1**, while its
retrieved pages were 16, 149, 136, 135 and 87. Its explanation described the
old restriction as support for the claim, then labelled that claim contradicted.
The verdict happened to match the official answer, but its explanation and
citation were unreliable.

## Implemented evidence checks

1. **Restore complete retrieved pages.** After ranking chunks, verification
   rejoins the chunks from the same source, document and page. Footnotes travel
   with the provision. This preserves the existing semantic top-20 and
   cross-encoder top-five retrieval ranking. The baseline continues using its
   original retrieved chunks.
2. **Require exact source quotations.** Every model-based SUPPORTED or
   CONTRADICTED claim judgment must cite passage IDs and quotations. Python checks
   that each ID exists and each quotation occurs in the corresponding checked text.
   Missing or invalid citations make the judgment INSUFFICIENT.
3. **Derive pages in Python.** The claim model returns passage IDs. Python obtains
   page numbers from the source metadata; the model cannot invent page numbers.
4. **Check legal status and counterevidence.** The shorter verification prompt
   explicitly covers omitted or invalidated wording, attached qualifications,
   negation, matching scope and counterexamples. Unclear cases require abstention.
5. **Record the checked context.** Traces retain original retrieval results and
   the expanded pages used by verification. The tutor shows validated quotations
   and lets the user inspect complete source context.
6. **Reject failed claim construction.** Claims marked non-propositional or
   containing introduced words that failed the source-word check become
   INSUFFICIENT before a model call. A list item such as “Preamble” must not be
   judged true merely because that word exists in the corpus. These are lexical
   construction checks, not a proof that every accepted claim is well formed.
7. **Require a blind second evidence review.** A valid SUPPORTED or CONTRADICTED
   model judgment is reviewed in a separate call using only the claim and source
   context. The reviewer receives neither the first verdict nor the official
   answer. Python validates its quotations and requires the same verdict.
   Disagreement or insufficient evidence leaves the claim INSUFFICIENT. A
   missing parsed response is a service error, never an accepted judgment.
8. **Keep the student answer policy strict.** The tutor resolves numbered
   questions with strict mapping and withholds an answer when required claims
   remain unresolved. Experimental mapping policies remain available for
   explicit research use in the CLI; the student UI uses the strict policy.
9. **Abstain on unsupported paired-statement formats.** Statement-I/II and
   Assertion/Reason labels require individual truth judgments and, when asked,
   a separate explanatory-relationship judgment. The current numbered parser
   does not provide that path. Python recognizes these labels, including common
   punctuation, inline labels and Unicode Roman numerals, and returns an
   explicit abstention before option retrieval or model calls. All four such
   questions remain in the 75-question benchmark and its scoring denominator.

The direct-MCQ path also restores page context and applies the legal-status
instruction before its option-fit and quotation checks. A candidate needs one
SUPPORTED option and three RULED_OUT alternatives, each with validated source
quotations. A blind second option analysis must satisfy those checks again and
select the same answer. Otherwise the direct path abstains. The generation
workflow uses the strengthened fact verifier through its existing gates.

The first and second calls use the same configured model, `gpt-4o-mini`.
“Independent review” means the first judgment is withheld from the reviewer;
the model's errors can still be correlated. Recognized literal absence claims
use the existing deterministic whole-corpus scan rather than two model calls.

Implemented in [evidence_context.py](../src/verification/evidence_context.py),
[fact_verifier.py](../src/verification/fact_verifier.py) and
[direct_mcq_verifier.py](../src/verification/direct_mcq_verifier.py), with the
format guard in [question_type.py](../src/verification/question_type.py).

## Blind review still accepted a wrong answer

The frozen `quoted-evidence-review-v2` test run completed all 42 questions with
no service errors: **zero correct, one wrong and 41 abstentions**. Its sole
accepted answer was `upsc-2023-a-q085`, an assertion/reason question:

- Statement I describes Constitution Day on 26 November.
- Statement II claims the Drafting Committee was set up on 26 November 1949.
- Both model calls selected B, meaning both statements were correct.
- The matched official key is C: Statement I correct, Statement II incorrect.

Both calls quoted a passage about the Constitution being developed over almost
three years and the Preamble's adoption date. Those quotations exist in the
corpus, but adoption on that date does not establish the date on which the
Drafting Committee was formed. The exact-date event was not verified.

The generic direct-option path compared the four coded answers without a
statement-level verification path for these labels. The second call repeated
the same date/event confusion. This is an observed counterexample to treating
valid quotations plus agreement as proof of correctness.

The failed report remains unchanged in
[test_reliability_v2.json](../data/evaluation/benchmark/test_reliability_v2.json).
The current `quoted-evidence-review-v3` guard abstains on the unsupported
format; it does not supply the official answer C. The same rule covers the
other three assertion/reason questions, without checking IDs or gold answers.

**Evaluation consequence:** this test failure has now informed implementation.
The original split IDs remain frozen, but subsequent runs are regression
checks. A fresh untouched question set is needed for an independent accuracy
estimate. The current runner records this scope in the report configuration,
and Evaluation Studio displays it.

## Correct letters do not establish faithful reasoning

Some previously correct answers also lacked evidence establishing the answer.
For development question `upsc-2019-a-q047`, the model connected a dated amendment
to a Prime Minister's tenure without a quotation establishing that link. For
`upsc-2019-a-q052`, it used Parliament's Article 368 power as support for an
option about the Supreme Court's Article 142 power. Both cases abstain in the
v2 reviewed run. Their earlier correct answer letters are not a reason to retain
unsupported reasoning.

The v3 rerun accepted `upsc-2019-a-q047` with the correct official answer A.
Both calls quoted the First Amendment's 1951 date, but that quotation still
does not establish the named Prime Ministers' tenures. The change from an
abstention to a correct letter demonstrates model variation, not a proof that
this evidence gap was resolved. Its explanation also needs manual evidence-fit
review.

The reviewed run's accepted development question, `upsc-2020-a-q011`, selects
the correct official answer B about parliamentary responsibility. Its quotes
for ruling out option A concern separation of powers, legislation and
collective responsibility; they do not explicitly settle the option's claim
that all political parties are represented in government. This still needs a
manual evidence-fit audit. The run does not establish 100% faithfulness, even
for its correct answer. Valid quotations and agreement are acceptance checks,
not a proof of entailment.

## Measuring the change

The intermediate runs remain separate and retain their actual policies, code
hashes, dataset/corpus hashes, model settings and full prediction traces.

| Stage | Split | Correct | Wrong | Abstained | Coverage |
|---|---|---:|---:|---:|---:|
| Original, 2 October | Development — 33 | 3 | 1 | 29 | 12.1% |
| Exact quotation checks, v1 | Development — 33 | 1 | 0 | 32 | 3.0% |
| Blind review, v2 | Development — 33 | 1 | 0 | 32 | 3.0% |
| Original, 2 October | Test — 42 | 1 | 1 | 40 | 4.8% |
| Blind review, v2 | Test — 42 | 0 | 1 | 41 | 2.4% |
| Format guard, v3 regression | Development — 33 | 2 | 0 | 31 | 6.1% |
| Format guard, v3 regression | Test — 42 | 0 | 0 | 42 | 0.0% |

All these runs completed with zero service errors. Original reports are
preserved in
[development_original_2026_10_02.json](../data/evaluation/benchmark/development_original_2026_10_02.json)
and [test_original_2026_10_02.json](../data/evaluation/benchmark/test_original_2026_10_02.json).
The first quotation-check run is
[development_evidence_checks_v1.json](../data/evaluation/benchmark/development_evidence_checks_v1.json),
and the reviewed development run is
[development_reliability_v2.json](../data/evaluation/benchmark/development_reliability_v2.json).

The final v3 source-verifier reruns completed without service errors and use
identical code, corpus, dataset and model settings. They run the verifier only;
the v2 comparison contains the separate vanilla-RAG measurements. There are
**two answered questions out of 75**. Test precision is undefined because no
test answer was attempted. This does not establish broad reliability or that
the abstentions were all necessary.

Full v3 snapshots are
[development_reliability_v3.json](../data/evaluation/benchmark/development_reliability_v3.json)
and [test_reliability_v3.json](../data/evaluation/benchmark/test_reliability_v3.json).
The dashboard's default development/test JSON files are byte-identical copies
of these snapshots. Earlier and failed runs remain unchanged.

```bash
.venv/bin/python -u -m src.evaluation.evaluate_benchmark --run \
  --systems verified --split development \
  --output data/evaluation/benchmark/development_reliability_v3.json
.venv/bin/python -u -m src.evaluation.evaluate_benchmark --run \
  --systems verified --split test \
  --output data/evaluation/benchmark/test_reliability_v3.json
```

Assess wrong answers together with correct answers, coverage and abstentions.
A zero-error observed run supports a claim about that run and its attempted
subset. It cannot establish error-free performance on arbitrary future questions.

Exact quotation checks verify provenance. Entailment, legal applicability and
complete coverage of a claim still require interpretation. Blind agreement
adds an acceptance check; it does not prove either interpretation correct.
Structured response schemas also leave room for factual mistakes, as described in the
[official OpenAI documentation](https://developers.openai.com/api/docs/guides/structured-outputs).

The offline suite currently passes **132 tests**, including split-page context,
fabricated quotations and passage IDs, failed claim construction, conflicting
review verdicts, invalid review quotations, missing parsed responses, blind
review inputs, unsupported paired formats, and the UI/CLI answering paths. These tests validate the
acceptance rules. They do not establish accuracy on future exam questions.
