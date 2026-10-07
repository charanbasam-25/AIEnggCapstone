# Verification reliability: what the checks do and what they do not prove

The current practice workflow uses source retrieval, page restoration, exact quotation binding and blind review. These controls protect the publication boundary; they do not prove that a model interpreted constitutional text correctly.

The principal observed historical failure was an answer about Article 368. The first retrieved chunk contained operative-looking wording, while a qualification on the same page was outside the top-five context. The model quoted the first passage and supported an over-broad claim. This led to complete-page restoration, quotation validation and stronger counterevidence instructions.

The current controls are:

1. construct complete claims before fact verification;
2. retrieve separately for each claim or option;
3. restore full pages and named Article continuations;
4. require source-reference IDs and exact text binding;
5. run a blind second review and require verdict agreement;
6. map statement truth patterns in Python;
7. require one supported and three ruled-out options for direct questions;
8. reject unsupported paired Statement-I/II and Assertion/Reason formats;
9. keep unresolved evidence as INSUFFICIENT;
10. revalidate every publication gate before saving.

Literal absence claims use a separate exhaustive lexical scan. A found term contradicts the absence claim. No occurrence supports a literal term claim only for the loaded corpus and only if extraction preserved the text. A top-five miss is never treated as corpus-wide absence.

The direct citation screen rejects wholly unrelated quotations through a coarse term-overlap check. It is a rejection filter, not a semantic entailment proof. Same-family blind review is a useful second acceptance check, not independent human adjudication.

Historical v3 benchmark results are 2 correct, 0 wrong and 31 abstentions on development, and 0 correct, 0 wrong and 42 abstentions on test. The prior test error, where agreeing reviewers confused adoption of the Constitution with formation of the Drafting Committee, is preserved as a counterexample. Correct letters with unrelated evidence are not counted as faithful reasoning.

Use the 75-question reports and the legacy 13-question traces for debugging. Use an independently reviewed static generated-MCQ benchmark for claims about learner-facing question accuracy.
