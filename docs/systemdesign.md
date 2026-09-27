System Design — Source-Verified UPSC Polity MCQ Verifier

Author: Charan Kumar Basam
Cohort: AI Engineering
Date: 2026-09-27

1. Overview

The Source-Verified UPSC Polity MCQ Verifier is an AI system designed to answer UPSC Civil Services Preliminary Examination questions on Indian Polity using a fixed, trusted knowledge base.

The system uses:

Constitution of India

NCERT Polity

Official UPSC Prelims questions and answer keys

The main goal is reliable, source-grounded answers rather than fluent but unsupported answers.

The system is designed around one principle:

The LLM verifies claims, but it does not decide the final answer.

Python-based deterministic logic handles question parsing, claim construction, answer mapping, and abstention.

2. Scope

In scope

UPSC Civil Services Preliminary Examination questions

Indian Polity

Multi-statement questions

Verification of individual statements

Source citations

Answer selection or abstention

Out of scope

Current affairs

Subjects other than Polity

Open-ended tutoring conversations

Questions whose answers cannot be established from the available sources

3. System Modes

The project contains two modes.

Mode 1 — Verify an Existing PYQ

This is the measured mode.

The system:

Takes an existing UPSC question.

Splits it into individual statements.

Converts each statement into a complete claim.

Retrieves relevant evidence.

Verifies each claim.

Combines the claim-level results.

Maps the results to an answer option.

Abstains when the available evidence is insufficient.

The evaluation contains 13 questions and 37 statements. fileciteturn0file0L433-L441

Mode 2 — Generate and Verify an MCQ

The system can also:

Generate an MCQ.

Extract its claims.

Verify the claims.

Verify the proposed answer key.

Check question quality.

Accept, revise, or reject the question.

This mode is implemented but was not measured in the final evaluation. fileciteturn0file0L45-L57

4. High-Level Architecture

                    UPSC Question
                         |
                         v
                +------------------+
                | Question Parser  |
                +------------------+
                         |
                         v
                +------------------+
                | Claim Builder    |
                +------------------+
                         |
              +----------+----------+
              |                     |
              v                     v
      Normal Claim          Absence Claim
              |                     |
              v                     v
       Retrieve Evidence      Full Corpus Scan
              |                     |
              +----------+----------+
                         |
                         v
                +------------------+
                | Fact Verifier    |
                |      (LLM)       |
                +------------------+
                         |
                         v
                +------------------+
                | Aggregate Claims |
                +------------------+
                         |
                         v
                +------------------+
                | Answer Mapping   |
                +------------------+
                         |
                 +-------+-------+
                 |               |
              Answer          Abstain

The important separation is that the LLM evaluates evidence, while deterministic Python logic performs the final decision. fileciteturn0file0L79-L95

5. Data and Knowledge Base

The knowledge base contains 422 pages and 1,149 chunks:

Source

Pages

Chunks

Constitution of India

402

1,107

NCERT Polity

20

42

Total

422

1,149

Each chunk stores:

Text

Source

Document

Page number

Chunk index

Page information is retained so that every verification result can point back to the evidence used. fileciteturn0file0L391-L429

6. Retrieval

The production retriever uses a two-stage process:

Semantic retrieval using bge-small-en-v1.5

Cross-encoder reranking using ms-marco-MiniLM-L-6-v2

The benchmark reported 93.75% Recall@5 for this configuration on 16 retrieval queries. fileciteturn0file0L267-L287

The production pipeline retrieves the top 5 pieces of evidence for a claim.

A key design finding was that increasing retrieval depth from 3 to 5 improved:

Coverage: 38.46% → 53.85%

Accuracy: 30.77% → 38.46%

with the claims and prompt held constant. fileciteturn0file0L571-L597

7. Claim Construction

UPSC questions often contain statements that are not complete sentences.

For example:

Statement: Police

Stem: Which of the following subjects are included in the Seventh Schedule?

The system converts the statement into a complete proposition before verification.

This step is deterministic rather than LLM-based because an LLM may paraphrase the statement and unintentionally create a different claim.

The measured result was:

Propositional claims: 70.27% → 100%

False SUPPORTED verdicts: 7–8 → 0

across the 37 evaluated claims. fileciteturn0file0L299-L327

8. Claim Verification

Each claim receives one of three verdicts:

SUPPORTED — the evidence supports the claim.

CONTRADICTED — the evidence contradicts the claim.

INSUFFICIENT — the available evidence does not establish the claim.

The LLM receives the claim and retrieved evidence and returns:

Verdict

Reasoning

Supporting page numbers

The output is structured so that malformed LLM responses are rejected instead of silently passing through the system. fileciteturn0file0L471-L519

9. Handling Absence Claims

Some questions ask whether something does not exist or is not mentioned in the Constitution.

A normal top-k retrieval approach cannot reliably prove absence because the relevant information may not appear in the retrieved chunks.

For these claims, the system performs a full scan of all 1,149 chunks instead of relying on retrieval.

This check is deterministic and does not use the LLM. fileciteturn0file0L343-L363

10. From Statements to a Final Answer

After every statement is verified, the system aggregates the results.

The rules are:

Any statement is CONTRADICTED
        → overall statement = CONTRADICTED

All evidence supports the statement
        → overall statement = SUPPORTED

Otherwise
        → overall statement = INSUFFICIENT

The resulting statement verdicts are then mapped to the available answer options.

The production policy is strict:

If any required statement remains insufficient, the system abstains instead of guessing.

The system also supports closed-world and elimination policies for evaluation. fileciteturn0file0L601-L635

11. Why Abstention Matters

The system does not treat every question as answerable.

If the available sources cannot establish the answer, it can return:

ABSTAIN

with a typed reason such as:

Statement could not be established

No answer option matches

Multiple options remain possible

This is important because forcing an answer can turn uncertainty into a confidently wrong answer.

12. Baseline Comparison

The project compares the verified system against a simple RAG baseline.

System A — Vanilla RAG

Question
   ↓
Retrieve evidence
   ↓
LLM answers directly

System B — Verified Pipeline

Question
   ↓
Split into claims
   ↓
Retrieve evidence for each claim
   ↓
Verify each claim
   ↓
Apply deterministic answer mapping
   ↓
Answer or abstain

The evaluated results were:

System

Coverage

Precision when answered

Overall Accuracy

Error Rate

Vanilla RAG

100%

46.15%

46.15%

53.85%

Verified System

30.77%

100%

30.77%

0%

The verified system therefore traded coverage for a much lower observed error rate on this evaluation set. fileciteturn0file0L639-L679

13. Evaluation Metrics

The system is evaluated using four metrics:

Coverage

Percentage of questions for which the system provides an answer rather than abstaining.

Precision when answered

Accuracy among only the questions that the system answered.

Overall accuracy

Correct answers divided by all questions.

Overall error rate

Confidently incorrect answers divided by all questions.

These metrics are reported together because optimizing only one metric can produce misleading results. fileciteturn0file0L639-L663

14. Reproducibility

LLM outputs are not perfectly deterministic.

The system therefore pins model temperature to zero for normal evaluation calls.

A stability experiment found:

Configuration

Stable claims

Unstable claims

Unpinned

28 / 37

9

Pinned

35 / 37

2

These results were measured over five repeats, so the instability counts are treated as lower bounds. fileciteturn0file0L683-L737

Because of this variability, small differences between configurations are not automatically treated as meaningful improvements.

15. Generation and Verification Workflow

The second system mode uses a verification loop:

Generate MCQ
     ↓
Extract claims
     ↓
Verify claims
     ↓
Verify answer key
     ↓
Audit question quality
     ↓
    Decide
   /      ACCEPT   REVISE
            |
            v
       Generate again

If retries are exhausted
            ↓
          REJECT

The verification process can identify failure reasons and feed them back into the next generation attempt. fileciteturn0file0L213-L247

This workflow is implemented but is currently an unmeasured capability.

16. Reliability and Failure Defenses

The main failure modes and corresponding defenses are:

Failure

Defense

LLM invents unsupported facts

Evidence and page citations are required

Statement is paraphrased incorrectly

Deterministic claim construction

Absence is inferred from top-k retrieval

Full-corpus absence check

Retrieval depth is too small

Production retrieval depth aligned with benchmark

Stale evaluation reports

Evaluation pipeline checks input/output freshness

LLM instability

Repeated runs and noise-band analysis

Temperature changes

Temperature explicitly controlled

Accuracy encourages unsafe guessing

Coverage, precision, accuracy and error rate reported together

fileciteturn0file0L837-L859

17. Current Limitations

The current system has several important limitations:

Generation mode is not evaluated.

The evaluation set contains only 13 questions.

Five repeats are not enough to precisely estimate model instability.

Retrieval benchmark numbers were not re-run during the final evaluation pass.

6 of the 9 abstentions appear to be caused by pipeline defects rather than genuine unanswerability.

The fact-verifier prompt has become large and should eventually be simplified.

The 2026 UPSC dataset is available but is not currently used in the evaluation.

These limitations mean that the current results should be interpreted as a measured evaluation of the existing 13-question benchmark, not as a general claim about all UPSC Polity questions. fileciteturn0file0L863-L905

18. Key Design Decisions

The most important design decisions are:

1. Verify claims instead of answering the whole question directly

This makes the reasoning process inspectable.

2. Keep claim construction deterministic

This prevents the LLM from changing the meaning of the original question.

3. Separate retrieval from verification

Retrieval finds evidence; the LLM judges whether that evidence supports the claim.

4. Handle absence claims separately

Absence requires searching the whole corpus rather than trusting a small retrieved set.

5. Keep final answer mapping outside the LLM

The LLM produces claim verdicts. Python determines the final option.

6. Allow abstention

The system should not manufacture an answer when the evidence is insufficient.

7. Evaluate the complete behavior

Coverage, precision, accuracy, and error rate are reported together.

19. Summary

The Source-Verified UPSC Polity MCQ Verifier is a claim-level RAG verification system rather than a conventional question-answering RAG system.

Its core pipeline is:

UPSC Question
     ↓
Parse
     ↓
Build Claims
     ↓
Retrieve Evidence
     ↓
Verify Claims
     ↓
Aggregate Verdicts
     ↓
Map to Answer
     ↓
Answer / Abstain

The central architectural idea is:

Use retrieval to provide evidence, use the LLM to verify individual claims, and use deterministic logic to make the final decision.

The final evaluation showed a clear reduction in observed confident errors compared with vanilla RAG, while also exposing a major coverage limitation and several areas that still require measurement and improvement. fileciteturn0file0L665-L679