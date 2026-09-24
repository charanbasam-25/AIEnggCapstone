Source-Verified UPSC Prelims Polity MCQ Tutor

Technical Documentation

Project: Source-Verified UPSC Prelims Polity MCQ Tutor
Domain: UPSC Civil Services Preliminary Examination — Indian Polity
Primary Architecture: RAG + Independent Verification + LangGraph Orchestration
Status: Working end-to-end prototype

1. Executive Summary

The Source-Verified UPSC Prelims Polity MCQ Tutor is an AI-based learning system for Indian Polity preparation for the UPSC Civil Services Preliminary Examination.

The project addresses a specific reliability problem in LLM-generated educational content: a language model can produce a plausible-looking multiple-choice question while introducing factual errors, incorrect answer keys, unsupported claims, or ambiguous distractors.

The central design principle is:

Generation and verification are separate responsibilities.

The system has two primary modes:

PYQ Tutor Mode

Explains official UPSC previous-year questions.

Retrieves evidence from the Constitution of India and NCERT Polity.

Uses official UPSC answer keys as ground truth.

Verified Practice Mode

Generates a new Indian Polity MCQ.

Extracts substantive factual claims.

Independently retrieves evidence.

Verifies individual claims.

Independently verifies the answer key.

Audits question quality.

Accepts, revises, or rejects the candidate.

The main research question is:

Does an independent verification pipeline reduce factual and answer-key errors in LLM-generated UPSC Polity MCQs compared with a standard RAG-based generator?

The project compares two systems:

System A — Vanilla RAG

User
  ↓
Retrieve relevant evidence
  ↓
LLM
  ↓
Answer / MCQ

System B — RAG + Verification

User
  ↓
MCQ Generator
  ↓
Claim Extraction
  ↓
Independent Evidence Retrieval
  ↓
Fact Verification
  ↓
Answer-Key Verification
  ↓
Quality Audit
  ↓
Decision
  ├── ACCEPT
  ├── REVISE → Generate again
  └── REJECT

2. Problem Statement

Large language models can generate UPSC-style multiple-choice questions with minimal input. However, generation quality does not guarantee factual correctness.

A generated MCQ can fail because:

a factual statement is incorrect;

a constitutional provision is misrepresented;

the declared answer is incorrect;

more than one option is correct;

no option is correct;

distractors are implausible;

wording introduces ambiguity;

the model relies on knowledge outside the trusted corpus.

For an educational tutor, these failures are important because an incorrect question can appear authoritative to a learner.

The project therefore treats every generated MCQ as an untrusted candidate until it passes independent checks.

Scope

The system is intentionally restricted to:

UPSC Civil Services Preliminary Examination;

Indian Polity;

static Polity;

Constitution of India;

NCERT Polity;

official UPSC previous-year questions;

official UPSC answer keys.

Current affairs are excluded from the project scope.

3. Objectives

3.1 Source-grounded generation

Generate UPSC-style Indian Polity MCQs within a controlled knowledge domain.

3.2 Evidence-based verification

Independently verify substantive factual claims against retrieved source evidence.

3.3 Independent answer verification

Do not trust the answer supplied by the question generator. Independently determine which option is supported by the evidence.

3.4 Quality verification

Detect ambiguity, multiple possible answers, poor distractors, inappropriate wording, and topic mismatch.

3.5 Failure-aware revision

When a candidate fails verification, pass the failure reasons back to the generator and attempt a revised candidate.

3.6 Quantitative evaluation

Compare a standard RAG baseline with the verification pipeline using measurable metrics.

4. Knowledge Base

The project uses a controlled, static knowledge base.

Knowledge Base
│
├── Constitution of India
├── NCERT Polity
├── UPSC Previous-Year Questions
└── UPSC Official Answer Keys

The Constitution and NCERT documents form the primary retrieval corpus for factual verification.

UPSC previous-year questions and official answer keys are used for tutoring and evaluation.

The system does not depend on unrestricted web search during the generation and verification flow. This keeps the experiment focused on a controlled RAG architecture.

5. Data Processing Pipeline

PDF
 ↓
Text Extraction
 ↓
Page Metadata Preservation
 ↓
Cleaning
 ↓
Page Documents
 ↓
Overlapping Chunks
 ↓
JSONL
 ↓
Retrieval Indexes

5.1 PDF extraction

PyMuPDF is used for PDF text extraction.

Page boundaries are preserved so that retrieved evidence can be traced back to the original document page.

Example:

SOURCE: constitution
DOCUMENT: Constitution of India
PAGE: 57

5.2 Chunking

Current configuration:

Chunk size: 1000
Chunk overlap: 150

The overlap reduces the chance that a relevant constitutional provision is split between unrelated chunks.

Current processed corpus:

Pages: 422
Chunks: 1149

Chunk metadata:

id
text
source
document
page
chunk_index

6. Retrieval Experiments

Several retrieval approaches were implemented and evaluated before selecting the final retrieval architecture.

The current retrieval benchmark contains 16 manually defined constitutional queries with expected Constitution pages.

Retrieval approach

Recall@5

BM25

75.00%

Hybrid BM25 + Semantic + RRF

81.25%

Semantic Retrieval

87.50%

Parent-Child + Reranker

87.50%

Semantic + Cross-Encoder Reranker

93.75%

These numbers describe performance on the project's 16-query benchmark and should not be interpreted as a general benchmark of Constitution retrieval.

7. BM25 Retrieval

BM25 provides lexical retrieval.

Query
 ↓
Tokenization
 ↓
BM25 scoring
 ↓
Top-K chunks

Measured result:

Recall@5: 75.00%

BM25 was retained as an experimental baseline and as a useful comparison against semantic retrieval.

8. Semantic Retrieval

Semantic retrieval uses:

BAAI/bge-small-en-v1.5

The query and document chunks are embedded and compared using normalized vector similarity.

Query
 ↓
Embedding
 ↓
Vector similarity
 ↓
Top-K chunks

Measured result:

Recall@5: 87.50%

9. Hybrid Retrieval

The hybrid approach combined BM25, semantic retrieval, and Reciprocal Rank Fusion (RRF).

                 ┌── BM25 ──────────┐
Query ───────────┤                   ├── RRF ── Top 5
                 └── Semantic ──────┘

Measured result:

Recall@5: 81.25%

On the current benchmark, this did not outperform semantic retrieval alone.

10. Semantic Retrieval + Cross-Encoder Reranking

The final retrieval architecture uses two stages.

Stage 1 — Candidate retrieval

BGE-small retrieves the top 20 candidate chunks.

Stage 2 — Cross-encoder reranking

Candidates are reranked using:

cross-encoder/ms-marco-MiniLM-L-6-v2

Query
 ↓
BGE-small
 ↓
Top 20 candidates
 ↓
Cross-Encoder
 ↓
Top 5
 ↓
LLM context

Measured result:

Recall@5: 93.75%

This is the current primary retrieval configuration.

11. Parent-Child Retrieval Experiment

A page-based parent-child architecture was also tested.

Page
 ├── Chunk 1
 ├── Chunk 2
 └── Chunk 3

The system retrieves child chunks and reconstructs/reranks their parent pages.

Measured result:

Recall@5: 87.50%

Since it did not improve over the selected semantic + cross-encoder configuration, it was not chosen as the primary retrieval mechanism.

This experiment is retained as part of the project's architectural decision record.

12. Final Retrieval Architecture

                    Query
                      │
                      ▼
              BGE-small Embedding
                      │
                      ▼
                 Top 20 chunks
                      │
                      ▼
          Cross-Encoder Reranking
                      │
                      ▼
                  Top 5 chunks
                      │
                      ▼
                LLM Context

The selection was based on the current retrieval benchmark rather than adding retrieval components purely for complexity.

13. Vanilla RAG Baseline — System A

The baseline represents a straightforward source-grounded RAG system.

User
 ↓
Retriever
 ↓
Relevant Constitution / NCERT chunks
 ↓
LLM
 ↓
Generated response

The baseline does not independently verify the generated output.

Its purpose is to establish a comparison point for the verification system.

14. Verification Pipeline — System B

System B adds independent validation after generation.

                 MCQ Generator
                       │
                       ▼
                Claim Extractor
                       │
                       ▼
           Independent Retrieval
                       │
                       ▼
                Fact Verifier
                       │
                       ▼
            Answer-Key Verifier
                       │
                       ▼
                Quality Auditor
                       │
                       ▼
                  Decision
              ┌─────┼─────┐
              ▼     ▼     ▼
           ACCEPT REVISE REJECT
                    │
                    └──────► Generator

The key principle is:

The generator creates a candidate; the verification pipeline decides whether the candidate is trustworthy enough to present.

15. MCQ Generator

The MCQ generator uses a structured Pydantic output:

question
option_a
option_b
option_c
option_d
correct_answer
explanation

The answer field is constrained to:

A | B | C | D

The generator is instructed to:

Generate exactly four options.

Produce one intended correct answer.

Stay within Indian Polity.

Avoid current affairs.

Avoid ambiguous wording.

Avoid multiple reasonably correct answers.

Avoid unsupported factual claims.

Provide the correct answer.

Provide a short explanation.

Treat the generated question as subject to independent verification.

The generator does not have authority to declare its own output verified.

16. Claim Extraction

The generated MCQ is passed to a separate claim extraction stage.

MCQ
 ↓
Claim Extractor
 ↓
Claim 1
Claim 2
Claim 3
...

The extractor identifies substantive Indian Polity claims.

Example:

Claim 1:
The Right to Education is a Fundamental Right under Article 21A.

Claim 2:
Article 21A mandates free and compulsory education for
children aged 6 to 14 years.

Each claim receives a unique identifier.

17. Independent Claim Retrieval

Each claim is independently passed to the retrieval layer.

Generated MCQ
     │
     ▼
Claim
     │
     ▼
Independent Retrieval
     │
     ▼
Evidence

The verifier does not simply trust the explanation produced by the generator.

Evidence is retrieved again from the controlled corpus.

This separates:

what the generator said;

what retrieval found;

what the verifier concluded.

18. Fact Verification

The Fact Verifier receives:

Claim
+
Retrieved Evidence

It produces:

SUPPORTED
CONTRADICTED
INSUFFICIENT

SUPPORTED

The evidence explicitly establishes the claim.

CONTRADICTED

The evidence explicitly conflicts with the claim.

INSUFFICIENT

The evidence is related but does not establish the claim strongly enough.

The verifier is instructed not to fill missing information using background knowledge.

For example, if a claim concerns the historical constitutional status of the Right to Property but the retrieved evidence does not establish the specific constitutional article relationship, the verifier should return INSUFFICIENT rather than infer the missing fact.

This conservative distinction is important for measuring both false acceptance and false rejection.

19. Answer-Key Verification

The answer-key verifier independently analyzes the options.

The LLM first identifies which options are supported by the supplied evidence.

Example:

Supported options:
[A]

Deterministic Python logic then checks:

Is at least one option supported?

Is exactly one option supported?

Does the supported option match the declared answer?

Possible results:

VALID
INVALID
INSUFFICIENT

Example:

Supported options = [A]
Declared answer   = A
Exactly one       = True

→ VALID

If multiple options are supported:

Supported options = [A, C]

→ INVALID

If no option can be supported:

Supported options = []

→ INSUFFICIENT

20. MCQ Quality Auditor

A separate quality auditor evaluates:

Unambiguous
Single best answer
Plausible distractors
Appropriate wording
Topic relevance

The result is:

PASS

or:

FAIL

with issue descriptions.

This is separate from factual verification because a question can be factually correct but poorly designed as an MCQ.

21. Decision Engine

The decision node combines verification results.

If:
    all factual claims are SUPPORTED
    AND answer verification is VALID
    AND quality audit is PASS

→ ACCEPT

If a verification stage fails and attempts remain:

→ REVISE

If the candidate still fails after the retry limit:

→ REJECT

22. Revision Loop

The workflow is failure-aware.

Attempt 1
   ↓
Claim verification
   ↓
Failure
   ↓
Failure reason
   ↓
Generate revised MCQ
   ↓
Attempt 2
   ↓
Verification

The actual failure reason is passed back to the generator.

This is different from simply generating another random question.

23. LangGraph Orchestration

LangGraph is used because the system contains explicit state, sequential processing, conditional branching, retry logic, and termination conditions.

START
  │
  ▼
Generate MCQ
  │
  ▼
Extract Claims
  │
  ▼
Verify Claims
  │
  ▼
Verify Answer
  │
  ▼
Quality Audit
  │
  ▼
Decision
  │
  ├──────────────► ACCEPT ──► END
  │
  ├──────────────► REJECT ──► END
  │
  └──────────────► REVISE
                       │
                       ▼
                 Generate MCQ

Workflow state:

topic
difficulty
mcq
claims
claim_evidence
fact_verifications
answer_evidence
answer_verification
quality_audit
decision
failure_reasons
retry_count
max_retries

24. Structured Outputs with Pydantic

Pydantic defines typed outputs for generation and verification.

Examples include:

MCQ

question
option_a
option_b
option_c
option_d
correct_answer
explanation

Fact Verification

verdict
reasoning
supporting_pages

Answer Verification

declared_answer
supported_options
exactly_one_correct
verdict
reasoning
supporting_pages

Quality Audit

unambiguous
single_best_answer
plausible_distractors
appropriate_wording
topic_relevant
issues
overall_quality

Structured outputs make downstream deterministic processing easier.

25. End-to-End Successful Run

The latest successful end-to-end run generated:

Which of the following Fundamental Rights is specifically aimed at guaranteeing educational rights to children?

Options:

A. Right to Education
B. Right to Equality
C. Right to Freedom of Speech
D. Right to Constitutional Remedies

Declared answer:

A

Fact Verification

Claim 1:

The Right to Education is a Fundamental Right under Article 21A of the Indian Constitution.

Result:

SUPPORTED

Supporting page:

14

Claim 2:

Article 21A mandates free and compulsory education for children aged 6 to 14 years.

Result:

SUPPORTED

Supporting page:

42

Answer-Key Verification

Declared answer: A
Supported options: [A]
Exactly one correct: True
Verdict: VALID

Quality Audit

Unambiguous: True
Single best answer: True
Plausible distractors: True
Appropriate wording: True
Topic relevant: True
Overall quality: PASS

Final Decision

Decision: ACCEPT
Attempts: 1

This demonstrates a complete successful execution of System B.

26. Failure Analysis

Failure analysis is part of the development process.

Earlier workflow executions produced candidates that were rejected because one or more claims could not be sufficiently verified.

One example involved a generated claim concerning the historical constitutional status of the Right to Property.

The retrieved evidence was not sufficient to establish the specific claim.

The verifier therefore returned:

INSUFFICIENT

The workflow subsequently rejected the candidate after the retry limit.

This revealed an important distinction:

Claim is false

is different from:

Claim cannot be established from the retrieved evidence

The three-way verdict was therefore retained:

SUPPORTED
CONTRADICTED
INSUFFICIENT

27. Architecture Design Principles

The project intentionally does not make every operation an autonomous agent.

LLM responsibilities

MCQ generation;

claim extraction;

factual verification;

option analysis;

quality auditing.

Deterministic/tool responsibilities

PDF processing;

chunking;

embeddings;

retrieval;

reranking;

state management;

retry counters;

final decision logic;

schema validation.

This separation keeps deterministic operations deterministic and makes the workflow easier to inspect.

28. Evaluation Methodology

The final evaluation compares two systems.

System A

Retrieval → Generation

System B

Retrieval
   ↓
Generation
   ↓
Independent Verification
   ↓
Quality Audit
   ↓
Decision / Revision

The primary hypothesis is:

An independent verification stage should reduce factual and answer-key errors in generated MCQs, while potentially increasing latency and false rejection due to retrieval insufficiency.

The evaluation therefore measures both reliability and verification cost.

29. Retrieval Metrics

Recall@5

For each benchmark query:

Hit = expected page appears in top 5 results

The current selected semantic + cross-encoder reranker achieved:

15 / 16
= 93.75% Recall@5

The benchmark currently contains 16 queries.

30. Generation and Verification Metrics

The final evaluation should measure:

Answer Accuracy

Compare system answers with official UPSC answer keys.

Evidence Support Rate

Supported claims
---------------- × 100
Evaluated claims

Verification Error Detection Rate

Measure how often the verification pipeline detects incorrect candidates.

False Acceptance Rate

Incorrect candidates accepted
----------------------------- × 100
Incorrect candidates evaluated

False Rejection Rate

Correct candidates rejected
--------------------------- × 100
Correct candidates evaluated

Answer-Key Validity

Measure whether exactly one option is supported and whether it matches the declared answer.

Quality Pass Rate

Percentage of generated MCQs that pass the quality audit.

31. LLM-as-a-Judge Evaluation

A separate evaluator can assess:

factual correctness;

evidence faithfulness;

clarity;

distractor quality;

ambiguity;

single-answer validity.

The judge should be treated as an evaluator rather than as the source of ground truth.

Where official UPSC answer keys exist, they remain the primary ground truth for answer-key evaluation.

32. Current Implementation Status

✓ Project structure
✓ Python environment
✓ Source ingestion
✓ PDF text extraction
✓ Page metadata preservation
✓ Document preprocessing
✓ Chunking
✓ JSONL corpus
✓ BM25 retrieval
✓ Semantic retrieval
✓ Hybrid retrieval
✓ Cross-encoder reranking
✓ Parent-child retrieval experiment
✓ Retrieval benchmark
✓ Vanilla RAG baseline
✓ MCQ generation
✓ Claim extraction
✓ Independent claim retrieval
✓ Fact verification
✓ Answer-key verification
✓ MCQ quality auditing
✓ LangGraph state
✓ LangGraph orchestration
✓ Revision loop
✓ ACCEPT / REVISE / REJECT logic
✓ Successful end-to-end ACCEPT run

The major remaining engineering phase is system-level comparative evaluation.

33. Known Limitations

33.1 Retrieval benchmark size

The current retrieval benchmark contains 16 queries. The reported retrieval numbers describe this benchmark only.

33.2 Static corpus

Current affairs are intentionally excluded.

33.3 Retrieval dependency

A correct claim may receive INSUFFICIENT if the retrieval layer fails to retrieve the necessary evidence.

33.4 LLM variability

Generation and verification use probabilistic models, so individual runs can differ.

33.5 Latency

System B performs multiple retrieval and LLM operations and therefore has greater latency than a single-pass RAG baseline.

33.6 Verification cost

The verification pipeline requires additional model calls compared with System A.

34. Performance Considerations

The current prototype repeatedly initializes retrieval resources in multiple workflow nodes.

This can create unnecessary overhead because embeddings, chunks, and the cross-encoder may be loaded or initialized repeatedly.

A production-oriented implementation should initialize shared retrieval resources once per process or workflow and reuse them.

Conceptually:

Application Startup
       │
       ├── Load chunks
       ├── Build / load embeddings
       ├── Load BM25
       └── Load cross-encoder
              │
              ▼
        Reuse across nodes

This optimization does not change the verification architecture.

35. Future Improvements

Potential improvements include:

Build a larger retrieval benchmark.

Add a larger official PYQ evaluation set.

Evaluate System A and System B on the same candidate set.

Measure false acceptance and false rejection.

Measure end-to-end latency.

Measure LLM calls per accepted MCQ.

Improve supporting-page attribution in answer-key verification.

Cache retrieval resources across workflow nodes.

Add deterministic checks for obvious problematic wording.

Add a user-facing UI after the evaluation pipeline is stable.

These should be prioritized based on evaluation needs rather than added purely for architectural complexity.

36. Final Architecture

                         ┌───────────────────────┐
                         │      User / UI        │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │   Tutor Orchestrator  │
                         └───────────┬───────────┘
                                     │
                       ┌─────────────┴─────────────┐
                       │                           │
                       ▼                           ▼
                PYQ Tutor Mode              Practice Mode
                       │                           │
                       │                           ▼
                       │                    MCQ Generator
                       │                           │
                       │                           ▼
                       │                    Claim Extractor
                       │                           │
                       │                           ▼
                       │                Independent Retrieval
                       │                           │
                       │                           ▼
                       │                     Fact Verifier
                       │                           │
                       │                           ▼
                       │                  Answer-Key Verifier
                       │                           │
                       │                           ▼
                       │                     Quality Auditor
                       │                           │
                       │                           ▼
                       │                       Decision
                       │                    ┌────┼────┐
                       │                    ▼    ▼    ▼
                       │                 ACCEPT REVISE REJECT
                       │                         │
                       │                         └──► Generator
                       │
                       ▼
                 Retrieval Layer
                       │
              ┌────────┴────────┐
              ▼                 ▼
        Constitution          NCERT
              │                 │
              └────────┬────────┘
                       ▼
                BGE-small
                       │
                    Top 20
                       │
                       ▼
              Cross-Encoder
                 Reranking
                       │
                    Top 5
                       │
                       ▼
                  Evidence

37. Conclusion

The Source-Verified UPSC Prelims Polity MCQ Tutor demonstrates a source-grounded approach to reliable educational generation.

Instead of treating an LLM-generated MCQ as trusted output, the system treats it as a candidate that must pass independent checks.

The verification pipeline combines:

Generation
    +
Claim Extraction
    +
Independent Retrieval
    +
Fact Verification
    +
Answer-Key Verification
    +
Quality Auditing
    +
Decision / Revision

The retrieval experiments provide an evidence-based architectural selection process. On the current 16-query benchmark, semantic retrieval followed by cross-encoder reranking achieved 93.75% Recall@5, compared with 75.00% for BM25.

The latest end-to-end run successfully produced an MCQ that passed:

Fact Verification      → SUPPORTED
Answer Verification    → VALID
Quality Audit          → PASS
Final Decision         → ACCEPT
Attempts               → 1

The central remaining experiment is the comparison between the vanilla RAG baseline and the verification pipeline.

That experiment will determine whether the additional verification complexity actually improves the reliability of generated UPSC Polity MCQs.