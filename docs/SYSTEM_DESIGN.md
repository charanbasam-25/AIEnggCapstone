# System design

This is the canonical technical description of the current static Polity practice implementation. Historical verifier experiments remain available in EVALUATION.md; they are not the product contract.

![Current architecture](diagrams/practice_architecture.png)

Editable sources and exportable images are listed in ARCHITECTURE_DIAGRAMS.md.

## Components

| Layer | Implementation | Responsibility |
|---|---|---|
| UI | app.py, src/ui | Practice, My Practice and read-only How it works views |
| Source preparation | src/ingestion | Extract PDF pages, clean text and create JSONL chunks |
| Retrieval | src/retrieval, src/verification/article_context.py | Semantic candidates, reranking, page restoration and named Article context |
| Generation | src/generation/mcq_generator.py | Structured candidate MCQ from an approved source packet |
| Orchestration | src/orchestration/graph.py and nodes.py | Fixed LangGraph stages, revisions and gate decision |
| Verification | src/verification | Claim parsing, fact checks, direct option checks and quotation binding |
| Persistence | src/practice/store.py | Accepted questions and JSON run reports in SQLite |
| Evaluation | src/evaluation, evals_app.py | Offline checks, benchmarks and report views |

## Source and retrieval contract

PyMuPDF extracts the Constitution and selected NCERT PDF pages. Chunks are 1,000 characters with 150 characters of overlap and carry source, document, page and chunk index. The active corpus is 1,149 chunks.

Each query first uses BAAI/bge-small-en-v1.5 to choose 20 candidates. The cross-encoder ms-marco-MiniLM-L-6-v2 reranks those candidates and the consumer receives five. PageEvidenceContext restores complete pages. ArticleEvidenceContext can add operative provisions, continuation pages and footnotes for named Articles in the approved Constitution. Added pages are retained separately from ranked results.

![Retrieval flow](diagrams/retrieval_flow.png)

Generation, statement verification and direct verification make their own retrieval calls. A retrieval hit is evidence to inspect, not a truth label.

## Verification contract

![Verification flow](diagrams/verification_flow.png)

The statement path parses the question, binds predicates into complete claims, retrieves per claim, validates quotation references, and requires an agreeing blind review. Python then maps resolved truth values to the coded option. The direct path retrieves the stem and each option, assesses A-D, validates citations and requires one supported option, three ruled-out options and an agreeing blind review.

![Literal absence flow](diagrams/absence_flow.png)

Recognised quoted-term absence claims are scanned across every loaded chunk. A match contradicts the absence claim. No match supports the literal claim only under the stated corpus-completeness and extraction assumptions. Conceptual absence is not handled by string matching.

## Generation contract

![Generation flow](diagrams/generation_flow.png)

The graph is retrieve_sources -> generate_mcq -> extract_claims -> verify_claims -> verify_answer_key -> audit_quality -> explain_question -> decide. REVISE returns to generation; ACCEPT and REJECT terminate. One candidate has at most two revisions. The service rechecks the terminal decision before persistence.

## Storage contract

![Data model](diagrams/data_model.png)

SQLite contains questions and runs. The questions table stores content, source hash, policy, gates and the complete validated payload. The runs table stores request metadata, stage measurements, model usage, retrieval traces and question records. Learner answers and scores are session memory, not database records. Dataset answer keys are grading inputs only.

## Evaluation contract

![Evaluation flow](diagrams/evaluation_flow.png)

The runner passes only question text and options to answering systems. Python grades predictions after the run. Reports record dataset, corpus, code, model, split and retrieval metadata. The offline check validates frozen inputs and report consistency without model calls. Generated-MCQ acceptance is a workflow measure; it is not expert accuracy.

## Failure handling

Missing sources, invalid structured output, unsupported formats, unresolved evidence, invalid quotations, disagreement, quality failure and explanation failure prevent publication. API timeout or connection errors are recorded separately from deliberate abstention. A source hash change retires the active set. Older policy records remain stored but are ineligible for current reuse.

## Runbook

    .venv/bin/python -m streamlit run app.py
    .venv/bin/python docs/diagrams/render_diagrams.py
    .venv/bin/python -m pytest
    .venv/bin/python -m src.evaluation.evaluate_benchmark --check
    .venv/bin/python -m src.evaluation.run_all --check

The active model is gpt-4o-mini with structured Responses calls. The local client has a 45-second timeout and one SDK retry. Cost estimates use recorded tokens and dated rates in src/practice/costs.py; missing usage is reported as incomplete rather than priced as zero.
