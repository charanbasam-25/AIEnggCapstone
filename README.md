# UPSC Practice: source-grounded Polity MCQs

UPSC Practice is a local learning application for static Indian Polity. A learner selects a topic, question style, requested difficulty and set size, attempts a quiz, and sees the checked answer with an explanation for every option after submission. Polity is available in the current release; Geography, History, Economy and Environment are visible as future subjects.

The application treats an MCQ as an untrusted candidate until it passes source, format, answer, quality and explanation checks. It uses retrieval before generation, separate review steps, exact source-reference validation and deterministic Python publication rules. When evidence is incomplete or checks disagree, the candidate is revised within a fixed budget or withheld.

## Start the app

    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
    touch .env
    .venv/bin/python -m streamlit run app.py

Add OPENAI_API_KEY=... to the local .env file and never commit it. Open http://localhost:8501. A checked question can be reused without a new model call when its topic, format, requested difficulty, corpus hash and generation policy still match.

The sidebar has three learner-facing areas: Practice prepares and delivers a quiz; My Practice shows current-session results and missed questions; How it works explains sources, privacy, guardrails, architecture, evaluations, failure handling, cost and latency.

The app does not provide a web-search fallback. The active corpus is data/processed/chunks.jsonl: 1,107 Constitution chunks and 42 chunks from a selected NCERT Grade 7 chapter. Its current SHA-256 is 9fe7b77d0b9dffaac73769d81dd926e99b6b4f9fc01b3082d51ec5ad974810a7.

## Architecture

![Current practice architecture](docs/diagrams/practice_architecture.png)

The editable source for this image is [practice_architecture.mmd](docs/diagrams/practice_architecture.mmd). Rebuild the complete diagram set with:

    .venv/bin/python docs/diagrams/render_diagrams.py

The command creates PNG, SVG, PDF and Mermaid files for the architecture, learner flow, retrieval, generation, verification, absence handling, data model, evaluation and baseline flows. It is offline and does not call a model. system-design.png, system-design.svg, system-design.pdf and overall-system.mmd remain compatibility copies of the current architecture for existing links.

## Technology and design

The UI uses Streamlit. LangGraph coordinates a fixed workflow; it is not an autonomous web agent. Pydantic models define structured boundaries. PyMuPDF extracts page-aware PDF text. BAAI/bge-small-en-v1.5 retrieves 20 candidates through a normalized NumPy index, and cross-encoder/ms-marco-MiniLM-L-6-v2 reranks them to five. SQLite stores accepted questions and run reports. OpenAI Responses structured outputs power the generator and review roles; the configured runtime model is gpt-4o-mini.

Statement questions are parsed and bound into complete claims before verification. Each claim is assessed as SUPPORTED, CONTRADICTED or INSUFFICIENT; Python maps a fully resolved truth pattern to an option. Direct questions compare all options and require one supported option, three ruled-out alternatives and an agreeing blind review. A literal quoted-term absence claim uses an exhaustive lexical scan of the loaded corpus; it is not inferred from missing top-five passages. This special path carries corpus-completeness and extraction assumptions.

The generation workflow has one initial draft and at most two MCQ revisions. Explanation writing has one local repair. Every revision clears prior gate results. The final service boundary validates the complete PracticeQuestion record before saving it.

## Evaluation

Evaluation is organized by task rather than one headline score:

- the 75-question mixed Polity/governance set is a regression diagnostic;
- a 16-query annotated page benchmark measures retrieval hits;
- legacy 13-question reports measure selective verification, grounding and stability;
- generation reports measure gate yield, revisions and blocking reasons;
- generated-MCQ expert accuracy is still unmeasured.

The preserved v3 source-verifier snapshots contain 2 correct, 0 wrong and 31 abstentions on development (33) and 0 correct, 0 wrong and 42 abstentions on test (42). These are regression results, not evidence of perfect future accuracy. The project explicitly does not use 75/75 as its acceptance criterion. See docs/BENCHMARK.md, EVALUATION.md and docs/BENCHMARK_NEXT_STEPS.md.

Run offline checks without model calls:

    .venv/bin/python -m pytest
    .venv/bin/python -m src.evaluation.evaluate_benchmark --check
    .venv/bin/python -m src.evaluation.run_all --check

For the separate evaluation dashboard:

    .venv/bin/python -m streamlit run evals_app.py --server.port 8502

The complete project map is in docs/PROJECT_BRIEF.md, the canonical technical design is docs/SYSTEM_DESIGN.md, and the practical learner flow is docs/PRACTICE_DESIGN.md.
