# Architecture diagrams

These diagrams describe the current implementation as of 7 October 2026. The PNG files are convenient for GitHub and presentations; SVG and PDF preserve a sharp export; MMD files remain editable in Mermaid editors and GitHub Markdown.

The source of truth for the image layouts is render_diagrams.py. Rebuild every export with:

    .venv/bin/python docs/diagrams/render_diagrams.py

## Diagram gallery

| Flow | PNG | Editable Mermaid | Meaning |
|---|---|---|---|
| Overall architecture | [PNG](diagrams/practice_architecture.png) | [MMD](diagrams/practice_architecture.mmd) | UI, source corpus, retrieval, workflow, storage and diagnostics |
| Learner practice | [PNG](diagrams/practice_flow.png) | [MMD](diagrams/practice_flow.mmd) | Selection, reuse, preparation, quiz and review |
| Retrieval | [PNG](diagrams/retrieval_flow.png) | [MMD](diagrams/retrieval_flow.mmd) | Page extraction, top 20, reranking, pages and quotes |
| Generation | [PNG](diagrams/generation_flow.png) | [MMD](diagrams/generation_flow.mmd) | Draft, gates, bounded revision and publication |
| Verification | [PNG](diagrams/verification_flow.png) | [MMD](diagrams/verification_flow.mmd) | Statement and direct question paths |
| Literal absence | [PNG](diagrams/absence_flow.png) | [MMD](diagrams/absence_flow.mmd) | Exhaustive quoted-term scan and its assumptions |
| Data model | [PNG](diagrams/data_model.png) | [MMD](diagrams/data_model.mmd) | SQLite tables, session memory and evaluation files |
| Evaluation | [PNG](diagrams/evaluation_flow.png) | [MMD](diagrams/evaluation_flow.mmd) | Answering, grading, checkpoints and Evals UI |
| Baseline | [PNG](diagrams/baseline_flow.png) | [MMD](diagrams/baseline_flow.mmd) | Vanilla RAG comparison protocol |

Each image also has matching SVG and PDF files beside it. The old system-design filenames are compatibility copies of the overall architecture.

## Reading conventions

Green means source or evidence data, blue means Python/UI control, purple means model work, amber means a decision, and grey means local storage. Solid arrows are data or control flow. Dashed arrows represent a model/revision relationship. A source passage is never itself a verdict.
