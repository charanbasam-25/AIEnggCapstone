# Retrieval baseline and selection

The source corpus contains 1,149 chunks: 1,107 Constitution chunks and 42 selected NCERT chunks. Page extraction keeps source, document, page and chunk index. Chunks are 1,000 characters with 150 characters of overlap.

![Retrieval flow](diagrams/retrieval_flow.png)

The annotated retrieval set contains 16 queries with expected source pages. A query is a hit when a relevant page appears in the top five. The current saved report records 14/16 hits (87.5%) for semantic retrieval and 14/16 for the semantic-plus-cross-encoder path. The report is a retrieval diagnostic, not MCQ accuracy.

The older bake-off also recorded BM25 at 12/16, hybrid RRF at 13/16, semantic at 14/16 and parent-child at 14/16. Those historical arms explain why the implementation retained semantic candidates plus reranking; they were not all rerun in the current report.

Run the deterministic benchmark with:

    .venv/bin/python -m src.evaluation.evaluate_retrieval

It records the corpus and query hashes, model names, candidate and result depths, timing and per-query traces. No LLM call is made. A ranking hit is evidence to inspect, not proof that a question is answerable.
