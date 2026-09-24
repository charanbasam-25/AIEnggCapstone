# Vanilla RAG Baseline

## Objective

Establish a simple source-grounded RAG baseline before adding independent
verification and orchestration.

## Retrieval Pipeline

The baseline uses the retrieval approach selected from the retrieval
experiments:

```text
User Query
    ↓
BGE-small semantic retrieval
    ↓
Top 20 candidate chunks
    ↓
MS MARCO MiniLM cross-encoder reranker
    ↓
Top 5 evidence chunks
    ↓
OpenAI LLM
    ↓
Grounded answer + source citations




Semantic retrieval followed by cross-encoder reranking was selected for the
Vanilla RAG baseline based on the benchmark results.


Model:gpt-4o-mini
Temperature:0

Example Result

The system retrieved Constitution of India, page 57 as the highest-ranked
evidence.

The generated answer explained that the President is elected by an electoral
college consisting of the elected members of both Houses of Parliament and
the elected members of the Legislative Assemblies of the States. It cited
page 57 of the Constitution.

The answer also referenced Article 55 regarding the manner of election.

Observations

The Vanilla RAG pipeline successfully connected the evaluated retrieval
system to an LLM and produced a source-grounded answer.

The retrieved top five results can contain multiple chunks from the same
page. For example, the President query returned page 57 more than once.
This creates redundant context.

This is a known limitation of the current baseline and can be addressed
later if necessary.