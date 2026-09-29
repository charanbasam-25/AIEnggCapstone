Retrieval Baseline and Comparative Evaluation
Project: Source-Verified UPSC Prelims Polity MCQ Tutor
PURPOSE
-------
This document records the retrieval experiments conducted before integrating retrieval into the RAG pipeline.
Three approaches were compared:
1. BM25 lexical retrieval
2. Semantic retrieval using BAAI/bge-small-en-v1.5
3. Hybrid BM25 + semantic retrieval using Reciprocal Rank Fusion (RRF)
All three approaches were evaluated on the same 16-query benchmark.

1. KNOWLEDGE BASE
-----------------
The current retrieval corpus contains:
- Constitution of India
- NCERT Polity textbook content
Corpus statistics:
Source              Pages       Chunks
---------------------------------------
Constitution        402         1,107
NCERT Polity         20            42
---------------------------------------
Total               422         1,149
Text extraction:
- Constitution: 845,502 characters
- NCERT Polity: 30,061 characters
Processed files:
data/processed/
    constitution.txt
    ncert_polity.txt
    prepared_documents.txt
    chunks.txt
    chunks.jsonl

2. CHUNKING STRATEGY
--------------------
The corpus was split into overlapping chunks.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
Each chunk preserves:
- source
- document
- page
- chunk index
Example:
{
  "id": 0,
  "text": "...",
  "source": "constitution",
  "document": "Constitution of India",
  "page": 1,
  "chunk_index": 0
}
Metadata provides provenance and allows retrieved evidence to be traced back to the original source and page.

3. EVALUATION BENCHMARK
-----------------------
A 16-query benchmark was created using specific constitutional provisions as retrieval targets.
Topics include:
- Fundamental Rights
- Directive Principles
- Fundamental Duties
- Election of the President
- Council of Ministers
- Parliament
- Rajya Sabha
- Lok Sabha
- High Courts
- Comptroller and Auditor-General
- Finance Commission
- Public Service Commissions
- Election Commission
- Legislative powers
- Emergency provisions
- Constitutional amendment
Gold pages:
q01  Fundamental Rights                         Page 37
q02  Directive Principles                      Page 52
q03  Fundamental Duties                        Page 56
q04  Election of President                     Page 57
q05  Council of Ministers                      Page 65
q06  Parliament                                Page 67
q07  Council of States                         Page 67
q08  House of the People                       Page 68
q09  High Courts                                Page 130
q10  Comptroller and Auditor-General            Page 99
q11  Finance Commission                         Page 193
q12  Public Service Commissions                 Page 208
q13  Election Commission                       Page 217
q14  Legislative powers                         Page 176
q15  Emergency                                  Page 239
q16  Constitutional amendment                   Page 259
The benchmark is stored in:
src/evaluation/retrieval_queries_gold.json

4. EVALUATION METRIC
--------------------
The primary metric is Recall@5.
A query is counted as a hit when at least one of the top five retrieved chunks:
1. comes from the expected source, and
2. comes from one of the expected gold pages.
Formula:
Recall@5 =
Number of queries with a relevant result in top 5
-------------------------------------------------
Total number of queries
Evaluation configuration:
- Queries: 16
- Top-k: 5

5. BM25 BASELINE
----------------
BM25 was used as the lexical retrieval baseline.
Implementation:
rank-bm25
BM25 uses lexical term matching based on factors including:
- term frequency
- inverse document frequency
- document-length normalization
Evaluation command:
python -m src.evaluation.evaluate_bm25
Result:
Hits: 12 / 16
Recall@5: 75.00%
BM25 missed:
- q03 — Fundamental Duties
- q04 — Election of President
- q06 — Parliament
- q16 — Constitutional amendment
Example q03 failure:
Query:
What are the Fundamental Duties of citizens?
Top results:
1. NCERT page 8
2. NCERT page 12
3. NCERT page 18
4. NCERT page 4
5. Constitution page 7
Observed failure pattern:
BM25 can rank passages highly when they contain overlapping vocabulary with the query, even when the query refers to a specific constitutional provision. NCERT passages can therefore rank above the expected Constitution provision.

6. SEMANTIC RETRIEVAL
---------------------
Semantic retrieval uses:
BAAI/bge-small-en-v1.5
Each chunk is converted into an embedding and the query is embedded using the same model. Results are ranked using vector similarity.
Evaluation command:
python -m src.evaluation.evaluate_semantic
Result:
Hits: 14 / 16
Recall@5: 87.50%
Improvement over BM25:
87.50% - 75.00% = 12.50 percentage points
Semantic retrieval missed two queries.
q06 — Parliament
Expected:
Constitution page 67
Top results:
1. Constitution page 218
2. Constitution page 78
3. NCERT page 3
4. Constitution page 301
5. Constitution page 145
q09 — High Courts
Expected:
Constitution page 130
Top results:
1. Constitution page 16
2. Constitution page 149
3. Constitution page 280
4. Constitution page 346
5. Constitution page 16
Observed failure pattern:
Semantic retrieval is better at conceptual similarity, but it can still fail at fine-grained section localization. It may retrieve constitutionally related content without retrieving the exact provision requested.

7. HYBRID BM25 + SEMANTIC
-------------------------
The hybrid approach combines BM25 and semantic retrieval using Reciprocal Rank Fusion (RRF).
Evaluation command:
python -m src.evaluation.evaluate_hybrid
Result:
Hits: 13 / 16
Recall@5: 81.25%
Compared with BM25:
81.25% - 75.00% = 6.25 percentage points
Compared with semantic retrieval:
81.25% - 87.50% = -6.25 percentage points
Hybrid missed:
- q06 — Parliament
- q09 — High Courts
- q16 — Constitutional amendment
Example q16 failure:
Expected:
Constitution page 259
Top results:
1. Constitution page 143
2. Constitution page 175
3. NCERT page 4
4. Constitution page 278
5. NCERT page 5
Observed failure pattern:
Combining lexical and semantic signals did not automatically improve ranking. RRF can promote documents that score reasonably across both systems without placing the exact target provision in the top five.

8. COMPARATIVE RESULTS
----------------------
All approaches used:
- Same 16 queries
- Same corpus
- Same top-k = 5
- Same Recall@5 metric
Approach                                  Hits       Recall@5
----------------------------------------------------------------
BM25                                      12/16      75.00%
Semantic — BAAI/bge-small-en-v1.5         14/16      87.50%
Hybrid — BM25 + Semantic + RRF            13/16      81.25%
Observed result:
Semantic retrieval produced the highest measured Recall@5 on this benchmark.
Hybrid improved over BM25 but did not outperform semantic retrieval.
This is an experimental observation for this 16-query benchmark, not a general claim that semantic retrieval is universally superior.

9. FAILURE ANALYSIS
-------------------
BM25:
Primary failure mode:
Lexical overlap without sufficiently precise provision-level retrieval.
Semantic retrieval:
Primary failure mode:
Semantic similarity without sufficiently precise section localization.
Hybrid retrieval:
Primary failure mode:
Combining retrieval signals does not automatically improve the final ranking.
The experiments demonstrate why retrieval strategies should be compared empirically rather than selected only because an approach appears more sophisticated.

10. RETRIEVAL DECISION
---------------------
Current measured results:
BM25       -> 75.00%
Semantic   -> 87.50%
Hybrid     -> 81.25%
For the next RAG stage, semantic retrieval using BAAI/bge-small-en-v1.5 will be used as the current retrieval baseline.
This decision is based on the measured results from the current benchmark.
It should not be interpreted as a universal conclusion about retrieval methods because the benchmark contains only 16 queries.
SUPERSEDED. This decision was the correct one on the evidence available at the
time, and two later arms changed it. On the same 16 queries, same corpus and
same Recall@5 metric:
Parent-child (page parents) + reranker             14/16      87.50%
Semantic -> ms-marco-MiniLM-L-6-v2 cross-encoder   15/16      93.75%   <- ships
The shipped retriever is therefore semantic retrieval reranked by a
cross-encoder over 20 first-stage candidates, not bare semantic retrieval.
Parent-child expansion matched bare semantic and did not justify its extra
machinery, so it is kept as a benchmark arm only.
Note that 93.75% is 15/16: one query is worth 6.25 points on this benchmark.
That is enough to reject hybrid RRF and parent-child, and not enough to claim
the reranker's margin is real rather than noise.
Both depths are now imported from src/retrieval/retrieval_config.py, so the
top-k this benchmark measures at and the top-k the pipeline queries at cannot
drift apart. See EVALUATION.md and docs/SYSTEM_DESIGN.md section 8 for why that
matters: they had drifted, and the pipeline ran at k=3 while this report
justified it at k=5.

11. LIMITATIONS
---------------
1. The benchmark contains only 16 queries.
2. Gold labels are based on specific constitutional provision pages.
3. The evaluation uses page-level relevance rather than detailed human chunk-level relevance judgments.
4. The current corpus contains the Constitution and NCERT Polity material.
5. Retrieval recall does not measure end-to-end answer quality.
6. This report predates the reranker and the parent-child arms. Both were
   evaluated later on the same benchmark and the reranker was selected; see
   section 10 and EVALUATION.md section 1. The three-approach comparison above
   is retained as the record of what was known at the time, not as the current
   result.
7. Qdrant has not yet been introduced into the retrieval pipeline.
8. The benchmark does not represent every possible UPSC Polity query.
9. Some questions may be answerable from multiple parts of the corpus, while this benchmark expects specific constitutional provision pages.
Therefore, these results should be treated as baseline experimental measurements.

12. REPRODUCIBILITY
-------------------
BM25:
python -m src.evaluation.evaluate_bm25
Semantic:
python -m src.evaluation.evaluate_semantic
Hybrid:
python -m src.evaluation.evaluate_hybrid
Gold benchmark:
src/evaluation/retrieval_queries_gold.json
Processed corpus:
data/processed/chunks.jsonl

13. RETRIEVAL ARCHITECTURE
--------------------------
                Constitution
                     +
                 NCERT Polity
                     |
                     v
              PDF Extraction
                     |
                     v
          Page-aware Processing
                     |
                     v
              Chunking + Metadata
                     |
                     v
                chunks.jsonl
                     |
          +----------+----------+
          |          |          |
          v          v          v
        BM25     Semantic     Hybrid
          |          |          |
          +----------+----------+
                     |
                     v
               Top-k Results
                     |
                     v
              Evaluation / RAG

14. NEXT STAGE
--------------
The retrieval baseline experiment is complete.
The next stage is to integrate semantic retrieval into the RAG pipeline and evaluate the complete retrieval-plus-generation flow.
Potential future retrieval experiments include:
- metadata-aware filtering
- improved chunk boundaries
- query rewriting
- reranking
- additional evaluation queries
- Qdrant-based vector retrieval
These should be treated as separate experiments rather than silently changing the current baseline.

15. FINAL BASELINE STATUS
-------------------------
Retrieval experimentation is complete for the current benchmark.
Final measured results:
BM25       : 75.00% Recall@5
Semantic   : 87.50% Recall@5
Hybrid RRF : 81.25% Recall@5
Current retrieval baseline:
Semantic retrieval using BAAI/bge-small-en-v1.5
