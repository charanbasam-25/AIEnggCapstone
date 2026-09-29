# Source-Verified UPSC Polity MCQ Verifier
A retrieval system that answers UPSC Prelims Polity multiple-choice questions
**only when the source material supports an answer**, and says so explicitly
when it does not.
The design decision the whole project rests on: **the language model never
picks the option letter.** It issues a SUPPORTED / CONTRADICTED /
INSUFFICIENT verdict on one statement at a time, each with a page citation.
Python maps those verdicts to an option. If the verdicts do not determine a
single option, the system abstains rather than guessing.
---
## The result, in one table
UPSC 2025 Prelims Polity, Q54–Q66 (n = 13).
| | Vanilla RAG | This system |
|---|---|---|
| Questions answered | 13 of 13 | 4 of 13 |
| Correct **when it answered** | **38.5%** | **100%** |
| Wrong answers presented as correct | **8** | **0** |
On the 4 questions both systems attempted, this system got 4 right; vanilla
RAG got 2.
The trade is deliberate and it is the point. Vanilla RAG answers everything
and is wrong 61.5% of the time, with no signal telling you which answers to
distrust. For preparing from a source of truth, eight confident wrong answers
are worse than nine honest refusals.
**Read that 100% as "1.00 on this run", not as a property of the system.**
Re-running the verifier five times at temperature 0 gives coverage between
23.1% and 38.5%, and precision-when-answered of 1.00 in four draws and 0.75
in the fifth — one wrong answer did get through. The honest summary is
*precision 0.95 ± 0.11 over 5 draws, versus vanilla RAG's 0.38*. Temperature
0 is not determinism; 35 of 37 claim verdicts were stable when pinned, 28 of
37 unpinned. Details in [EVALUATION.md §0.1](EVALUATION.md).
## Why the answer is decomposed
The rejected architecture asked the model to judge a whole question in one
call. On Q54 it returned **three of the four options as supported** — it
decided nothing, expensively. The same model, asked about one statement at a
time, produced clean verdicts with page citations.
That comparison is the justification for the architecture, not a
post-hoc rationalisation of it. Full numbers in
[EVALUATION.md §2](EVALUATION.md).
## Retrieval
Five configurations on a 16-query gold benchmark, Recall@5:
| Retriever | Recall@5 |
|---|---|
| BM25 (lexical baseline) | 75.00% |
| Hybrid (RRF) | 81.25% |
| Semantic (`bge-small-en-v1.5`) | 87.50% |
| Parent-child + reranker | 87.50% |
| **Semantic + cross-encoder reranker** | **93.75%** |
The baseline came first and the reranker was added because the baseline's
misses were measured, not assumed. Hybrid RRF and parent–child were both
built and both lost, and were rejected on the measurement rather than kept
because they were more sophisticated.
93.75% is 15/16, so one query is 6.25 points. That is enough to reject hybrid
and parent–child; it is not enough to claim the reranker's margin over bare
semantic is real rather than noise.
`Recall@5` and the k the pipeline queries at are the same constant —
`RETRIEVAL_TOP_K` in `src/retrieval/retrieval_config.py`. They have to be: the
metric is a claim about the consumer's evidence window, so measuring at one
depth and running at another measures nothing.
---
## Run it
```bash
pip install -r requirements.txt
# create .env in the repo root with your key:
#   OPENAI_API_KEY=sk-...
# optional, for the System C ablation's lexical arm:
#   SYSTEM_C_RETRIEVER=bm25
streamlit run app.py        # demo UI
```
First launch embeds all 1,149 chunks and takes roughly 70 seconds; after
that the models are cached for the session.
Or from the command line:
```bash
# a stored 2025 question
python -m src.verify_cli --pyq 54
# your own question
python -m src.verify_cli --file myquestion.txt --show-evidence
```
Questions must be in the multi-statement form, because that is the form the
system can decompose:
```
Consider the following statements:
I.  The Governor is appointed by the President.
II. The Governor holds office for a term of five years.
Which of the statements given above is/are correct?
A) I only
B) II only
C) Both I and II
D) Neither I nor II
```
A single-fact question (*"Who appoints the Governor?"*) has no independent
statements to check, and the system declines it rather than falling back on
the model's memory. That is a boundary of the design, and it is enforced in
code.
## Reproduce the numbers
```bash
python -m src.evaluation.run_all --check   # verify every report is
                                           # newer than its inputs; 0 API calls
python -m src.evaluation.run_all           # re-run the full chain
```
`--check` exists because the evaluation modules read each other's output
files, and running them out of order does not fail — it silently produces a
document-ready table of wrong numbers. That happened twice. See
`src/evaluation/run_all.py`.
It asserts two different things. The first is ordering: every report must be at
least as new as every file it was derived from. The second is not about file
times at all — editing `RETRIEVAL_TOP_K` in `src/retrieval/retrieval_config.py`
invalidates every number in `data/evaluation/` **without modifying a single file
in it**, so the ordering check passes while the reports have quietly stopped
describing the system. `check_retrieval_depth` reads the `claim_top_k` recorded
in `verified_pyq_results.json` back off disk and fails if it no longer matches
the constant the code imports.
That second check exists because the failure it catches already happened: the
retriever was selected on Recall@5 and the pipeline queried it at k=3. Both
depths now come from one module, so the invariant holds because there is one
value rather than because someone remembered to keep twenty literals in step.
---
## Documentation
| Document | What's in it |
|---|---|
| [DESIGN.md](DESIGN.md) | Problem, data surface, architecture, core flow, comparative evaluation |
| [EVALUATION.md](EVALUATION.md) | Every measurement, with noise bands, threats to validity, and 10 failure analyses |
| [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) | Module-level design and interfaces |
| [docs/retrieval_baseline.md](docs/retrieval_baseline.md) | The retrieval ablation in detail |
**Read EVALUATION.md §0 first if you intend to quote any number from this
repo.** It states which run produced which figure and what the measurement
noise is. At n = 13, one question moves a percentage by 7.7 points, and
verdicts are not fully reproducible even at temperature 0 (35 of 37 stable
when pinned, 28 of 37 unpinned).
## Scope
The contribution is **verification**. A generation loop
(`src/orchestration/`) is included as a second consumer of the same verifier
— it generates a question, then gates it through the same claim
verification before accepting it. It is a stress test on unseen input, not a
second product. Its results are in
[EVALUATION.md](EVALUATION.md).
## Known limitations
- **n = 13.** Small. Every percentage carries a wide interval, stated
  wherever one is quoted.
- **Coverage is 30.8%.** Nine of thirteen abstentions; six are traceable to
  pipeline defects rather than genuine evidence gaps, and are itemised in
  EVALUATION.md §6.
- **One domain, one corpus.** Indian Polity, 1,149 chunks. Nothing here has
  been tested outside it.
