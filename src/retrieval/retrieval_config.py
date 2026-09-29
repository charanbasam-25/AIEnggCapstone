"""
The retrieval depths, defined once.

Why this file exists
--------------------

The project's largest measured defect was one integer in the wrong place.
The retriever was chosen by a Recall@5 ablation and the verification
pipeline then queried it at k=3, so the depth that justified the choice
was never the depth that shipped. The 93.75% figure quoted for
semantic+cross-encoder described a configuration nothing ran at.

That instance was fixed by editing the 3 to a 5. The condition that
produced it was not: the benchmark depth and the pipeline depth were two
unrelated literals in two unrelated files, and nothing connected them.
Before this module there were roughly twenty independent `5`s and `20`s
across src/, agreeing only because they had each been edited by hand to
agree. Change the ablation to Recall@10, miss one of the production
constants, and the original bug reappears with no error and no diff to
notice - the numbers just quietly stop describing the system.

So the depths live here, and both the benchmarks that measure retrieval
and the pipeline that consumes it import them. The invariant is no longer
maintained by remembering; it holds because there is only one value.

RETRIEVAL_TOP_K
    How many chunks reach the consumer - the LLM's evidence window for a
    claim, and the cutoff in `Recall@k`. These are the same number by
    definition: `Recall@k` is a claim about what the consumer will see,
    so measuring at one depth and running at another measures nothing.

RERANK_CANDIDATE_K
    How many chunks the first stage hands the cross-encoder. This is a
    ceiling the second stage cannot exceed: a page the bi-encoder leaves
    out of the top 20 cannot be recovered by reranking, however good the
    reranker is. It is the same class of coupling as the above - the
    ablation's candidate depth and the pipeline's have to match or the
    ablation is not measuring the pipeline - which is why it is here too.

Changing either value invalidates every number in data/evaluation/.
`python -m src.evaluation.run_all --check` asserts that the depth recorded
in verified_pyq_results.json still equals RETRIEVAL_TOP_K and fails if it
does not, so the invalidation is an error rather than a silent drift. That
check costs no API calls; run it after editing this file.

The one place a literal depth is still correct
---------------------------------------------

src/evaluation/system_c_pipeline.py hardcodes k=3 and k=5 in its arm
definitions, and should. It is the ablation that *varies* depth, so there
the literal is the measurement rather than a setting - and the C0 arm
replays a frozen k=3 archive that must not move when this file changes.
Its RETRIEVER_TOP_K equivalent would be a bug, not a fix.

Its `candidate_k` is a different matter and does import from here. No arm
varies the first-stage depth, so if that number drifted from the shipped
one the ablation would quietly stop describing the shipped retriever -
the same failure, one stage earlier.
"""

# Chunks returned to the consumer, and the k in Recall@k.
RETRIEVAL_TOP_K = 5

# Chunks the bi-encoder hands the cross-encoder to rerank.
RERANK_CANDIDATE_K = 20
