from src.retrieval.retrieval_config import (
    RERANK_CANDIDATE_K,
    RETRIEVAL_TOP_K,
)
from src.retrieval.semantic_reranker import (
    SemanticReranker,
    load_chunks,
)


class ClaimRetriever:
    """
    Independently retrieves evidence for factual claims.

    This component does not use the evidence that was used
    during MCQ generation.
    """

    def __init__(
        self,
        chunks: list[dict],
        candidate_k: int = RERANK_CANDIDATE_K,
    ):
        self.retriever = SemanticReranker(
            chunks,
            candidate_k=candidate_k,
        )

    def retrieve(
        self,
        claim: str,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> list[dict]:
        """
        Return the top_k reranked chunks for one claim.

        The default was 3, which is where the project's largest measured
        defect came from: the retriever was selected on a Recall@5
        benchmark and then queried at k=3, so the depth that justified it
        was never the depth it ran at.

        What the change is worth
        ------------------------

        The only comparison in the project that varies depth and nothing
        else is C0 vs C1d in src/evaluation/system_c_pipeline.py: both
        arms use stored claims and the base prompt, so k is the single
        moving part. From data/evaluation/system_c_results.json, STRICT
        against stored labels:

                                 C0 (k=3)   C1d (k=5)
            coverage              30.77%      38.46%    4/13 -> 5/13
            precision answered    50.00%      80.00%
            accuracy overall      15.38%      30.77%
            error rate overall    15.38%       7.69%

        One question moved from abstain to correct and one from wrong to
        correct. Against the audited labels the same change is flat on
        accuracy (23.08% both arms) and slightly worse on precision
        (75.00% -> 60.00%), so the size of the win depends on which label
        set you score, and one of the 13 stored labels is known wrong
        (Q58; see src/evaluation/label_audit.py). n=13 and
        verdict_stability.py measures a 15.38-point accuracy spread
        across repeats, so read this as directional, not as two decimals.

        An earlier version of this docstring quoted coverage
        38.46% -> 53.85% and accuracy 30.77% -> 38.46%. Those are the
        table above misread one row across: 38.46% is where coverage
        ended, not where it started, and 53.85% appears in no arm of any
        stored run. The claim "with claims and prompt held fixed" was
        true of C0/C1d but was attached to numbers taken from somewhere
        else. Same error as the bug this docstring describes, one level
        up - a figure sitting next to a setting it was not measured at -
        so it is corrected here rather than quietly deleted.
        """

        results = self.retriever.retrieve(
            claim,
            top_k=top_k,
        )

        return results


if __name__ == "__main__":
    chunks = load_chunks("data/processed/chunks.jsonl")

    retriever = ClaimRetriever(chunks)

    claim = (
        "The Right to Property was originally a Fundamental "
        "Right under Article 31 of the Constitution of India."
    )

    results = retriever.retrieve(claim)

    print("\n=== CLAIM ===\n")
    print(claim)

    print("\n=== INDEPENDENT EVIDENCE ===\n")

    for rank, result in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print(f"Source: {result['source']}")
        print(f"Page: {result['page']}")
        print(f"Score: {result['reranker_score']:.4f}")
        print(f"Text:\n{result['text']}")