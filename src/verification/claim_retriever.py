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
        candidate_k: int = 20,
    ):
        self.retriever = SemanticReranker(
            chunks,
            candidate_k=candidate_k,
        )

    def retrieve(
        self,
        claim: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Return the top_k reranked chunks for one claim.

        The default was 3, which is where the project's largest measured
        defect came from: the retriever was selected on a Recall@5
        benchmark and then queried at k=3, so the depth that justified it
        was never the depth it ran at. Raising it to 5 moved coverage
        38.46% -> 53.85% and accuracy 30.77% -> 38.46% with claims and
        prompt held fixed.

        Every caller now passes top_k explicitly, so this default should
        not be load-bearing. It is aligned with the benchmark anyway, so
        that a future caller who omits it inherits the measured setting
        rather than the one that caused the bug.
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