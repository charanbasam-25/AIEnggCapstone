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
        top_k: int = 3,
    ) -> list[dict]:

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