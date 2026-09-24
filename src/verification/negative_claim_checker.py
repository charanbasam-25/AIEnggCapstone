import re

from src.retrieval.bm25_retriever import BM25Retriever


class NegativeClaimChecker:
    """
    Handles explicit lexical absence claims such as:

    "There is no mention of the word 'political party'..."
    "The Constitution does not mention 'political party'..."

    This component only handles claims where the target term
    is explicitly quoted. It does not attempt to reason about
    conceptual absence.
    """

    ABSENCE_PATTERNS = [
        r"no mention of (?:the word|the term)?\s*[\"'‘“]([^\"'’”]+)[\"'’”]",
        r"does not mention (?:the word|the term)?\s*[\"'‘“]([^\"'’”]+)[\"'’”]",
        r"does not contain (?:the word|the term)?\s*[\"'‘“]([^\"'’”]+)[\"'’”]",
        r"not mentioned (?:as|by)?\s*[\"'‘“]([^\"'’”]+)[\"'’”]",
    ]

    def __init__(
        self,
        chunks: list[dict],
    ):
        self.retriever = BM25Retriever(
            chunks
        )

    def extract_target_term(
        self,
        claim: str,
    ) -> str | None:

        for pattern in self.ABSENCE_PATTERNS:

            match = re.search(
                pattern,
                claim,
                flags=re.IGNORECASE,
            )

            if match:
                return match.group(1).strip()

        return None

    def check(
        self,
        claim: str,
        top_k: int = 5,
    ) -> dict:

        target_term = self.extract_target_term(
            claim
        )

        if not target_term:
            return {
                "is_absence_claim": False,
                "target_term": None,
                "found": False,
                "evidence": [],
            }

        evidence = self.retriever.retrieve(
            target_term,
            top_k=top_k,
        )

        normalized_term = (
            target_term.lower()
        )

        matching_evidence = []

        for item in evidence:

            text = item["text"].lower()

            if normalized_term in text:

                matching_evidence.append(
                    item
                )

        return {
            "is_absence_claim": True,
            "target_term": target_term,
            "found": bool(
                matching_evidence
            ),
            "evidence": matching_evidence,
        }


if __name__ == "__main__":

    from src.retrieval.semantic_reranker import (
        load_chunks,
    )

    chunks = load_chunks(
        "data/processed/chunks.jsonl"
    )

    checker = NegativeClaimChecker(
        chunks
    )

    claim = (
        "There is no mention of the word "
        "'political party' in the Constitution of India."
    )

    result = checker.check(
        claim
    )

    print(
        "\n=== NEGATIVE CLAIM CHECK ===\n"
    )

    print(
        f"Absence claim: "
        f"{result['is_absence_claim']}"
    )

    print(
        f"Target term: "
        f"{result['target_term']}"
    )

    print(
        f"Found in corpus: "
        f"{result['found']}"
    )

    print(
        f"Evidence count: "
        f"{len(result['evidence'])}"
    )

    for item in result["evidence"]:

        print(
            f"\n--- {item['source']} "
            f"p.{item['page']} ---"
        )

        print(
            item["text"][:500]
        )
