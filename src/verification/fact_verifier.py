import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from src.verification.negative_claim_checker import (
    NegativeClaimChecker,
)


load_dotenv()

MODEL_NAME = "gpt-4o-mini"


class FactVerificationResult(BaseModel):
    verdict: str = Field(
        pattern="^(SUPPORTED|CONTRADICTED|INSUFFICIENT)$"
    )
    reasoning: str
    supporting_pages: list[int]


class FactVerifier:
    def __init__(
        self,
        chunks: list[dict] | None = None,
    ):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.negative_claim_checker = None

        if chunks is not None:
            self.negative_claim_checker = (
                NegativeClaimChecker(chunks)
            )

    def verify(
        self,
        claim: str,
        evidence: list[dict],
    ) -> FactVerificationResult:

        # ==================================================
        # Explicit lexical absence claim handling
        # ==================================================

        if self.negative_claim_checker is not None:

            negative_result = (
                self.negative_claim_checker.check(
                    claim
                )
            )

            if negative_result[
                "is_absence_claim"
            ]:

                matching_evidence = (
                    negative_result["evidence"]
                )

                target_term = (
                    negative_result["target_term"]
                )

                if matching_evidence:

                    supporting_pages = sorted(
                        {
                            item["page"]
                            for item in matching_evidence
                        }
                    )

                    return FactVerificationResult(
                        verdict="CONTRADICTED",
                        reasoning=(
                            f"The claim states that "
                            f"'{target_term}' is absent "
                            f"or not mentioned. However, "
                            f"the corpus explicitly contains "
                            f"'{target_term}' in the retrieved "
                            f"source evidence."
                        ),
                        supporting_pages=(
                            supporting_pages
                        ),
                    )

                return FactVerificationResult(
                    verdict="INSUFFICIENT",
                    reasoning=(
                        f"The claim is an explicit absence "
                        f"claim about '{target_term}', but "
                        f"no matching occurrence was found "
                        f"through the lexical corpus search. "
                        f"Absence cannot be established from "
                        f"failure to find a match."
                    ),
                    supporting_pages=[],
                )

        # ==================================================
        # Normal evidence-based LLM verification
        # ==================================================

        evidence_text = []

        for rank, item in enumerate(
            evidence,
            start=1,
        ):

            evidence_text.append(
                f"""
EVIDENCE {rank}
Source: {item["source"]}
Page: {item["page"]}

{item["text"]}
"""
            )

        context = "\n".join(
            evidence_text
        )

        prompt = f"""
You are a factual verification component for a UPSC Indian
Polity MCQ system.

Your task is to determine whether the provided source evidence
supports, contradicts, or is insufficient to verify the claim.

IMPORTANT RULES:

1. Use ONLY the provided source evidence.

2. Do not use outside knowledge.

3. Do not infer facts that are not supported by the evidence.

4. Do not assume that semantically related text proves the claim.

5. Return SUPPORTED only when the provided evidence explicitly
   establishes the complete claim.

6. Do not use background knowledge to connect separate facts.

7. Do not infer missing constitutional Articles, provisions,
   dates, relationships, authorities, or conclusions.

8. If the claim requires information that is not explicitly
   present in the evidence, return INSUFFICIENT.

9. Return CONTRADICTED only when the provided evidence explicitly
   establishes information that conflicts with the claim.

10. If the evidence is relevant but insufficient to establish
    the complete claim, return INSUFFICIENT.

11. Cite only pages that actually support your verdict.

12. A semantically related passage is not sufficient evidence.

13. Be especially careful with negative or absence claims.

14. If the claim says that something does not exist, is not
    mentioned, is absent, or is not provided for, failure to
    find that information in the supplied evidence does NOT
    prove the claim.

15. For an absence claim, return INSUFFICIENT unless the
    provided evidence explicitly establishes the absence.

16. Do not treat "the evidence does not mention X" as evidence
    that "the Constitution does not mention X".

17. Do not assume that the retrieved top-k evidence represents
    the entire Constitution or the entire knowledge base.

18. For multi-part claims, verify the entire claim. If only part
    of the claim is supported, return INSUFFICIENT unless the
    evidence explicitly contradicts the complete claim.
19. Do not use the wording of the claim itself as evidence.
20. Evaluate the claim within its exact scope and subject.
21. Evidence about a different constitutional institution, office,
    House, legislature, authority, or category must not be treated
    as contradictory merely because it states a different rule.
22. Do not construct a contradiction by comparing the claim with
    a different constitutional provision unless the evidence
    explicitly states that the claimed rule is not applicable.
23. When one passage directly supports the claim and another passage
    concerns a different scope or institution, treat the latter as
    irrelevant rather than contradictory.
24. For a claim about Parliament or the House of the People, do not
    use provisions concerning State Legislatures as contradictory
    evidence unless the claim itself covers both.

CLAIM:

{claim}

SOURCE EVIDENCE:

{context}
"""

        response = self.client.responses.parse(
            model=MODEL_NAME,
            input=prompt,
            text_format=FactVerificationResult,
        )

        result = response.output_parsed

        if result is None:
            raise ValueError(
                "Fact verification response was not parsed"
            )

        return result


if __name__ == "__main__":

    from src.retrieval.semantic_reranker import (
        load_chunks,
    )
    from src.verification.claim_retriever import (
        ClaimRetriever,
    )

    chunks = load_chunks(
        "data/processed/chunks.jsonl"
    )

    retriever = ClaimRetriever(
        chunks
    )

    verifier = FactVerifier(
        chunks
    )

    claim = (
        "There is no mention of the word "
        "'political party' in the Constitution of India."
    )

    evidence = retriever.retrieve(
        claim,
        top_k=3,
    )

    result = verifier.verify(
        claim,
        evidence,
    )

    print(
        "\n=== CLAIM ===\n"
    )

    print(claim)

    print(
        "\n=== VERIFICATION RESULT ===\n"
    )

    print(
        f"Verdict: {result.verdict}"
    )

    print(
        f"Reasoning: {result.reasoning}"
    )

    print(
        f"Supporting pages: "
        f"{result.supporting_pages}"
    )
