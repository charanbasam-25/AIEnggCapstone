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

    # temperature is a parameter only so that the stability harness in
    # evaluation/verdict_stability.py can measure the unpinned behaviour
    # through this exact code path instead of reimplementing it. Passing
    # None omits the parameter from the request, which is what this class
    # did before it was pinned. Production callers should leave it alone.
    def verify(
        self,
        claim: str,
        evidence: list[dict],
        temperature: float | None = 0,
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

                    return FactVerificationResult(
                        verdict="CONTRADICTED",
                        reasoning=(
                            f"The claim states that "
                            f"'{target_term}' is absent or not "
                            f"mentioned. A scan of all "
                            f"{negative_result['chunks_scanned']} "
                            f"corpus chunks finds it "
                            f"{negative_result['occurrence_count']} "
                            f"time(s), on page(s) "
                            f"{negative_result['pages']}."
                        ),
                        supporting_pages=(
                            negative_result["pages"]
                        ),
                    )

                # The scan covered every chunk, so a null result is
                # informative and this returns SUPPORTED rather than
                # INSUFFICIENT. The earlier INSUFFICIENT was right only
                # because the search was a top-5 retrieval: absence from
                # 5 ranked chunks says nothing about the corpus.
                #
                # Two assumptions remain, and they are named in the
                # reasoning rather than buried here, because neither can
                # be checked from inside this system: that the corpus is
                # complete for the claim's scope, and that PDF extraction
                # preserved the term.
                return FactVerificationResult(
                    verdict="SUPPORTED",
                    reasoning=(
                        f"The claim asserts that '{target_term}' "
                        f"does not appear. An exhaustive "
                        f"case-insensitive scan of all "
                        f"{negative_result['chunks_scanned']} "
                        f"corpus chunks found no occurrence. This "
                        f"establishes the claim for this corpus, "
                        f"assuming the corpus covers the claim's "
                        f"scope and that text extraction preserved "
                        f"the term."
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

        # Rules 25-26 were added on measurement, not on intuition: in the
        # ablation they resolved one further question (Q63 II, an
        # INSUFFICIENT that became a correct CONTRADICTED) and cost
        # nothing elsewhere. Rule 25's second sentence is the important
        # half. Without it "the evidence states no age" collapses into
        # "the evidence states a different age", which would turn every
        # thin retrieval into a confident contradiction.
        #
        # Known problem with this prompt, recorded rather than hidden.
        # Appending those two rules also flipped an unrelated claim
        # (Q58 I) from SUPPORTED to CONTRADICTED on identical evidence,
        # and neither rule mentions anything in it. At 26 rules the prompt
        # is long enough that adding one perturbs the others, so a "rule"
        # here is not an isolated control. The verdict it flipped to is
        # the correct one, which is luck, not evidence that the mechanism
        # is sound. Further verifier work should decompose this prompt
        # rather than extend it.
        #
        # An earlier version of this comment said that flip happened "at
        # temperature 0". It did not - temperature was unset until the
        # call below was fixed, so that observation was one sample from an
        # unpinned sampler and the two-rule change was never the isolated
        # cause it was written up as. Q58 I turns out to sit on the
        # decision boundary: pinned at temperature 0 it is INSUFFICIENT on
        # this exact prompt, but SUPPORTED if three whitespace characters
        # change or if the same prompt goes to chat-completions instead.
        # Neither factor alone does it; the conjunction does. Rules 25-26
        # perturbing it is the same phenomenon, not a separate one.
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
25. If the claim states a specific number, age, duration, fraction,
    majority or threshold, and the evidence states a different value
    for the same provision and the same subject, return CONTRADICTED.
    The evidence must actually state a value; the absence of a value
    is INSUFFICIENT, never a contradiction.

26. When checking a number, confirm the evidence concerns the same
    office, body or provision as the claim before treating the values
    as comparable.

CLAIM:

{claim}

SOURCE EVIDENCE:

{context}
"""

        # temperature is pinned because it was not, and that invalidated
        # more than it looked like it would. The Responses API defaults to
        # 1.0, so every verdict this class has ever produced was a single
        # draw from a distribution. Sampling one claim (Q58 I) seven times
        # on fixed evidence returned CONTRADICTED three times,
        # INSUFFICIENT three times and SUPPORTED once - all three
        # verdicts. Meanwhile the baseline in rag/vanilla_rag.py was
        # pinned all along, so the headline comparison was a stochastic
        # system measured against a deterministic one.
        #
        # Pinning does not make the verdict correct, only repeatable. See
        # the boundary-case note in docs/ for what survives pinning.
        request = {
            "model": MODEL_NAME,
            "input": prompt,
            "text_format": FactVerificationResult,
        }

        if temperature is not None:
            request["temperature"] = temperature

        response = self.client.responses.parse(**request)

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

    evidence = retriever.retrieve(claim)

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
