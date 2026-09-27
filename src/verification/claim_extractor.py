"""
Turn a question's numbered statements into independently verifiable
claims.

Why this is deterministic first
-------------------------------

This component used to be a single LLM call. The ablation run measured
both paths on the same 13 questions and the same verifier, and the
deterministic builder won on the one axis that does not depend on an
answer key:

    stored (LLM-shaped) claims   70.27% propositional,  8-9 false SUPPORTED
    bound (deterministic) claims  100.00% propositional,   0 false SUPPORTED

A non-propositional claim is the problem worth fixing. When a question
distributes its predicate across the stem -- "Which of the following are
included in the Seventh Schedule? I. Police" -- copying the item
verbatim yields the claim "Police", a bare noun phrase with no truth
value. SUPPORTED on it is not a judgement that happens to be wrong; it
is wrong by construction, because there was no proposition to support.
Those verdicts then flow into the answer mapping as if they meant
something. The builder binds the stem's predicate back onto each item
so the claim states something that can be true or false.

The honest counter-evidence: on final answer accuracy the two tie at
53.85%, and the LLM-shaped claims reach it with one fewer wrong answer
(7.69% vs 15.38% error). That is a single question out of 13, measured
against a label set where one gold answer was itself found to be wrong
(see src/evaluation/label_audit.py). Claim soundness is measured over 37
claims and needs no gold label at all, so it is the stronger signal, and
it is what this default follows. The tie is recorded rather than buried.

The LLM path is kept as a fallback for questions the parser cannot
decompose, and is reachable on demand. It is not the default.
"""

import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.generation.mcq_generator import MCQ
from src.verification.claim_builder import (
    BINDING_NONE,
    build_claims_for_question,
)
from src.verification.question_parser import is_propositional


load_dotenv()

MODEL_NAME = "gpt-4o-mini"

SOURCE_DETERMINISTIC = "deterministic"
SOURCE_LLM = "llm"


class Claim(BaseModel):
    claim_id: str

    # The numbered statement this claim belongs to.
    # Examples: "I", "II", "III", "IV".
    # Use None for contextual claims that are not one of
    # the numbered statements being tested.
    statement_number: str | None = None

    group_id: str
    claim: str

    # Provenance. These carry the builder's own account of what it did to
    # the item, so a downstream surprise can be traced to a binding
    # decision instead of guessed at. They default to the values a
    # verbatim pass-through would produce, which is what the LLM path
    # effectively is.
    source: str = SOURCE_DETERMINISTIC
    binding: str = BINDING_NONE
    needed_binding: bool = False
    is_propositional: bool = True
    unsupported_tokens: list[str] = Field(default_factory=list)


class ClaimExtractionResult(BaseModel):
    claims: list[Claim] = Field(min_length=1)


class LLMClaim(BaseModel):
    """
    The fallback path's output shape.

    Deliberately identical to the original schema and separate from
    `Claim`. The provenance fields above must not appear in a schema sent
    to the model: under strict structured output every field is required,
    so the model would be asked to invent a `binding` value it has no way
    to know.
    """

    claim_id: str
    statement_number: str | None = None
    group_id: str
    claim: str


class LLMClaimExtractionResult(BaseModel):
    claims: list[LLMClaim] = Field(min_length=1)


class ClaimExtractor:
    """
    Extract claims deterministically, falling back to an LLM call.

    `allow_llm_fallback=False` makes the component pure: no network, no
    API key, no nondeterminism. That is how the evaluation harness uses
    it, so a failure to parse surfaces as an error rather than as an
    unremarked change of method mid-run.
    """

    def __init__(self, allow_llm_fallback: bool = True):
        self.allow_llm_fallback = allow_llm_fallback

        # Constructed on first use. Building the client eagerly would
        # make the deterministic path require an API key it never uses.
        self._client = None

    @property
    def client(self):

        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )

        return self._client

    def extract(self, mcq: MCQ) -> ClaimExtractionResult:

        try:
            return self.extract_deterministic(mcq.question)

        except ValueError as error:

            if not self.allow_llm_fallback:
                raise

            self.last_fallback_reason = str(error)

            return self.extract_with_llm(mcq)

    def extract_deterministic(
        self,
        question_text: str,
    ) -> ClaimExtractionResult:
        """
        Parse the question and bind each item into a standalone claim.

        Raises ValueError when the question has no numbered items, which
        is the signal the caller uses to decide on a fallback.
        """

        _, built = build_claims_for_question(question_text)

        claims = [
            Claim(
                claim_id=f"claim_{item.label}",
                statement_number=item.label,
                group_id="main",
                claim=item.claim,
                source=SOURCE_DETERMINISTIC,
                binding=item.binding,
                needed_binding=item.needed_binding,
                is_propositional=item.is_propositional,
                unsupported_tokens=item.unsupported,
            )
            for item in built
        ]

        if not claims:
            raise ValueError(
                "Question parsed but produced no claims: "
                f"{question_text[:120]!r}"
            )

        return ClaimExtractionResult(claims=claims)

    def extract_with_llm(self, mcq: MCQ) -> ClaimExtractionResult:
        """
        Fallback for questions the parser cannot decompose.

        Rule 4 asks for the same predicate binding the deterministic path
        performs, because a verbatim copy of a stem-distributed item has
        no truth value. The instruction is weaker than the template
        substitution it imitates -- the model can paraphrase, and there
        is no reconstruction check -- which is the reason this is the
        fallback and not the default.
        """

        prompt = f"""
You are a factual claim extraction component for a UPSC
Indian Polity verification system.

Your input is a UPSC question.

Your task is to restate the numbered statements of the question
as standalone propositions that can be checked one at a time.

IMPORTANT:
This is an extraction task only.

Do NOT solve the question.
Do NOT determine which statements are correct.
Do NOT use outside knowledge.

Rules:

1. Extract EVERY numbered statement in the question.

2. Preserve the statement number exactly:
   I, II, III, IV, etc.

3. Every numbered statement must produce exactly one claim.

4. A numbered statement is often not a sentence on its own,
   because the question's predicate sits in the stem. In that
   case combine the stem's predicate with the item so that the
   claim asserts something that could be true or false.

   Question: "Which of the following are included in the
   Seventh Schedule? I. Police"
   Claim:    "Police is included in the Seventh Schedule."

   Take the predicate from the question's own wording. Do not
   introduce any term that is not in the question.

5. For pair-matching questions, state that the two halves of
   the pair are correctly matched.

6. Do NOT extract answer options.

7. Do NOT extract the declared answer.

8. Do NOT extract explanations.

9. Do NOT add any fact that is not present in the question.

10. Do NOT correct wording or factual errors in the statement.

11. Do NOT decide whether a claim is true or false.

12. Use:
       group_id = "main"

13. If the question contains no numbered statements,
    extract the substantive factual claim(s) from the
    question and use:
       statement_number = null

Question:

{mcq.question}
"""

        # Pinned for the same reason as in fact_verifier.py: the Responses
        # API default is 1.0, so leaving this unset made claim extraction
        # a fresh sample on every run and nothing downstream reproducible.
        response = self.client.responses.parse(
            model=MODEL_NAME,
            input=prompt,
            temperature=0,
            text_format=LLMClaimExtractionResult,
        )

        if response.output_parsed is None:
            raise ValueError(
                "Claim extraction response was not parsed"
            )

        claims = [
            Claim(
                claim_id=item.claim_id,
                statement_number=item.statement_number,
                group_id=item.group_id,
                claim=item.claim,
                source=SOURCE_LLM,
                # The model is not asked to report on its own binding,
                # so the only provenance available is what can be
                # computed from the text it returned.
                is_propositional=is_propositional(item.claim),
            )
            for item in response.output_parsed.claims
        ]

        return ClaimExtractionResult(claims=claims)


if __name__ == "__main__":

    from src.verification.question_parser import ascii_safe

    # A stem-distributed question, which is the case the deterministic
    # path exists for. No API key and no network are needed to run this.
    sample = (
        "Which of the following are included in the Seventh Schedule "
        "of the Constitution of India?\n"
        "I. Police\n"
        "II. Public health\n"
        "III. Defence of India\n"
        "Select the correct answer using the code given below."
    )

    extractor = ClaimExtractor(allow_llm_fallback=False)

    result = extractor.extract_deterministic(sample)

    print("\n=== DETERMINISTIC CLAIMS ===\n")

    for claim in result.claims:

        print(
            f"{claim.claim_id} | "
            f"statement={claim.statement_number} | "
            f"binding={claim.binding} | "
            f"propositional={claim.is_propositional}"
        )
        print(f"    {ascii_safe(claim.claim)}")

        if claim.unsupported_tokens:
            print(
                "    unsupported tokens: "
                f"{claim.unsupported_tokens}"
            )

    assert all(
        claim.is_propositional for claim in result.claims
    ), "binding must leave every claim propositional"

    print("\nok\n")
