import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from src.generation.mcq_generator import MCQ


load_dotenv()

MODEL_NAME = "gpt-4o-mini"


class Claim(BaseModel):
    claim_id: str

    # The numbered statement this claim belongs to.
    # Examples: "I", "II", "III", "IV".
    # Use None for contextual claims that are not one of
    # the numbered statements being tested.
    statement_number: str | None = None

    group_id: str
    claim: str


class ClaimExtractionResult(BaseModel):
    claims: list[Claim] = Field(min_length=1)


class ClaimExtractor:

    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def extract(
        self,
        mcq: MCQ,
    ) -> ClaimExtractionResult:

        prompt = f"""
You are a factual claim extraction component for a UPSC
Indian Polity verification system.

Your input is a UPSC question.

Your task is to extract the substantive factual claims
contained in the numbered statements of the question.

IMPORTANT:
This is an extraction task only.

Do NOT solve the question.
Do NOT determine which statements are correct.
Do NOT use outside knowledge.

Rules:

1. Extract EVERY numbered statement in the question.

2. Preserve the statement number exactly:
   I, II, III, IV, etc.

3. Every numbered statement must produce at least one claim.

4. If a numbered statement contains multiple independent
   factual assertions, split them into atomic claims.

5. All claims belonging to the same numbered statement
   must have the same statement_number.

6. For pair-matching questions, preserve the entire
   factual pair as one claim when the pair represents
   a single proposition.

7. Do NOT extract answer options.

8. Do NOT extract the declared answer.

9. Do NOT extract explanations.

10. Do NOT infer or add facts that are not explicitly
    present in the question.

11. Do NOT correct wording or factual errors.

12. Do NOT decide whether a claim is true or false.

13. Use:
       group_id = "main"

14. If the question contains no numbered statements,
    extract the substantive factual claim(s) from the
    question and use:
       statement_number = null

15. For a question containing numbered statements,
    NEVER omit a numbered statement.

Question:

{mcq.question}
"""

        response = self.client.responses.parse(
            model=MODEL_NAME,
            input=prompt,
            text_format=ClaimExtractionResult,
        )

        if response.output_parsed is None:
            raise ValueError(
                "Claim extraction response was not parsed"
            )

        result = response.output_parsed

        # Deterministic validation:
        # if the question contains numbered statements,
        # verify that the model extracted every one.
        import re

        expected_statements = set(
            re.findall(
                r"(?m)^\\s*(I|II|III|IV)\\.",
                mcq.question,
            )
        )

        extracted_statements = {
            claim.statement_number
            for claim in result.claims
            if claim.statement_number is not None
        }

        missing_statements = (
            expected_statements
            - extracted_statements
        )

        if missing_statements:
            raise ValueError(
                "Claim extraction omitted numbered "
                f"statements: "
                f"{sorted(missing_statements)}"
            )

        return result



if __name__ == "__main__":

    from src.generation.mcq_generator import MCQGenerator

    generator = MCQGenerator()

    mcq = generator.generate(
        topic="Fundamental Rights",
        difficulty="medium",
    )

    extractor = ClaimExtractor()

    result = extractor.extract(mcq)

    print("\n=== MCQ ===\n")
    print(mcq.question)

    print("\n=== OPTIONS ===\n")
    print(f"A. {mcq.option_a}")
    print(f"B. {mcq.option_b}")
    print(f"C. {mcq.option_c}")
    print(f"D. {mcq.option_d}")

    print("\n=== EXTRACTED CLAIMS ===\n")

    for claim in result.claims:

        statement = (
            claim.statement_number
            if claim.statement_number is not None
            else "context"
        )

        print(
            f"{claim.claim_id} | "
            f"statement={statement} | "
            f"group={claim.group_id} | "
            f"{claim.claim}"
        )
