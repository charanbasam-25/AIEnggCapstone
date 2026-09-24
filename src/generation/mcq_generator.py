import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

MODEL_NAME = "gpt-4o-mini"


class MCQ(BaseModel):
    question: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    correct_answer: str = Field(pattern="^[ABCD]$")
    explanation: str


class MCQGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(
        self,
        topic: str,
        difficulty: str = "medium",
        failure_reasons: list[str] | None = None,
    ) -> MCQ:

        revision_feedback = ""

        if failure_reasons:
            revision_feedback = f"""
This is a revision attempt.

The previous candidate MCQ failed verification for the
following reasons:

{chr(10).join(f"- {reason}" for reason in failure_reasons)}

Generate a NEW MCQ that specifically avoids these problems.

Do not simply repeat the previous question.
"""

        prompt = f"""
Generate one UPSC Civil Services Preliminary Examination
Indian Polity multiple-choice question.

Topic:
{topic}

Difficulty:
{difficulty}

{revision_feedback}

Requirements:

1. Generate exactly four options: A, B, C and D.
2. There must be exactly one correct answer.
3. The question must test Indian Polity.
4. Avoid current affairs.
5. Avoid ambiguous wording.
6. Avoid questions where more than one option could reasonably
   be considered correct.
7. Do not use unsupported factual claims.
8. Provide the correct answer as A, B, C or D.
9. Provide a short explanation of why the correct option is correct.
10. The candidate will be independently verified against the
    Constitution of India and NCERT sources.
11. Do not claim that the question has already been verified.
"""

        response = self.client.responses.parse(
            model=MODEL_NAME,
            input=prompt,
            text_format=MCQ,
        )

        return response.output_parsed


if __name__ == "__main__":
    generator = MCQGenerator()

    mcq = generator.generate(
        topic="Fundamental Rights",
        difficulty="medium",
    )

    print("\n=== GENERATED MCQ ===\n")
    print(f"Question: {mcq.question}")
    print(f"A. {mcq.option_a}")
    print(f"B. {mcq.option_b}")
    print(f"C. {mcq.option_c}")
    print(f"D. {mcq.option_d}")
    print(f"\nCorrect Answer: {mcq.correct_answer}")
    print(f"Explanation: {mcq.explanation}")