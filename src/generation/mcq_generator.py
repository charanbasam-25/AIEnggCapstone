import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

MODEL_NAME = "gpt-4o-mini"

# Question formats.
#
# FORMAT_SIMPLE is what this generator has always produced: a stem and four
# option strings, one of which is correct. FORMAT_STATEMENTS produces the
# multi-statement form that actually dominates UPSC Prelims Polity and that
# the rest of this project is built around.
#
# The distinction is not cosmetic, and measuring it is the reason the
# parameter exists. A FORMAT_SIMPLE question decomposes into exactly one
# claim, so gating it exercises none of the project's core machinery -
# question_parser's STATEMENTS/STEM_DISTRIBUTED/PAIRS classification,
# claim_builder's predicate binding, and answer_mapping's subset and count
# parsing are all unreachable from a generated question. The generate-and-
# gate loop was therefore measuring a degenerate case without saying so.
#
# FORMAT_SIMPLE remains the default and its prompt is byte-identical to the
# one this module shipped with, so the default path is not silently
# re-measured. The seam follows the same idiom as FactVerifier.verify's
# `temperature` parameter: an argument that exists so a harness can measure
# an alternative, not so callers can drift.
FORMAT_SIMPLE = "simple"
FORMAT_STATEMENTS = "statements"

FORMATS = (FORMAT_SIMPLE, FORMAT_STATEMENTS)

# The label syntax here is not a stylistic choice. question_parser locates
# items with the pattern `(?<![A-Za-z])I\.(?=\s|$)` (question_parser.py:372),
# so Roman numerals followed by a period are the only markers it will find.
# Arabic numerals would generate questions that the claim builder silently
# fails to decompose, which would look like a verifier defect rather than a
# format mismatch.
STATEMENTS_REQUIREMENTS = """
Format requirements, which override requirement 1 above:

12. The question must use the UPSC multi-statement form. Write a short
    stem, then two to four statements, then a closing instruction.
13. Label the statements with Roman numerals followed by a period:
    "I.", "II.", "III.". Put each statement on its own line.
14. End the stem with a closing instruction, either
    "How many of the statements given above are correct?" or
    "Which of the statements given above is/are correct?".
15. The four options must describe subsets or counts of those
    statements, not restate them. Use the same Roman numerals.
    For "Which ... is/are correct?" use forms such as
    "I and II only", "II and III only", "I only", "I, II and III".
    For "How many ... are correct?" use exactly
    "Only one", "Only two", "Only three", "None".
16. Each statement must be independently true or false, and at least
    one must be false, so that the option set is not degenerate.
"""


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
        question_format: str = FORMAT_SIMPLE,
    ) -> MCQ:

        if question_format not in FORMATS:
            raise ValueError(
                f"question_format must be one of {FORMATS}, "
                f"got {question_format!r}"
            )

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

        # Appended rather than interpolated, so that the FORMAT_SIMPLE
        # prompt is byte-identical to the one this module shipped with. An
        # empty `{format_requirements}` slot would still add a newline, and
        # at temperature 0 a whitespace change is enough to move a verdict
        # (see EVALUATION.md on the decision-boundary experiment).
        if question_format == FORMAT_STATEMENTS:
            prompt += STATEMENTS_REQUIREMENTS

        # Pinned for reproducibility, same as the verification modules.
        # Question generation is the one place where sampling diversity
        # would arguably be wanted, but the generated questions feed the
        # evaluation set, so a fixed corpus matters more than variety.
        response = self.client.responses.parse(
            model=MODEL_NAME,
            input=prompt,
            temperature=0,
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