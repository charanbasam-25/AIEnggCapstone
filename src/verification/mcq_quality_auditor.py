import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from src.generation.mcq_generator import MCQ

load_dotenv()

MODEL_NAME = "gpt-4o-mini"


class QualityAuditResult(BaseModel):
    unambiguous: bool
    single_best_answer: bool
    plausible_distractors: bool
    appropriate_wording: bool
    topic_relevant: bool
    issues: list[str] = Field(default_factory=list)
    overall_quality: str = Field(
        pattern="^(PASS|FAIL)$"
    )


class MCQQualityAuditor:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def audit(
        self,
        mcq: MCQ,
    ) -> QualityAuditResult:

        prompt = f"""
You are a quality auditor for a UPSC Civil Services Preliminary
Examination Indian Polity MCQ.

Evaluate the candidate question as an examination question.

QUESTION:

{mcq.question}

OPTIONS:

A. {mcq.option_a}
B. {mcq.option_b}
C. {mcq.option_c}
D. {mcq.option_d}

DECLARED ANSWER:

{mcq.correct_answer}

EXPLANATION:

{mcq.explanation}

Evaluate the following dimensions:

1. UNAMBIGUOUS
   The question should have a clear interpretation.

2. SINGLE BEST ANSWER
   The wording should allow one clearly defensible answer.
   Do not assume the declared answer is correct.

3. PLAUSIBLE DISTRACTORS
   The incorrect options should be reasonable alternatives,
   rather than obviously irrelevant or nonsensical choices.

4. APPROPRIATE WORDING
   The question should be reasonably consistent with the style
   expected in a UPSC Prelims Indian Polity question.

5. TOPIC RELEVANT
   The question should actually test Indian Polity and the
   requested subject matter.

IMPORTANT:

- Do not use outside sources.
- Do not verify constitutional facts.
- Do not change or rewrite the question.
- Identify concrete quality problems.
- If there are no meaningful quality problems, return PASS.
- If one or more important quality problems exist, return FAIL.
- Explain each identified issue briefly.
"""

        response = self.client.responses.parse(
            model=MODEL_NAME,
            input=prompt,
            text_format=QualityAuditResult,
        )

        return response.output_parsed


if __name__ == "__main__":
    from src.generation.mcq_generator import MCQGenerator

    generator = MCQGenerator()

    mcq = generator.generate(
        topic="Fundamental Rights",
        difficulty="medium",
    )

    auditor = MCQQualityAuditor()

    result = auditor.audit(mcq)

    print("\n=== MCQ ===\n")
    print(mcq.question)
    print(f"A. {mcq.option_a}")
    print(f"B. {mcq.option_b}")
    print(f"C. {mcq.option_c}")
    print(f"D. {mcq.option_d}")

    print("\n=== QUALITY AUDIT ===\n")
    print(f"Unambiguous: {result.unambiguous}")
    print(f"Single best answer: {result.single_best_answer}")
    print(f"Plausible distractors: {result.plausible_distractors}")
    print(f"Appropriate wording: {result.appropriate_wording}")
    print(f"Topic relevant: {result.topic_relevant}")
    print(f"Overall quality: {result.overall_quality}")

    if result.issues:
        print("\nIssues:")
        for issue in result.issues:
            print(f"- {issue}")