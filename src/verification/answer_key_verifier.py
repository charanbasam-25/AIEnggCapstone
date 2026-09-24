import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from src.generation.mcq_generator import MCQ

load_dotenv()

MODEL_NAME = "gpt-4o-mini"


class AnswerAnalysis(BaseModel):
    supported_options: list[str] = Field(
        description="Options that the provided evidence supports as the correct answer."
    )
    reasoning: str


class AnswerKeyVerificationResult(BaseModel):
    declared_answer: str = Field(pattern="^[ABCD]$")
    supported_options: list[str]
    exactly_one_correct: bool
    verdict: str = Field(
        pattern="^(VALID|INVALID|INSUFFICIENT)$"
    )
    reasoning: str
    supporting_pages: list[int]


class AnswerKeyVerifier:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def analyze(
        self,
        mcq: MCQ,
        evidence: list[dict],
    ) -> AnswerAnalysis:

        options = f"""
A. {mcq.option_a}
B. {mcq.option_b}
C. {mcq.option_c}
D. {mcq.option_d}
"""

        evidence_text = []

        for rank, item in enumerate(evidence, start=1):
            evidence_text.append(
                f"""
EVIDENCE {rank}
Source: {item["source"]}
Page: {item["page"]}

{item["text"]}
"""
            )

        context = "\n".join(evidence_text)

        prompt = f"""
You are an independent answer analysis component for a UPSC
Indian Polity MCQ verification system.

Determine which options, if any, are supported as the correct
answer by the provided source evidence.

QUESTION:

{mcq.question}

OPTIONS:

{options}

DECLARED ANSWER:

{mcq.correct_answer}

SOURCE EVIDENCE:

{context}

RULES:

1. Use ONLY the provided source evidence.
2. Do not use outside knowledge.
3. Do not assume that a semantically related passage proves an option.
4. An option should be included in supported_options only when
   the evidence explicitly supports that option as the answer.
5. If the evidence is insufficient to establish an option,
   do not include it.
6. If multiple options are explicitly supported, include all of them.
7. Do not use the declared answer as evidence that an option is correct.
8. Do not make a final VALID or INVALID decision.
9. Return only option letters A, B, C or D in supported_options.
"""

        response = self.client.responses.parse(
            model=MODEL_NAME,
            input=prompt,
            text_format=AnswerAnalysis,
        )

        return response.output_parsed

    def verify(
        self,
        mcq: MCQ,
        evidence: list[dict],
    ) -> AnswerKeyVerificationResult:

        analysis = self.analyze(mcq, evidence)

        supported_options = analysis.supported_options

        # Defensive validation of the model output.
        supported_options = [
            option
            for option in supported_options
            if option in {"A", "B", "C", "D"}
        ]

        # Remove duplicates while preserving order.
        supported_options = list(dict.fromkeys(supported_options))

        exactly_one_correct = len(supported_options) == 1

        if not supported_options:
            verdict = "INSUFFICIENT"

        elif not exactly_one_correct:
            verdict = "INVALID"

        elif supported_options[0] != mcq.correct_answer:
            verdict = "INVALID"

        else:
            verdict = "VALID"

        supporting_pages = []

        if verdict == "VALID":
            for item in evidence:
                supporting_pages.append(item["page"])

        return AnswerKeyVerificationResult(
            declared_answer=mcq.correct_answer,
            supported_options=supported_options,
            exactly_one_correct=exactly_one_correct,
            verdict=verdict,
            reasoning=analysis.reasoning,
            supporting_pages=list(dict.fromkeys(supporting_pages)),
        )


if __name__ == "__main__":
    from src.generation.mcq_generator import MCQGenerator
    from src.verification.claim_retriever import ClaimRetriever
    from src.retrieval.semantic_reranker import load_chunks

    generator = MCQGenerator()

    mcq = generator.generate(
        topic="Fundamental Rights",
        difficulty="medium",
    )

    chunks = load_chunks("data/processed/chunks.jsonl")

    retriever = ClaimRetriever(chunks)

    evidence = retriever.retrieve(
        mcq.question,
        top_k=5,
    )

    verifier = AnswerKeyVerifier()

    result = verifier.verify(
        mcq,
        evidence,
    )

    print("\n=== MCQ ===\n")
    print(mcq.question)
    print(f"A. {mcq.option_a}")
    print(f"B. {mcq.option_b}")
    print(f"C. {mcq.option_c}")
    print(f"D. {mcq.option_d}")

    print(f"\nDeclared answer: {result.declared_answer}")

    print("\n=== ANSWER-KEY VERIFICATION ===\n")
    print(f"Supported options: {result.supported_options}")
    print(f"Exactly one correct: {result.exactly_one_correct}")
    print(f"Verdict: {result.verdict}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Supporting pages: {result.supporting_pages}")