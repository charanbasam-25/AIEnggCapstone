"""Contracts at the boundary between candidates and learner-visible questions."""

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from src.generation.mcq_generator import MCQ
from src.orchestration.telemetry import GENERATION_POLICY
from src.practice.catalog import TOPICS
from src.verification.learning_notes import LearningNotes, notes_issues


REQUIRED_GATES = {"sources", "format", "answer", "quality", "explanations"}


def question_fingerprint(question: str) -> str:
    normalized = re.sub(r"\W+", " ", question.casefold()).strip()
    return hashlib.sha256(normalized.encode()).hexdigest()


def record_id(payload: dict) -> str:
    content = {key: value for key, value in payload.items() if key != "id"}
    return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class PracticeRequest(BaseModel):
    subject: Literal["Polity"] = "Polity"
    topic: str = "Fundamental Rights"
    count: int = Field(default=5, ge=1, le=10)
    question_format: Literal["simple", "statements", "mixed"] = "simple"
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    max_retries: int = Field(default=2, ge=0, le=2)
    reuse_checked: bool = True
    exclude_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def supported_topic(self):
        if self.topic not in TOPICS:
            raise ValueError("Choose an available Polity topic.")
        return self


class PracticeQuestion(BaseModel):
    id: str
    topic: str
    subject: Literal["Polity"] = "Polity"
    question_format: Literal["simple", "statements"]
    difficulty: Literal["easy", "medium", "hard"]
    mcq: MCQ
    notes: LearningNotes
    evidence: list[dict]
    corpus_hash: str
    generation_policy: str
    created_at: str
    gates: dict[str, Literal["PASS"]]
    review_type: Literal["automated_evidence_review"] = "automated_evidence_review"

    @model_validator(mode="after")
    def publication_contract(self):
        if self.topic not in TOPICS:
            raise ValueError("This question does not use an available Polity topic.")
        if self.generation_policy != GENERATION_POLICY:
            raise ValueError("This question was checked under a different generation policy.")
        if set(self.gates) != REQUIRED_GATES:
            raise ValueError("Every publication gate must be present.")
        if not self.mcq.question.strip():
            raise ValueError("The question is empty.")
        options = [getattr(self.mcq, f"option_{letter}") for letter in "abcd"]
        if any(not option.strip() for option in options) or len({option.strip().casefold() for option in options}) != 4:
            raise ValueError("Four nonempty, distinct options are required.")
        if notes_issues(self.notes, self.evidence, self.mcq):
            raise ValueError("Saved learning notes fail source or option validation.")
        if self.mcq.explanation != self.notes.summary.text:
            raise ValueError("The answer explanation must be the reviewed learning summary.")
        if self.id != record_id(self.model_dump(mode="json")):
            raise ValueError("The saved question content has changed.")
        return self


@dataclass
class PreparationResult:
    questions: list[PracticeQuestion]
    report: dict


def score_answers(questions: list[PracticeQuestion], answers: dict[str, str | None]) -> dict:
    correct = sum(answers.get(question.id) == question.mcq.correct_answer for question in questions)
    skipped = sum(answers.get(question.id) not in set("ABCD") for question in questions)
    return {
        "total": len(questions), "correct": correct,
        "wrong": len(questions) - correct - skipped, "skipped": skipped,
        "accuracy": correct / len(questions) if questions else None,
    }
