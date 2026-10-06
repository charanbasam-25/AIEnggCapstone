from typing import TypedDict

from src.generation.mcq_generator import MCQ
from src.verification.claim_extractor import Claim
from src.verification.fact_verifier import FactVerificationResult
from src.verification.answer_key_verifier import AnswerKeyVerificationResult
from src.verification.mcq_quality_auditor import QualityAuditResult
from src.verification.direct_mcq_verifier import DirectVerificationResult
from src.verification.learning_notes import LearningNotesResult


class MCQVerificationState(TypedDict, total=False):
    # User request
    topic: str
    difficulty: str
    focus: str
    avoid_questions: list[str]
    generation_evidence: list[dict]
    structural_issues: list[str]

    # Direct ("simple") and numbered ("statements") questions use different
    # answer-verification paths. The default is a direct question.
    question_format: str

    # Generated candidate
    mcq: MCQ

    # Extracted factual claims
    claims: list[Claim]

    # Evidence retrieved independently for each claim
    claim_evidence: dict[str, list[dict]]

    # Results of factual verification
    fact_verifications: dict[str, FactVerificationResult]

    # Evidence retrieved independently for answer verification
    answer_evidence: list[dict]

    # Result of answer-key verification
    answer_verification: AnswerKeyVerificationResult | None
    direct_verification: DirectVerificationResult | None

    # Result of quality audit
    quality_audit: QualityAuditResult | None
    learning_notes: LearningNotesResult | None

    # Final workflow decision
    decision: str

    # Reasons that caused revision/rejection
    failure_reasons: list[str]

    # Number of generation attempts
    retry_count: int

    # Maximum number of retries
    max_retries: int
