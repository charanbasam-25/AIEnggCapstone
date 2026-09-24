from typing import TypedDict

from src.generation.mcq_generator import MCQ
from src.verification.claim_extractor import Claim
from src.verification.fact_verifier import FactVerificationResult
from src.verification.answer_key_verifier import AnswerKeyVerificationResult
from src.verification.mcq_quality_auditor import QualityAuditResult


class MCQVerificationState(TypedDict, total=False):
    # User request
    topic: str
    difficulty: str

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
    answer_verification: AnswerKeyVerificationResult

    # Result of quality audit
    quality_audit: QualityAuditResult

    # Final workflow decision
    decision: str

    # Reasons that caused revision/rejection
    failure_reasons: list[str]

    # Number of generation attempts
    retry_count: int

    # Maximum number of retries
    max_retries: int