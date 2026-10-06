"""Source-first generation with deterministic publication gates."""

import hashlib
import re
from functools import lru_cache
from pathlib import Path

from src.evaluation.answer_mapping import STRICT, determine_statement_statuses, map_answer
from src.generation.mcq_generator import FORMAT_SIMPLE, FORMAT_STATEMENTS, MCQGenerator
from src.orchestration.state import MCQVerificationState
from src.orchestration.telemetry import create_model_client, measured_node, record_evidence_packet
from src.retrieval.retrieval_config import RETRIEVAL_TOP_K
from src.verification.answer_key_verifier import AnswerKeyVerificationResult
from src.verification.article_context import ArticleEvidenceContext, anchor_metadata
from src.verification.claim_extractor import ClaimExtractor
from src.verification.direct_mcq_verifier import DirectMCQVerifier, verify_direct_question
from src.verification.fact_verifier import FactVerifier, verify_constructed_claim
from src.verification.learning_notes import (
    LearningNotesVerifier, citation_issues, explanation_review_passes, notes_issues,
)
from src.verification.mcq_quality_auditor import MCQQualityAuditor, QUALITY_FIELDS, quality_passes
from src.verification.question_type import STATEMENT_MCQ, classify_question, has_labelled_statement_pair


CHUNKS_PATH = str(Path(__file__).resolve().parents[2] / "data/processed/chunks.jsonl")
CLAIM_TOP_K = RETRIEVAL_TOP_K
ANSWER_TOP_K = RETRIEVAL_TOP_K


def _corpus_signature():
    path = Path(CHUNKS_PATH)
    stat = path.stat()
    return str(path.resolve()), stat.st_mtime_ns, stat.st_size, corpus_hash()


def corpus_hash() -> str:
    return hashlib.sha256(Path(CHUNKS_PATH).read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def _cached_chunks(signature) -> tuple:
    from src.retrieval.semantic_reranker import load_chunks
    return tuple(load_chunks(signature[0]))


def get_chunks() -> tuple:
    return _cached_chunks(_corpus_signature())


@lru_cache(maxsize=1)
def _cached_retriever(signature):
    from src.verification.claim_retriever import ClaimRetriever
    return ClaimRetriever(list(_cached_chunks(signature)))


def get_retriever():
    return _cached_retriever(_corpus_signature())


@lru_cache(maxsize=1)
def _cached_fact_verifier(signature):
    return FactVerifier(
        chunks=list(_cached_chunks(signature)), client=create_model_client(), reference_quotes=True,
    )


def get_fact_verifier() -> FactVerifier:
    return _cached_fact_verifier(_corpus_signature())


@lru_cache(maxsize=1)
def _cached_direct_verifier(signature):
    return DirectMCQVerifier(
        chunks=list(_cached_chunks(signature)), client=create_model_client(), reference_quotes=True,
    )


def get_direct_verifier() -> DirectMCQVerifier:
    return _cached_direct_verifier(_corpus_signature())


def mcq_options(mcq) -> dict[str, str]:
    return {letter: getattr(mcq, f"option_{letter.lower()}") for letter in "ABCD"}


def _normal(text: str) -> str:
    return re.sub(r"\W+", " ", text.casefold()).strip()


def _unique_evidence(passages) -> list[dict]:
    unique = {}
    for passage in passages:
        key = (passage.get("source"), passage.get("document"), passage.get("page"), passage["text"])
        unique.setdefault(key, passage)
    return list(unique.values())


def _key_result(mcq, predicted, reasoning, pages=()) -> AnswerKeyVerificationResult:
    return AnswerKeyVerificationResult(
        declared_answer=mcq.correct_answer,
        supported_options=[predicted] if predicted else [],
        exactly_one_correct=bool(predicted),
        verdict=("VALID" if predicted == mcq.correct_answer else "INVALID") if predicted else "INSUFFICIENT",
        reasoning=reasoning,
        supporting_pages=sorted(set(pages)),
    )


@measured_node
def retrieve_sources(state: MCQVerificationState) -> dict:
    query = f"{state['topic']}. {state.get('focus', state['topic'])}"
    retrieved = get_retriever().retrieve(query, top_k=RETRIEVAL_TOP_K)
    # A named Article is also resolved against its actual source heading.
    # This adds evidence, without replacing the measured semantic ranking.
    evidence = ArticleEvidenceContext(list(get_chunks())).supplement(query, retrieved)
    evidence = [
        passage for passage in evidence
        if passage.get("source") and passage.get("text", "").strip()
        and isinstance(passage.get("page"), int) and not isinstance(passage["page"], bool)
        and passage["page"] > 0
    ]
    if not evidence:
        raise ValueError("No approved source passages are available for this topic.")
    record_evidence_packet("Generation sources", evidence, article_anchors=anchor_metadata(evidence))
    return {"generation_evidence": evidence}


@measured_node
def generate_mcq(state: MCQVerificationState) -> dict:
    if not state.get("generation_evidence"):
        raise ValueError("Generation cannot run without retrieved source passages.")
    mcq = MCQGenerator().generate(
        topic=state["topic"],
        difficulty=state.get("difficulty", "medium"),
        failure_reasons=state.get("failure_reasons", []),
        question_format=state.get("question_format", FORMAT_SIMPLE),
        evidence=state["generation_evidence"],
        focus=state.get("focus"),
        avoid_questions=state.get("avoid_questions", []),
    )
    # Clear every gate on every attempt. A previous PASS cannot approve a
    # revised question whose corresponding check was skipped or failed.
    return {
        "mcq": mcq, "retry_count": state.get("retry_count", 0) + 1,
        "failure_reasons": [], "structural_issues": [], "claims": [],
        "claim_evidence": {}, "fact_verifications": {}, "answer_evidence": [],
        "answer_verification": None, "direct_verification": None,
        "quality_audit": None, "learning_notes": None,
    }


@measured_node
def extract_claims(state: MCQVerificationState) -> dict:
    mcq = state["mcq"]
    issues = []
    options = list(mcq_options(mcq).values())
    if not mcq.question.strip() or any(not option.strip() for option in options):
        issues.append("The question or an option is empty.")
    if len({_normal(option) for option in options}) != 4:
        issues.append("The options are not distinct.")
    if any(_normal(mcq.question) == _normal(old) for old in state.get("avoid_questions", [])):
        issues.append("This question has already been used.")
    if has_labelled_statement_pair(mcq.question):
        issues.append("Paired Statement-I/II and Assertion/Reason questions are not supported.")
    kind = classify_question(mcq.question)
    requested = state.get("question_format", FORMAT_SIMPLE)
    if (requested == FORMAT_STATEMENTS) != (kind == STATEMENT_MCQ):
        issues.append("The generated question does not match the requested format.")
    claims = []
    if kind == STATEMENT_MCQ and not issues:
        try:
            claims = ClaimExtractor(allow_llm_fallback=False).extract(mcq).claims
            labels = [claim.statement_number for claim in claims]
            if not 2 <= len(labels) <= 4 or len(set(labels)) != len(labels):
                issues.append("Statement questions require two to four distinct statements.")
        except ValueError:
            issues.append("The statements could not be parsed and bound safely.")
    return {"claims": claims, "structural_issues": issues}


@measured_node
def verify_claims(state: MCQVerificationState) -> dict:
    if state.get("structural_issues") or not state.get("claims"):
        return {"claim_evidence": {}, "fact_verifications": {}}
    retriever, verifier = get_retriever(), get_fact_verifier()
    context = ArticleEvidenceContext(list(get_chunks()))
    evidence_by_claim, verdicts = {}, {}
    for claim in state["claims"]:
        evidence = context.supplement(
            claim.claim, retriever.retrieve(claim.claim, top_k=CLAIM_TOP_K),
        )
        result = verify_constructed_claim(claim, evidence, verifier)
        evidence_by_claim[claim.claim_id] = result.checked_evidence or evidence
        verdicts[claim.claim_id] = result
        record_evidence_packet(
            f"Statement {claim.statement_number}", evidence_by_claim[claim.claim_id],
            claim_id=claim.claim_id, verdict=result.verdict, method=result.verification_method,
            article_anchors=anchor_metadata(evidence),
        )
    return {"claim_evidence": evidence_by_claim, "fact_verifications": verdicts}


@measured_node
def verify_answer_key(state: MCQVerificationState) -> dict:
    mcq = state["mcq"]
    if state.get("structural_issues"):
        return {
            "answer_verification": _key_result(mcq, None, "The candidate did not pass format validation."),
            "answer_evidence": [],
        }
    if classify_question(mcq.question) == STATEMENT_MCQ:
        claims, verdicts = state.get("claims", []), state.get("fact_verifications", {})
        if not claims or any(claim.claim_id not in verdicts for claim in claims):
            return {"answer_verification": _key_result(mcq, None, "Statement checks are incomplete."),
                    "answer_evidence": []}
        statuses = determine_statement_statuses(claims, verdicts)
        predicted, debug = map_answer(statuses, mcq_options(mcq), mcq.question, policy=STRICT)
        pages = [page for result in verdicts.values() for page in result.supporting_pages]
        evidence = _unique_evidence(
            passage for result in verdicts.values() for passage in result.checked_evidence
        )
        record_evidence_packet("Answer verification", evidence)
        return {
            "answer_verification": _key_result(
                mcq, predicted,
                f"Strict statement mapping: {statuses}. "
                + (f"The evidence implies option {predicted}." if predicted else
                   f"No unique answer: {debug.get('abstention_reason', 'unresolved evidence')}."),
                pages,
            ),
            "answer_evidence": evidence,
        }
    result = verify_direct_question(
        {"question_text": mcq.question, "options": mcq_options(mcq)},
        get_retriever(), get_direct_verifier(),
    )
    record_evidence_packet("Answer verification", result.evidence)
    issues = []
    for assessment in result.option_assessments:
        if assessment.answer_fit != "INSUFFICIENT":
            issues.extend(citation_issues(assessment.citations, result.evidence))
    predicted = result.predicted_answer
    if result.status != "ANSWERED" or not result.independently_reviewed or issues:
        predicted = None
    pages = [
        result.evidence[citation.evidence_id - 1]["page"]
        for assessment in result.option_assessments for citation in assessment.citations
        if 1 <= citation.evidence_id <= len(result.evidence)
        and isinstance(result.evidence[citation.evidence_id - 1].get("page"), int)
    ]
    return {
        "direct_verification": result,
        "answer_evidence": result.evidence,
        "answer_verification": _key_result(
            mcq, predicted,
            "; ".join(issues) if issues else " ".join(
                part for part in [
                    result.abstention_reason,
                    *[
                        f"Option {assessment.option} unresolved: {assessment.reasoning}"
                        for assessment in result.option_assessments
                        if assessment.answer_fit == "INSUFFICIENT"
                    ],
                    result.reasoning,
                ] if part
            ), pages,
        ),
    }


@measured_node
def audit_quality(state: MCQVerificationState) -> dict:
    answer = state.get("answer_verification")
    if answer is None or answer.verdict != "VALID":
        return {"quality_audit": None}
    return {"quality_audit": MCQQualityAuditor().audit(state["mcq"], topic=state["topic"])}


@measured_node
def explain_question(state: MCQVerificationState) -> dict:
    if not quality_passes(state.get("quality_audit")):
        return {"learning_notes": None}
    result = LearningNotesVerifier().verify(state["mcq"], state.get("answer_evidence", []))
    return {"learning_notes": result}


@measured_node
def decide(state: MCQVerificationState) -> dict:
    reasons = list(state.get("structural_issues", []))
    if not state.get("generation_evidence"):
        reasons.append("Grounded generation evidence is missing.")
    mcq = state.get("mcq")
    if mcq is None:
        reasons.append("No generated candidate is available.")
    elif classify_question(mcq.question) == STATEMENT_MCQ:
        claims, verdicts = state.get("claims", []), state.get("fact_verifications", {})
        if not claims:
            reasons.append("The numbered statements were not verified.")
        for claim in claims:
            result = verdicts.get(claim.claim_id)
            if result is None:
                reasons.append(f"{claim.claim_id}: verification is missing.")
            elif result.verdict == "INSUFFICIENT":
                reasons.append(
                    f"{claim.claim_id}: INSUFFICIENT. Claim: {claim.claim}. "
                    f"Reason: {result.reasoning}"
                )
            elif (
                not result.independently_reviewed
                or citation_issues(result.citations, result.checked_evidence)
            ):
                reasons.append(f"{claim.claim_id}: quoted independent review is incomplete.")
        # CONTRADICTED is a resolved false statement, not a defective MCQ.
        # The independently resolved truth pattern must match its answer key.
    else:
        direct = state.get("direct_verification")
        if direct is None or direct.status != "ANSWERED" or not direct.independently_reviewed:
            reasons.append("The direct answer did not pass independent option review.")
    answer = state.get("answer_verification")
    if answer is None:
        reasons.append("Answer-key verification is missing.")
    elif (
        answer.verdict != "VALID" or not answer.exactly_one_correct
        or answer.supported_options != [mcq.correct_answer]
    ):
        reasons.append(f"answer_key: {answer.verdict}. {answer.reasoning}")
    quality = state.get("quality_audit")
    if not quality_passes(quality):
        if quality is None:
            reasons.append("Quality audit did not complete.")
        else:
            reasons.extend(f"quality: {issue}" for issue in quality.issues)
            reasons.extend(f"quality: {name} failed." for name in QUALITY_FIELDS if not getattr(quality, name))
            if quality.overall_quality != "PASS":
                reasons.append("quality: the auditor returned FAIL.")
    lesson = state.get("learning_notes")
    if (
        lesson is None or lesson.verdict != "PASS" or lesson.notes is None
        or lesson.review is None or not explanation_review_passes(lesson.review)
    ):
        reasons.extend(f"explanation: {issue}" for issue in (lesson.issues if lesson else []))
        reasons.append("Explanation verification did not pass.")
    elif notes_issues(lesson.notes, state.get("answer_evidence", []), state["mcq"]):
        reasons.append("Explanation citations or option coverage are invalid.")
    reasons = list(dict.fromkeys(reasons))
    decision = (
        "ACCEPT" if not reasons else
        "REVISE" if state.get("retry_count", 0) <= state.get("max_retries", 2) else
        "REJECT"
    )
    return {"decision": decision, "failure_reasons": reasons}
