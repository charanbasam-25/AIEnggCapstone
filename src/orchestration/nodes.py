from src.generation.mcq_generator import MCQGenerator
from src.orchestration.state import MCQVerificationState
from src.verification.claim_extractor import ClaimExtractor
from src.verification.claim_retriever import ClaimRetriever
from src.verification.fact_verifier import FactVerifier
from src.retrieval.semantic_reranker import load_chunks
from src.verification.answer_key_verifier import AnswerKeyVerifier
from src.verification.mcq_quality_auditor import MCQQualityAuditor

def generate_mcq(state: MCQVerificationState) -> dict:
    generator = MCQGenerator()

    failure_reasons = state.get("failure_reasons", [])

    mcq = generator.generate(
        topic=state["topic"],
        difficulty=state.get("difficulty", "medium"),
        failure_reasons=failure_reasons,
    )

    return {
        "mcq": mcq,
        "retry_count": state.get("retry_count", 0) + 1,
        "failure_reasons": [],
    }
    
def extract_claims(state: MCQVerificationState) -> dict:
    extractor = ClaimExtractor()

    result = extractor.extract(state["mcq"])

    return {
        "claims": result.claims,
    }



def verify_claims(state: MCQVerificationState) -> dict:
    chunks = load_chunks("data/processed/chunks.jsonl")

    retriever = ClaimRetriever(chunks)
    verifier = FactVerifier()

    claim_evidence = {}
    fact_verifications = {}

    for claim in state["claims"]:
        evidence = retriever.retrieve(
            claim.claim,
            top_k=3,
        )

        result = verifier.verify(
            claim.claim,
            evidence,
        )

        claim_evidence[claim.claim_id] = evidence
        fact_verifications[claim.claim_id] = result

    return {
        "claim_evidence": claim_evidence,
        "fact_verifications": fact_verifications,
    }
    



def verify_answer_key(state: MCQVerificationState) -> dict:
    chunks = load_chunks("data/processed/chunks.jsonl")

    retriever = ClaimRetriever(chunks)
    verifier = AnswerKeyVerifier()

    evidence = retriever.retrieve(
        state["mcq"].question,
        top_k=5,
    )

    result = verifier.verify(
        state["mcq"],
        evidence,
    )

    return {
        "answer_evidence": evidence,
        "answer_verification": result,
    }


def audit_quality(state: MCQVerificationState) -> dict:
    auditor = MCQQualityAuditor()

    result = auditor.audit(
        state["mcq"],
    )

    return {
        "quality_audit": result,
    }

def decide(state: MCQVerificationState) -> dict:
    failure_reasons = []

    # 1. Fact verification
    claims_by_id = {
        claim.claim_id: claim
        for claim in state["claims"]
    }

    for claim_id, result in state["fact_verifications"].items():
        if result.verdict != "SUPPORTED":
            claim = claims_by_id.get(claim_id)

            if claim:
                failure_reasons.append(
                    f"{claim_id}: {result.verdict}. "
                    f"Claim: {claim.claim}"
                )
            else:
                failure_reasons.append(
                    f"{claim_id}: {result.verdict}"
                )

    # 2. Answer-key verification
    answer_result = state["answer_verification"]

    if answer_result.verdict != "VALID":
        failure_reasons.append(
            "answer_key: "
            f"{answer_result.verdict}. "
            f"Reason: {answer_result.reasoning}"
        )

    # 3. Quality audit
    quality_result = state["quality_audit"]

    if quality_result.overall_quality != "PASS":
        for issue in quality_result.issues:
            failure_reasons.append(
                f"quality: {issue}"
            )

    # Determine final decision.
    if not failure_reasons:
        decision = "ACCEPT"

    elif state.get("retry_count", 0) < state.get("max_retries", 2):
        decision = "REVISE"

    else:
        decision = "REJECT"

    return {
        "decision": decision,
        "failure_reasons": failure_reasons,
    }