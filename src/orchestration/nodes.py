from functools import lru_cache

from src.generation.mcq_generator import FORMAT_SIMPLE, MCQGenerator
from src.orchestration.state import MCQVerificationState
from src.verification.claim_extractor import ClaimExtractor
from src.verification.claim_retriever import ClaimRetriever
from src.verification.fact_verifier import FactVerifier
from src.retrieval.semantic_reranker import load_chunks
from src.verification.answer_key_verifier import AnswerKeyVerifier
from src.verification.mcq_quality_auditor import MCQQualityAuditor


CHUNKS_PATH = "data/processed/chunks.jsonl"

# Retrieval depth for claim verification.
#
# This was 3 while the retrieval benchmark that justified the retriever
# was tuned on Recall@5, so the pipeline ran one setting and was argued
# for with another. Measured on the ablation, raising it to 5 was the
# single largest accuracy change of any intervention in the project:
# coverage 38.46% -> 53.85% and accuracy 30.77% -> 38.46% with claims and
# prompt held fixed. Four claims that had been INSUFFICIENT resolved
# purely because the supporting page was now in range.
#
# Worth noting how dull the winning fix was. It is one integer, not an
# architecture, and it beat both of the designed interventions.
CLAIM_TOP_K = 5

# Retrieval depth for whole-question answer-key verification.
ANSWER_TOP_K = 5


@lru_cache(maxsize=1)
def get_chunks() -> tuple:
    """
    Load the corpus once per process.

    Both verification nodes used to call load_chunks and rebuild a
    ClaimRetriever on every invocation, so a single revision loop parsed
    the corpus and re-embedded it four times over. Returned as a tuple
    because lru_cache requires the retriever factory's argument to be
    hashable.
    """

    return tuple(load_chunks(CHUNKS_PATH))


@lru_cache(maxsize=1)
def get_retriever() -> ClaimRetriever:
    """Build the retriever once; index construction is the expensive part."""

    return ClaimRetriever(list(get_chunks()))


@lru_cache(maxsize=1)
def get_fact_verifier() -> FactVerifier:
    """
    Build the fact verifier once, with the corpus attached.

    Passing chunks is not optional in practice. FactVerifier only
    constructs its NegativeClaimChecker when given them, so the previous
    bare FactVerifier() call here silently disabled absence-claim
    handling: an absence claim fell through to the LLM path, where "the
    evidence does not mention X" is exactly the inference rules 13-17 of
    that prompt exist to forbid. The evaluation harness passed chunks;
    this node did not, so the orchestrated pipeline was weaker than the
    thing being measured.
    """

    return FactVerifier(list(get_chunks()))


def generate_mcq(state: MCQVerificationState) -> dict:
    generator = MCQGenerator()

    failure_reasons = state.get("failure_reasons", [])

    mcq = generator.generate(
        topic=state["topic"],
        difficulty=state.get("difficulty", "medium"),
        failure_reasons=failure_reasons,
        question_format=state.get("question_format", FORMAT_SIMPLE),
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
    retriever = get_retriever()
    verifier = get_fact_verifier()

    claim_evidence = {}
    fact_verifications = {}

    for claim in state["claims"]:
        evidence = retriever.retrieve(
            claim.claim,
            top_k=CLAIM_TOP_K,
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
    retriever = get_retriever()
    verifier = AnswerKeyVerifier()

    evidence = retriever.retrieve(
        state["mcq"].question,
        top_k=ANSWER_TOP_K,
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
    #
    # The bound is `<=`, not `<`, and the difference was a real off-by-one.
    # `retry_count` is incremented in generate_mcq *before* the gates run
    # (nodes.py:86), so by the time it is read here it counts attempts
    # made, not retries taken: it is 1 on the first pass, never 0. Compared
    # with `<`, a max_retries of 2 therefore granted exactly one revision
    # and the loop could never reach a third attempt - measured over 15
    # topics, every REJECT terminated at attempt 2 and attempt 3 never
    # occurred. That silently halved the revision budget the loop claimed
    # to have, and understated the repair rate, which is the one metric
    # that says whether feeding failure_reasons back into the prompt does
    # anything at all.
    #
    # The counter's name is the trap. Renaming it would touch state.py and
    # every caller, so it is documented instead: retry_count counts
    # attempts, max_retries counts retries, and attempts = retries + 1.
    if not failure_reasons:
        decision = "ACCEPT"

    elif state.get("retry_count", 0) <= state.get("max_retries", 2):
        decision = "REVISE"

    else:
        decision = "REJECT"

    return {
        "decision": decision,
        "failure_reasons": failure_reasons,
    }