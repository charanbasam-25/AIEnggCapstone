"""
Run the claim-level verification pipeline over the 2025 Polity PYQs.

What changed here, and why
--------------------------

This module was the project's main measurement instrument, and four
things in it were distorting what it measured.

1. It retrieved 3 chunks per claim while the retrieval benchmark that
   justified the retriever was tuned on Recall@5. Raising it to 5 turned
   out to be the largest single accuracy change in the project, ahead of
   both deliberately designed interventions.

2. It carried its own copy of the option-mapping logic, and that copy
   contained the bugs `src/evaluation/answer_mapping.py` was written to
   fix: `negated` was computed and then ignored, so "Neither I nor II"
   could only match by accident, and "None" parsed as neither a set nor
   a count so it could never match at all. The duplicates are now gone
   and the fixed module is imported, which also means the abstention
   policy is a named argument rather than an implicit gate.

3. It scored an abstention as an incorrect answer. Those are not the same
   failure: an abstention costs coverage, a wrong answer costs a user a
   wrong fact. Status is now three-way and the summary reports selective
   prediction metrics.

4. It built claims by copying each numbered statement verbatim. Measured
   over the same 37 claims, that leaves 70.27% of them propositional and
   produces 8-9 SUPPORTED verdicts on text with no truth value; binding
   the question's predicate onto each item gives 100% and zero. The
   deterministic builder is now the default.

Reproducibility, stated rather than hidden. Two things worth knowing:

The file on disk is current. verified_pyq_results.json was regenerated
under the defaults below (CLAIM_MODE_BOUND, top_k=5, STRICT) with
temperature pinned, and every downstream report was regenerated after it
via src/evaluation/run_all.py, which refuses to finish if any report is
older than an input it was derived from. The earlier verbatim/top_k=3 run
is preserved at verified_pyq_results.k3_verbatim.json because
system_c_pipeline.py replays it as the C0 arm, and the pre-pinning run is
preserved at verified_pyq_results.unpinned.json.

Re-running will not reproduce it exactly. Pinning temperature to 0 cut
verdict instability from 9 of 37 claims to 2, but did not remove it:
src/evaluation/verdict_stability.py measures a residual accuracy spread
of 15.38 points - two questions - across independent repeats of this
pipeline. Read any single run's accuracy as a draw from that range, not
as a point estimate. The error-rate gap against vanilla RAG is the
headline because it is roughly six questions wide and survives it.
"""

import json
from pathlib import Path

from src.evaluation.answer_mapping import (
    STRICT,
    determine_statement_statuses,
    map_answer,
)
from src.evaluation.label_audit import audit_status, corrected_label
from src.evaluation.system_b_metrics import percent, selective_metrics
from src.generation.mcq_generator import MCQ
from src.retrieval.retrieval_config import RETRIEVAL_TOP_K
from src.retrieval.semantic_reranker import load_chunks
from src.verification.answer_key_verifier import AnswerKeyVerifier
from src.verification.claim_builder import build_claims_for_question
from src.verification.claim_retriever import ClaimRetriever
from src.verification.fact_verifier import FactVerifier
from src.evaluation.pyq_statement_parser import extract_numbered_statements


DATASET_PATH = "data/evaluation/pyq_2025_polity.json"
CHUNKS_PATH = "data/processed/chunks.jsonl"
OUTPUT_PATH = "data/evaluation/verified_pyq_results.json"

# Retrieval depth per claim. See note 1 in the module docstring: this was
# 3, which is the defect that cost the most. Imported rather than written
# here so that it cannot drift from the depth the retrieval benchmarks
# measure; run_all.py --check asserts the value recorded in the stored
# results still matches it.
CLAIM_TOP_K = RETRIEVAL_TOP_K

# Retrieval depth for the whole-question answer-key verifier.
ANSWER_TOP_K = RETRIEVAL_TOP_K

CLAIM_MODE_BOUND = "bound"
CLAIM_MODE_VERBATIM = "verbatim"

# Default to bound claims. The verbatim mode is kept because it is the
# setting the stored results were produced under, so it has to remain
# reachable for anyone reproducing them.
CLAIM_MODE = CLAIM_MODE_BOUND

# Abstention policy used for the headline numbers. STRICT and ELIMINATION
# measured identically on this data; CLOSED_WORLD is reported for
# risk-coverage comparison only and is an assumption, not a result.
POLICY = STRICT


class EvaluationClaim:
    """
    Lightweight claim object compatible with the existing
    verification pipeline.
    """

    def __init__(
        self,
        claim_id: str,
        statement_number: str,
        claim: str,
        binding: str = "none",
        is_propositional: bool = True,
        unsupported_tokens: "list[str] | None" = None,
    ):
        self.claim_id = claim_id
        self.statement_number = statement_number
        self.group_id = "main"
        self.claim = claim
        self.binding = binding
        self.is_propositional = is_propositional
        self.unsupported_tokens = unsupported_tokens or []

    def model_dump(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "statement_number": self.statement_number,
            "group_id": self.group_id,
            "claim": self.claim,
            "binding": self.binding,
            "is_propositional": self.is_propositional,
            "unsupported_tokens": self.unsupported_tokens,
        }


def pyq_to_mcq(question: dict) -> MCQ:
    options = question["options"]

    return MCQ(
        question=question["question_text"],
        option_a=options["A"],
        option_b=options["B"],
        option_c=options["C"],
        option_d=options["D"],
        correct_answer=question["official_answer"],
        explanation="Official UPSC PYQ used for evaluation.",
    )


def build_pyq_claims(
    question_text: str,
    mode: str = CLAIM_MODE,
) -> list[EvaluationClaim]:
    """
    Turn a question's numbered statements into claims.

    CLAIM_MODE_BOUND binds the question's predicate onto each item so the
    claim states a proposition. CLAIM_MODE_VERBATIM copies the item, which
    is what produced the stored results and is retained only to reproduce
    them.

    claim_id keeps the original `claim_{index}` form in both modes. It is
    the key under which verdicts are stored in verified_pyq_results.json,
    and several downstream modules join on it.
    """

    if mode not in (CLAIM_MODE_BOUND, CLAIM_MODE_VERBATIM):
        raise ValueError(f"Unknown claim mode: {mode}")

    if mode == CLAIM_MODE_BOUND:

        _, built = build_claims_for_question(question_text)

        return [
            EvaluationClaim(
                claim_id=f"claim_{index}",
                statement_number=item.label,
                claim=item.claim,
                binding=item.binding,
                is_propositional=item.is_propositional,
                unsupported_tokens=item.unsupported,
            )
            for index, item in enumerate(built, start=1)
        ]

    statements = extract_numbered_statements(
        question_text
    )

    claims = []

    for index, statement in enumerate(
        statements,
        start=1,
    ):

        claims.append(
            EvaluationClaim(
                claim_id=f"claim_{index}",
                statement_number=statement[
                    "statement_number"
                ],
                claim=statement["text"],
            )
        )

    return claims


def get_statement_claims(
    claims: list,
) -> dict[str, list[str]]:

    statement_claims: dict[str, list[str]] = {}

    for claim in claims:

        if claim.statement_number is None:
            continue

        statement = (
            claim.statement_number.upper()
        )

        statement_claims.setdefault(
            statement,
            [],
        ).append(
            claim.claim_id
        )

    return statement_claims


def determine_answer_from_claims(
    claims: list,
    fact_verifications: dict,
    options: dict,
    question_text: str,
    policy: str = POLICY,
) -> "tuple[str | None, dict]":
    """
    Aggregate verdicts into statement statuses and map them to an option.

    Both steps are delegated to src/evaluation/answer_mapping.py. This
    file used to carry its own copy of each, and the copies were the
    buggy ones: `negated` was computed and discarded so "Neither I nor
    II" matched only by accident, "None" parsed as neither a statement
    set nor a count so it could never match, and abstention was hard-wired
    rather than being a policy the caller chooses. Keeping a second
    implementation is what let those defects persist while the fixed
    version sat one import away.
    """

    statement_claims = get_statement_claims(claims)

    statement_statuses = determine_statement_statuses(
        claims,
        fact_verifications,
    )

    predicted_answer, mapping_debug = map_answer(
        statement_statuses=statement_statuses,
        options=options,
        question_text=question_text,
        policy=policy,
    )

    # Preserved for the stored records, which index claims by statement.
    mapping_debug["statement_claims"] = statement_claims

    # The old field name, kept so downstream readers of existing result
    # files do not have to branch on schema version. It means what it
    # always meant: nothing was left unresolved.
    mapping_debug["complete_evidence"] = not mapping_debug[
        "insufficient_statements"
    ]

    return predicted_answer, mapping_debug


STATUS_CORRECT = "CORRECT"
STATUS_INCORRECT = "INCORRECT"
STATUS_ABSTAINED = "ABSTAINED"


def score_status(
    predicted_answer: "str | None",
    label: str,
) -> str:
    """
    Three-way outcome for one question.

    This used to be two-way, with `predicted_answer is None` falling into
    INCORRECT. That made the headline number unable to express the thing
    the architecture was built to do. Declining to answer costs coverage;
    asserting a wrong constitutional fact costs a user. Collapsing them
    meant every abstention the verifier added looked like a regression,
    which is the wrong gradient for a system whose stated purpose is to
    refuse to guess.
    """

    if predicted_answer is None:
        return STATUS_ABSTAINED

    if predicted_answer == label:
        return STATUS_CORRECT

    return STATUS_INCORRECT


def report_scoring(
    results: list[dict],
    label_field: str,
    title: str,
) -> dict:
    """Print and return selective-prediction metrics for one label set."""

    metrics = selective_metrics(
        [
            (result["predicted_answer"], result[label_field])
            for result in results
        ]
    )

    print()
    print(f"--- {title} ---")

    print(
        f"Total:                   "
        f"{metrics['total']}"
    )

    print(
        f"Answered:                "
        f"{metrics['answered']}"
    )

    print(
        f"Abstained:               "
        f"{metrics['abstained']}"
    )

    print(
        f"Correct:                 "
        f"{metrics['correct']}"
    )

    print(
        f"Wrong:                   "
        f"{metrics['wrong']}"
    )

    print(
        f"Coverage:                "
        f"{percent(metrics['coverage'])}"
    )

    print(
        f"Precision when answered: "
        f"{percent(metrics['precision_when_answered'])}"
    )

    print(
        f"Accuracy overall:        "
        f"{percent(metrics['accuracy_overall'])}"
    )

    print(
        f"Error rate overall:      "
        f"{percent(metrics['error_rate_overall'])}"
    )

    return metrics


def main() -> "dict | None":

    dataset = json.loads(
        Path(DATASET_PATH).read_text(
            encoding="utf-8"
        )
    )

    chunks = load_chunks(
        CHUNKS_PATH
    )

    claim_retriever = ClaimRetriever(
        chunks
    )

    fact_verifier = FactVerifier(chunks)

    answer_key_verifier = (
        AnswerKeyVerifier()
    )

    results = []

    questions = dataset[
        "questions"
    ]

    for index, question in enumerate(
        questions,
        start=1,
    ):

        print(
            f"Evaluating {index}/{len(questions)}: "
            f"Q{question['q_number']}"
        )

        mcq = pyq_to_mcq(
            question
        )

        # ==================================================
        # 1. Deterministic PYQ statement extraction
        # ==================================================

        claims = build_pyq_claims(
            question["question_text"]
        )

        if not claims:

            raise ValueError(
                f"No numbered statements extracted "
                f"for Q{question['q_number']}"
            )

        # ==================================================
        # 2. Independently verify every statement
        # ==================================================

        claim_evidence = {}
        fact_verifications = {}

        for claim in claims:

            evidence = (
                claim_retriever.retrieve(
                    claim.claim,
                    top_k=CLAIM_TOP_K,
                )
            )

            verification = (
                fact_verifier.verify(
                    claim.claim,
                    evidence,
                )
            )

            claim_evidence[
                claim.claim_id
            ] = evidence

            fact_verifications[
                claim.claim_id
            ] = verification

        # ==================================================
        # 3. Whole-question verifier retained for comparison
        # ==================================================

        answer_evidence = (
            claim_retriever.retrieve(
                mcq.question,
                top_k=ANSWER_TOP_K,
            )
        )

        answer_verification = (
            answer_key_verifier.verify(
                mcq,
                answer_evidence,
            )
        )

        # ==================================================
        # 4. Deterministic answer mapping
        # ==================================================

        (
            predicted_answer,
            claim_mapping,
        ) = determine_answer_from_claims(
            claims=claims,
            fact_verifications=fact_verifications,
            options=question["options"],
            question_text=question[
                "question_text"
            ],
        )

        stored_label = question["official_answer"]

        audited_label = corrected_label(
            question["q_number"],
            stored_label,
        )

        results.append(
            {
                "q_number":
                    question["q_number"],
                "question":
                    question["question_text"],
                "options":
                    question["options"],
                "official_answer":
                    stored_label,
                "audited_answer":
                    audited_label,
                "label_audit_status":
                    audit_status(
                        question["q_number"]
                    ),
                "predicted_answer":
                    predicted_answer,
                "status":
                    score_status(
                        predicted_answer,
                        stored_label,
                    ),
                "status_audited":
                    score_status(
                        predicted_answer,
                        audited_label,
                    ),
                "claim_mode": CLAIM_MODE,
                "policy": POLICY,
                "claim_top_k": CLAIM_TOP_K,
                "claims": [
                    claim.model_dump()
                    for claim in claims
                ],
                "fact_verifications": {
                    claim_id:
                        result.model_dump()
                    for claim_id, result
                    in fact_verifications.items()
                },
                "claim_mapping":
                    claim_mapping,
                "answer_verification":
                    answer_verification.model_dump(),
                "answer_evidence":
                    answer_evidence,
                "claim_evidence":
                    claim_evidence,
            }
        )

    Path(OUTPUT_PATH).write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=== VERIFIED PYQ EVALUATION ==="
    )

    print(
        f"claim mode: {CLAIM_MODE}   "
        f"policy: {POLICY}   "
        f"claim top_k: {CLAIM_TOP_K}   "
        f"answer top_k: {ANSWER_TOP_K}"
    )

    if not results:

        print("\nNo questions evaluated.")

        return

    stored = report_scoring(
        results,
        "official_answer",
        "stored labels",
    )

    # Both scorings are printed because one of the 13 stored labels is
    # wrong (Q58; see src/evaluation/label_audit.py). Reporting only the
    # audited numbers would look like grading against a key chosen after
    # seeing the predictions, and reporting only the stored ones scores a
    # correct answer as a failure. Printing both makes the size of the
    # correction visible instead of absorbing it.
    audited = report_scoring(
        results,
        "audited_answer",
        "audited labels",
    )

    abstained = [
        result["q_number"]
        for result in results
        if result["status"] == STATUS_ABSTAINED
    ]

    wrong = [
        result["q_number"]
        for result in results
        if result["status_audited"] == STATUS_INCORRECT
    ]

    print()

    print(
        f"Abstained on:            {abstained}"
    )

    print(
        f"Wrong (audited labels):  {wrong}"
    )

    non_propositional = [
        claim["claim"]
        for result in results
        for claim in result["claims"]
        if not claim["is_propositional"]
    ]

    invented = [
        claim["claim"]
        for result in results
        for claim in result["claims"]
        if claim["unsupported_tokens"]
    ]

    total_claims = sum(
        len(result["claims"]) for result in results
    )

    # Claim soundness is measured over every claim rather than every
    # question, and it needs no gold label, so it is the one quality
    # signal the Q58 label error cannot contaminate.
    print()

    print(
        f"Claims:                  {total_claims}"
    )

    print(
        f"Propositional:           "
        f"{total_claims - len(non_propositional)}/{total_claims}"
    )

    print(
        f"Claims with new tokens:  {len(invented)}"
    )

    for claim_text in non_propositional:
        print(f"  not propositional: {claim_text}")

    for claim_text in invented:
        print(f"  invented tokens:   {claim_text}")

    print()

    print(
        f"Saved results to: "
        f"{OUTPUT_PATH}"
    )

    return {
        "stored_labels": stored,
        "audited_labels": audited,
    }


if __name__ == "__main__":
    main()
