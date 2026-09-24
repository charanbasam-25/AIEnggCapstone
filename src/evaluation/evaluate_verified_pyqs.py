import json
import re
from pathlib import Path

from src.generation.mcq_generator import MCQ
from src.retrieval.semantic_reranker import load_chunks
from src.verification.claim_retriever import ClaimRetriever
from src.verification.fact_verifier import FactVerifier
from src.verification.answer_key_verifier import AnswerKeyVerifier
from src.evaluation.pyq_statement_parser import extract_numbered_statements


DATASET_PATH = "data/evaluation/pyq_2025_polity.json"
CHUNKS_PATH = "data/processed/chunks.jsonl"
OUTPUT_PATH = "data/evaluation/verified_pyq_results.json"


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
    ):
        self.claim_id = claim_id
        self.statement_number = statement_number
        self.group_id = "main"
        self.claim = claim

    def model_dump(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "statement_number": self.statement_number,
            "group_id": self.group_id,
            "claim": self.claim,
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
) -> list[EvaluationClaim]:

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


def determine_statement_statuses(
    claims: list,
    fact_verifications: dict,
) -> dict[str, str]:

    statement_claims = get_statement_claims(
        claims
    )

    statuses = {}

    for statement, claim_ids in statement_claims.items():

        verdicts = [
            fact_verifications[
                claim_id
            ].verdict
            for claim_id in claim_ids
        ]

        if any(
            verdict == "CONTRADICTED"
            for verdict in verdicts
        ):

            statuses[
                statement
            ] = "CONTRADICTED"

        elif all(
            verdict == "SUPPORTED"
            for verdict in verdicts
        ):

            statuses[
                statement
            ] = "SUPPORTED"

        else:

            statuses[
                statement
            ] = "INSUFFICIENT"

    return statuses


def parse_statement_set(
    option_text: str,
) -> tuple[set[str], bool] | None:

    text = option_text.upper().strip()

    if "NEITHER" in text:

        matches = re.findall(
            r"\b(?:I|II|III|IV)\b",
            text,
        )

        statements = set(matches)

        if statements:
            return statements, True

    if "NONE OF THE ABOVE" in text:

        return set(), True

    matches = re.findall(
        r"\b(?:I|II|III|IV)\b",
        text,
    )

    if not matches:
        return None

    return set(matches), False


def parse_count_option(
    option_text: str,
) -> int | None:

    text = option_text.upper().strip()

    patterns = {
        1: ["ONLY ONE"],
        2: ["ONLY TWO"],
        3: [
            "ONLY THREE",
            "ALL THE THREE",
            "ALL THREE",
        ],
        4: [
            "ONLY FOUR",
            "ALL THE FOUR",
            "ALL FOUR",
        ],
    }

    for count, phrases in patterns.items():

        for phrase in phrases:

            if phrase in text:
                return count

    return None


def question_asks_for_incorrect(
    question_text: str,
) -> bool:

    text = question_text.upper()

    return any(
        phrase in text
        for phrase in [
            "NOT CORRECT",
            "INCORRECT",
            "NOT CORRECTLY",
        ]
    )


def determine_answer_from_claims(
    claims: list,
    fact_verifications: dict,
    options: dict,
    question_text: str,
) -> tuple[str | None, dict]:

    statement_claims = get_statement_claims(
        claims
    )

    statement_statuses = (
        determine_statement_statuses(
            claims,
            fact_verifications,
        )
    )

    all_statements = set(
        statement_claims.keys()
    )

    supported_statements = {
        statement
        for statement, status
        in statement_statuses.items()
        if status == "SUPPORTED"
    }

    contradicted_statements = {
        statement
        for statement, status
        in statement_statuses.items()
        if status == "CONTRADICTED"
    }

    insufficient_statements = {
        statement
        for statement, status
        in statement_statuses.items()
        if status == "INSUFFICIENT"
    }

    asks_for_incorrect = (
        question_asks_for_incorrect(
            question_text
        )
    )

    target_statements = (
        contradicted_statements
        if asks_for_incorrect
        else supported_statements
    )

    complete_evidence = (
        len(insufficient_statements) == 0
        and len(statement_statuses)
        == len(all_statements)
    )

    option_mapping = {}
    matching_options = []

    for option_letter, option_text in options.items():

        statement_option = parse_statement_set(
            option_text
        )

        if statement_option is not None:

            required_statements, is_negated = (
                statement_option
            )

            option_mapping[
                option_letter
            ] = {
                "type": "explicit",
                "required_statements": sorted(
                    required_statements
                ),
                "negated": is_negated,
            }

            if complete_evidence:

                if (
                    required_statements
                    == target_statements
                ):

                    matching_options.append(
                        option_letter
                    )

            continue

        required_count = (
            parse_count_option(
                option_text
            )
        )

        if required_count is not None:

            option_mapping[
                option_letter
            ] = {
                "type": "count",
                "required_count": required_count,
                "counts": (
                    "contradicted"
                    if asks_for_incorrect
                    else "supported"
                ),
            }

            if complete_evidence:

                actual_count = len(
                    target_statements
                )

                if (
                    actual_count
                    == required_count
                ):

                    matching_options.append(
                        option_letter
                    )

            continue

        option_mapping[
            option_letter
        ] = {
            "type": "unknown",
            "text": option_text,
        }

    predicted_answer = None

    if len(matching_options) == 1:
        predicted_answer = matching_options[0]

    mapping_debug = {
        "question_asks_for_incorrect":
            asks_for_incorrect,
        "complete_evidence":
            complete_evidence,
        "statement_claims":
            statement_claims,
        "statement_statuses":
            statement_statuses,
        "supported_statements":
            sorted(supported_statements),
        "contradicted_statements":
            sorted(contradicted_statements),
        "insufficient_statements":
            sorted(insufficient_statements),
        "target_statements":
            sorted(target_statements),
        "option_mapping":
            option_mapping,
        "matching_options":
            matching_options,
    }

    return (
        predicted_answer,
        mapping_debug,
    )


def main() -> None:

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
                    top_k=3,
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
                top_k=5,
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

        status = (
            "CORRECT"
            if predicted_answer
            == question["official_answer"]
            else "INCORRECT"
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
                    question["official_answer"],
                "predicted_answer":
                    predicted_answer,
                "status":
                    status,
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

    correct = sum(
        result["status"]
        == "CORRECT"
        for result in results
    )

    incorrect = sum(
        result["status"]
        == "INCORRECT"
        for result in results
    )

    print()
    print(
        "=== VERIFIED PYQ EVALUATION ==="
    )

    print(
        f"Total:     {len(results)}"
    )

    print(
        f"Correct:   {correct}"
    )

    print(
        f"Incorrect: {incorrect}"
    )

    if results:

        print(
            f"Accuracy:  "
            f"{correct / len(results):.2%}"
        )

    else:

        print(
            "Accuracy:  N/A"
        )

    print()

    print(
        f"Saved results to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
