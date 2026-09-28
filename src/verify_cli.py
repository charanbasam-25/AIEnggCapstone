"""
Verify one MCQ from the command line.

Why this file exists
--------------------

Until now the verifier had no front door. There were exactly two ways to
put a question into it, and neither was usable by a person:

  1. src/evaluation/evaluate_verified_pyqs.py reads a fixed dataset file
     and loops over all 13 questions. To check a new question you had to
     edit data/evaluation/pyq_2025_polity.json.

  2. src/orchestration/run_workflow.py has the topic hardcoded, and it
     *generates* a question rather than accepting one.

That is a reasonable state for a benchmark and an indefensible one for a
system whose claim is "it verifies MCQs". This module closes the gap: one
question in, verdicts and a decision out.

What it deliberately does not do
--------------------------------

It does not reimplement anything. Claim construction, retrieval,
verification and option mapping are all imported from the modules the
evaluation harness uses, so a number printed here and a number in
data/evaluation/verified_pyq_results.json come from the same code.

That constraint is not stylistic. evaluate_verified_pyqs.py used to carry
its own copy of the mapping logic, and the copy was the buggy one: the
`negated` flag was computed and then discarded, so "Neither I nor II"
matched only by accident, and "None" parsed as neither a statement set nor
a count so it could never match at all. A second entry point that
re-derived "which option do these verdicts imply" would recreate exactly
that defect, in the one place a reader is most likely to look.

Why it lives at src/ root rather than in a package
--------------------------------------------------

It imports from src.verification (the verifier), src.retrieval (the
index) and src.evaluation (claim building and mapping). A module that
depends on every layer is a composition root, and putting it inside any
one of those packages would create a dependency pointing the wrong way -
src.verification importing src.evaluation in particular.

Usage
-----

    # one of the 13 stored 2025 Prelims questions, by its number
    python -m src.verify_cli --pyq 54

    # a question pasted as plain text (ends at EOF: Ctrl-Z then Enter)
    python -m src.verify_cli

    # the same text from a file
    python -m src.verify_cli --file my_question.txt

    # exact input, no text parsing
    python -m src.verify_cli --json my_question.json

    # show what a different abstention policy would have answered
    python -m src.verify_cli --pyq 54 --policy closed_world

The plain-text format is what you would paste out of a question paper:

    Consider the following statements:
    I. The President is elected by direct election.
    II. The Vice-President is the ex-officio Chairman of the Rajya Sabha.
    Which of the statements given above is/are correct?
    (a) I only
    (b) II only
    (c) Both I and II
    (d) Neither I nor II

Everything before the first option line is the question; the four option
lines must be labelled a-d or A-D in order. Roman-numeral statement
labels are never mistaken for option labels because the option pattern
only accepts A-D.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from src.evaluation.answer_mapping import (
    CLOSED_WORLD,
    ELIMINATION,
    POLICIES,
    STRICT,
)
from src.evaluation.evaluate_verified_pyqs import (
    ANSWER_TOP_K,
    CLAIM_MODE,
    CLAIM_MODE_BOUND,
    CLAIM_MODE_VERBATIM,
    CLAIM_TOP_K,
    CHUNKS_PATH,
    DATASET_PATH,
    build_pyq_claims,
    determine_answer_from_claims,
    pyq_to_mcq,
)
from src.retrieval.semantic_reranker import load_chunks
from src.verification.answer_key_verifier import AnswerKeyVerifier
from src.verification.claim_retriever import ClaimRetriever
from src.verification.fact_verifier import FactVerifier

# Only A-D. A statement labelled "I." or "III." cannot collide with this,
# which is the whole reason the option label set is kept this narrow.
OPTION_LINE = re.compile(
    r"^\s*\(?\s*([A-Da-d])\s*[\).:\-]\s*(\S.*?)\s*$"
)

OPTION_LETTERS = ("A", "B", "C", "D")

# Human wording for the typed reasons produced by answer_mapping. The
# abstention is the product, so it gets a sentence rather than a code.
ABSTENTION_WORDING = {
    "policy_insufficient_evidence": (
        "at least one statement could not be settled from the corpus, "
        "and the strict policy declines rather than guesses"
    ),
    "no_option_matched": (
        "the verdicts are internally consistent but no option describes "
        "them, which usually means a statement was misread or the "
        "official key is disputable"
    ),
    "ambiguous_match": (
        "more than one option matches the verdicts, so answering would "
        "mean picking arbitrarily"
    ),
    "multiple_options_possible": (
        "elimination narrowed the field but more than one option is "
        "still viable"
    ),
}


def parse_plain_text(text: str) -> dict:
    """
    Split a pasted question into question_text and four options.

    Fails loudly. A half-parsed question would be verified against the
    wrong statements and still print a confident-looking verdict, which
    is worse than refusing to start.
    """

    lines = text.splitlines()

    # Find a run of option lines labelled A, B, C, D in order. Scanning
    # for the ordered run rather than for any four matches means a stray
    # "b)" earlier in the stem cannot capture the parse.
    start = None

    for index, line in enumerate(lines):

        match = OPTION_LINE.match(line)

        if match is None or match.group(1).upper() != "A":
            continue

        candidate = {}
        cursor = index

        for expected in OPTION_LETTERS:

            while cursor < len(lines) and not lines[cursor].strip():
                cursor += 1

            if cursor >= len(lines):
                break

            step = OPTION_LINE.match(lines[cursor])

            if step is None or step.group(1).upper() != expected:
                break

            candidate[expected] = step.group(2).strip()
            cursor += 1

        if len(candidate) == 4:
            start = index
            options = candidate
            break

    if start is None:
        raise SystemExit(
            "Could not find four option lines labelled (a) (b) (c) (d).\n"
            "Expected something like:\n"
            "    (a) I only\n"
            "    (b) II only\n"
            "    (c) Both I and II\n"
            "    (d) Neither I nor II\n"
            "Use --json to bypass text parsing entirely."
        )

    question_text = " ".join(
        line.strip()
        for line in lines[:start]
        if line.strip()
    )

    if not question_text:
        raise SystemExit(
            "Found the options but no question text above them."
        )

    return {
        "q_number": None,
        "question_text": question_text,
        "options": options,
        "official_answer": None,
    }


def load_from_dataset(q_number: int) -> dict:

    dataset = json.loads(
        Path(DATASET_PATH).read_text(encoding="utf-8")
    )

    for question in dataset["questions"]:
        if question["q_number"] == q_number:
            return question

    available = ", ".join(
        str(question["q_number"])
        for question in dataset["questions"]
    )

    raise SystemExit(
        f"Q{q_number} is not in {DATASET_PATH}. Available: {available}"
    )


def load_from_json(path: str) -> dict:

    question = json.loads(
        Path(path).read_text(encoding="utf-8")
    )

    missing = [
        key
        for key in ("question_text", "options")
        if key not in question
    ]

    if missing:
        raise SystemExit(
            f"{path} is missing required key(s): {missing}. Expected the "
            f"same shape as one entry in {DATASET_PATH}."
        )

    unexpected = set(question["options"]) - set(OPTION_LETTERS)

    if unexpected or len(question["options"]) != 4:
        raise SystemExit(
            f"{path}: options must be exactly the keys A, B, C, D; got "
            f"{sorted(question['options'])}"
        )

    question.setdefault("q_number", None)
    question.setdefault("official_answer", None)

    return question


def read_question(arguments: argparse.Namespace) -> dict:

    if arguments.pyq is not None:
        return load_from_dataset(arguments.pyq)

    if arguments.json:
        return load_from_json(arguments.json)

    if arguments.file:
        return parse_plain_text(
            Path(arguments.file).read_text(encoding="utf-8")
        )

    if sys.stdin.isatty():
        print(
            "Paste the question and its four options, then press "
            "Ctrl-Z and Enter (Windows) or Ctrl-D (Unix):\n"
        )

    text = sys.stdin.read()

    if not text.strip():
        raise SystemExit(
            "No input. Use --pyq N, --file PATH, --json PATH, or pipe "
            "the question in on stdin."
        )

    return parse_plain_text(text)


def print_question(question: dict) -> None:

    print("=" * 66)
    print("QUESTION")
    print("=" * 66)
    print()
    print(question["question_text"])
    print()

    for letter in OPTION_LETTERS:
        print(f"  {letter}. {question['options'][letter]}")

    print()


def print_claim_result(claim, evidence, verification, show_evidence):

    label = (
        f"statement {claim.statement_number}"
        if claim.statement_number
        else claim.claim_id
    )

    print("-" * 66)
    print(f"{label} -> {verification.verdict}")
    print("-" * 66)
    print(f"  claim:  {claim.claim}")
    print(f"  reason: {verification.reasoning}")

    pages = verification.supporting_pages

    if pages:
        print(f"  pages:  {', '.join(str(page) for page in pages)}")
    else:
        # Worth naming rather than printing an empty list: no cited page
        # is how an INSUFFICIENT verdict is supposed to look, and it is
        # also how a SUPPORTED verdict looks when it should not.
        print("  pages:  none cited")

    if show_evidence:

        print("  evidence retrieved:")

        for chunk in evidence:
            snippet = " ".join(chunk["text"].split())[:200]
            print(
                f"    [{chunk['source']} p.{chunk['page']}] {snippet}..."
            )

    print()


def print_decision(question, predicted, debug, answer_verification):

    print("=" * 66)
    print("DECISION")
    print("=" * 66)
    print()

    statuses = debug["statement_statuses"]

    for statement in sorted(statuses):
        print(f"  statement {statement}: {statuses[statement]}")

    if debug["question_asks_for_incorrect"]:
        print()
        print(
            "  note: this question asks which statements are NOT "
            "correct, so the target set is the contradicted ones"
        )

    print()

    if predicted is None:

        reason = debug["abstention_reason"]

        print("  ABSTAINED - no answer given")
        print()
        print(
            f"  why: {ABSTENTION_WORDING.get(reason, reason)}"
        )
        print()
        print(
            "  This is a designed outcome, not a crash. On the 13 "
            "audited questions this pipeline states a false answer 0 "
            "times; the unverified RAG baseline does so 7 times."
        )

    else:
        print(f"  ANSWER: {predicted}. "
              f"{question['options'][predicted]}")

        official = question.get("official_answer")

        if official:
            verdict = (
                "matches the official key"
                if predicted == official
                else f"DISAGREES with the official key ({official})"
            )
            print(f"  {verdict}")

    print()
    print("-" * 66)
    print("whole-question verifier (comparison only, not used above)")
    print("-" * 66)
    print(f"  verdict:           {answer_verification.verdict}")
    print(f"  supported options: {answer_verification.supported_options}")
    print(f"  reasoning:         {answer_verification.reasoning}")
    print()
    print(
        "  Reported because it is the architecture this project "
        "measured and rejected: one LLM call judging the whole question "
        "at once. The answer above comes from per-statement verdicts "
        "mapped by plain Python instead."
    )
    print()


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Verify one multiple-choice Polity question against the "
            "Constitution and NCERT Polity corpus."
        ),
        epilog=(
            "With no input flag, the question is read from stdin as "
            "plain text."
        ),
    )

    source = parser.add_mutually_exclusive_group()

    source.add_argument(
        "--pyq",
        type=int,
        metavar="N",
        help=f"question number from {DATASET_PATH}",
    )
    source.add_argument(
        "--json",
        metavar="PATH",
        help="JSON file with question_text and options (A-D)",
    )
    source.add_argument(
        "--file",
        metavar="PATH",
        help="plain-text file holding the question and four options",
    )

    parser.add_argument(
        "--policy",
        choices=POLICIES,
        default=STRICT,
        help=(
            f"abstention policy (default {STRICT}). "
            f"{CLOSED_WORLD} answers anyway by treating unsettled "
            f"statements as false, which measured higher accuracy and "
            f"took the error rate from 0%% to 30.77%%. {ELIMINATION} "
            f"answers only when exactly one option survives."
        ),
    )
    parser.add_argument(
        "--claim-mode",
        choices=(CLAIM_MODE_BOUND, CLAIM_MODE_VERBATIM),
        default=CLAIM_MODE,
        help=(
            f"how statements become claims (default {CLAIM_MODE}). "
            f"{CLAIM_MODE_BOUND} binds the stem's predicate onto each "
            f"item so the claim is a standalone proposition."
        ),
    )
    parser.add_argument(
        "--show-evidence",
        action="store_true",
        help="print the retrieved passages behind each verdict",
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        help="also write the full result, including debug, as JSON",
    )

    arguments = parser.parse_args()

    question = read_question(arguments)

    print_question(question)

    try:
        claims = build_pyq_claims(
            question["question_text"],
            mode=arguments.claim_mode,
        )

    except ValueError:
        # question_parser.parse_question raises on a question with no
        # numbered items rather than returning an empty list. Testing
        # `not claims` alone was wrong: the call never returned, so the
        # message below was unreachable and the user got a traceback.
        claims = []

    if not claims:
        raise SystemExit(
            "No numbered statements found, so there is nothing to "
            "decompose and verify.\n"
            "This verifier works on the 'Consider the following "
            "statements: I. ... II. ...' form. A single-fact question "
            "has no statements to check independently, which is a real "
            "limit of the design and not a parsing bug."
        )

    print(f"Loading corpus and models (first run takes ~70s)...\n")

    chunks = load_chunks(CHUNKS_PATH)

    claim_retriever = ClaimRetriever(chunks)

    # Constructed WITH chunks. A bare FactVerifier() silently disables
    # absence-claim handling, and that omission has already shipped twice
    # in this codebase.
    fact_verifier = FactVerifier(chunks)

    answer_key_verifier = AnswerKeyVerifier()

    print("=" * 66)
    print(f"PER-STATEMENT VERIFICATION ({len(claims)} statement(s))")
    print("=" * 66)
    print()

    claim_evidence = {}
    fact_verifications = {}

    for claim in claims:

        evidence = claim_retriever.retrieve(
            claim.claim,
            top_k=CLAIM_TOP_K,
        )

        verification = fact_verifier.verify(claim.claim, evidence)

        claim_evidence[claim.claim_id] = evidence
        fact_verifications[claim.claim_id] = verification

        print_claim_result(
            claim,
            evidence,
            verification,
            arguments.show_evidence,
        )

    mcq = pyq_to_mcq(
        {
            **question,
            # pyq_to_mcq expects a key to put in MCQ.correct_answer. For
            # an unlabelled question there is none, and "A" is a
            # placeholder: the whole-question verifier below reports
            # which options it finds supported, and nothing downstream
            # reads this field.
            "official_answer": question.get("official_answer") or "A",
        }
    )

    answer_evidence = claim_retriever.retrieve(
        mcq.question,
        top_k=ANSWER_TOP_K,
    )

    answer_verification = answer_key_verifier.verify(
        mcq,
        answer_evidence,
    )

    predicted_answer, mapping_debug = determine_answer_from_claims(
        claims=claims,
        fact_verifications=fact_verifications,
        options=question["options"],
        question_text=question["question_text"],
        policy=arguments.policy,
    )

    print_decision(
        question,
        predicted_answer,
        mapping_debug,
        answer_verification,
    )

    if arguments.save:

        record = {
            "q_number": question.get("q_number"),
            "question": question["question_text"],
            "options": question["options"],
            "official_answer": question.get("official_answer"),
            "policy": arguments.policy,
            "claim_mode": arguments.claim_mode,
            "predicted_answer": predicted_answer,
            "claims": [
                {
                    "claim_id": claim.claim_id,
                    "statement_number": claim.statement_number,
                    "claim": claim.claim,
                    "verdict": fact_verifications[
                        claim.claim_id
                    ].verdict,
                    "reasoning": fact_verifications[
                        claim.claim_id
                    ].reasoning,
                    "supporting_pages": fact_verifications[
                        claim.claim_id
                    ].supporting_pages,
                }
                for claim in claims
            ],
            "mapping_debug": mapping_debug,
            "answer_key_verification": {
                "verdict": answer_verification.verdict,
                "supported_options": (
                    answer_verification.supported_options
                ),
                "reasoning": answer_verification.reasoning,
            },
        }

        Path(arguments.save).write_text(
            json.dumps(record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"Saved to: {arguments.save}")


if __name__ == "__main__":
    main()
