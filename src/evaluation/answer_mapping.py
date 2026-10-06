"""
Deterministic mapping from statement verdicts to an answer option.

This module extracts and fixes the mapping logic that currently lives
inside evaluate_verified_pyqs.py, and makes the abstention policy an
explicit parameter instead of a hidden all-or-nothing gate.

Three defects in the original logic are corrected here.

1. The `negated` flag was computed for options such as
   "Neither I nor II" and then never used. Options were matched with
   `required_statements == target_statements`, so a negated option could
   only match by accident. A negated option is now correctly treated as
   "none of the listed statements are in the target set".

2. The option text "None" did not parse as either a statement set or a
   count, so it fell through to type "unknown" and could never match.
   Q56 and Q64 both use it. It now parses as count 0.

3. Abstention was hard-wired: if any statement came back INSUFFICIENT,
   `complete_evidence` was False and no option was ever evaluated, so
   the system abstained. That is a defensible policy but it was applied
   silently and was not comparable against a baseline that always
   answers. It is now a named policy.

Policies
--------

STRICT
    Abstain if any statement is INSUFFICIENT. Only fully resolved
    questions are answered. This is the original behaviour.

CLOSED_WORLD
    Treat INSUFFICIENT as "not established", so an unresolved statement
    is simply absent from the supported set.

    This is an assumption, not a finding. It is the "absence of evidence
    is evidence of absence" move that the development log warns against,
    so it must never be presented as a verification result. It is
    included only to quantify how much accuracy the strict policy gives
    up, i.e. to produce a risk-coverage comparison.

ELIMINATION
    Keep every option that is logically consistent with the verdicts, and
    answer only if exactly one survives.

    This exists because STRICT throws away usable information. STRICT
    abstains the moment any statement is INSUFFICIENT, even when the
    verdicts already rule out every option but one. An unresolved
    statement genuinely leaves several answers open sometimes, but not
    always, and STRICT cannot tell those two situations apart.

    An option is eliminated only by a verdict that contradicts it:

        - it omits a statement that is established as being in the
          target set, or
        - it includes a statement that is established as being outside
          the target set, or
        - it states a count outside the range still reachable given how
          many statements remain unresolved.

    INSUFFICIENT never counts for or against membership, so this makes no
    closed-world assumption. When nothing is INSUFFICIENT the policy
    reduces exactly to STRICT, so it is a strict generalisation of it and
    cannot answer fewer questions.

The verifier itself is unchanged. Nothing here edits a verdict; this
module only decides how to act on the verdicts it is given.
"""

import re


STRICT = "strict"
CLOSED_WORLD = "closed_world"
ELIMINATION = "elimination"

POLICIES = (STRICT, CLOSED_WORLD, ELIMINATION)

STATEMENT_TOKEN = r"\b(?:I|II|III|IV|V)\b"

NONE_TEXTS = frozenset(
    {
        "NONE",
        "NONE OF THE ABOVE",
        "NONE OF THEM",
    }
)

COUNT_PHRASES = {
    1: ("ONLY ONE",),
    2: ("ONLY TWO",),
    3: ("ONLY THREE", "ALL THE THREE", "ALL THREE"),
    4: ("ONLY FOUR", "ALL THE FOUR", "ALL FOUR"),
}

INCORRECT_PHRASES = (
    "NOT CORRECT",
    "INCORRECT",
    "NOT CORRECTLY",
)


def question_asks_for_incorrect(question_text: str) -> bool:
    """True when the question asks which statements are NOT correct."""

    text = question_text.upper()

    return any(phrase in text for phrase in INCORRECT_PHRASES)


def parse_statement_set(
    option_text: str,
) -> tuple[set[str], bool] | None:
    """
    Parse an option that names specific statements.

    Returns (statements, negated) or None when the option does not name
    statements.

    "I and III only"   -> ({"I", "III"}, False)
    "Neither I nor II" -> ({"I", "II"}, True)
    "None"             -> (set(), True)
    """

    text = option_text.upper().strip()

    if text in NONE_TEXTS:
        return set(), True

    statements = set(re.findall(STATEMENT_TOKEN, text))

    if not statements:
        return None

    negated = "NEITHER" in text or "NONE" in text

    return statements, negated


def parse_count_option(option_text: str) -> int | None:
    """
    Parse an option that states how many statements are correct.

    Returns the required count, or None when the option is not a count.
    """

    text = option_text.upper().strip()

    if text in NONE_TEXTS:
        return 0

    for count, phrases in COUNT_PHRASES.items():
        if any(phrase in text for phrase in phrases):
            return count

    return None


def determine_statement_statuses(
    claims: list,
    fact_verifications: dict,
) -> dict[str, str]:
    """
    Aggregate per-claim verdicts into one status per statement.

    A statement is CONTRADICTED if any of its claims is contradicted,
    SUPPORTED only if every claim is supported, and INSUFFICIENT
    otherwise.
    """

    statement_claims: dict[str, list[str]] = {}

    for claim in claims:

        if claim.statement_number is None:
            continue

        statement_claims.setdefault(
            claim.statement_number.upper(),
            [],
        ).append(claim.claim_id)

    statuses = {}

    for statement, claim_ids in statement_claims.items():

        verdicts = [
            fact_verifications[claim_id].verdict
            for claim_id in claim_ids
        ]

        if any(verdict == "CONTRADICTED" for verdict in verdicts):
            statuses[statement] = "CONTRADICTED"

        elif all(verdict == "SUPPORTED" for verdict in verdicts):
            statuses[statement] = "SUPPORTED"

        else:
            statuses[statement] = "INSUFFICIENT"

    return statuses


def explicit_option_survives(
    required_statements: set[str],
    is_negated: bool,
    definitely_in: set[str],
    definitely_out: set[str],
) -> bool:
    """
    Decide whether an option naming specific statements is still possible.

    An unresolved statement is neither required nor forbidden, so it
    cannot eliminate anything. Only a definite verdict can:

        omitting a statement known to be in the target set  -> impossible
        including a statement known to be outside it        -> impossible

    A negated option such as "Neither I nor II" asserts the target set is
    empty, which survives exactly while nothing is known to be in it.
    """

    if is_negated:
        return not definitely_in

    return definitely_in <= required_statements and not (
        required_statements & definitely_out
    )


def count_option_survives(
    required_count: int,
    definitely_in: set[str],
    unresolved: set[str],
) -> bool:
    """
    Decide whether a count option is still reachable.

    The true count is at least the number of statements already
    established as being in the target set, and at most that plus the
    number still unresolved. Any count in between remains possible.
    """

    return (
        len(definitely_in)
        <= required_count
        <= len(definitely_in) + len(unresolved)
    )


def map_answer(
    statement_statuses: dict[str, str],
    options: dict[str, str],
    question_text: str,
    policy: str = STRICT,
) -> tuple[str | None, dict]:
    """
    Choose the option implied by the statement verdicts.

    Returns (predicted_answer, debug). predicted_answer is None when the
    policy declines to answer, or when zero or several options match.

    An ambiguous match is deliberately reported as an abstention rather
    than resolved by picking the first candidate, so that mapping
    ambiguity cannot be mistaken for a confident answer.
    """

    if policy not in POLICIES:
        raise ValueError(
            f"Unknown policy: {policy}. Expected one of {POLICIES}."
        )

    supported = {
        statement
        for statement, status in statement_statuses.items()
        if status == "SUPPORTED"
    }

    contradicted = {
        statement
        for statement, status in statement_statuses.items()
        if status == "CONTRADICTED"
    }

    insufficient = {
        statement
        for statement, status in statement_statuses.items()
        if status == "INSUFFICIENT"
    }

    asks_for_incorrect = question_asks_for_incorrect(question_text)

    target_statements = (
        contradicted if asks_for_incorrect else supported
    )

    # Which verdict places a statement inside the target set depends on
    # what the question asks for. When it asks which statements are NOT
    # correct, a CONTRADICTED statement belongs in the answer and a
    # SUPPORTED one is excluded from it.
    definitely_in = target_statements
    definitely_out = supported if asks_for_incorrect else contradicted

    abstained_by_policy = policy == STRICT and bool(insufficient)

    option_mapping = {}
    matching_options = []

    for option_letter, option_text in options.items():

        # Count options are checked first. "None" is both a valid count
        # of zero and a degenerate statement set, and reading it as a
        # count keeps it comparable with "Only one" / "Only two".
        required_count = parse_count_option(option_text)

        if required_count is not None:

            option_mapping[option_letter] = {
                "type": "count",
                "required_count": required_count,
                "counts": (
                    "contradicted"
                    if asks_for_incorrect
                    else "supported"
                ),
            }

            if policy == ELIMINATION:
                matched = count_option_survives(
                    required_count, definitely_in, insufficient
                )

            else:
                matched = (
                    not abstained_by_policy
                    and len(target_statements) == required_count
                )

            if matched:
                matching_options.append(option_letter)

            continue

        statement_option = parse_statement_set(option_text)

        if statement_option is not None:

            required_statements, is_negated = statement_option

            option_mapping[option_letter] = {
                "type": "explicit",
                "required_statements": sorted(required_statements),
                "negated": is_negated,
            }

            if policy == ELIMINATION:
                matched = explicit_option_survives(
                    required_statements,
                    is_negated,
                    definitely_in,
                    definitely_out,
                )

            elif abstained_by_policy:
                matched = False

            else:
                # A negated option asserts that none of the listed
                # statements belong to the target set.
                matched = (
                    not target_statements
                    if is_negated
                    else required_statements == target_statements
                )

            if matched:
                matching_options.append(option_letter)

            continue

        option_mapping[option_letter] = {
            "type": "unknown",
            "text": option_text,
        }

    predicted_answer = (
        matching_options[0] if len(matching_options) == 1 else None
    )

    if abstained_by_policy:
        abstention_reason = "policy_insufficient_evidence"

    elif not matching_options:
        abstention_reason = "no_option_matched"

    elif len(matching_options) > 1:
        abstention_reason = (
            "multiple_options_possible"
            if policy == ELIMINATION
            else "ambiguous_match"
        )

    else:
        abstention_reason = None

    debug = {
        "policy": policy,
        "question_asks_for_incorrect": asks_for_incorrect,
        "statement_statuses": statement_statuses,
        "supported_statements": sorted(supported),
        "contradicted_statements": sorted(contradicted),
        "insufficient_statements": sorted(insufficient),
        "target_statements": sorted(target_statements),
        "surviving_options": (
            matching_options if policy == ELIMINATION else None
        ),
        "option_mapping": option_mapping,
        "matching_options": matching_options,
        "abstention_reason": abstention_reason,
    }

    return predicted_answer, debug


# ==========================================================
# Self-check
# ==========================================================
#
# This module is now the only mapping implementation in the project:
# evaluate_verified_pyqs.py, system_c_pipeline.py and
# replay_answer_mapping.py all call it. It is also stdlib-only, so unlike
# the verifier it can be tested here without an API key. The three
# defects named in the module docstring each get an assertion, so a
# regression shows up as a failure rather than as a quietly different
# accuracy number.


class _Claim:
    """Minimal stand-in for an EvaluationClaim."""

    def __init__(self, claim_id: str, statement_number: "str | None"):
        self.claim_id = claim_id
        self.statement_number = statement_number


class _Verdict:
    """Minimal stand-in for a FactVerificationResult."""

    def __init__(self, verdict: str):
        self.verdict = verdict


TWO_STATEMENT_OPTIONS = {
    "A": "I only",
    "B": "II only",
    "C": "Both I and II",
    "D": "Neither I nor II",
}

COUNT_OPTIONS = {
    "A": "Only one",
    "B": "Only two",
    "C": "Only three",
    "D": "None",
}


def self_check() -> None:

    # ---- parsing -----------------------------------------------------

    assert parse_statement_set("I and III only") == ({"I", "III"}, False)
    assert parse_statement_set("Neither I nor II") == ({"I", "II"}, True)
    assert parse_statement_set("None") == (set(), True)
    assert parse_statement_set("Only two") is None

    # Defect 2: "None" must parse as a count of zero. It previously fell
    # through to type "unknown" and could never match, which silently
    # removed one option from Q56 and Q64.
    assert parse_count_option("None") == 0
    assert parse_count_option("None of the above") == 0
    assert parse_count_option("Only one") == 1
    assert parse_count_option("All the three") == 3
    assert parse_count_option("I and II only") is None

    assert question_asks_for_incorrect(
        "Which of the statements given above is not correct?"
    )
    assert not question_asks_for_incorrect(
        "Which of the statements given above is correct?"
    )

    # ---- verdict aggregation ----------------------------------------

    claims = [
        _Claim("claim_1", "I"),
        _Claim("claim_2", "I"),
        _Claim("claim_3", "II"),
        _Claim("claim_4", None),
    ]

    statuses = determine_statement_statuses(
        claims,
        {
            "claim_1": _Verdict("SUPPORTED"),
            "claim_2": _Verdict("INSUFFICIENT"),
            "claim_3": _Verdict("CONTRADICTED"),
            "claim_4": _Verdict("SUPPORTED"),
        },
    )

    # One unresolved claim makes the whole statement unresolved; one
    # contradicted claim contradicts it. A claim with no statement number
    # belongs to no statement and must not create one.
    assert statuses == {"I": "INSUFFICIENT", "II": "CONTRADICTED"}, statuses

    # ---- defect 1: negated options -----------------------------------

    # Both statements false, question asks which are correct, so the
    # answer is "Neither I nor II". Before the fix this matched only by
    # accident, because `negated` was computed and then discarded.
    answer, debug = map_answer(
        {"I": "CONTRADICTED", "II": "CONTRADICTED"},
        TWO_STATEMENT_OPTIONS,
        "Which of the statements given above is/are correct?",
    )

    assert answer == "D", (answer, debug["matching_options"])
    assert debug["target_statements"] == []
    assert debug["abstention_reason"] is None

    # The same verdicts under a count-style option set must pick "None".
    answer, _ = map_answer(
        {"I": "CONTRADICTED", "II": "CONTRADICTED"},
        COUNT_OPTIONS,
        "How many of the statements given above are correct?",
    )

    assert answer == "D", answer

    # A question asking which statements are NOT correct inverts which
    # verdict puts a statement in the target set.
    answer, debug = map_answer(
        {"I": "CONTRADICTED", "II": "SUPPORTED"},
        TWO_STATEMENT_OPTIONS,
        "Which of the statements given above is not correct?",
    )

    assert debug["question_asks_for_incorrect"]
    assert answer == "A", answer

    # ---- defect 3: abstention as an explicit policy ------------------

    unresolved = {"I": "SUPPORTED", "II": "INSUFFICIENT"}

    strict_answer, strict_debug = map_answer(
        unresolved,
        TWO_STATEMENT_OPTIONS,
        "Which of the statements given above is/are correct?",
        policy=STRICT,
    )

    assert strict_answer is None
    assert (
        strict_debug["abstention_reason"] == "policy_insufficient_evidence"
    )

    # CLOSED_WORLD reads the unresolved statement as false. It answers
    # here, and that is the assumption being priced, not a result.
    closed_answer, _ = map_answer(
        unresolved,
        TWO_STATEMENT_OPTIONS,
        "Which of the statements given above is/are correct?",
        policy=CLOSED_WORLD,
    )

    assert closed_answer == "A", closed_answer

    # ELIMINATION keeps every option still consistent with the verdicts.
    # Here two remain, so it abstains rather than picking one.
    elim_answer, elim_debug = map_answer(
        unresolved,
        TWO_STATEMENT_OPTIONS,
        "Which of the statements given above is/are correct?",
        policy=ELIMINATION,
    )

    assert elim_answer is None
    assert elim_debug["abstention_reason"] == "multiple_options_possible"
    assert elim_debug["surviving_options"] == ["A", "C"], elim_debug

    # But when the verdicts rule out all but one option, ELIMINATION
    # answers where STRICT cannot. This is the whole claim made for the
    # policy, so it needs a case that actually exercises it.
    partial = {"I": "SUPPORTED", "II": "CONTRADICTED", "III": "INSUFFICIENT"}

    narrow_options = {
        "A": "I and II only",
        "B": "I and III only",
        "C": "II only",
        "D": "II and III only",
    }

    assert (
        map_answer(
            partial,
            narrow_options,
            "Which of the statements given above is/are correct?",
            policy=STRICT,
        )[0]
        is None
    )

    elim_answer, elim_debug = map_answer(
        partial,
        narrow_options,
        "Which of the statements given above is/are correct?",
        policy=ELIMINATION,
    )

    assert elim_answer == "B", (elim_answer, elim_debug["surviving_options"])

    # ELIMINATION must reduce exactly to STRICT when nothing is
    # unresolved, otherwise it is not the generalisation it claims to be.
    for statuses in (
        {"I": "SUPPORTED", "II": "SUPPORTED"},
        {"I": "SUPPORTED", "II": "CONTRADICTED"},
        {"I": "CONTRADICTED", "II": "CONTRADICTED"},
    ):
        for options in (TWO_STATEMENT_OPTIONS, COUNT_OPTIONS):

            question = "Which of the statements given above is/are correct?"

            assert (
                map_answer(statuses, options, question, policy=STRICT)[0]
                == map_answer(
                    statuses, options, question, policy=ELIMINATION
                )[0]
            ), (statuses, options)

    # An unknown policy must fail loudly rather than default to one.
    try:
        map_answer({}, TWO_STATEMENT_OPTIONS, "", policy="lenient")

    except ValueError:
        pass

    else:
        raise AssertionError("unknown policy must raise")


if __name__ == "__main__":

    self_check()

    print("answer mapping self-check passed")
