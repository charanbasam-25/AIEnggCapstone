"""
Turn parsed question items into verifiable propositions.

The problem
-----------

The verifier was being asked to judge things that have no truth value.
Item I of Q54 is the noun phrase "List I-Union List, in the Seventh
Schedule"; item I of Q65 is the fragment "the State Government loses its
executive power in such areas", whose "such areas" refers to a condition
stated in the lead-in; item II of Q56 was handed over as the
concatenation "Valuing and preserving of the rich heritage of our
composite culture The Fundamental Duties".

None of those can be true or false, so a SUPPORTED verdict on them means
only "the corpus discusses this topic". Measured on the evaluation set,
6 of 37 claims came back SUPPORTED on exactly this basis. A false
SUPPORTED is worse than an abstention, because the mapping layer then
commits to an answer on the strength of it.

The approach
------------

Bind the stem to the item, so the claim carries the question's
predicate. Which side the stem sits on depends on the question shape
that question_parser identified:

    Q54  the predicate follows the items
         "For a constitutional amendment with respect to <ITEM>,
          ratification by the Legislatures of not less than one-half of
          the States is required before the President gives assent."

    Q65  the antecedent precedes the items
         "With reference to the Constitution of India, if an area in a
          State is declared as Scheduled Area under the Fifth Schedule,
          <ITEM>."

    Q56  the two columns are joined by the linking predicate implied by
         the column headings
         "In the Constitution of India, '<LEFT>' is stated under
          <RIGHT>."

Why binding is done with templates rather than by an LLM
-------------------------------------------------------

Rewriting a claim is exactly the step where an LLM can quietly introduce
a fact that the question never asserted, and a fabricated claim would be
verified faithfully and scored as a real result. Substituting an item
into a clause that already exists in the question text cannot do that:
every content word in the output came from the question.

That property is enforced rather than assumed. `unsupported_tokens`
checks each built claim against the question text plus a fixed template
vocabulary, and `build_claims` records any violation on the claim. The
one genuinely ambiguous sub-task, deciding where the left column of a
pair ends, is isolated behind the `splitter` argument and guarded by
reconstruction: whatever splits the item must produce two halves that
rejoin to the original, so a splitter cannot invent or drop words.

Standard library only.
"""

import re

from src.verification.question_parser import (
    PREDICATE_MARKERS,
    QUESTION_TYPE_PAIRS,
    QUESTION_TYPE_STATEMENTS,
    QUESTION_TYPE_STEM_DISTRIBUTED,
    ParsedQuestion,
    is_propositional,
    needs_binding,
    parse_question,
)


BINDING_NONE = "none"
BINDING_TRAILING_PREDICATE = "trailing_predicate"
BINDING_LEADING_PREDICATE = "leading_predicate"
BINDING_LEADING_CONDITION = "leading_condition"
BINDING_PAIR = "pair"
BINDING_FAILED = "failed"

# The back-reference a stem-distributed question uses to point at its own
# item list. Substituting the item for this phrase rebuilds the sentence
# UPSC intended the reader to construct.
BACK_REFERENCE = re.compile(
    r"\b(which|how\s+many)\s+(?:one\s+)?of\s+the\s+"
    r"(?:above|statements\s+given\s+above|pairs\s+given\s+above)"
    r"(?:\s+(?:statements|pairs|activities|subjects))?",
    re.IGNORECASE,
)

# The back-reference a lead-in uses when the question's predicate sits
# *before* the item list: "Which of the following are included in the
# Seventh Schedule of the Constitution of India?" Substituting the item
# for this phrase is the mirror image of what BACK_REFERENCE does for a
# trailing stem.
LEADING_BACK_REFERENCE = re.compile(
    r"^\s*(?:which|how\s+many)\s+(?:one\s+)?of\s+the\s+following"
    r"(?:\s+(?:statements|pairs|activities|subjects))?\s*",
    re.IGNORECASE,
)

# Boilerplate that introduces the item list and carries no content.
# Stripped before a lead-in is used as a claim's antecedent, so the claim
# does not begin "Consider the following statements: ...".
LEAD_IN_BOILERPLATE = re.compile(
    r"^consider\s+the\s+following"
    r"(\s+(statements|pairs|subjects|activities))?\s*:?\s*",
    re.IGNORECASE,
)

# Capitalised words that open the right-hand column of a pair once the
# PDF extractor has flattened the two columns into one line. UPSC's
# right-hand columns begin with a determiner or a sentence adverb, which
# is what makes a mid-item capital here a reliable column boundary.
PAIR_BOUNDARY_OPENERS = frozenset(
    """
    The A An It Its This These Initially Originally Formerly Currently
    Presently Both Only Situated Named Known
    """.split()
)

# Words the binding templates are allowed to introduce. Anything outside
# this set that is not already in the question text is a fabrication, so
# the list is deliberately short and closed.
TEMPLATE_VOCABULARY = frozenset(
    """
    under stated regarding following correct correctly matched pair
    constitution india reference
    """.split()
)

# Function words are exempt from the fabrication check. They carry no
# content, so they cannot assert a fact the question did not, and
# grammatical repair such as turning "are" into "is" legitimately
# introduces them.
FUNCTION_WORDS = frozenset(
    """
    a an the of in on at to for from by with as and or but not is are
    was were be been being has have had this that these those it its
    such their his her which who whom whose than then there here any
    all both each no nor if when while into out up down over under
    again further once only same so too very s t
    """.split()
)


class BuiltClaim:
    """One verifiable proposition derived from one question item."""

    def __init__(
        self,
        label: str,
        original_item: str,
        claim: str,
        binding: str,
        needed_binding: bool,
        unsupported: "list[str]",
    ) -> None:
        self.label = label
        self.original_item = original_item
        self.claim = claim
        self.binding = binding
        self.needed_binding = needed_binding
        self.unsupported = unsupported

    @property
    def is_propositional(self) -> bool:
        return is_propositional(self.claim)

    def to_dict(self) -> dict:
        return {
            "statement_number": self.label,
            "original_item": self.original_item,
            "claim": self.claim,
            "binding": self.binding,
            "needed_binding": self.needed_binding,
            "is_propositional": self.is_propositional,
            "unsupported_tokens": self.unsupported,
        }


def content_tokens(text: str) -> "list[str]":
    """Lower-cased alphabetic tokens, which is what comparison needs."""

    return re.findall(r"[a-z]+", text.lower())


def unsupported_tokens(claim: str, source: str) -> "list[str]":
    """
    Words in the claim that came from neither the question nor a template.

    This is the guard that makes template binding trustworthy: a claim
    should be a rearrangement of the question's own words, so any token
    outside the question text and outside TEMPLATE_VOCABULARY means the
    binding added information the question never asserted.
    """

    permitted = (
        set(content_tokens(source))
        | TEMPLATE_VOCABULARY
        | FUNCTION_WORDS
    )

    return sorted(
        {
            token
            for token in content_tokens(claim)
            if token not in permitted
        }
    )


def as_sentence(text: str) -> str:
    """Normalise a built claim to a single terminated sentence."""

    cleaned = " ".join(text.split()).rstrip(" ,;:")

    if cleaned.endswith("?"):
        cleaned = cleaned[:-1].rstrip(" ,;:")

    if not cleaned.endswith("."):
        cleaned += "."

    return cleaned


def is_plural_subject(item: str) -> bool:
    """
    Decide whether a substituted item takes a plural verb.

    Needed because a "How many of the above are ..." closing is written
    for a plural subject, and substituting a single item can leave
    "Production of crude oil are regulated". A coordinated head such as
    "Refining, storage and distribution of petroleum" stays plural; a
    single head such as "Production of natural gas" becomes singular.
    """

    head = item.split(" of ")[0]

    return " and " in f" {head} "


def fix_agreement(tail: str, plural: bool) -> str:
    """Correct the verb of a closing clause after item substitution."""

    if plural:
        return tail

    return re.sub(r"^(\s*)are\b", r"\1is", tail, count=1)


def bind_trailing_predicate(
    item: str,
    trailing_stem: str,
) -> "str | None":
    """
    Substitute the item into the closing clause that carries the
    predicate, as in Q54 and Q66.

    Returns None when the closing holds no back-reference to substitute
    for, so the caller can fall back rather than emit a mangled claim.
    """

    match = BACK_REFERENCE.search(trailing_stem)

    if match is None:
        return None

    head = trailing_stem[: match.start()]
    tail = fix_agreement(
        trailing_stem[match.end() :], is_plural_subject(item)
    )

    # Strip a trailing comma from the item so "List I-Union List, in the
    # Seventh Schedule," does not collide with the clause's own comma.
    return as_sentence(f"{head}{item.rstrip(' ,')}{tail}")


def bind_leading_predicate(item: str, lead_in: str) -> "str | None":
    """
    Substitute the item into an interrogative lead-in that carries the
    predicate.

        "Which of the following are included in the Seventh Schedule of
         the Constitution of India? I. Police"
      -> "Police is included in the Seventh Schedule of the Constitution
          of India."

    This is the mirror of `bind_trailing_predicate`, for the questions
    that ask first and list second.

    Returns None unless what follows the back-reference opens with a
    verb. That guard is what keeps the function from mangling a lead-in
    of the form "Which of the following statements about the Lokpal
    is/are correct?", where the remainder is a noun phrase and gluing the
    item onto it would produce "Police statements about the Lokpal
    is/are correct." Failing here is the right outcome: the caller falls
    through, and an unbindable item is reported as BINDING_FAILED rather
    than dressed up as a proposition.
    """

    match = LEADING_BACK_REFERENCE.match(lead_in)

    if match is None:
        return None

    predicate = lead_in[match.end() :].strip(" :;,").rstrip("?").strip()
    words = predicate.split()

    # A question mark surviving the rstrip means the lead-in holds more
    # than one sentence, so the predicate boundary is not where this
    # template assumes it is.
    if not words or "?" in predicate:
        return None

    if words[0].lower() not in PREDICATE_MARKERS:
        return None

    return as_sentence(
        f"{item.rstrip(' ,')} "
        f"{fix_agreement(predicate, is_plural_subject(item))}"
    )


def bind_leading_condition(item: str, lead_in: str) -> "str | None":
    """
    Prefix the item with the lead-in that supplies its antecedent, as in
    Q65, where the item is a lower-case continuation of a conditional.
    """

    antecedent = LEAD_IN_BOILERPLATE.sub("", lead_in).strip(" :;,")

    if not antecedent:
        return None

    # An interrogative antecedent is not an antecedent. Prefixing the
    # item with one produced claims such as "Which of the following are
    # included in the Seventh Schedule of the Constitution of India?,
    # Police." -- which passes the propositional check on the strength of
    # the stem's own verb while asserting nothing. Such a question is
    # `bind_leading_predicate`'s case, and if that declined it, this must
    # decline too rather than paper over the failure.
    if "?" in antecedent or LEADING_BACK_REFERENCE.match(antecedent):
        return None

    return as_sentence(f"{antecedent}, {item}")


def split_pair_by_opener(item: str) -> "tuple[str, str] | None":
    """
    Split a flattened pair item at the start of its right-hand column.

    The PDF extractor joins the two printed columns into one line, so the
    boundary has to be recovered from wording. A capitalised determiner or
    sentence adverb appearing mid-item is that boundary, because UPSC's
    right-hand column always opens with one.
    """

    words = item.split()

    for index in range(1, len(words)):

        if words[index].strip("\"'") in PAIR_BOUNDARY_OPENERS:
            return (
                " ".join(words[:index]).strip(" ,;:"),
                " ".join(words[index:]).strip(" ,;:"),
            )

    return None


def reconstructs(
    item: str,
    halves: "tuple[str, str]",
) -> bool:
    """
    Confirm a split neither invented nor dropped words.

    Applied to every splitter, including an LLM one, so a split that
    paraphrases or omits part of the item is rejected instead of being
    bound into a claim that misstates the pair.
    """

    left, right = halves

    return content_tokens(f"{left} {right}") == content_tokens(item)


def bind_pair(
    item: str,
    pair_columns: "tuple[str, str] | None",
    splitter,
) -> "str | None":
    """
    Build a proposition asserting that the item's two columns correspond.

    The linking predicate comes from the right-hand column heading, since
    that is what the question says the correspondence is: "Stated under"
    means the left column is stated under the right, while "Description"
    means the right column describes the left.
    """

    halves = splitter(item)

    if halves is None or not reconstructs(item, halves):
        return None

    left, right = halves

    if not left or not right:
        return None

    heading = (pair_columns[1] if pair_columns else "").lower()

    if "stated under" in heading:
        return as_sentence(f'"{left}" is stated under {right}')

    if "description" in heading:
        return as_sentence(
            f"Regarding {left}, the following is correct: {right}"
        )

    return as_sentence(f'"{left}" and "{right}" are correctly matched')


def build_claims(
    parsed: ParsedQuestion,
    splitter=split_pair_by_opener,
) -> "list[BuiltClaim]":
    """
    Produce one verifiable claim per numbered item.

    An item that already stands alone is passed through unchanged, so
    this cannot regress the eight questions whose items are complete
    sentences. Items that need binding get the treatment their question
    shape calls for, and a binding that fails is reported as
    BINDING_FAILED with the original text rather than silently dropped,
    because a missing claim would show up as an unexplained abstention.
    """

    source = " ".join(
        [parsed.lead_in, parsed.closing, *parsed.items.values()]
    )

    needing = set(parsed.items_needing_binding())
    claims: "list[BuiltClaim]" = []

    for label, item in parsed.items.items():

        bound: "str | None" = None
        binding = BINDING_NONE

        if label not in needing:
            bound = as_sentence(item)

        elif parsed.question_type == QUESTION_TYPE_PAIRS:
            binding = BINDING_PAIR
            bound = bind_pair(item, parsed.pair_columns, splitter)

        elif parsed.question_type == QUESTION_TYPE_STEM_DISTRIBUTED:

            # Ordered most specific first. A trailing stem is the
            # strongest signal, because it names the predicate outright;
            # an interrogative lead-in is next; prefixing a conditional
            # antecedent is the weakest and so goes last.
            if parsed.trailing_stem:
                binding = BINDING_TRAILING_PREDICATE
                bound = bind_trailing_predicate(
                    item, parsed.trailing_stem
                )

            if bound is None:
                binding = BINDING_LEADING_PREDICATE
                bound = bind_leading_predicate(item, parsed.lead_in)

            if bound is None:
                binding = BINDING_LEADING_CONDITION
                bound = bind_leading_condition(item, parsed.lead_in)

        if bound is None:
            binding = BINDING_FAILED
            bound = as_sentence(item)

        claims.append(
            BuiltClaim(
                label=label,
                original_item=item,
                claim=bound,
                binding=binding,
                needed_binding=label in needing,
                unsupported=unsupported_tokens(bound, source),
            )
        )

    return claims


def build_claims_for_question(
    question_text: str,
    splitter=split_pair_by_opener,
) -> "tuple[ParsedQuestion, list[BuiltClaim]]":
    """Parse and bind in one call, for callers that only have raw text."""

    parsed = parse_question(question_text)

    return parsed, build_claims(parsed, splitter=splitter)


def self_check() -> None:
    """
    Assertions covering each binding path and each guard.

    Run on every invocation of the module's report so a regression in
    binding surfaces here rather than as a silent drop in claim quality
    several stages downstream.
    """

    # Trailing-predicate binding must carry the predicate into the claim.
    bound = bind_trailing_predicate(
        "List I-Union List, in the Seventh Schedule",
        "For a constitutional amendment with respect to which of the "
        "above, ratification by the Legislatures of not less than "
        "one-half of the States is required?",
    )
    assert bound is not None
    assert bound.startswith("For a constitutional amendment")
    assert "List I-Union List" in bound
    assert "ratification" in bound
    assert is_propositional(bound), bound

    # Number agreement must be repaired for a singular item.
    singular = bind_trailing_predicate(
        "Production of natural gas",
        "How many of the above activities are regulated by the Board?",
    )
    assert singular == (
        "Production of natural gas is regulated by the Board."
    ), singular

    # A coordinated head stays plural.
    plural = bind_trailing_predicate(
        "Refining, storage and distribution of petroleum",
        "How many of the above activities are regulated by the Board?",
    )
    assert plural is not None and " are regulated" in plural, plural

    # Leading-predicate binding is the mirror of the trailing form: the
    # question asks first and lists second.
    leading = bind_leading_predicate(
        "Police",
        "Which of the following are included in the Seventh Schedule "
        "of the Constitution of India?",
    )
    assert leading == (
        "Police is included in the Seventh Schedule of the "
        "Constitution of India."
    ), leading

    # It must decline a lead-in whose remainder is a noun phrase, rather
    # than gluing the item onto it. Declining costs a claim; not
    # declining costs a claim that asserts nothing and says it does.
    assert (
        bind_leading_predicate(
            "Lokpal",
            "Which of the following statements about the Lokpal "
            "is/are correct?",
        )
        is None
    )

    # The same lead-in must not be salvaged by the conditional form
    # either, which is what produced claims of the shape
    # "Which of the following are included ...?, Police."
    assert (
        bind_leading_condition(
            "Police",
            "Which of the following are included in the Seventh "
            "Schedule of the Constitution of India?",
        )
        is None
    )

    # The closing instruction is exam apparatus. It must not end up
    # inside an item, which is what happened when the question's
    # interrogative sat in the lead-in and there was no "which of the
    # above" for the closing detector to find.
    _, instructed = build_claims_for_question(
        "Which of the following are included in the Seventh Schedule "
        "of the Constitution of India? I. Police II. Public health "
        "III. Defence of India "
        "Select the correct answer using the code given below."
    )
    assert len(instructed) == 3, instructed
    assert all(
        "select the correct answer" not in claim.claim.lower()
        for claim in instructed
    ), [claim.claim for claim in instructed]
    assert all(claim.is_propositional for claim in instructed)
    assert all(not claim.unsupported for claim in instructed)

    # Leading-condition binding must restore the antecedent.
    conditional = bind_leading_condition(
        "the State Government loses its executive power in such areas",
        "Consider the following statements: With reference to the "
        "Constitution of India, if an area in a State is declared as "
        "Scheduled Area under the Fifth Schedule",
    )
    assert conditional is not None
    assert conditional.startswith("With reference to")
    assert "Fifth Schedule, the State Government loses" in conditional

    # Pair binding must join the columns with the heading's predicate.
    pair = bind_pair(
        "Valuing and preserving of the rich heritage of our composite "
        "culture The Fundamental Duties",
        ("Provision in the Constitution of India", "Stated under"),
        split_pair_by_opener,
    )
    assert pair is not None
    assert pair.endswith("is stated under The Fundamental Duties.")

    # A splitter that drops or invents words must be rejected.
    assert (
        bind_pair(
            "Nagaland The State came into existence",
            ("State", "Description"),
            lambda item: ("Nagaland", "The State was created"),
        )
        is None
    )

    # The fabrication guard must notice a word the question never used.
    assert unsupported_tokens(
        "Nagaland attained statehood in 1963.", "Nagaland The State"
    ) == ["attained", "statehood"]

    # Rearranging the question's own words must pass cleanly, otherwise
    # the guard would fire on every legitimate binding.
    assert (
        unsupported_tokens(
            "Nagaland is a State.", "Nagaland The State"
        )
        == []
    )


if __name__ == "__main__":

    import json
    from pathlib import Path

    from src.verification.question_parser import ascii_safe

    self_check()
    print("binding self-check passed\n")

    records = json.loads(
        Path("data/evaluation/verified_pyq_results.json").read_text(
            encoding="utf-8"
        )
    )

    binding_counts: "dict[str, int]" = {}
    total = 0
    propositional = 0
    fabrications = 0

    for record in records:

        parsed, claims = build_claims_for_question(record["question"])

        print("=" * 70)
        print(f"Q{record['q_number']}  {parsed.question_type}")

        for claim in claims:

            total += 1
            propositional += int(claim.is_propositional)
            fabrications += int(bool(claim.unsupported))

            binding_counts[claim.binding] = (
                binding_counts.get(claim.binding, 0) + 1
            )

            if claim.binding in (BINDING_NONE,):
                continue

            print(f"  {claim.label}. [{claim.binding}]")
            print(f"     was : {ascii_safe(claim.original_item)[:100]}")
            print(f"     now : {ascii_safe(claim.claim)[:180]}")

            if claim.unsupported:
                print(f"     !! invented tokens: {claim.unsupported}")

    print("=" * 70)
    print(f"claims               : {total}")
    print(
        f"propositional        : {propositional}/{total} "
        f"({propositional / total:.1%})"
    )
    print(f"binding methods      : {binding_counts}")
    print(f"claims with invented tokens : {fabrications}")
