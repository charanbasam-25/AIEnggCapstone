"""
Deterministic structural parsing of UPSC statement-style questions.

Why this module exists
----------------------

Verification can only be as good as the claim it is given, and the
claims being handed to the verifier were often not propositions at all.
Measured on the 13-question evaluation set, 8 of 37 extracted claims
(21.6%) had no truth value, and 6 of those came back SUPPORTED, because
the verifier found the corpus *mentioned* the topic. A bare noun phrase
cannot be true or false, so SUPPORTED silently degenerated into "this
topic exists in the corpus", and the mapping layer then acted on it.
That is a false SUPPORTED, which is strictly worse than an abstention.

The cause is structural, not a prompt-tuning problem. A UPSC question
distributes its predicate across the stem and the numbered items in
three distinct shapes, and only the first is safe to read item-by-item:

STATEMENTS
    Each numbered item is already a complete sentence.
        "An Ordinance can amend any Central Act."
    Safe to verify as-is.

STEM_DISTRIBUTED
    The items are noun phrases or sentence fragments, and the assertion
    lives in the stem. Q54's item I is the noun phrase "List I-Union
    List, in the Seventh Schedule"; the actual claim sits in the closing
    clause about ratification. Q65 is worse: its items are fragments
    continuing a conditional lead-in, so item I reads "the State
    Government loses its executive power in such areas" where "such
    areas" has no referent once the lead-in is dropped. Verifying that
    in isolation is meaningless, which is why Q65 failed even though the
    Fifth Schedule evidence is present in the corpus.

    The stem can sit on either side of the items, so both sides are
    returned: Q54 and Q66 carry it in the closing clause, Q65 in the
    lead-in.

PAIRS
    Two columns to be matched. The proposition is the *pairing*, so
    concatenating the columns without a linking predicate produces word
    salad, as in Q56: "Valuing and preserving of the rich heritage of
    our composite culture The Fundamental Duties". Pair items always
    need binding even when the right-hand column happens to contain a
    verb, because the claim under test is the correspondence, not either
    column alone.

Scope
-----

This module only finds structure. It decides where the stem ends, which
items exist, what shape the question is, and whether each item can stand
alone. It never rewrites an item into a claim and never judges truth;
rewriting belongs to claim_builder and truth to the verifier.

Text normalisation
------------------

The extracted question text carries PDF artifacts that break both
matching and parsing:

    U+0007 / U+0008   46 and 2 occurrences, used as paragraph breaks
    U+F02A            123 occurrences, a private-use symbol-font glyph
    U+2014 U+2013 U+23AF   em dash, en dash and a line-extension glyph
    U+2018 U+2019 U+201C U+201D   typographic quotes

These are *not* character corruption: the corpus contains zero U+FFFD
replacement characters. They are legitimate typography plus a font
artifact, and they only look like corruption on a cp1252 console.
Normalising them still matters, because "List I-Union List" is written
with a line-extension glyph, so a literal ASCII-hyphen search misses it.

The BEL paragraph markers are deliberately NOT used for segmentation.
They fall inconsistently: in Q55 and Q60 a marker such as "II." trails
at the end of the preceding segment rather than leading its own, and Q66
splits mid-list. Item markers are located by ordered scan instead, which
is reliable across all 13 questions.

Standard library only, so this runs in the evaluation environment where
the package index is unreachable.
"""

import re
import unicodedata


# Statement labels in the order UPSC uses them. Parsing scans for these
# in sequence rather than with a single alternation, because an
# alternation matches the "I." inside "List I-Union List" and inside
# ordinary prose, while an ordered scan cannot get ahead of itself.
STATEMENT_LABELS = ("I", "II", "III", "IV", "V")

QUESTION_TYPE_STATEMENTS = "STATEMENTS"
QUESTION_TYPE_STEM_DISTRIBUTED = "STEM_DISTRIBUTED"
QUESTION_TYPE_PAIRS = "PAIRS"

# Characters to fold to an ASCII equivalent before any matching.
CHARACTER_FOLDING = {
    "—": "-",
    "–": "-",
    "⎯": "-",
    "‒": "-",
    "−": "-",
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "″": '"',
    "…": "...",
    " ": " ",
    " ": " ",
    "​": " ",
}

# The interrogative core that closes a UPSC question, e.g. "which of the
# above" or "how many of the statements given above". Located first, then
# the sentence it belongs to is recovered by walking backwards, because
# in a stem-distributed question the predicate sits *before* this core
# ("For a constitutional amendment with respect to which of the above,
# ratification ... is required?").
INTERROGATIVE_CORE = re.compile(
    r"\b(which|how many)\b[^?]{0,120}?\babove\b",
    re.IGNORECASE,
)

# Words that open the closing clause in a stem-distributed question.
# Used to find the clause start when the item list has no terminal
# punctuation, which is the normal UPSC layout.
CLAUSE_OPENERS = frozenset(
    """
    for in with among from under how which if on at by after before
    consider
    """.split()
)

# A closing that matches this carries no predicate of its own; it is
# pure interrogative framing, so nothing needs to be bound from it.
BARE_CLOSING = re.compile(
    r"""^
    (which|how\s+many)
    \s+(one\s+)?of\s+the
    (\s+(statements|pairs|activities|subjects|given))*
    \s*(above|given\s+above)?
    \s*(statements|pairs|activities|subjects)?
    \s*(is\s*/\s*are|is|are)?
    \s*(not\s+)?(correct|correctly\s+matched|incorrect|true)?
    \s*\??$""",
    re.IGNORECASE | re.VERBOSE,
)

# Phrases indicating the question asks which statements are NOT correct.
# Getting this backwards inverts the answer, so it is read explicitly
# from the text rather than assumed.
INCORRECT_PATTERNS = (
    r"\bare not correct\b",
    r"\bis not correct\b",
    r"\bis\s*/\s*are not correct\b",
    r"\bnot correctly matched\b",
    r"\bincorrect\b",
)

COUNTING_PATTERNS = (r"\bhow many\b",)

# The instruction line UPSC prints as the last sentence of a question,
# e.g. "Select the correct answer using the code given below."
#
# It needs its own detector because it contains no "above"
# back-reference, so INTERROGATIVE_CORE cannot see it. When the
# question's interrogative sits in the lead-in instead of the closing,
# this instruction is the only closing there is, and without this the
# last item absorbs it: item III becomes "Defence of India Select the
# correct answer using the code given below", which then reads as
# propositional and gets sent to the verifier as a factual assertion.
#
# Anchored at the end because it is always the final sentence, which is
# what makes it safe to delete outright: it carries no predicate any item
# could need and no fact any claim could rest on.
CLOSING_INSTRUCTION = re.compile(
    r"(?:select|choose|indicate)\s+the\s+(?:correct\s+)?"
    r"(?:answer|code|option|response)\b[^.?]*[.?]?\s*$",
    re.IGNORECASE,
)

PAIR_PATTERNS = (
    r"consider the following pairs",
    r"correctly matched",
)

# Introductory phrases stripped before reading pair column headings, so
# the left-hand heading is not polluted with "Consider the following
# pairs:".
LEAD_IN_PREFIXES = (
    r"^consider the following pairs\s*:?\s*",
    r"^consider the following\s*:?\s*",
    r"^with reference to [^,]{0,60},\s*consider the following\s*:?\s*",
)

# A topic tag injected by the dataset builder, e.g.
# "[Salient Features of Indian Constitution]". Not part of the question.
TOPIC_TAG = re.compile(r"\[[^\]]{3,80}\]")

# Tokens that signal an item asserts something.
#
# Deliberately excludes "state" and "states": in a constitutional corpus
# these are overwhelmingly the noun, and including them made Q56's pair
# fragment "Separation of Judiciary from the Executive in the public
# services of the State ..." look like a proposition when it is not.
PREDICATE_MARKERS = frozenset(
    """
    is are was were be been being has have had shall will would can
    could may might must does do did requires require required
    provides provide provided includes include included stated
    appoints appoint appointed holds hold held consists consist
    consisted needs need needed becomes become became remains remain
    gives give given makes make made takes take taken cannot not no
    enjoys enjoy exercises exercise empowered entitled elected
    nominated prescribed declared deemed subject bound liable loses
    lose lost assumes assume vacate vacates resign resigns removed
    applies apply exist exists attain attains constitutes constitute
    regulated regulates amend amends abridge abridges come comes
    answerable instituted continued mention mentions mentioned
    """.split()
)


class ParsedQuestion:
    """The structural decomposition of one question."""

    def __init__(
        self,
        raw: str,
        normalised: str,
        lead_in: str,
        items: "dict[str, str]",
        closing: str,
        trailing_stem: str,
        question_type: str,
        pair_columns: "tuple[str, str] | None",
        asks_for_incorrect: bool,
        asks_how_many: bool,
    ) -> None:
        self.raw = raw
        self.normalised = normalised
        self.lead_in = lead_in
        self.items = items
        self.closing = closing
        self.trailing_stem = trailing_stem
        self.question_type = question_type
        self.pair_columns = pair_columns
        self.asks_for_incorrect = asks_for_incorrect
        self.asks_how_many = asks_how_many

    def items_needing_binding(self) -> "list[str]":
        """
        Labels whose item cannot stand alone as a proposition.

        For PAIRS every item needs binding regardless of wording, since
        the claim under test is the correspondence between the columns
        and not the content of either column.
        """

        if self.question_type == QUESTION_TYPE_PAIRS:
            return list(self.items)

        return [
            label
            for label, text in self.items.items()
            if needs_binding(text)
        ]

    def to_dict(self) -> dict:
        return {
            "lead_in": self.lead_in,
            "items": self.items,
            "closing": self.closing,
            "trailing_stem": self.trailing_stem,
            "question_type": self.question_type,
            "pair_columns": (
                list(self.pair_columns) if self.pair_columns else None
            ),
            "asks_for_incorrect": self.asks_for_incorrect,
            "asks_how_many": self.asks_how_many,
            "items_needing_binding": self.items_needing_binding(),
        }


def normalise_text(text: str) -> str:
    """
    Fold PDF typography to ASCII and strip control and font artifacts.

    Applied before any matching so that a literal search for
    "List I-Union List" finds text written with a line-extension glyph.
    """

    folded = "".join(
        CHARACTER_FOLDING.get(character, character) for character in text
    )

    kept = []

    for character in folded:
        category = unicodedata.category(character)

        # Co is the private-use area, which here holds symbol-font
        # glyphs such as U+F02A. Cc is control characters, including the
        # BEL and backspace used as paragraph breaks. Both become spaces
        # rather than being deleted, so they cannot fuse two words.
        kept.append(" " if category in ("Co", "Cc") else character)

    return " ".join("".join(kept).split())


def is_propositional(text: str) -> bool:
    """
    True when the text contains something that could be true or false.

    A lexical proxy for the presence of a predicate, not a parser. Shared
    with the metrics module so the parser's notion of a verifiable claim
    and the reported non-propositional rate cannot drift apart.
    """

    tokens = re.findall(r"[a-z]+", text.lower())

    return any(token in PREDICATE_MARKERS for token in tokens)


def is_continuation_fragment(text: str) -> bool:
    """
    True when an item is a grammatical continuation of the lead-in.

    UPSC signals this by starting the item in lower case, as in Q65's
    "the State Government loses its executive power in such areas".
    Such an item may well contain a verb, so the propositional check
    passes, yet it is still unverifiable alone because its subject or
    its conditional antecedent lives in the lead-in.
    """

    stripped = text.lstrip("\"'(")

    return bool(stripped) and stripped[0].islower()


def needs_binding(text: str) -> bool:
    """True when an item cannot be verified without the stem."""

    return is_continuation_fragment(text) or not is_propositional(text)


def find_item_spans(text: str) -> "list[tuple[str, int, int]]":
    """
    Locate the numbered item markers by ordered scan.

    Returns (label, marker_start, content_start) per item found.

    Scanning in label order is what makes this reliable. A single
    alternation over I|II|III|IV matches the "I." inside "List I-Union
    List" and any sentence-initial "I."; an ordered scan requires each
    marker to appear after the previous one, so those cannot be mistaken
    for item boundaries.
    """

    spans: "list[tuple[str, int, int]]" = []
    search_from = 0

    for label in STATEMENT_LABELS:

        # The marker must not be preceded by a letter, so "List I" does
        # not supply the "I" of a marker, and must be followed by a
        # separator rather than more label letters.
        pattern = re.compile(
            rf"(?<![A-Za-z]){re.escape(label)}\.(?=\s|$)"
        )

        match = pattern.search(text, search_from)

        if match is None:
            break

        spans.append((label, match.start(), match.end()))
        search_from = match.end()

    return spans


def find_closing_start(text: str, search_from: int) -> int:
    """
    Find where the question's closing sentence begins.

    Two steps, because the interrogative core is not always the start of
    its own sentence. In Q54 the core is "which of the above" but the
    sentence begins earlier, at "For a constitutional amendment ...",
    and that earlier part is exactly the predicate the items need bound
    to them. Cutting at the core would discard it.

    `search_from` is the end of the last item marker, so the walk back
    cannot cross into the item list and swallow an item.

    Returns len(text) when no closing is present.
    """

    match = INTERROGATIVE_CORE.search(text, search_from)

    if match is None:
        # A question whose interrogative sits in the lead-in has no
        # "which of the above" to find; its closing is the bare
        # instruction line. Missing it makes the last item swallow the
        # instruction.
        match = CLOSING_INSTRUCTION.search(text, search_from)

    if match is None:
        return len(text)

    # A sentence terminator between the last item and the core is the
    # most reliable boundary when the items are punctuated.
    terminator = max(
        text.rfind(". ", search_from, match.start()),
        text.rfind("? ", search_from, match.start()),
    )

    if terminator != -1:
        return terminator + 2

    # Otherwise fall back on the capitalised clause opener that UPSC
    # uses to start the closing clause after an unpunctuated list.
    for word in reversed(
        list(re.finditer(r"\b[A-Z][a-z]+", text[: match.start()]))
    ):

        if word.start() < search_from:
            break

        if word.group().lower() in CLAUSE_OPENERS:
            return word.start()

    return match.start()


def strip_lead_in_prefix(lead_in: str) -> str:
    """Remove the boilerplate that precedes pair column headings."""

    for pattern in LEAD_IN_PREFIXES:
        stripped = re.sub(pattern, "", lead_in, flags=re.IGNORECASE)

        if stripped != lead_in:
            return stripped.strip(" :;,")

    return lead_in


def detect_pair_columns(lead_in: str) -> "tuple[str, str] | None":
    """
    Recover the two column headings of a pair-matching question.

    UPSC prints these as an unpunctuated two-column header, which the
    PDF extractor flattens into a run of title-case words, e.g.
    "Provision in the Constitution of India Stated under" or
    "State Description". Splitting on the known right-hand headings is
    more reliable than trying to segment title case.
    """

    header = strip_lead_in_prefix(lead_in)
    lowered = header.lower()

    for heading in (
        "stated under",
        "description",
        "provision in",
        "article",
    ):
        position = lowered.rfind(heading)

        if position <= 0:
            continue

        left = header[:position].strip(" :;,")
        right = header[position:].strip(" :;,")

        if left and right:
            return left, right

    return None


def classify(
    lead_in: str,
    items: "dict[str, str]",
    closing: str,
) -> str:
    """Decide which of the three question shapes this is."""

    haystack = f"{lead_in} {closing}".lower()

    if any(re.search(pattern, haystack) for pattern in PAIR_PATTERNS):
        return QUESTION_TYPE_PAIRS

    # Every item must stand alone for the item-by-item reading to be
    # sound. One fragment is enough to make the whole question
    # stem-distributed, because the stem has to be recovered either way.
    if items and not any(needs_binding(text) for text in items.values()):
        return QUESTION_TYPE_STATEMENTS

    return QUESTION_TYPE_STEM_DISTRIBUTED


def parse_question(question_text: str) -> ParsedQuestion:
    """
    Decompose a question into stem, numbered items and closing.

    Raises ValueError when no numbered items are found, so a question
    this parser cannot handle fails loudly instead of silently yielding
    zero claims. The guard it replaces could never fire: it anchored on
    ``^`` in text that contains no newlines, and its ``\\s`` escape was
    written inside a raw string so it matched a literal backslash.
    """

    normalised = normalise_text(question_text)

    body = " ".join(TOPIC_TAG.sub(" ", normalised).split())

    spans = find_item_spans(body)

    if not spans:
        raise ValueError(
            f"No numbered items found in question: {body[:120]!r}"
        )

    closing_start = find_closing_start(body, spans[-1][2])

    lead_in = body[: spans[0][1]].strip(" :;,")

    # Drop the "Select the correct answer ..." instruction. It is exam
    # apparatus, not question content, and leaving it in the closing puts
    # it into every claim bound from a trailing predicate.
    closing = CLOSING_INSTRUCTION.sub(
        "", body[closing_start:].strip()
    ).strip(" :;,")

    items: "dict[str, str]" = {}

    for index, (label, _, content_start) in enumerate(spans):

        content_end = (
            spans[index + 1][1]
            if index + 1 < len(spans)
            else closing_start
        )

        items[label] = body[content_start:content_end].strip(" :;,.")

    question_type = classify(lead_in, items, closing)

    # A bare closing is pure interrogative framing and carries nothing
    # worth binding; anything else holds the question's predicate.
    trailing_stem = "" if BARE_CLOSING.match(closing) else closing

    pair_columns = (
        detect_pair_columns(lead_in)
        if question_type == QUESTION_TYPE_PAIRS
        else None
    )

    haystack = f"{lead_in} {closing}".lower()

    return ParsedQuestion(
        raw=question_text,
        normalised=normalised,
        lead_in=lead_in,
        items=items,
        closing=closing,
        trailing_stem=trailing_stem,
        question_type=question_type,
        pair_columns=pair_columns,
        asks_for_incorrect=any(
            re.search(pattern, haystack)
            for pattern in INCORRECT_PATTERNS
        ),
        asks_how_many=any(
            re.search(pattern, haystack)
            for pattern in COUNTING_PATTERNS
        ),
    )


def ascii_safe(text: str) -> str:
    """Render text printable on a cp1252 console."""

    return text.encode("ascii", "replace").decode("ascii")


if __name__ == "__main__":

    import json
    from pathlib import Path

    records = json.loads(
        Path("data/evaluation/verified_pyq_results.json").read_text(
            encoding="utf-8"
        )
    )

    type_counts: "dict[str, int]" = {}
    binding_total = 0
    item_total = 0

    for record in records:

        parsed = parse_question(record["question"])

        type_counts[parsed.question_type] = (
            type_counts.get(parsed.question_type, 0) + 1
        )

        needing = parsed.items_needing_binding()
        binding_total += len(needing)
        item_total += len(parsed.items)

        print("=" * 70)
        print(
            f"Q{record['q_number']}  {parsed.question_type}"
            f"   how_many={parsed.asks_how_many}"
            f"   not_correct={parsed.asks_for_incorrect}"
        )
        print(f"  lead-in : {ascii_safe(parsed.lead_in)[:110]}")
        print(f"  closing : {ascii_safe(parsed.closing)[:110]}")

        if parsed.trailing_stem:
            print("  -> closing carries the predicate")

        if parsed.pair_columns:
            print(
                f"  columns : {ascii_safe(parsed.pair_columns[0])[:45]}"
                f"  ||  {ascii_safe(parsed.pair_columns[1])[:35]}"
            )

        for label, text in parsed.items.items():
            flag = "BIND" if label in needing else "OK  "
            print(f"  [{flag}] {label}: {ascii_safe(text)[:95]}")

    print("=" * 70)
    print(f"question types      : {type_counts}")
    print(f"items needing bind  : {binding_total}/{item_total}")
