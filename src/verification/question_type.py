"""Choose a verification path without using a model or an answer key."""

import re
import unicodedata

from src.verification.question_parser import find_item_spans


STATEMENT_MCQ = "statement_mcq"
DIRECT_MCQ = "direct_mcq"
BEST_ANSWER_MCQ = "best_answer_mcq"

BEST_ANSWER_WORDING = re.compile(
    r"\bbest\b"
    r"|\b(?:chief|primary|principal|main)\s+(?:purpose|function|objective|aim|goal|reason)\b"
    r"|\bmost\s+(?:appropriate|accurate|correct|likely|suitable|important)\b",
    re.IGNORECASE,
)

STATEMENT_LABELS = tuple(re.compile(
    rf"\bstatement\s*[-–—]?\s*{label}\b[ \t]*(?::|[.)\-–—]|$)",
    re.IGNORECASE | re.MULTILINE,
) for label in (r"(?:I|1)", r"(?:II|2)"))
ASSERTION_LABELS = tuple(re.compile(
    rf"\b{name}(?:[ \t]*\([ \t]*{label}[ \t]*\))?[ \t]*(?::|[.\-–—]|$)",
    re.IGNORECASE | re.MULTILINE,
) for name, label in (("assertion", "A"), ("reason", "R")))


def has_labelled_statement_pair(question_text: str) -> bool:
    """Recognize paired labels that the numbered claim parser does not handle.

    Truth and explanatory relationships cannot be inferred from generic
    option-fit judgments. These formats need their own statement-level path.
    """
    normalized = unicodedata.normalize("NFKC", question_text)
    return any(all(pattern.search(normalized) for pattern in pair)
               for pair in (STATEMENT_LABELS, ASSERTION_LABELS))


def classify_question(question_text: str) -> str:
    """Numbered questions retain the existing statement-verification path.

    Best-answer wording changes how direct options are compared; it is not
    a truth test of each option in isolation.
    """

    # Keep paired statement labels on the guarded direct entry point even
    # when a statement contains its own numbered list.
    if has_labelled_statement_pair(question_text):
        return DIRECT_MCQ
    # A lone "I." can be part of an entity such as "World War I.".
    # The supported multi-statement format has two to four ordered items.
    if len(find_item_spans(question_text)) >= 2:
        return STATEMENT_MCQ
    if BEST_ANSWER_WORDING.search(question_text):
        return BEST_ANSWER_MCQ
    return DIRECT_MCQ
