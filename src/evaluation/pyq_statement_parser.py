import re


STATEMENT_PATTERN = re.compile(
    r"(?<![A-Za-z])\b(I|II|III|IV)\.\s*"
)


def clean_question_text(text: str) -> str:
    """
    Remove control characters introduced by PDF extraction
    while preserving the actual question text.
    """

    return (
        text
        .replace("\x07", " ")
        .replace("\x08", " ")
        .replace("\x0b", " ")
        .replace("\x0c", " ")
        .strip()
    )


def extract_numbered_statements(
    question_text: str,
) -> list[dict]:
    """
    Deterministically extract numbered statements from a
    UPSC PYQ.

    Returns:

        [
            {
                "statement_number": "I",
                "text": "..."
            },
            ...
        ]

    The source wording is preserved as much as possible.

    This parser does not interpret or correct the statement.
    """

    text = clean_question_text(
        question_text
    )

    matches = list(
        STATEMENT_PATTERN.finditer(text)
    )

    if not matches:
        return []

    statements = []

    question_markers = [
        "Which of the statements",
        "Which of the above",
        "How many of the above",
        "For a constitutional amendment",
    ]

    for index, match in enumerate(matches):

        start = match.end()

        if index + 1 < len(matches):

            end = matches[index + 1].start()

        else:

            end = len(text)

            for marker in question_markers:

                marker_position = text.find(
                    marker,
                    start,
                )

                if marker_position != -1:
                    end = min(
                        end,
                        marker_position,
                    )

        statement_text = text[
            start:end
        ].strip()

        if not statement_text:
            continue

        statements.append(
            {
                "statement_number": match.group(1),
                "text": statement_text,
            }
        )

    return statements


if __name__ == "__main__":

    examples = [
        "I. First statement II. Second statement III. Third statement",
        "I. First statement II. Second statement",
    ]

    for example in examples:

        print(
            extract_numbered_statements(
                example
            )
        )
