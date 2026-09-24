import json
import re
from pathlib import Path

import pymupdf


PDF_PATH = "data/raw/upsc/prelims2025Questions.pdf"
OUTPUT_PATH = "data/evaluation/pyq_2025_raw.json"


def extract_pdf_text(pdf_path: str) -> str:
    pdf = pymupdf.open(pdf_path)

    pages = []
    for page in pdf:
        pages.append(page.get_text())

    return "\n".join(pages)


def remove_answer_section(text: str) -> str:
    # The PDF contains explanations after the ANSWERS heading.
    # Keep only the actual question paper before that section.
    parts = re.split(r"\bANSWERS\b", text, maxsplit=1)

    return parts[0]


def parse_questions(text: str) -> list[dict]:
    # Find question-number positions first, then slice each question block.
    # This is more robust to PDF line/control-character formatting.
    question_starts = [m for m in re.finditer(r'(?m)^\s*(\d+)\.\s+', text) if int(m.group(1)) <= 100]

    questions = []

    for index, match in enumerate(question_starts):
        question_number = int(match.group(1))

        block_start = match.end()
        block_end = question_starts[index + 1].start() if index + 1 < len(question_starts) else len(text)
        block = text[block_start:block_end].strip()

        # Extract A-D options.
        option_matches = list(
            re.finditer(
                r'(?ms)(?:^|\n)\s*\(([a-dA-D])\)\s*(.*?)(?=(?:\n\s*\([a-dA-D]\)\s)|\Z)',
                block,
            )
        )

        if len(option_matches) < 4:
            continue

        first_option_start = option_matches[0].start()
        question_text = block[:first_option_start].strip()

        options = {}
        for option_match in option_matches[:4]:
            option = option_match.group(1).upper()
            option_text = re.sub(r'\s+', ' ', option_match.group(2).strip()); option_text = re.split(r'\b(?:CURRENT AFFAIRS|ANSWERS)\b', option_text, maxsplit=1)[0].strip()
            options[option] = option_text

        questions.append(
            {
                'q_number': question_number,
                'question_text': re.sub(r'\s+', ' ', question_text),
                'options': options,
            }
        )

    return questions

def main() -> None:
    text = extract_pdf_text(PDF_PATH)

    question_section = remove_answer_section(text)

    questions = parse_questions(question_section)

    output = {
        "year": 2025,
        "source": PDF_PATH,
        "questions": questions,
    }

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    Path(OUTPUT_PATH).write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Extracted questions: {len(questions)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
