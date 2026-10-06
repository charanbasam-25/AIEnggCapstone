"""Offline loading, validation and scoring for the frozen Polity benchmark."""

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from src.verification.question_type import classify_question


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = ROOT / "data/benchmarks/polity_v1.json"
LOCK_PATH = ROOT / "data/benchmarks/polity_v1.lock.json"
SPLITS = ("development", "test")
LETTERS = set("ABCD")
ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalise_numbered_question(text: str, options: dict) -> tuple[str, dict, list]:
    """Convert sequential line-start item IDs and coded options to Roman IDs.

    Only an ordered 1., 2., ... list is changed. Dates, article numbers,
    quantities and assertion/reason labels stay intact. Raw text is retained
    in the dataset so the transformation can always be inspected.
    """
    matches = list(re.finditer(r"(?m)^\s*([1-5])\.\s+", text))
    numbers = [int(match[1]) for match in matches]
    if len(numbers) < 2 or numbers != list(range(1, len(numbers) + 1)):
        return text, deepcopy(options), []
    converted = text
    for match in reversed(matches):
        prefix = match[0]
        converted = converted[:match.start()] + re.sub(
            r"[1-5]\.", ROMAN[int(match[1])] + ".", prefix, count=1
        ) + converted[match.end():]
    converted_options = {}
    for letter, option in options.items():
        converted_options[letter] = re.sub(
            r"\b[1-5]\b", lambda match: ROMAN[int(match[0])], option
        )
    return converted, converted_options, ["arabic_item_ids_to_roman"]


def question_input(question: dict) -> dict:
    """The sole boundary passed to answering code; gold and provenance stay out."""
    return {
        "question_text": question["question_text"],
        "options": deepcopy(question["options"]),
    }


def content_fingerprint(question: dict) -> str:
    text = " ".join([
        question["question_text"],
        *(question["options"][letter] for letter in "ABCD"),
    ])
    # Ignore typography and statement numeral conventions for duplicate checks.
    text = re.sub(r"\b(?:I|II|III|IV|V|[1-5])\b", "item", text)
    text = " ".join(re.findall(r"[a-z0-9]+", text.lower()))
    return hashlib.sha256(text.encode()).hexdigest()


def validate_dataset(dataset: dict, *, check_sources: bool = False) -> None:
    if dataset.get("schema_version") != 1:
        raise ValueError("Unsupported benchmark schema")
    questions = dataset.get("questions", [])
    if not questions:
        raise ValueError("The benchmark has no questions")
    sources = dataset["sources"]
    ids, content = set(), set()
    for question in questions:
        qid = question["id"]
        if qid in ids:
            raise ValueError(f"Duplicate question ID: {qid}")
        ids.add(qid)
        if question["split"] not in SPLITS:
            raise ValueError(f"Unknown split: {qid}")
        if not question["question_text"].strip() or not question["topic"].strip():
            raise ValueError(f"Missing question text or topic: {qid}")
        options = question["options"]
        if set(options) != LETTERS or not all(
            isinstance(value, str) and value.strip() for value in options.values()
        ):
            raise ValueError(f"Expected four nonempty options: {qid}")
        if question["official_answer"] not in LETTERS:
            raise ValueError(f"Invalid or dropped answer: {qid}")
        source = sources[question["source_id"]]
        if (question["year"], question["booklet_series"]) != (
            source["year"], source["booklet_series"]
        ):
            raise ValueError(f"Paper/key series mismatch: {qid}")
        number = question["q_number"]
        if not 1 <= number <= 100 or source["answers"].get(str(number)) != question["official_answer"]:
            raise ValueError(f"Official-key mismatch: {qid}")
        pages = question["source_pdf_pages"]
        if not pages or any(not 1 <= page <= source["paper_pages"] for page in pages):
            raise ValueError(f"Invalid PDF page reference: {qid}")
        text, normal_options, operations = normalise_numbered_question(
            question["raw_question_text"], question["raw_options"]
        )
        if (text, normal_options, operations) != (
            question["question_text"], options, question["normalisation"]
        ):
            raise ValueError(f"Unrecorded text transformation: {qid}")
        if question["question_type"] != classify_question(text):
            raise ValueError(f"Question routing changed: {qid}")
        fingerprint = content_fingerprint(question)
        if fingerprint in content:
            raise ValueError(f"Duplicate question content across splits: {qid}")
        content.add(fingerprint)
        if question.get("previously_used_for_development") and question["split"] == "test":
            raise ValueError(f"Known development example in test split: {qid}")
    if {q["split"] for q in questions} != set(SPLITS):
        raise ValueError("Both development and test splits are required")
    if check_sources:
        for source in sources.values():
            for kind in ("paper", "key"):
                path = ROOT / source[f"{kind}_path"]
                if sha256_file(path) != source[f"{kind}_sha256"]:
                    raise ValueError(f"Source PDF changed: {path.name}")


def load_benchmark(path: Path = BENCHMARK_PATH, *, check_sources: bool = False) -> dict:
    dataset = json.loads(path.read_text(encoding="utf-8"))
    validate_dataset(dataset, check_sources=check_sources)
    if path.resolve() == BENCHMARK_PATH.resolve():
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        if sha256_file(path) != lock["dataset_sha256"]:
            raise ValueError("The frozen benchmark changed; create a new version before evaluating")
    return dataset


def inventory(dataset: dict) -> dict:
    questions = dataset["questions"]
    return {
        "dataset_id": dataset["dataset_id"],
        "total": len(questions),
        "splits": dict(Counter(q["split"] for q in questions)),
        "years": dict(sorted(Counter(str(q["year"]) for q in questions).items())),
        "question_types": dict(Counter(q["question_type"] for q in questions)),
        "topics": dict(Counter(q["topic"] for q in questions)),
        "formats_by_split": {
            split: dict(Counter(q["question_type"] for q in questions if q["split"] == split))
            for split in SPLITS
        },
    }


def score_records(questions: list[dict], records: list[dict]) -> dict:
    """Keep wrong answers, abstentions, service errors and pending work distinct."""
    by_id = {}
    expected = {q["id"] for q in questions}
    for row in records:
        if row["question_id"] not in expected or row["question_id"] in by_id:
            raise ValueError("Results contain an unknown or duplicate question")
        prediction = row.get("predicted_answer")
        if prediction is not None and prediction not in LETTERS:
            raise ValueError("Invalid prediction")
        if row["status"] not in ("ANSWERED", "ABSTAINED", "ERROR"):
            raise ValueError("Invalid result status")
        if (row["status"] == "ANSWERED") != (prediction is not None):
            raise ValueError("Prediction and status disagree")
        by_id[row["question_id"]] = row

    def group_metrics(group):
        selected = [by_id[q["id"]] for q in group if q["id"] in by_id]
        answered = [row for row in selected if row["status"] == "ANSWERED"]
        gold = {q["id"]: q["official_answer"] for q in group}
        correct = sum(row["predicted_answer"] == gold[row["question_id"]] for row in answered)
        total = len(group)
        return {
            "total": total,
            "completed": len(selected),
            "pending": total - len(selected),
            "answered": len(answered),
            "correct": correct,
            "wrong": len(answered) - correct,
            "abstained": sum(row["status"] == "ABSTAINED" for row in selected),
            "errors": sum(row["status"] == "ERROR" for row in selected),
            "coverage": len(answered) / total if total else None,
            "precision_when_answered": correct / len(answered) if answered else None,
            "accuracy_overall": correct / total if total else None,
            "error_rate_overall": (len(answered) - correct) / total if total else None,
            "upsc_marks": 2 * correct - (2 / 3) * (len(answered) - correct),
        }

    return {
        "overall": group_metrics(questions),
        "by_question_type": {
            kind: group_metrics([q for q in questions if q["question_type"] == kind])
            for kind in sorted({q["question_type"] for q in questions})
        },
        "by_topic": {
            topic: group_metrics([q for q in questions if q["topic"] == topic])
            for topic in sorted({q["topic"] for q in questions})
        },
        "by_year": {
            str(year): group_metrics([q for q in questions if q["year"] == year])
            for year in sorted({q["year"] for q in questions})
        },
    }
