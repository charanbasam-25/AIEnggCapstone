"""Rebuild the frozen dataset from source transcriptions and official keys."""

import json
from pathlib import Path

from src.evaluation.benchmark import (
    BENCHMARK_PATH,
    LOCK_PATH,
    ROOT,
    inventory,
    normalise_numbered_question,
    sha256_file,
    validate_dataset,
)
from src.verification.question_type import classify_question


CURATED_PATH = ROOT / "data/benchmarks/curated_questions.json"
SOURCES_PATH = ROOT / "data/benchmarks/upsc_sources.json"


def build_dataset() -> dict:
    sources = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    rows = json.loads(CURATED_PATH.read_text(encoding="utf-8"))
    questions = []
    for row in rows:
        year, number = row["year"], row["q_number"]
        source_id = f"upsc_{year}_gs1_a"
        label = sources[source_id]["answers"][str(number)]
        if label == "X":
            raise ValueError(f"Dropped UPSC question cannot be scored: {year} Q{number}")
        text, options, operations = normalise_numbered_question(
            row["raw_question_text"], row["raw_options"]
        )
        previously_used = year == 2023 and number == 33
        questions.append({
            **row,
            "id": f"upsc-{year}-a-q{number:03d}",
            "source_id": source_id,
            "booklet_series": "A",
            "question_text": text,
            "options": options,
            "normalisation": operations,
            "official_answer": label,
            "question_type": classify_question(text),
            "format": "assertion_reason" if "Statement-I:" in text else (
                "coded_statements" if operations else "direct"
            ),
            "split": "development" if year <= 2020 or previously_used else "test",
            "previously_used_for_development": previously_used,
            "corpus_answerability": "unreviewed",
            "gold_evidence_chunk_ids": [],
        })
    dataset = {
        "schema_version": 1,
        "dataset_id": "upsc_polity_v1",
        "frozen_on": "2026-10-01",
        "description": "75 UPSC Polity and governance PYQs from official 2019–2023 GS I papers and booklet-matched official answer keys.",
        "split_protocol": "2019–2020 development; 2021–2023 test, except the previously used 2023 Q33 development example. Splits fixed before model evaluation. Existing 13 questions remain a separate regression set.",
        "transcription_protocol": "OCR-assisted transcription of official English pages; line wrapping, typography and scan noise cleaned. Original statement IDs and options are retained; an ordered Arabic item list is converted to Roman IDs for the existing parser.",
        "answerability_protocol": "Official exam answers are gold labels. Corpus answerability and gold evidence are not yet manually audited; no answerability or retrieval recall claim is made from this dataset.",
        "temporal_scope": "Labels refer to the exam year. The current Constitution/NCERT corpus is not an archived corpus for every exam year. Time-sensitive claims require a separate evidence audit.",
        "sources": sources,
        "excluded_dropped_questions": [
            {"year": 2021, "booklet_series": "A", "q_number": 80, "reason": "X in official answer key"},
            {"year": 2023, "booklet_series": "A", "q_number": 34, "reason": "X in official answer key"},
        ],
        "questions": questions,
    }
    validate_dataset(dataset, check_sources=True)
    return dataset


def main() -> None:
    dataset = build_dataset()
    encoded = (json.dumps(dataset, indent=2, ensure_ascii=False) + "\n").encode()
    # Rebuilding v1 is permitted only if it reproduces the frozen bytes.
    # Intentional changes should receive a new version and a fresh lock.
    if LOCK_PATH.exists():
        import hashlib
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        if hashlib.sha256(encoded).hexdigest() != lock["dataset_sha256"]:
            raise ValueError("Rebuild changes the frozen v1 dataset; create a new benchmark version")
    BENCHMARK_PATH.write_bytes(encoded)
    lock = {
        "dataset_id": dataset["dataset_id"],
        "dataset_sha256": sha256_file(BENCHMARK_PATH),
        "curated_sha256": sha256_file(CURATED_PATH),
        "sources_sha256": sha256_file(SOURCES_PATH),
        "question_ids_by_split": {
            split: [q["id"] for q in dataset["questions"] if q["split"] == split]
            for split in ("development", "test")
        },
    }
    LOCK_PATH.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(inventory(dataset), indent=2))


if __name__ == "__main__":
    main()
