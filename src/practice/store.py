"""Persist accepted questions and versioned runtime measurements."""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from pydantic import ValidationError

from src.orchestration.telemetry import GENERATION_POLICY
from src.practice.models import PracticeQuestion, question_fingerprint


DEFAULT_STORE = Path(__file__).resolve().parents[2] / "data/practice/practice.sqlite3"


class PracticeStore:
    def __init__(self, path: Path = DEFAULT_STORE):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS questions (
                    id TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    difficulty TEXT NOT NULL,
                    format TEXT NOT NULL,
                    corpus_hash TEXT NOT NULL,
                    policy TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    UNIQUE(fingerprint, corpus_hash, policy)
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
            """)
            connection.commit()

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.path, timeout=5)
        try:
            yield connection
        finally:
            connection.close()

    def save_question(self, question: PracticeQuestion) -> bool:
        # Revalidate even if a caller used model_construct or edited an object.
        question = PracticeQuestion.model_validate(question.model_dump(mode="json"))
        with self._connection() as connection:
            cursor = connection.execute(
                """INSERT OR IGNORE INTO questions
                   (id, fingerprint, topic, difficulty, format, corpus_hash, policy, created_at, payload)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (question.id, question_fingerprint(question.mcq.question), question.topic,
                 question.difficulty, question.question_format, question.corpus_hash,
                 question.generation_policy, question.created_at, question.model_dump_json()),
            )
            connection.commit()
            return cursor.rowcount == 1

    def load_questions(
        self, topic: str, difficulty: str, question_format: str,
        corpus_hash: str, exclude_ids=(),
    ) -> list[PracticeQuestion]:
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT payload FROM questions
                   WHERE topic=? AND difficulty=? AND format=? AND corpus_hash=? AND policy=?
                   ORDER BY created_at DESC""",
                (topic, difficulty, question_format, corpus_hash, GENERATION_POLICY),
            ).fetchall()
        questions = []
        excluded = set(exclude_ids)
        for (payload,) in rows:
            try:
                question = PracticeQuestion.model_validate_json(payload)
            except (ValidationError, ValueError):
                continue
            if question.id not in excluded:
                questions.append(question)
        return questions

    def used_stems(self, topic: str, corpus_hash: str) -> list[str]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT payload FROM questions WHERE topic=? AND corpus_hash=? AND policy=? ORDER BY created_at DESC",
                (topic, corpus_hash, GENERATION_POLICY),
            ).fetchall()
        stems = []
        for (payload,) in rows:
            try:
                stems.append(PracticeQuestion.model_validate_json(payload).mcq.question)
            except (ValidationError, ValueError):
                continue
        return stems

    def save_run(self, report: dict) -> None:
        with self._connection() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO runs(id, created_at, payload) VALUES(?, ?, ?)",
                (report["id"], report["created_at"], json.dumps(report)),
            )
            connection.commit()

    def load_runs(self, limit: int = 50) -> list[dict]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT payload FROM runs ORDER BY created_at DESC LIMIT ?", (limit,),
            ).fetchall()
        runs = []
        for (payload,) in rows:
            try:
                report = json.loads(payload)
                if isinstance(report, dict) and report.get("generation_policy"):
                    runs.append(report)
            except (TypeError, ValueError):
                continue
        return runs
