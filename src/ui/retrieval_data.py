"""Read stored practice diagnostics without creating a DB or loading models."""

import json
from pathlib import Path
import sqlite3


def _read_rows(path: Path, query: str, parameters: tuple) -> list:
    path = Path(path)
    if not path.is_file():
        return []
    try:
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            return connection.execute(query, parameters).fetchall()
        finally:
            connection.close()
    except (sqlite3.Error, OSError):
        return []


def load_practice_runs(path: Path, limit: int = 50) -> list[dict]:
    rows = _read_rows(path, "SELECT payload FROM runs ORDER BY created_at DESC LIMIT ?", (limit,))
    runs = []
    for (payload,) in rows:
        try:
            run = json.loads(payload)
            if isinstance(run, dict) and isinstance(run.get("id"), str) and run.get("generation_policy"):
                runs.append(run)
        except (TypeError, ValueError):
            continue
    return runs


def load_checked_question(path: Path, question_id: str):
    from src.practice.models import PracticeQuestion

    rows = _read_rows(path, "SELECT payload FROM questions WHERE id=?", (question_id,))
    if not rows:
        return None
    try:
        return PracticeQuestion.model_validate_json(rows[0][0])
    except (ValueError, TypeError):
        return None


def passage_rows(passages: list[dict], cited_pages=()) -> list[dict]:
    cited = set(cited_pages)
    return [{
        "Semantic rank": item.get("semantic_rank"),
        "Reranked rank": item.get("reranker_rank"),
        "Source": item.get("source"), "PDF page": item.get("page"),
        "Chunk": item.get("id"), "Semantic score": item.get("semantic_score"),
        "Cross-encoder score": item.get("reranker_score"),
        "Selected": item.get("selected", False),
        "Cited page": (item.get("source"), item.get("page")) in cited,
    } for item in passages]
