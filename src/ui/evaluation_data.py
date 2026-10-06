"""Read saved evaluations for display without loading models or calling APIs."""

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
from io import StringIO
import json
import math
from pathlib import Path

from src.evaluation.benchmark import score_records, sha256_file


ROOT = Path(__file__).resolve().parents[2]
SYSTEM_NAMES = {"verified": "Source verifier", "vanilla": "Vanilla RAG"}
REPORT_FILES = (
    ("benchmark_development", "Development benchmark", "75-question benchmark", "benchmark/development_results.json"),
    ("benchmark_test", "Test benchmark", "75-question benchmark", "benchmark/test_results.json"),
    ("selective", "Answer quality", "13-question regression", "selective_metrics.json"),
    ("grounding", "Grounding and claim checks", "13-question regression", "system_b_metrics.json"),
    ("judge", "Explanation judge", "13-question regression", "judge_results.json"),
    ("stability", "Verdict stability", "13-question regression", "verdict_stability.json"),
    ("retrieval", "RAG retrieval benchmark", "Annotated source-page queries", "retrieval_results.json"),
    ("generation_grounded_simple", "Source-grounded simple generation", "Current generation workflow", "generation_grounded_results.json"),
    ("generation_grounded_statements", "Source-grounded statement generation", "Current generation workflow", "generation_grounded_results.statements.json"),
    ("generation_simple", "Earlier simple-question generation", "Historical generation workflow", "generation_loop_results.json"),
    ("generation_statements", "Earlier statement-question generation", "Historical generation workflow", "generation_loop_results.statements.json"),
    ("verified_records", "Verifier question traces", "13-question regression", "verified_pyq_results.json"),
    ("vanilla_records", "Baseline question traces", "13-question regression", "vanilla_rag_scored.json"),
)


@dataclass(frozen=True)
class SavedReport:
    key: str
    label: str
    scope: str
    path: Path
    data: dict | list | None = None
    modified_at: float | None = None
    error: str | None = None

    @property
    def status(self) -> str:
        if self.error:
            return "Unreadable"
        if self.data is None:
            return "Not run"
        return "Saved"

    @property
    def updated(self) -> str:
        if self.modified_at is None:
            return "—"
        return datetime.fromtimestamp(self.modified_at, timezone.utc).strftime("%d %b %Y, %H:%M UTC")


def _valid_shape(key: str, data: object) -> bool:
    if key == "verified_records":
        return isinstance(data, list) and all(
            isinstance(row, dict) and "q_number" in row and row.get("official_answer") in set("ABCD")
            for row in data
        )
    if not isinstance(data, dict):
        return False
    if key == "vanilla_records":
        return isinstance(data.get("results"), list) and all(
            isinstance(row, dict) and "q_number" in row for row in data["results"]
        )
    if key.startswith("generation_grounded_"):
        return (
            isinstance(data.get("config"), dict)
            and isinstance(data.get("summary"), dict)
            and isinstance(data.get("results"), list)
        )
    if key == "retrieval" or key.startswith("retrieval_snapshot_"):
        return (
            isinstance(data.get("config"), dict) and isinstance(data.get("metrics"), dict)
            and isinstance(data.get("results"), list)
        )
    required = {
        "selective": ("system_a_vanilla_rag", "system_b_verified"),
        "grounding": ("grounding", "verdict_distributions"),
        "judge": ("summary",),
        "stability": ("conditions",),
        "generation_simple": ("summary",),
        "generation_statements": ("summary",),
    }.get(key, ("config", "metrics", "results"))
    return all(isinstance(data.get(field), dict) for field in required)


def read_report(root: Path, key: str, label: str, scope: str, filename: str) -> SavedReport:
    path = root / "data/evaluation" / filename
    try:
        # Deliberately read on every UI rerun: newly created and replaced
        # checkpoints must appear without a stale Streamlit data cache.
        data = json.loads(path.read_text(encoding="utf-8"))
        modified_at = path.stat().st_mtime
        if not _valid_shape(key, data):
            raise ValueError("Unexpected report shape")
        return SavedReport(key, label, scope, path, data, modified_at)
    except FileNotFoundError:
        return SavedReport(key, label, scope, path)
    except (OSError, ValueError, TypeError):
        return SavedReport(key, label, scope, path, error="The saved report could not be read. Regenerate it and refresh.")


def load_reports(root: Path = ROOT) -> dict[str, SavedReport]:
    reports = {key: read_report(root, key, label, scope, filename)
               for key, label, scope, filename in REPORT_FILES}
    known_paths = {report.path for report in reports.values()}
    # Smoke runs and explicitly named snapshots stay available for
    # inspection without taking the place of a full-split comparison.
    for path in sorted((root / "data/evaluation/benchmark").glob("*.json")):
        if path in known_paths:
            continue
        key = f"benchmark_snapshot_{path.stem}"
        reports[key] = read_report(root, key, path.stem.replace("_", " ").title(),
                                   "Benchmark snapshot", f"benchmark/{path.name}")
    for path in sorted((root / "data/evaluation").glob("retrieval_*.json")):
        if path in known_paths:
            continue
        key = f"retrieval_snapshot_{path.stem}"
        reports[key] = read_report(root, key, path.stem.replace("_", " ").title(),
                                   "Retrieval snapshot", path.name)
    return reports


def number(value: object) -> float | int | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return value
    return None


def percent(value: object, *, scale: int = 1) -> str:
    value = number(value)
    return "—" if value is None else f"{value / scale:.1%}"


def answer_metrics(report: SavedReport) -> dict[str, dict]:
    if not isinstance(report.data, dict):
        return {}
    if report.key == "selective":
        pairs = (("verified", "system_b_verified"), ("vanilla", "system_a_vanilla_rag"))
        metrics = {system: report.data[field] for system, field in pairs}
    else:
        metrics = {system: values.get("overall", {}) for system, values in report.data["metrics"].items()
                   if system in SYSTEM_NAMES and isinstance(values, dict)}
    normalised = {}
    for system, values in metrics.items():
        values = dict(values)
        answered, correct = number(values.get("answered")), number(values.get("correct"))
        if "wrong" not in values and answered is not None and correct is not None:
            values["wrong"] = answered - correct
        normalised[system] = values
    return normalised


def comparison_rows(metrics: dict[str, dict]) -> list[dict]:
    return [{
        "System": SYSTEM_NAMES[system],
        "Answered": values.get("answered"),
        "Correct": values.get("correct"),
        "Wrong": values.get("wrong"),
        "Abstained": values.get("abstained"),
        "Run errors": values.get("errors"),
        "Coverage": percent(values.get("coverage")),
        "Precision when answered": percent(values.get("precision_when_answered")),
        "Overall accuracy": percent(values.get("accuracy_overall")),
    } for system, values in metrics.items()]


def benchmark_issue(report: SavedReport, root: Path, split: str, dataset: dict) -> str | None:
    """Check that a report really describes the dataset and scope being shown."""
    if not isinstance(report.data, dict):
        return report.error
    config = report.data["config"]
    if config.get("dataset_sha256") != sha256_file(root / "data/benchmarks/polity_v1.json"):
        return "This saved run belongs to a different dataset version. Generate a report for the current benchmark."
    if config.get("split") != split:
        return "This saved run belongs to a different split. Its scores cannot be shown for this split."
    selected = config.get("question_ids", [])
    expected = {q["id"] for q in dataset["questions"] if q["split"] == split}
    if (not isinstance(selected, list) or not selected
            or not all(isinstance(qid, str) for qid in selected)
            or len(set(selected)) != len(selected) or not set(selected) <= expected):
        return "The run's question IDs do not match the selected benchmark split."
    if not config.get("is_full_split") or set(selected) != expected:
        return "This is a smoke run, not a complete benchmark split. Inspect it under Reports."
    corpus = root / "data/processed/chunks.jsonl"
    if not corpus.exists() or config.get("corpus_sha256") != sha256_file(corpus):
        return "This saved run uses a different corpus. Generate a report for the current reference library."
    metrics = report.data["metrics"]
    results = report.data["results"]
    if not metrics or set(metrics) != set(results) or not set(metrics) <= set(SYSTEM_NAMES):
        return "The saved systems and their result records do not match. Regenerate this report."
    for system, values in metrics.items():
        overall = values.get("overall") if isinstance(values, dict) else None
        records = results[system]
        if not isinstance(overall, dict) or not isinstance(records, list):
            return "The saved system metrics are incomplete or unreadable. Regenerate this report."
        if overall.get("total") != len(expected):
            return "The saved metric totals do not match the selected question set."
        questions = [q for q in dataset["questions"] if q["split"] == split]
        try:
            calculated = score_records(questions, records)["overall"]
        except (ValueError, KeyError, TypeError):
            return "The saved results contain invalid, unknown or duplicate question records."
        if any(overall.get(field) != value for field, value in calculated.items()):
            return "The saved metrics disagree with their question records. Regenerate this report."
    return None


def question_rows(questions: list[dict], results: dict[str, list[dict]]) -> list[dict]:
    by_id = {q["id"]: q for q in questions}
    rows = []
    for system, records in results.items():
        if system not in SYSTEM_NAMES:
            continue
        for record in records:
            question = by_id.get(record.get("question_id"))
            if question is None:
                continue
            prediction = record.get("predicted_answer")
            gold = question["official_answer"]
            status = record.get("status")
            outcome = ("Correct" if prediction == gold else "Wrong") if status == "ANSWERED" else {
                "ABSTAINED": "Abstained", "ERROR": "Run error",
            }.get(status, "Unknown")
            rows.append({
                "Question": question["id"], "System": SYSTEM_NAMES[system], "Outcome": outcome,
                "Prediction": prediction or "—", "Answer key": gold,
                "Topic": question.get("topic", "—"), "Year": question.get("year"),
                "Seconds": record.get("elapsed_seconds"),
                "_question": question, "_record": record,
            })
    return rows


def legacy_question_rows(reports: dict[str, SavedReport]) -> list[dict]:
    verified = reports["verified_records"].data
    baseline = reports["vanilla_records"].data
    if not isinstance(verified, list):
        return []
    questions = [{
        "id": f"Q{q['q_number']}", "question_text": q.get("question", ""),
        "options": q.get("options", {}), "official_answer": q["official_answer"],
        "year": 2025,
    } for q in verified]
    results = {"verified": [{
        "question_id": f"Q{q['q_number']}", "predicted_answer": q.get("predicted_answer"),
        "status": "ANSWERED" if q.get("predicted_answer") else "ABSTAINED", "details": q,
    } for q in verified]}
    if isinstance(baseline, dict):
        results["vanilla"] = [{
            "question_id": f"Q{q['q_number']}", "predicted_answer": q.get("predicted_answer"),
            "status": "ANSWERED" if q.get("predicted_answer") else "ABSTAINED", "details": q,
        } for q in baseline.get("results", [])]
    return question_rows(questions, results)


def table_csv(rows: list[dict]) -> str:
    if not rows:
        return ""
    columns = [key for key in rows[0] if not key.startswith("_")]
    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        values = {key: row.get(key) for key in columns}
        for key, value in values.items():
            if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
                values[key] = "'" + value
        writer.writerow(values)
    return stream.getvalue()
