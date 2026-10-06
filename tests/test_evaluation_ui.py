"""Exercise report freshness, scope integrity and UI behavior without model calls."""

import json
from pathlib import Path
import shutil

import pytest
from streamlit.testing.v1 import AppTest

from src.evaluation.benchmark import BENCHMARK_PATH, load_benchmark, sha256_file
from src.evaluation.evaluate_benchmark import save_checkpoint
from src.ui.evaluation_data import (
    ROOT, answer_metrics, benchmark_issue, comparison_rows,
    load_reports, percent, question_rows, table_csv,
)


@pytest.fixture
def eval_root(tmp_path):
    (tmp_path / "data/benchmarks").mkdir(parents=True)
    shutil.copyfile(BENCHMARK_PATH, tmp_path / "data/benchmarks/polity_v1.json")
    (tmp_path / "data/processed").mkdir()
    # No actual retrieval runs in these tests. This stand-in is only used
    # to verify that saved reports are bound to the recorded corpus bytes.
    (tmp_path / "data/processed/chunks.jsonl").write_text('{"text": "Synthetic test corpus"}\n')
    return tmp_path


def write_report(root, filename, payload):
    path = root / "data/evaluation" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def dashboard(root):
    return AppTest.from_string(
        f"from pathlib import Path\n"
        f"from src.ui.evaluations import render_evaluation_dashboard\n"
        f"render_evaluation_dashboard(Path({str(root)!r}))\n",
        default_timeout=30,
    ).run()


def metric(app, label):
    return next(item.value for item in app.metric if item.label == label)


def regression_payload(answered=0, correct=0):
    def values(attempted, right):
        return {
            "total": 13, "answered": attempted, "correct": right,
            "abstained": 13 - attempted, "coverage": attempted / 13,
            "precision_when_answered": right / attempted if attempted else None,
            "accuracy_overall": right / 13,
        }
    return {"system_a_vanilla_rag": values(13, 5), "system_b_verified": values(answered, correct)}


def benchmark_report(root, *, records=None):
    dataset = load_benchmark(root / "data/benchmarks/polity_v1.json")
    questions = [q for q in dataset["questions"] if q["split"] == "development"]
    config = {
        "systems": ["verified"], "split": "development", "is_full_split": True,
        "question_ids": [q["id"] for q in questions],
        "dataset_sha256": sha256_file(root / "data/benchmarks/polity_v1.json"),
        "corpus_sha256": sha256_file(root / "data/processed/chunks.jsonl"),
    }
    if records is None:
        records = [{"question_id": q["id"], "status": "ABSTAINED", "predicted_answer": None,
                    "details": {"abstention_reason": "Synthetic missing-evidence case"}}
                   for q in questions]
    path = root / "data/evaluation/benchmark/development_results.json"
    report = save_checkpoint(path, config, questions, {"verified": records}, "2026-10-02T00:00:00Z")
    return dataset, report


def test_reports_appear_and_update_without_a_stale_cache(eval_root):
    assert load_reports(eval_root)["selective"].status == "Not run"
    write_report(eval_root, "selective_metrics.json", regression_payload())
    first = load_reports(eval_root)["selective"]
    assert first.status == "Saved"
    assert answer_metrics(first)["verified"]["precision_when_answered"] is None
    write_report(eval_root, "selective_metrics.json", regression_payload(2, 1))
    second = load_reports(eval_root)["selective"]
    assert answer_metrics(second)["verified"]["precision_when_answered"] == .5
    assert comparison_rows(answer_metrics(second))[0]["Wrong"] == 1


@pytest.mark.parametrize("payload", ('{broken', 'null', '[]', '{"summary": 3}'))
def test_unreadable_reports_do_not_break_other_measurements(eval_root, payload):
    path = eval_root / "data/evaluation/judge_results.json"
    path.parent.mkdir(parents=True)
    path.write_text(payload)
    write_report(eval_root, "selective_metrics.json", regression_payload(2, 1))
    reports = load_reports(eval_root)
    assert reports["judge"].status == "Unreadable"
    assert reports["selective"].status == "Saved"
    app = dashboard(eval_root)
    assert not app.exception, [item.message for item in app.exception]
    assert any("could not be read" in item.value for item in app.warning)


@pytest.mark.parametrize("change, message", [
    (lambda r: r["config"].update(dataset_sha256="changed"), "different dataset"),
    (lambda r: r["config"].update(corpus_sha256="changed"), "different corpus"),
    (lambda r: r["config"].update(split="test"), "different split"),
    (lambda r: r["config"].update(is_full_split=False), "smoke run"),
    (lambda r: r["config"].update(question_ids=["unknown"]), "question IDs"),
    (lambda r: r["config"].update(question_ids=[{}]), "question IDs"),
    (lambda r: r["metrics"]["verified"]["overall"].update(total=1), "metric totals"),
    (lambda r: r["metrics"]["verified"]["overall"].update(correct=10), "metrics disagree"),
    (lambda r: r["results"]["verified"][0].update(question_id=[]), "invalid, unknown"),
])
def test_wrong_scope_is_not_presented_as_a_full_benchmark(eval_root, change, message):
    dataset, payload = benchmark_report(eval_root)
    change(payload)
    write_report(eval_root, "benchmark/development_results.json", payload)
    report = load_reports(eval_root)["benchmark_development"]
    assert message in benchmark_issue(report, eval_root, "development", dataset)
    app = dashboard(eval_root)
    assert not app.exception, [item.message for item in app.exception]
    assert any(message in item.value for item in app.warning)
    assert not any(item.label == "Overall accuracy" for item in app.metric)


def test_empty_dashboard_never_initialises_answering_services(eval_root, monkeypatch):
    from src.evaluation import evaluate_benchmark
    def forbidden(*args, **kwargs):
        raise AssertionError("Evaluation viewing initialized answering services")
    monkeypatch.setattr(evaluate_benchmark, "make_services", forbidden)
    app = dashboard(eval_root)
    assert not app.exception, [item.message for item in app.exception]
    assert metric(app, "Benchmark questions") == "75"
    assert [tab.label for tab in app.tabs] == [
        "Overview", "Task specification", "Answer quality", "RAG / Retrieval", "Grounding & faithfulness", "Generation", "Stability", "Reports",
    ]
    assert any("has not been recorded" in item.value for item in app.info)


def test_zero_answers_and_refresh_are_rendered_correctly(eval_root):
    write_report(eval_root, "selective_metrics.json", regression_payload())
    app = dashboard(eval_root)
    app.radio(key="eval_question_set").set_value("13-question regression").run()
    assert not app.exception
    assert metric(app, "Precision when answered") == "—"
    write_report(eval_root, "selective_metrics.json", regression_payload(2, 1))
    app.button(key="eval_refresh_reports").click().run()
    assert not app.exception
    assert metric(app, "Precision when answered") == "50.0%"
    assert metric(app, "Coverage") == "15.4%"


def test_incomplete_benchmark_distinguishes_errors_from_abstentions(eval_root):
    dataset = load_benchmark(eval_root / "data/benchmarks/polity_v1.json")
    question = next(q for q in dataset["questions"] if q["split"] == "development")
    records = [{"question_id": question["id"], "status": "ERROR", "predicted_answer": None,
                "error_type": "SyntheticServiceError"}]
    benchmark_report(eval_root, records=records)
    app = dashboard(eval_root)
    assert not app.exception, [item.message for item in app.exception]
    assert any("Run incomplete" in item.value for item in app.warning)
    assert any("separate from evidence-based abstentions" in item.value for item in app.warning)
    assert metric(app, "Precision when answered") == "—"
    app.selectbox(key="eval_result_outcome").set_value("Abstained").run()
    assert not app.exception
    assert any("No saved results match" in item.value for item in app.info)


def test_question_outcomes_and_exports_keep_errors_distinct(eval_root):
    dataset = load_benchmark(eval_root / "data/benchmarks/polity_v1.json")
    questions = dataset["questions"][:3]
    rows = question_rows(questions, {"verified": [
        {"question_id": questions[0]["id"], "status": "ANSWERED", "predicted_answer": questions[0]["official_answer"]},
        {"question_id": questions[1]["id"], "status": "ABSTAINED", "predicted_answer": None},
        {"question_id": questions[2]["id"], "status": "ERROR", "predicted_answer": None},
    ]})
    assert [row["Outcome"] for row in rows] == ["Correct", "Abstained", "Run error"]
    export = table_csv(rows)
    assert "_question" not in export and "_record" not in export
    assert "Run error" in export
    assert "'=dangerous" in table_csv([{"Topic": "=dangerous"}])
    assert percent(None) == "—" and percent(float("nan")) == "—"


def test_smoke_snapshots_are_available_without_becoming_full_results(eval_root):
    _, report = benchmark_report(eval_root)
    path = eval_root / "data/evaluation/benchmark/development_results.json"
    path.unlink()
    report["config"]["is_full_split"] = False
    write_report(eval_root, "benchmark/development_first3_results.json", report)
    reports = load_reports(eval_root)
    assert reports["benchmark_development"].status == "Not run"
    assert reports["benchmark_snapshot_development_first3_results"].status == "Saved"


def test_generation_and_stability_controls_select_the_saved_experiment(eval_root):
    for arm, claims, decision, accepted in (("simple", 1, "ACCEPT", 100), ("statements", 3, "REJECT", 0)):
        filename = "generation_loop_results" + (".statements" if arm == "statements" else "") + ".json"
        write_report(eval_root, filename, {
            "summary": {
                "topics_completed": 1, "topics_errored": 0,
                "terminal_decisions": {"accept_rate": accepted},
                "repair": {"repair_rate": None},
                "claims": {"mean_claims_per_attempt": claims},
                "attempts": {"total_attempts": 1},
                "unverified_baseline": {"defect_rate": 100 - accepted},
            },
            "results": [{"topic": "Synthetic topic", "terminal_decision": decision,
                         "attempts": [{"attempt": 1, "decision": decision, "mcq": {"question": "Synthetic question"}}]}],
        })
    write_report(eval_root, "verdict_stability.json", {
        "model": "synthetic", "repeats": 2,
        "conditions": {"pinned": {
            "temperature": 0, "summary": {"claims": 2, "stable_claims": 1, "unstable_claims": 1, "stability_rate": .5},
            "rows": [
                {"q_number": 1, "claim": "Stable test", "stable": True, "verdicts": ["SUPPORTED", "SUPPORTED"]},
                {"q_number": 2, "claim": "Changing test", "stable": False, "verdicts": ["SUPPORTED", "INSUFFICIENT"]},
            ],
        }},
    })
    app = dashboard(eval_root)
    assert not app.exception
    assert metric(app, "Mean claims per attempt") == "1"
    app.radio(key="eval_generation_arm").set_value("statements").run()
    assert not app.exception
    assert metric(app, "Mean claims per attempt") == "3"
    stability_tables = [table.value for table in app.dataframe if "Verdicts across repeats" in table.value.columns]
    assert len(stability_tables[0]) == 1
    app.checkbox(key="eval_unstable_only").uncheck().run()
    assert not app.exception
    stability_tables = [table.value for table in app.dataframe if "Verdicts across repeats" in table.value.columns]
    assert len(stability_tables[0]) == 2


def test_standalone_and_main_app_show_the_shared_dashboard():
    standalone = AppTest.from_file(str(ROOT / "evals_app.py"), default_timeout=30).run()
    assert not standalone.exception, [item.message for item in standalone.exception]
    assert any(tab.label == "Grounding & faithfulness" for tab in standalone.tabs)
    assert any(item.label == "Verifier faithfulness" for item in standalone.metric)
    main = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not main.exception, [item.message for item in main.exception]
    shared_labels = [tab.label for tab in standalone.tabs]
    assert not main.tabs
    assert not any(item.label == "Benchmark questions" for item in main.metric)
    main.radio(key="workspace").set_value("How it works").run()
    assert not main.exception
    assert any(item.value == "About this site" for item in main.subheader)
    next(button for button in main.button if button.label == "Explore guardrails").click().run()
    assert not main.exception, [item.message for item in main.exception]
    assert main.radio(key="about_section").value == "Guardrails"
    assert any(item.value == "Guardrails" for item in main.subheader)
    assert not main.metric
    main.radio(key="about_section").set_value("Evals").run()
    assert not main.exception
    assert [tab.label for tab in main.tabs if tab.label in shared_labels] == shared_labels
    main.sidebar.radio[0].set_value("Practice").run()
    assert not main.exception
    assert any(button.label == "Generate MCQs" for button in main.button)
    assert "Verify a question" not in main.radio(key="workspace").options
    legacy = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    legacy.session_state["workspace"] = "Verify a question"
    legacy.run()
    assert not legacy.exception, [item.message for item in legacy.exception]
    assert legacy.radio(key="workspace").value == "Practice"


def test_current_generation_report_requires_current_policy_and_source_snapshot(eval_root):
    from src.orchestration.telemetry import GENERATION_POLICY

    legacy = {"summary": {"claims": {"mean_claims_per_attempt": 99}}, "results": []}
    write_report(eval_root, "generation_loop_results.json", legacy)
    current = {
        "config": {
            "generation_policy": GENERATION_POLICY,
            "corpus_hash": sha256_file(eval_root / "data/processed/chunks.jsonl"),
            "corpus_changed_during_run": False,
        },
        "summary": {"claims": {"mean_claims_per_attempt": 0}}, "results": [],
    }
    write_report(eval_root, "generation_grounded_results.json", current)
    app = dashboard(eval_root)
    assert not app.exception
    assert metric(app, "Mean claims per attempt") == "0"
    current["config"]["generation_policy"] = "superseded-policy"
    write_report(eval_root, "generation_grounded_results.json", current)
    app.run()
    assert not app.exception
    assert metric(app, "Mean claims per attempt") == "99"
    current["config"]["generation_policy"] = GENERATION_POLICY
    write_report(eval_root, "generation_grounded_results.json", current)
    (eval_root / "data/processed/chunks.jsonl").write_text("changed source")
    app.run()
    assert not app.exception
    assert metric(app, "Mean claims per attempt") == "99"
