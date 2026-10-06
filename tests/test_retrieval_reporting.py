"""Check trace provenance, retrieval scoring, freshness and read-only inspection."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from src.evaluation.benchmark import sha256_file
from src.evaluation.evaluate_retrieval import run_retrieval_benchmark
from src.evaluation.retrieval_report import RETRIEVAL_REPORT_VERSION, retrieval_report_issue
from src.orchestration.telemetry import (
    GENERATION_POLICY, RETRIEVAL_TRACE_VERSION, capture_telemetry, measured_stage,
    record_evidence_packet,
)
from src.practice.store import PracticeStore
from src.retrieval.semantic_reranker import EMBEDDING_MODEL_NAME, MODEL_NAME, SemanticReranker
from src.ui.evaluation_data import load_reports
from src.ui.retrieval_data import load_checked_question, load_practice_runs


def fake_reranker(candidates, *, fail=False):
    instance = SemanticReranker.__new__(SemanticReranker)
    instance.candidate_k = 20
    instance.retriever = SimpleNamespace(retrieve=lambda query, top_k: deepcopy(candidates[:top_k]))

    def predict(pairs):
        if fail:
            raise RuntimeError("PRIVATE_ERROR_MESSAGE")
        return list(range(1, len(pairs) + 1))

    instance.reranker = SimpleNamespace(predict=predict)
    return instance


@pytest.fixture
def candidates():
    return [{
        "id": rank, "source": "synthetic", "document": "test.txt", "page": rank,
        "chunk_index": 0, "text": f"SOURCE_EXCERPT_{rank}", "score": 1 - rank * .1,
    } for rank in range(1, 7)]


@pytest.fixture
def retrieval_root(tmp_path, candidates):
    corpus = tmp_path / "data/processed/chunks.jsonl"
    corpus.parent.mkdir(parents=True)
    corpus.write_text("\n".join(json.dumps(row) for row in candidates))
    queries = [
        {"query_id": "q1", "query": "Find the sixth passage", "expected_source": "synthetic", "expected_pages": [6]},
        {"query_id": "q2", "query": "Find the first passage", "expected_source": "synthetic", "expected_pages": [1]},
    ]
    query_path = tmp_path / "src/evaluation/retrieval_queries_gold.json"
    query_path.parent.mkdir(parents=True)
    query_path.write_text(json.dumps(queries))
    config = {
        "report_version": RETRIEVAL_REPORT_VERSION, "corpus_sha256": sha256_file(corpus),
        "queries_sha256": sha256_file(query_path), "query_ids": [row["query_id"] for row in queries],
        "is_full_benchmark": True, "top_k": 5, "candidate_k": 20,
        "embedding_model": EMBEDDING_MODEL_NAME, "reranker_model": MODEL_NAME,
        "corpus_changed_during_run": False, "queries_changed_during_run": False,
    }
    report = run_retrieval_benchmark(fake_reranker(candidates), queries, config)
    path = tmp_path / "data/evaluation/retrieval_results.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(report))
    return tmp_path, report


def dashboard(root):
    return AppTest.from_string(
        "from pathlib import Path\nfrom src.ui.evaluations import render_evaluation_dashboard\n"
        f"render_evaluation_dashboard(Path({str(root)!r}))\n", default_timeout=30,
    ).run()


def test_actual_rank_changes_and_stage_timings_are_recorded_without_mutating_candidates(candidates):
    original = deepcopy(candidates)
    with capture_telemetry() as run:
        run.question_number = 3
        with measured_stage("verify_claims", attempt=2):
            output = fake_reranker(candidates).retrieve("Find evidence", top_k=5)
            record_evidence_packet("Checked statement", output)
    assert candidates == original
    assert [row["id"] for row in output] == [6, 5, 4, 3, 2]
    trace = run.retrievals[0]
    assert (trace["question"], trace["attempt"], trace["stage"]) == (3, 2, "verify_claims")
    assert [row["id"] for row in trace["candidates"]] == [1, 2, 3, 4, 5, 6]
    assert trace["selected"][0]["semantic_rank"] == 6
    assert trace["selected"][0]["reranker_rank"] == 1
    assert trace["semantic_seconds"] >= 0 and trace["rerank_seconds"] >= 0
    assert trace["elapsed_seconds"] >= trace["semantic_seconds"] + trace["rerank_seconds"]
    assert run.evidence_packets[0]["retrieval_ids"] == [trace["id"]]
    assert run.summary()["api_calls"] == 0
    with capture_telemetry() as separate:
        fake_reranker(candidates).retrieve("Another query")
    assert separate.retrievals[0]["stage"] == "Model call"
    assert separate.retrievals[0]["attempt"] == 0


def test_failed_reranking_retains_search_candidates_but_not_provider_messages(candidates):
    with capture_telemetry() as run, pytest.raises(RuntimeError):
        fake_reranker(candidates, fail=True).retrieve("A safe query")
    trace = run.retrievals[0]
    assert not trace["success"] and trace["error_type"] == "RuntimeError"
    assert len(trace["candidates"]) == 6 and trace["selected"] == []
    assert "PRIVATE_ERROR_MESSAGE" not in json.dumps(run.retrievals)


def test_empty_candidate_search_returns_no_passages_or_invented_scores():
    with capture_telemetry() as run:
        assert fake_reranker([]).retrieve("Missing sources") == []
    assert run.retrievals[0]["candidates"] == []
    assert run.retrievals[0]["selected"] == []
    assert run.retrievals[0]["rerank_seconds"] is None


def test_benchmark_uses_actual_first_stage_results_and_preserves_the_page_hit_definition(retrieval_root):
    root, report = retrieval_root
    assert report["metrics"]["semantic"]["hits"] == 1
    assert report["metrics"]["reranked"]["hits"] == 1
    assert report["results"][0]["trace"]["selected"][0]["page"] == 6
    assert retrieval_report_issue(report, root) is None
    assert retrieval_report_issue(json.loads(json.dumps(report)), root) is None


@pytest.mark.parametrize("change,message", [
    (lambda r: r["config"].update(corpus_sha256="different"), "different corpus"),
    (lambda r: r["config"].update(queries_sha256="different"), "different query set"),
    (lambda r: r["config"].update(top_k=3), "depths differ"),
    (lambda r: r["config"].update(is_full_benchmark=False), "partial retrieval"),
    (lambda r: r["config"].update(corpus_changed_during_run=True), "changed during"),
    (lambda r: r["metrics"]["reranked"].update(hits=2), "metrics disagree"),
    (lambda r: r["results"][0].update(expected_pages=[1]), "source-page labels"),
    (lambda r: r["results"][0]["trace"]["selected"].reverse(), "disagree with the recorded reranking"),
])
def test_stale_or_inconsistent_retrieval_reports_cannot_be_presented_as_current(retrieval_root, change, message):
    root, report = retrieval_root
    change(report)
    assert message in retrieval_report_issue(report, root)


def test_retrieval_dashboard_is_read_only_and_never_initializes_models(retrieval_root, monkeypatch):
    root, _ = retrieval_root
    def forbidden(*args, **kwargs):
        raise AssertionError("Viewing retrieval results constructed a model")
    monkeypatch.setattr(SemanticReranker, "__init__", forbidden)
    app = dashboard(root)
    assert not app.exception, [item.message for item in app.exception]
    assert "RAG / Retrieval" in [tab.label for tab in app.tabs]
    assert next(item.value for item in app.metric if item.label == "Reranked Recall@5") == "50.0%"
    ranking_tables = [table.value for table in app.dataframe if "Semantic rank" in table.value.columns]
    assert [len(table) for table in ranking_tables] == [6, 5]
    assert not (root / "data/practice").exists()


def test_invalid_benchmark_is_warned_about_without_displaying_current_scores(retrieval_root):
    root, report = retrieval_root
    report["metrics"]["semantic"]["hits"] = 99
    (root / "data/evaluation/retrieval_results.json").write_text(json.dumps(report))
    app = dashboard(root)
    assert not app.exception
    assert any("metrics disagree" in item.value for item in app.warning)
    assert not any(item.label == "Reranked Recall@5" for item in app.metric)


def test_retrieval_smoke_reports_remain_snapshots_without_replacing_full_benchmark(retrieval_root):
    root, report = retrieval_root
    report["config"]["is_full_benchmark"] = False
    (root / "data/evaluation/retrieval_smoke_results.json").write_text(json.dumps(report))
    (root / "data/evaluation/retrieval_results.json").unlink()
    reports = load_reports(root)
    assert reports["retrieval"].status == "Not run"
    assert reports["retrieval_snapshot_retrieval_smoke_results"].status == "Saved"


def test_reading_an_absent_practice_store_does_not_create_files(tmp_path):
    path = tmp_path / "missing/practice.sqlite3"
    assert load_practice_runs(path) == []
    assert load_checked_question(path, "unknown") is None
    assert not path.parent.exists()


def test_active_quiz_hides_its_trace_and_matching_bank_history_until_submission(retrieval_root):
    root, report = retrieval_root
    (root / "data/evaluation/retrieval_results.json").unlink()
    store = PracticeStore(root / "data/practice/practice.sqlite3")
    run = {
        "id": "archived-run", "created_at": "2026-10-06T00:00:00Z", "topic": "Test topic",
        "generation_policy": GENERATION_POLICY, "corpus_hash": report["config"]["corpus_sha256"],
        "accepted": 1, "reused": 0, "status": "complete", "candidates": [],
        "retrieval_trace_version": RETRIEVAL_TRACE_VERSION,
        "retrievals": [report["results"][0]["trace"]], "evidence_packets": [],
        "question_records": [{"slot": 1, "question_id": "active-question", "origin": "generated"}],
    }
    trace = run["retrievals"][0]
    passage = trace["selected"][0]
    run["evidence_packets"] = [{
        "label": "Statement I", "retrieval_ids": [trace["id"]], "evidence": [passage],
        "article_anchors": [{"source": passage["source"], "document": passage.get("document"),
                             "page": passage["page"], "articles": ["15"]}],
    }]
    store.save_run(run)
    app = AppTest.from_string(
        "from pathlib import Path\nimport streamlit as st\n"
        "from src.ui.evaluations import render_evaluation_dashboard\n"
        "if 'practice_session' not in st.session_state:\n"
        "    st.session_state.practice_session = {'run_id': 'new-bank-reuse', 'submitted': False, "
        "'questions': [{'id': 'active-question'}]}\n"
        f"render_evaluation_dashboard(Path({str(root)!r}))\n", default_timeout=30,
    ).run()
    assert not app.exception
    assert any("Submit your active practice" in item.value for item in app.info)
    assert not any("SOURCE_EXCERPT_" in item.value for item in app.text)
    assert not any("Named Article lookup" in item.value for item in app.caption)
    app.session_state["practice_session"]["submitted"] = True
    app.run()
    assert not app.exception
    assert any("SOURCE_EXCERPT_" in item.value for item in app.text)
    assert any("Named Article lookup" in item.value for item in app.caption)
    assert len(load_practice_runs(store.path)) == 1
