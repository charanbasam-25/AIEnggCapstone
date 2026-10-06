"""Inspect real practice searches and saved retrieval benchmarks without rerunning them."""

from pathlib import Path
import json

import streamlit as st

from src.evaluation.benchmark import sha256_file
from src.evaluation.retrieval_report import (
    RETRIEVAL_SYSTEM_NAMES, page_hit, retrieval_report_issue,
)
from src.orchestration.telemetry import RETRIEVAL_TRACE_VERSION
from src.ui.evaluation_data import ROOT, SavedReport, percent
from src.ui.retrieval_data import load_checked_question, load_practice_runs, passage_rows


STAGE_NAMES = {
    "retrieve_sources": "Generation sources", "verify_claims": "Statement verification",
    "verify_answer_key": "Answer verification", "retrieval_benchmark": "Benchmark search",
}


def _records(value) -> list[dict]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _pending_run(run: dict) -> bool:
    session = st.session_state.get("practice_session")
    if not session or session.get("submitted"):
        return False
    active_ids = {row.get("id") for row in _records(session.get("questions"))}
    run_ids = {row.get("question_id") for row in _records(run.get("question_records"))}
    return session.get("run_id") == run["id"] or bool(active_ids & run_ids)


def _checked_question(root: Path, question_id: str):
    from src.practice.models import PracticeQuestion

    sessions = [st.session_state.get("practice_session") or {},
                *st.session_state.get("practice_history", [])]
    for session in sessions:
        for payload in _records(session.get("questions")):
            if payload.get("id") == question_id:
                try:
                    return PracticeQuestion.model_validate(payload)
                except ValueError:
                    return None
    return load_checked_question(root / "data/practice/practice.sqlite3", question_id)


def _cited_pages(question) -> set:
    if question is None:
        return set()
    return {(question.evidence[citation.evidence_id - 1]["source"],
             question.evidence[citation.evidence_id - 1]["page"])
            for note in [question.notes.summary, *question.notes.options] for citation in note.citations}


def _render_trace(trace: dict, key: str, cited_pages=()) -> None:
    st.markdown("**Search query**")
    st.write(trace.get("query", "Query not recorded"))
    st.caption(
        f"Embedding: {trace.get('embedding_model', 'Not recorded')} · "
        f"Cross-encoder: {trace.get('reranker_model', 'Not recorded')}"
    )
    if not trace.get("success", False):
        st.warning(f"This search stopped: {trace.get('error_type') or 'No completed result'}.")
    candidates, selected = _records(trace.get("candidates")), _records(trace.get("selected"))
    st.markdown(f"**Semantic candidates · {len(candidates)} returned**")
    st.dataframe(passage_rows(candidates, cited_pages), hide_index=True, width="stretch")
    st.markdown(f"**Reranked passages · top {trace.get('top_k', '?')} requested, {len(selected)} returned**")
    st.dataframe(passage_rows(selected, cited_pages), hide_index=True, width="stretch")
    st.caption("Cross-encoder scores rank relevance; they are not confidence percentages. ‘Cited page’ means the final learner explanation cites that page, not that every chunk on it proves the answer.")
    if candidates:
        with st.expander("Read a candidate passage"):
            choice = st.selectbox(
                "Candidate passage", range(len(candidates)), key=f"{key}_candidate",
                format_func=lambda index:
                    f"Semantic #{candidates[index].get('semantic_rank')} · "
                    f"{candidates[index].get('source')} · PDF page {candidates[index].get('page')}",
            )
            st.text(candidates[choice].get("text") or "Text not recorded")
    with st.expander("Read the selected passages"):
        for item in selected:
            st.caption(f"Reranked #{item.get('reranker_rank')} · {item.get('source')} · PDF page {item.get('page')} · Chunk {item.get('id')}")
            st.text(item.get("text") or "Text not recorded")


def _render_question_outcome(run: dict, slot: int, question) -> None:
    if run.get("error_type") == "CorpusChanged":
        st.warning("This run was retired because its source snapshot changed. No question from it was released as an active practice set.")
        return
    candidate = next((row for row in _records(run.get("candidates")) if row.get("slot") == slot), None)
    if candidate:
        st.markdown(f"**Preparation slot {slot} · {candidate.get('decision', 'Not recorded')}**")
        history = _records(candidate.get("attempt_history"))
        if history:
            st.dataframe([{
                "Draft": item.get("attempt"), "Decision": item.get("decision"),
                "Answer check": item.get("answer_verdict"), "Quality": item.get("quality_verdict"),
                "Explanations": item.get("explanation_verdict"),
                "Reasons": " · ".join(item.get("failure_reasons", [])),
            } for item in history], hide_index=True, width="stretch")
        elif candidate.get("failure_reasons"):
            for reason in candidate["failure_reasons"]:
                st.write(reason)
    if question is None:
        st.caption("No current-policy approved question record is available for this slot. Retrieved passages alone do not approve an answer.")
        return
    with st.container(border=True):
        st.markdown("**Source-checked question**")
        st.write(question.mcq.question)
        for letter in "ABCD":
            st.write(f"{letter}. {getattr(question.mcq, f'option_{letter.lower()}')}")
        st.write(f"Reviewed answer: {question.mcq.correct_answer}")
        st.write(question.notes.summary.text)
        st.caption("All publication checks passed. This is an automated review, not an independent correctness score.")
        with st.expander("Exact quotations used in the learner explanations"):
            seen = set()
            for note in [question.notes.summary, *question.notes.options]:
                for citation in note.citations:
                    evidence = question.evidence[citation.evidence_id - 1]
                    identity = (evidence["source"], evidence["page"], citation.quote)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    st.caption(f"{evidence['source']} · PDF page {evidence['page']}")
                    st.text(citation.quote)


def _render_practice_searches(root: Path) -> None:
    st.markdown("### Practice searches")
    runs = load_practice_runs(root / "data/practice/practice.sqlite3")
    current = st.session_state.get("practice_last_run")
    if current and not any(run["id"] == current["id"] for run in runs):
        runs.insert(0, current)
    if not runs:
        st.info("Generate a practice set to record its search queries, rankings and evidence.")
        return
    by_id = {run["id"]: run for run in runs}
    current_id = current["id"] if current else None
    if current_id and current_id != st.session_state.get("eval_rag_current_id"):
        st.session_state["eval_rag_run"] = current_id
        st.session_state["eval_rag_current_id"] = current_id
    if st.session_state.get("eval_rag_run") not in by_id:
        st.session_state["eval_rag_run"] = current_id or runs[0]["id"]
    selected_id = st.selectbox(
        "Practice retrieval run", list(by_id), key="eval_rag_run",
        format_func=lambda run_id:
            f"{by_id[run_id].get('created_at', '')[:19].replace('T', ' ')} UTC · "
            f"{by_id[run_id].get('topic', 'Topic')} · {by_id[run_id].get('status', 'Not recorded')} · {run_id[:6]}",
    )
    run = by_id[selected_id]
    traces = _records(run.get("retrievals"))
    if run.get("retrieval_trace_version") != RETRIEVAL_TRACE_VERSION:
        st.info("This earlier run has no recorded candidate-search trace. New generation runs record the actual searches; earlier rankings cannot be reconstructed from final citations.")
        return
    corpus = root / "data/processed/chunks.jsonl"
    if not corpus.is_file() or run.get("corpus_hash") != sha256_file(corpus):
        st.info("Historical source snapshot: these searches used a different corpus from the current reference library.")
    for column, label, value in zip(
        st.columns(3), ("Recorded searches", "Passages reranked", "Search & rerank time"),
        (len(traces), sum(len(_records(trace.get("candidates"))) for trace in traces),
         f"{sum(trace.get('elapsed_seconds', 0) or 0 for trace in traces):.2f}s"),
    ):
        column.metric(label, value)
    st.caption("Timings cover actual search calls and exclude initial model loading. Preparation time in Cost & latency includes loading, model calls, retries and other checks.")
    if _pending_run(run):
        st.info("Submit your active practice set to unlock its search passages and reviewed answer here.")
        return
    if not traces:
        if run.get("reused"):
            st.info(f"{run['reused']} question(s) reused from the checked bank. This run made no new retrieval searches; stored answer evidence is shown below.")
        else:
            st.info("No retrieval search completed or was recorded for this run.")
    records = _records(run.get("question_records"))
    slots = sorted({trace.get("question", 0) for trace in traces}
                   | {row.get("slot", 0) for row in records}
                   | {row.get("slot", 0) for row in _records(run.get("candidates"))})
    if not slots:
        return
    slot = st.selectbox("Preparation slot", slots, key=f"eval_rag_slot_{selected_id}",
                        format_func=lambda value: f"Question slot {value}")
    record = next((row for row in records if row.get("slot") == slot), None)
    question = _checked_question(root, record["question_id"]) if record else None
    question_traces = [trace for trace in traces if trace.get("question") == slot]
    if question_traces:
        st.dataframe([{
            "Draft": str(trace["attempt"]) if trace.get("attempt") else "Source search",
            "Stage": STAGE_NAMES.get(trace.get("stage"), trace.get("stage")),
            "Query": trace.get("query"), "Candidates": len(_records(trace.get("candidates"))),
            "Selected": len(_records(trace.get("selected"))), "Semantic seconds": trace.get("semantic_seconds"),
            "Rerank seconds": trace.get("rerank_seconds"), "Total seconds": trace.get("elapsed_seconds"),
            "Outcome": "Completed" if trace.get("success") else trace.get("error_type", "Failed"),
        } for trace in question_traces], hide_index=True, width="stretch")
        trace_by_id = {trace["id"]: trace for trace in question_traces}
        trace_id = st.selectbox(
            "Search to inspect", list(trace_by_id), key=f"eval_rag_search_{selected_id}_{slot}",
            format_func=lambda value:
                f"{STAGE_NAMES.get(trace_by_id[value].get('stage'), 'Search')} · "
                f"Draft {trace_by_id[value].get('attempt') or 'source'} · {trace_by_id[value].get('query', '')[:100]}",
        )
        _render_trace(trace_by_id[trace_id], f"rag_{selected_id}_{trace_id}", _cited_pages(question))
        packets = [packet for packet in _records(run.get("evidence_packets"))
                   if trace_id in packet.get("retrieval_ids", [])]
        if packets:
            with st.expander("Expanded source pages passed to this stage"):
                seen = set()
                for packet in packets:
                    st.markdown(f"**{packet.get('label', 'Checked evidence')}**")
                    anchors = _records(packet.get("article_anchors"))
                    if anchors:
                        st.caption("Named Article lookup adds its source pages alongside ranked passages. These pages still require evidence review.")
                        st.dataframe([{
                            "Article": ", ".join(row.get("articles", [])),
                            "Source": row.get("source"), "Document": row.get("document"),
                            "PDF page": row.get("page"),
                        } for row in anchors], hide_index=True, width="stretch")
                    for passage in _records(packet.get("evidence")):
                        identity = (passage.get("source"), passage.get("document"), passage.get("page"), passage.get("text"))
                        if identity in seen:
                            continue
                        seen.add(identity)
                        st.caption(f"{passage.get('source')} · PDF page {passage.get('page')} · {passage.get('context_kind', 'Recorded passage')}")
                        st.text(passage.get("text") or "Text not recorded")
                st.caption("Complete source pages preserve qualifications and footnotes. An evidence packet records what was supplied; relevance and support still require verification.")
    _render_question_outcome(run, slot, question)
    st.download_button(
        "Download retrieval diagnostics", json.dumps(run, indent=2),
        "practice_retrieval.json", "application/json", key=f"eval_rag_download_{selected_id}",
    )


def _render_retrieval_benchmark(report: SavedReport, root: Path) -> None:
    st.markdown("### Saved retrieval benchmark")
    st.write("Check whether an annotated relevant source page appears in the top five. Semantic search and reranked search use the same recorded candidate pool.")
    if report.error:
        st.warning(report.error)
        return
    if report.data is None:
        st.info("No structured retrieval benchmark has been saved yet.")
        st.code(".venv/bin/python -m src.evaluation.evaluate_retrieval", language="bash")
        st.caption("This runs the local embedding and cross-encoder models. It does not call an LLM API.")
        return
    issue = retrieval_report_issue(report.data, root)
    if issue:
        st.warning(issue)
        st.code(".venv/bin/python -m src.evaluation.evaluate_retrieval", language="bash")
        return
    payload = report.data
    metrics, config = payload["metrics"], payload["config"]
    for column, label, value in zip(
        st.columns(3), ("Annotated retrieval queries", "Semantic Recall@5", "Reranked Recall@5"),
        (len(payload["results"]), percent(metrics["semantic"]["hit_rate"]), percent(metrics["reranked"]["hit_rate"])),
    ):
        column.metric(label, value)
    st.dataframe([{
        "Retriever": RETRIEVAL_SYSTEM_NAMES[system], "Hits": values["hits"], "Misses": values["misses"],
        "Page hit rate": percent(values["hit_rate"]), "Mean search seconds": values["mean_search_seconds"],
    } for system, values in metrics.items()], hide_index=True, width="stretch")
    st.caption(f"Saved {report.updated} · {config['candidate_k']} semantic candidates → top {config['top_k']} reranked chunks. Recall@5 here means queries with at least one annotated source-page hit ÷ all queries; it is not answer accuracy or a complete relevance audit.")
    st.caption("Semantic timing is the shared first-stage search; reranked timing includes that search plus the cross-encoder. Initial model loading is excluded from these per-query times.")
    rows = _records(payload["results"])
    st.dataframe([{
        "Query ID": row["query_id"], "Query": row["query"], "Expected source": row["expected_source"],
        "Expected PDF pages": ", ".join(map(str, row["expected_pages"])),
        "Semantic": "Hit" if page_hit(row["trace"]["candidates"][:config["top_k"]], row) else "Miss",
        "Reranked": "Hit" if page_hit(row["trace"]["selected"], row) else "Miss",
    } for row in rows], hide_index=True, width="stretch")
    index = st.selectbox("Inspect a benchmark query", range(len(rows)), key="eval_rag_benchmark_query",
                         format_func=lambda value: f"{rows[value]['query_id']} · {rows[value]['query']}")
    _render_trace(rows[index]["trace"], f"rag_benchmark_{rows[index]['query_id']}")


def render_retrieval_dashboard(report: SavedReport, root: Path = ROOT) -> None:
    st.subheader("Follow the evidence through RAG")
    st.write("Semantic search → cross-encoder reranking → complete source-page context → answer and explanation checks.")
    _render_practice_searches(root)
    st.divider()
    _render_retrieval_benchmark(report, root)
    st.caption("Viewing, filtering or downloading these diagnostics reads saved data and makes no retrieval or model calls.")
