"""Evaluation dashboard shared by app.py and the standalone evals app."""

import json
from pathlib import Path
import re

import streamlit as st

from src.evaluation.benchmark import inventory, load_benchmark
from src.evaluation.evaluate_benchmark import VERIFICATION_PIPELINE
from src.ui.evaluation_tasks import render_task_specification
from src.ui.evaluation_data import (
    ROOT, SYSTEM_NAMES, SavedReport, answer_metrics, benchmark_issue,
    comparison_rows, legacy_question_rows, load_reports, number, percent,
    question_rows, table_csv,
)


SCORE_LABELS = {
    "factual_correctness": "Factual correctness",
    "evidence_faithfulness": "Evidence faithfulness",
    "reasoning_validity": "Reasoning validity",
    "epistemic_honesty": "Epistemic honesty",
}


def _text(value: object) -> str:
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", str(value)).strip()


def _mapping(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _records(value: object) -> list[dict]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _score(value: object) -> str:
    value = number(value)
    return "—" if value is None else f"{value:.2f} / 5"


def _available(report: SavedReport, message: str) -> bool:
    if report.error:
        st.warning(f"{report.label}: {report.error}")
        return False
    if report.data is None:
        st.info(message)
        return False
    return True


def _report_caption(report: SavedReport) -> None:
    st.caption(f"{report.scope} · saved {report.updated}")


def _download(rows: list[dict], filename: str, key: str) -> None:
    st.download_button("Download table as CSV", table_csv(rows), file_name=filename,
                       mime="text/csv", key=key, disabled=not rows)


def _render_overview(reports: dict[str, SavedReport], dataset: dict | None) -> None:
    st.subheader("A clear view of system quality")
    st.write("Use RAG / Retrieval to inspect search evidence, answer quality to compare answers and abstentions, and the remaining checks to assess explanations, revisions and consistency.")
    for column, title, description in zip(st.columns(3),
        ("Answer quality", "Evidence quality", "Workflow quality"),
        ("Accuracy, coverage and precision on fixed question sets.",
         "Page citations and an LLM judge's assessment of explanations.",
         "Generated-question checks, revisions and repeated verdicts.")):
        with column, st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(description)

    st.subheader("Available measurements")
    st.dataframe([{
        "Measurement": report.label, "Question set / workflow": report.scope,
        "Status": report.status, "Last saved": report.updated,
    } for key, report in reports.items() if not key.endswith("records")], hide_index=True, width="stretch")
    st.caption("The 75-question benchmark and the earlier 13-question regression set are separate measurements. Grounding, judging and stability reports currently describe the regression set.")

    if dataset:
        counts = inventory(dataset)
        with st.expander("Benchmark composition and source papers"):
            st.dataframe([{
                "Question format": label,
                "Development": counts["formats_by_split"]["development"].get(kind, 0),
                "Test": counts["formats_by_split"]["test"].get(kind, 0),
            } for kind, label in (
                ("statement_mcq", "Numbered statements"),
                ("direct_mcq", "Direct options / assertion and reason"),
                ("best_answer_mcq", "Best answer"),
            )], hide_index=True, width="stretch")
            for source in dataset["sources"].values():
                st.markdown(f"**{source['year']}** · [Question paper]({source['paper_url']}) · [Official answer key]({source['key_url']})")
            st.caption("The fixed splits retain their original names. The test split has now been inspected during verifier debugging and is used for regression checks. A fresh untouched set is needed for independent evaluation. Corpus answerability and gold evidence still need manual review.")

    with st.expander("What each metric means"):
        st.markdown("""
        - **Coverage:** answered questions ÷ all questions.
        - **Precision when answered:** correct answers ÷ answered questions. It is undefined when no questions were answered.
        - **Overall accuracy:** correct answers ÷ all questions, including abstentions.
        - **Citation coverage:** committed verdicts with page citations ÷ all committed verdicts. Citations need a separate check for actual support.
        - **Faithfulness:** the model judge's assessment of whether an explanation stays within its evidence, scored from 1 to 5.
        - **Repair rate:** topics blocked on the first attempt that were eventually accepted after revision.
        - **Stability:** claims receiving the same verdict in every recorded repeat. A stable verdict can still be wrong.
        """)


def _render_answer_quality(reports: dict[str, SavedReport], root: Path, dataset: dict | None) -> None:
    st.subheader("Compare answers and abstentions")
    chosen = st.radio("Question set", ("75-question benchmark", "13-question regression"),
                      horizontal=True, key="eval_question_set")
    if chosen == "75-question benchmark":
        if dataset is None:
            st.info("The benchmark dataset is unavailable. The saved regression reports remain accessible in the other question set.")
            return
        split = st.selectbox("Benchmark results", ("development", "test"),
                             format_func=str.title, key="benchmark_split")
        report = reports[f"benchmark_{split}"]
        if not _available(report, "Dataset ready. A model evaluation has not been recorded for this split yet."):
            with st.expander("Produce results for this split"):
                st.code(f".venv/bin/python -m src.evaluation.evaluate_benchmark --run --split {split} --output data/evaluation/benchmark/{split}_reference_context_v2.json", language="bash")
                st.caption("This calls the configured model and saves a resumable comparison of the source verifier and vanilla RAG.")
            return
        issue = benchmark_issue(report, root, split, dataset)
        if issue:
            st.warning(issue)
            return
        payload = report.data
        if payload["config"].get("evaluation_scope"):
            st.caption(payload["config"]["evaluation_scope"])
        if "verified" in payload["results"] and payload["config"].get("verification_pipeline") != VERIFICATION_PIPELINE:
            st.caption("These saved scores use an earlier verification pipeline. The current quotation references, Article context and citation relevance screen need a new full run. Development smoke checks are available under Reports.")
        questions = [q for q in dataset["questions"] if q["split"] == split]
        explorer = question_rows(questions, payload["results"])
    else:
        report = reports["selective"]
        if not _available(report, "No answer-quality report has been saved for the regression set yet."):
            return
        payload = report.data
        explorer = legacy_question_rows(reports)
        st.caption("Historical 2025 regression results · 13 numbered-statement questions · scores use the stored answer keys.")

    metrics = answer_metrics(report)
    if not metrics:
        st.info("This report has no system metrics to display yet.")
        return
    if chosen == "75-question benchmark":
        pending = sum(values.get("pending", 0) for values in metrics.values())
        errors = sum(values.get("errors", 0) for values in metrics.values())
        if not payload.get("complete") or pending:
            st.warning(f"Run incomplete: {pending} system/question results are still pending. The displayed metrics are preliminary.")
        if errors:
            st.warning(f"This run contains {errors} run error(s). Errors are separate from evidence-based abstentions; resolve them before quoting performance.")
    _report_caption(report)
    focus = st.selectbox("Focus system", list(metrics), format_func=SYSTEM_NAMES.get, key="eval_focus_system")
    values = metrics[focus]
    for column, label, value, help_text in zip(st.columns(4),
        ("Overall accuracy", "Precision when answered", "Coverage", "Wrong answers"),
        (percent(values.get("accuracy_overall")), percent(values.get("precision_when_answered")),
         percent(values.get("coverage")), values.get("wrong", "—")),
        ("Correct answers divided by every question in the set.",
         "Correct answers divided by attempted questions. — means no defined score.",
         "The fraction of questions for which the system selected an answer.",
         "Answered questions that disagree with the stored answer key.")):
        column.metric(label, value, help=help_text)

    st.subheader("System comparison")
    comparison = comparison_rows(metrics)
    st.dataframe(comparison, hide_index=True, width="stretch")
    chart = [{"System": SYSTEM_NAMES[system], **{
        label: number(values.get(field)) for label, field in
        (("Correct", "correct"), ("Wrong", "wrong"), ("Abstained", "abstained"))
    }} for system, values in metrics.items()]
    st.bar_chart(chart, x="System", y=["Correct", "Wrong", "Abstained"], stack=True,
                 color=["#16a34a", "#e35d6a", "#94a3b8"], y_label="Questions", height=280)
    st.caption("Read precision together with coverage. A system can be accurate on its few attempted questions while leaving many unanswered.")
    _download(comparison, f"{report.path.stem}_comparison.csv", "eval_comparison_csv")

    if chosen == "75-question benchmark":
        with st.expander("Breakdown by question format, topic or year"):
            dimension = st.selectbox("Breakdown", ("Question format", "Topic", "Year"), key="eval_breakdown")
            field = {"Question format": "by_question_type", "Topic": "by_topic", "Year": "by_year"}[dimension]
            groups = []
            for system, system_metrics in payload["metrics"].items():
                if system not in SYSTEM_NAMES:
                    continue
                for group, group_values in _mapping(system_metrics.get(field)).items():
                    groups.append({"Group": group, **comparison_rows({system: group_values})[0]})
            st.dataframe(groups, hide_index=True, width="stretch")
    _render_question_explorer(explorer)


def _render_question_explorer(rows: list[dict]) -> None:
    st.subheader("Inspect individual results")
    if not rows:
        st.info("Question traces have not been saved for this report.")
        return
    left, right = st.columns(2)
    system = left.selectbox("System", ["All systems", *dict.fromkeys(row["System"] for row in rows)], key="eval_result_system")
    outcome = right.selectbox("Outcome", ("All outcomes", "Correct", "Wrong", "Abstained", "Run error"), key="eval_result_outcome")
    filtered = [row for row in rows if (system == "All systems" or row["System"] == system)
                and (outcome == "All outcomes" or row["Outcome"] == outcome)]
    st.caption(f"{len(filtered)} matching results")
    if not filtered:
        st.info("No saved results match these filters.")
        return
    visible = [{key: value for key, value in row.items() if not key.startswith("_")} for row in filtered]
    st.dataframe(visible, hide_index=True, width="stretch")
    _download(visible, "question_results.csv", "eval_questions_csv")
    index = st.selectbox("Inspect a result", range(len(filtered)), key="eval_inspect_result",
                         format_func=lambda i: f"{filtered[i]['Question']} · {filtered[i]['System']} · {filtered[i]['Outcome']}")
    selected = filtered[index]
    question, record = selected["_question"], selected["_record"]
    with st.container(border=True):
        st.markdown(f"**{selected['Question']} · {selected['Outcome']}**")
        st.write(_text(question.get("question_text", "")))
        for letter, option in question.get("options", {}).items():
            st.write(f"{letter}. {_text(option)}")
        st.caption(f"System answer: {selected['Prediction']} · stored answer key: {selected['Answer key']}")
        details = _mapping(record.get("details"))
        explanation = details.get("answer") or details.get("rag_answer") or details.get("reasoning")
        if explanation:
            st.write(_text(explanation))
        if details.get("abstention_reason"):
            st.info(_text(details["abstention_reason"]))
        if record.get("error_type"):
            st.warning(f"Run error: {record['error_type']}")
        with st.expander("Saved reasoning, citations and verification trace"):
            st.json(details or record, expanded=False)


def _render_grounding_and_faithfulness(reports: dict[str, SavedReport]) -> None:
    st.subheader("Does the evidence support the explanation?")
    st.caption("These saved checks describe the earlier 13-question regression set. They have not been run on the new 75-question benchmark.")
    grounding = reports["grounding"]
    if _available(grounding, "No citation-grounding report has been saved yet."):
        payload = grounding.data
        values = _mapping(payload.get("grounding"))
        st.markdown("**Grounding · page citation coverage**")
        for column, label, value in zip(st.columns(3),
            ("Citation coverage", "Committed verdicts", "Verdicts with citations"),
            (percent(values.get("evidence_support_rate")), values.get("committed_verdicts", "—"), values.get("grounded_verdicts", "—"))):
            column.metric(label, value)
        st.write("This check asks whether supported or contradicted verdicts cite pages. It does not establish that those pages actually prove the conclusion.")
        _report_caption(grounding)
        counts = _mapping(_mapping(_mapping(payload.get("verdict_distributions")).get("claim_level")).get("counts"))
        if counts:
            st.bar_chart([{"Verdict": label.title(), "Claims": value} for label, value in counts.items()],
                         x="Verdict", y="Claims", horizontal=True, height=240)
        if values.get("ungrounded_verdicts"):
            with st.expander("Verdicts missing citations"):
                st.dataframe(values["ungrounded_verdicts"], hide_index=True, width="stretch")

    st.divider()
    judge = reports["judge"]
    st.markdown("**Faithfulness · LLM assessment of reasoning**")
    if not _available(judge, "No explanation-judge report has been saved yet."):
        return
    payload = judge.data
    summary = payload["summary"]
    a, b = _mapping(summary.get("A")), _mapping(summary.get("B"))
    for column, label, value in zip(st.columns(3),
        ("Verifier faithfulness", "Baseline faithfulness", "Questions judged"),
        (_score(b.get("evidence_faithfulness")), _score(a.get("evidence_faithfulness")), payload.get("questions_judged", "—"))):
        column.metric(label, value)
    st.caption(f"Judge: {payload.get('judge_model', 'Not recorded')} · scores from 1 to 5 · anonymized responses with alternating positions")
    rows = [{"Dimension": label, "Vanilla RAG": number(a.get(field)), "Source verifier": number(b.get(field))}
            for field, label in SCORE_LABELS.items()]
    st.dataframe(rows, hide_index=True, width="stretch")
    st.bar_chart(rows, x="Dimension", y=["Vanilla RAG", "Source verifier"], stack=False,
                 color=["#94a3b8", "#2457d6"], y_label="Judge score (1–5)", height=280)
    st.caption("Faithfulness measures whether the explanation follows its supplied evidence. Judge scores are model assessments; answer correctness is scored separately against the stored key.")
    _report_caption(judge)
    judged = _records(payload.get("rows"))
    if judged:
        with st.expander("Inspect a question's judge feedback"):
            index = st.selectbox("Judged question", range(len(judged)), key="eval_judged_question",
                                 format_func=lambda i: f"Q{judged[i].get('q_number', i + 1)}")
            row = judged[index]
            for column, system, name in zip(st.columns(2), ("A", "B"), ("Vanilla RAG", "Source verifier")):
                with column, st.container(border=True):
                    st.markdown(f"**{name}**")
                    scores = _mapping(_mapping(row.get("scores")).get(system))
                    st.metric("Faithfulness", _score(scores.get("evidence_faithfulness")))
                    st.write(_mapping(row.get("justification")).get(system, "No feedback recorded."))
                    flagged = _mapping(row.get("hallucinated_claims")).get(system, [])
                    if flagged:
                        st.markdown("**Claims flagged by the judge**")
                        for claim in flagged:
                            st.write(_text(claim))
                    else:
                        st.caption("The judge flagged no unsupported claims in this response.")


def _render_generation(reports: dict[str, SavedReport], root: Path = ROOT) -> None:
    from src.orchestration.telemetry import GENERATION_POLICY
    from src.evaluation.benchmark import sha256_file

    st.subheader("Inspect the generate → verify → revise loop")
    st.write("Compare simple and numbered-statement questions, then follow the checks and revisions for a topic.")
    selected, current = {}, {}
    for arm in ("simple", "statements"):
        latest = reports[f"generation_grounded_{arm}"]
        config = _mapping(latest.data.get("config")) if isinstance(latest.data, dict) else {}
        source = root / "data/processed/chunks.jsonl"
        current[arm] = bool(
            latest.data is not None and config.get("generation_policy") == GENERATION_POLICY
            and config.get("corpus_changed_during_run") is False and source.is_file()
            and config.get("corpus_hash") == sha256_file(source)
        )
        selected[arm] = latest if current[arm] else reports[f"generation_{arm}"]
    if not all(current.values()):
        st.info("The current generation workflow has not been evaluated for both formats with this source snapshot. Earlier experiments, where available, are labelled historical below. Cost & latency shows your actual preparation runs.")
    for column, arm, label in zip(st.columns(2),
        ("simple", "statements"), ("Simple questions", "Numbered statements")):
        report = selected[arm]
        with column, st.container(border=True):
            st.markdown(f"**{label}**")
            if not _available(report, "This generation format has not been evaluated yet."):
                continue
            st.caption("Current source-grounded workflow" if current[arm] else "Historical workflow · earlier prompts and gates")
            summary = report.data["summary"]
            st.caption(f"{summary.get('topics_completed', '—')} topics completed · {summary.get('topics_errored', '—')} run errors")
            st.metric("Accepted by the gates", percent(_mapping(summary.get("terminal_decisions")).get("accept_rate"), scale=100))
            st.metric("Repair rate", percent(_mapping(summary.get("repair")).get("repair_rate"), scale=100))
            st.caption(f"Saved {report.updated}")
    st.caption("Acceptance means the system's own checks passed. These workflow measurements do not independently establish that an accepted question is correct.")
    arm = st.radio("Question format", ("simple", "statements"), horizontal=True,
                   format_func=lambda value: "Simple questions" if value == "simple" else "Numbered statements", key="eval_generation_arm")
    report = selected[arm]
    if not isinstance(report.data, dict):
        return
    summary = report.data["summary"]
    for column, label, value in zip(st.columns(3),
        ("First-attempt block rate", "Total attempts", "Mean claims per attempt"),
        (percent(_mapping(summary.get("unverified_baseline")).get("block_rate", _mapping(summary.get("unverified_baseline")).get("defect_rate")), scale=100),
         _mapping(summary.get("attempts")).get("total_attempts", "—"),
         _mapping(summary.get("claims")).get("mean_claims_per_attempt", "—"))):
        column.metric(label, value)
    st.caption("A block means the workflow's checks withheld a first draft. It can reflect missing evidence, ambiguity or a detected inconsistency; it is not a measured error rate.")
    attribution = _mapping(summary.get("gate_attribution"))
    blocked, sole = _mapping(attribution.get("blocked_by")), _mapping(attribution.get("sole_blocker"))
    st.dataframe([{
        "Gate": name, "Attempts blocked": blocked.get(key, 0), "Only blocker": sole.get(key, 0),
    } for key, name in (
        (("sources", "Source passages"), ("format", "Question format")) if current[arm] else ()
    ) + (("fact", "Statement verification"), ("answer_key", "Answer-key check"), ("quality", "Question quality")) + (
        (("explanations", "Explanation review"),) if current[arm] else ()
    )],
                 hide_index=True, width="stretch")
    st.caption("Several gates can block the same attempt. ‘Only blocker’ counts attempts blocked by that gate alone.")
    topics = _records(report.data.get("results"))
    if not topics:
        return
    index = st.selectbox("Inspect a topic", range(len(topics)), key="eval_generation_topic",
                         format_func=lambda i: f"{topics[i].get('topic', 'Topic')} · {topics[i].get('terminal_decision', 'Run error')}")
    topic = topics[index]
    st.markdown(f"**{topic.get('topic', 'Topic')} · {topic.get('terminal_decision', 'Run error')}**")
    for attempt in _records(topic.get("attempts")):
        with st.expander(f"Attempt {attempt.get('attempt', '—')} · {attempt.get('decision', '—')}"):
            mcq = _mapping(attempt.get("mcq"))
            st.write(_text(mcq.get("question", "Question not recorded.")))
            for letter in "abcd":
                if mcq.get(f"option_{letter}"):
                    st.write(f"{letter.upper()}. {_text(mcq[f'option_{letter}'])}")
            st.caption(f"Declared answer: {mcq.get('correct_answer', '—')} · answer-key verdict: {attempt.get('answer_verdict', '—')} · quality: {attempt.get('quality', '—')}")
            st.write("Fact verdicts", attempt.get("fact_verdicts", {}))
            reasons = attempt.get("failure_reasons", [])
            if reasons:
                st.markdown("**Feedback for revision**")
                for reason in reasons:
                    st.write(_text(reason))
            else:
                st.caption("No gate failures were recorded for this attempt.")


def _render_stability(reports: dict[str, SavedReport]) -> None:
    st.subheader("Do repeated checks return the same verdict?")
    report = reports["stability"]
    if not _available(report, "No repeated-verdict evaluation has been saved yet."):
        return
    payload = report.data
    conditions = payload["conditions"]
    st.caption(f"{report.scope} · model: {payload.get('model', 'Not recorded')} · {payload.get('repeats', '—')} repeats per condition")
    rows, claim_rows = [], []
    for name, condition in conditions.items():
        if not isinstance(condition, dict):
            continue
        label = "Temperature 0" if condition.get("temperature") == 0 else "Temperature unset"
        summary = _mapping(condition.get("summary"))
        rows.append({"Condition": label, "Claims": summary.get("claims"),
                     "Stable": summary.get("stable_claims"), "Changed": summary.get("unstable_claims"),
                     "Observed stability": percent(summary.get("stability_rate"))})
        for row in _records(condition.get("rows")):
            claim_rows.append({"Condition": label, "Question": f"Q{row.get('q_number', '—')}",
                               "Claim": row.get("claim", ""), "Stable": row.get("stable"),
                               "Verdicts across repeats": " → ".join(row.get("verdicts", []))})
    st.dataframe(rows, hide_index=True, width="stretch")
    if rows:
        st.bar_chart(rows, x="Condition", y=["Stable", "Changed"], stack=True,
                     color=["#2457d6", "#eab308"], y_label="Claims", height=280)
    st.write("Stable means the verdict stayed the same in every recorded repeat. Temperature 0 can improve consistency, but these observations do not prove determinism or correctness.")
    _report_caption(report)
    changed_only = st.checkbox("Show only claims with changing verdicts", value=True, key="eval_unstable_only")
    visible = [row for row in claim_rows if not changed_only or row["Stable"] is False]
    if visible:
        st.dataframe(visible, hide_index=True, width="stretch")
        _download(visible, "verdict_stability.csv", "eval_stability_csv")
    else:
        st.info("No claims with changing verdicts match this view.")


def _render_reports(reports: dict[str, SavedReport], root: Path, dataset: dict | None) -> None:
    st.subheader("Saved reports and reproducibility")
    rows = [{"Report": report.label, "Scope": report.scope, "Status": report.status,
             "Saved": report.updated, "File": str(report.path.relative_to(root))}
            for report in reports.values()]
    st.dataframe(rows, hide_index=True, width="stretch")
    _download(rows, "evaluation_report_inventory.csv", "eval_inventory_csv")
    available = {key: report for key, report in reports.items() if report.data is not None}
    if available:
        key = st.selectbox("Open a saved report", list(available), key="eval_saved_report",
                           format_func=lambda value: available[value].label)
        report = available[key]
        _report_caption(report)
        st.download_button("Download full report as JSON", json.dumps(report.data, indent=2, ensure_ascii=False),
                           file_name=report.path.name, mime="application/json", key="eval_report_json")
        with st.expander("Inspect full report"):
            st.json(report.data, expanded=False)
    with st.expander("Validate data and reproduce evaluations"):
        st.write("Validate the frozen benchmark and its original source files before running model evaluations.")
        st.code(".venv/bin/python -m src.evaluation.evaluate_benchmark --check", language="bash")
        if dataset and st.button("Validate benchmark data", key="eval_validate_data"):
            try:
                with st.spinner("Checking labels, splits and source PDF hashes…"):
                    load_benchmark(root / "data/benchmarks/polity_v1.json", check_sources=True)
                st.success("Benchmark labels, splits, frozen data and source PDF hashes passed validation.")
            except (ValueError, KeyError, OSError):
                st.error("Benchmark validation failed. Run the check command in your terminal to inspect the issue.")
        st.write("Generate a development comparison, then refresh this dashboard. Model evaluation calls the configured OpenAI API.")
        st.code(".venv/bin/python -m src.evaluation.evaluate_benchmark --run --split development --output data/evaluation/benchmark/development_reference_context_v2.json", language="bash")
        st.write("Check the earlier reports for freshness, or reproduce the existing evaluation chain.")
        st.code(".venv/bin/python -m src.evaluation.run_all --check\n.venv/bin/python -m src.evaluation.run_all", language="bash")
        st.caption("The freshness check is offline. Reproducing the evaluation chain runs verification, stability sampling and model judging.")


def render_evaluation_dashboard(root: Path = ROOT) -> None:
    left, right = st.columns([4, 1])
    left.caption("CUSTOM PYTHON EVALUATION HARNESS · SAVED EXPERIMENTS")
    right.button("Refresh reports", key="eval_refresh_reports", width="stretch",
                 help="Read the latest files, including reports produced while this page was open.")
    # Reading follows the button on every rerun; refresh never regenerates
    # answers, initializes a model or clears the tutor's model cache.
    reports = load_reports(root)
    try:
        dataset = load_benchmark(root / "data/benchmarks/polity_v1.json")
    except (OSError, ValueError, KeyError):
        dataset = None
        st.warning("The benchmark dataset could not be loaded. Saved regression and workflow reports are still available.")
    counts = inventory(dataset) if dataset else {}
    for column, label, value in zip(st.columns(4),
        ("Benchmark questions", "Development", "Test", "Saved reports"),
        (counts.get("total", "—"), counts.get("splits", {}).get("development", "—"),
         counts.get("splits", {}).get("test", "—"), sum(report.data is not None for report in reports.values()))):
        column.metric(label, value)
    overview, tasks, answers, retrieval, evidence, generation, stability, files = st.tabs(
        ("Overview", "Task specification", "Answer quality", "RAG / Retrieval", "Grounding & faithfulness", "Generation", "Stability", "Reports")
    )
    with overview:
        _render_overview(reports, dataset)
    with tasks:
        render_task_specification()
    with answers:
        _render_answer_quality(reports, root, dataset)
    with retrieval:
        from src.ui.retrieval import render_retrieval_dashboard
        render_retrieval_dashboard(reports["retrieval"], root)
    with evidence:
        _render_grounding_and_faithfulness(reports)
    with generation:
        _render_generation(reports, root)
    with stability:
        _render_stability(reports)
    with files:
        _render_reports(reports, root, dataset)
