"""State the evaluation contract without treating internal gates as gold labels."""

import streamlit as st


def render_task_specification() -> None:
    st.subheader("Task specification")
    st.write(
        "The product task is to prepare useful, source-backed Polity MCQs. "
        "Answering an existing PYQ, retrieving a source page and grading an explanation "
        "are separate evaluation tasks with different scoring units."
    )
    with st.container(border=True):
        st.markdown("**Primary task: generate a practice set**")
        st.write(
            "Input: a supported Polity topic, direct/statement/mixed style, requested "
            "practice level and question count. Output: up to that count of accepted "
            "four-option MCQs, each with one checked key, a summary, explanations for "
            "A–D and source quotations. An unresolved draft must be withheld."
        )
        st.write(
            "An expert evaluation would label each delivered question for key correctness, "
            "ambiguity, option-note correctness, source support and topic/style fit. "
            "The target is no known erroneous key or explanation in a released set. "
            "An independently labeled, frozen benchmark for generated MCQs has not yet been built or run."
        )
        st.caption(
            "Example request: Fundamental Rights · Direct questions · Standard · 1. "
            "Returning one internally accepted question records workflow success; expert adjudication "
            "is still needed to measure whether that question and every explanation are correct."
        )
    st.markdown("**What the existing harness measures**")
    st.table([
        {"Task": "Answer an existing PYQ", "Input → output": "Stem + A–D, with official key hidden → predicted letter or abstention", "Scoring unit / reference": "75 questions (33 development, 42 test); stored official answer keys", "Measure": "Correct/wrong/abstained/errors, precision, coverage, overall accuracy"},
        {"Task": "Retrieve relevant evidence", "Input → output": "Annotated search query → semantic/reranked source pages", "Scoring unit / reference": "16 queries; expected source-page labels", "Measure": "Fraction with any expected page in the top 5 (legacy name Recall@5)"},
        {"Task": "Ground a verdict / assess faithfulness", "Input → output": "Claim, verdict, explanation and cited evidence → citation checks and model-judge scores", "Scoring unit / reference": "Earlier 13-question regression; provenance checks and judge rubric", "Measure": "Citation coverage and 1–5 judge ratings; not expert factual accuracy"},
        {"Task": "Check generated workflow behavior", "Input → output": "Topic/style → drafts, revisions, terminal gate outcomes", "Scoring unit / reference": "Per topic run/candidate; source and policy must match", "Measure": "Acceptance, rejection, repair and blocking gates; current full-format report not run"},
        {"Task": "Repeat verdicts", "Input → output": "Same claim/evidence across repeat calls → verdict sequence", "Scoring unit / reference": "37 claims from the earlier 13 questions; 5 repetitions", "Measure": "Consistency and changes; a stable verdict can still be wrong"},
        {"Task": "Enforce software boundaries", "Input → output": "Synthetic valid/invalid sources, outputs and errors → publication decision", "Scoring unit / reference": "Offline pytest cases; fake models and known outcomes", "Measure": "Gate, quote-binding, reuse, UI and error behavior; no model-accuracy score"},
        {"Task": "Measure preparation", "Input → output": "Actual run trace → elapsed seconds, calls, tokens and estimated USD", "Scoring unit / reference": "Per saved preparation run; recorded usage and dated supported pricing", "Measure": "Cost & latency, including rejected work; no production latency SLO measured"},
    ], hide_index=True, width="stretch")
    with st.expander("Scoring rules and failure categories", expanded=True):
        st.write(
            "For N questions, let C be correct answers and W be wrong answers. "
            "Precision when answered = C / (C + W); coverage = (C + W) / N; "
            "overall accuracy = C / N. If the system answers nothing, precision is undefined, "
            "not 100%. Run errors stay separate from deliberate abstentions and make a report preliminary."
        )
        st.write(
            "Useful error categories include missed evidence, unsupported or conflicting keys, "
            "ambiguous distractors, incorrect option explanations, malformed output, stale sources "
            "and service failures. Page citation presence measures provenance; it does not prove entailment."
        )
    with st.expander("What is still needed for a trustworthy release"):
        st.write(
            "Build an expert-labeled generated-question set across the supported topics and formats, "
            "including exceptions, footnotes, weak distractors and insufficient evidence. Freeze an "
            "untouched holdout after development, report expert key and explanation error rates alongside "
            "yield, and measure latency percentiles and cost per delivered question over repeated runs."
        )
        st.caption(
            "The existing 42-question test split has informed debugging and now serves as regression coverage. "
            "It is not an untouched holdout. The 13-question grounding/stability reports are not 75-question evaluations."
        )
