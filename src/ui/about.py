"""Explain the practice experience and link to optional system diagnostics."""

from html import escape
from pathlib import Path

import streamlit as st

from src.practice.catalog import TOPICS


def _open_section(section: str) -> None:
    st.session_state["about_section"] = section


def render_about_site() -> None:
    st.subheader("About this site")
    st.write(
        "UPSC Practice helps you study Polity through topic-based, UPSC-style MCQs. "
        "Attempt a question first, then learn from explanations for every option "
        "and quotations from the reference material."
    )

    st.subheader("Problem statement")
    st.write(
        "UPSC aspirants need topic-wise practice with explanations they can check. "
        "An AI-generated question can have a wrong key, ambiguous alternatives or a convincing "
        "but unsupported explanation. Studying those errors can teach the wrong concept."
    )
    st.write(
        "This project prepares Polity MCQs from a limited reference library and checks the "
        "answer and every option explanation before release. If the evidence cannot resolve "
        "a draft, the system revises it within a limit or withholds it."
    )
    st.caption(
        "Success means useful practice with correct keys, clear alternatives and source-supported "
        "explanations. Passing automated gates alone does not establish that success; expert evaluation is still needed."
    )

    st.subheader("Scope")
    source_column, scope_column = st.columns(2, gap="medium")
    with source_column, st.container(border=True):
        st.markdown("**Available in this MVP**")
        st.write(
            f"Polity: {len(TOPICS)} curated topics, direct/statement/mixed sets, and "
            "1, 5 or 10 requested questions. Accepted items include a checked key, "
            "a summary, explanations for A–D and source quotations."
        )
        st.caption("The output can be a smaller set or no set when drafts do not pass all checks.")
    with scope_column, st.container(border=True):
        st.markdown("**Source and product boundaries**")
        st.write(
            "Constitution of India and a selected NCERT Grade 7 chapter introducing the "
            "Constitution. No live web search or comprehensive UPSC/current-affairs coverage. "
            "Geography, History, Economy and Environment remain locked."
        )
        st.caption(
            "Foundation, Standard and Challenging are requested levels, not calibrated scores. "
            "This is a local capstone MVP, without learner accounts or production user isolation."
        )

    st.subheader("The learner experience")
    steps = (
        ("01", "Choose a topic", "Set your question style, practice level and number of questions."),
        ("02", "Attempt the quiz", "Work through the questions before seeing the answer key."),
        ("03", "Understand the answer", "Review every option and read the cited source passages."),
        ("04", "Revisit mistakes", "Use My Practice to return to incorrect and skipped questions."),
    )
    cards = "".join(
        f'<div class="about-step"><span class="about-step-number">{number}</span>'
        f'<h3>{escape(title)}</h3><p>{escape(description)}</p></div>'
        for number, title, description in steps
    )
    st.markdown(f'<div class="about-steps">{cards}</div>', unsafe_allow_html=True)

    st.markdown("**How a question reaches your quiz**")
    st.write(
        "Find evidence → draft a question → check its format and answer → review its quality "
        "and every explanation → accept or withhold. The generated key is a proposal: "
        "Python resolves the published answer only after the evidence checks. System design "
        "shows the architecture and tradeoffs; Guardrails explains each mandatory gate."
    )
    st.caption(
        "Previously checked questions can be reused when their source snapshot and checking "
        "rules still match. Automated checks can miss errors; the citations help you review the reasoning."
    )
    st.button("Explore guardrails", on_click=_open_section, args=("Guardrails",))

    st.subheader("Explore the work behind the questions")
    evaluation_column, usage_column = st.columns(2, gap="medium")
    with evaluation_column, st.container(border=True):
        st.markdown("**Evaluations & retrieval**")
        st.write("Inspect saved answer-quality checks, evidence reviews, generation experiments and the actual RAG search passages.")
        st.caption("Answer quality: 75 PYQs. Earlier grounding, faithfulness and stability: 13 questions. Retrieval: 16 annotated queries.")
        st.button("Explore Evals", on_click=_open_section, args=("Evals",), width="stretch")
    with usage_column, st.container(border=True):
        st.markdown("**Cost & latency**")
        st.write("Inspect preparation time, stages, recorded calls and tokens, and a dated estimate of model cost.")
        st.caption("These are measurements from saved practice runs. Viewing them makes no new model calls.")
        st.button("View cost & latency", on_click=_open_section, args=("Cost & latency",), width="stretch")
    brief = Path(__file__).resolve().parents[2] / "docs/PROJECT_BRIEF.md"
    if brief.is_file():
        st.download_button(
            "Download project brief", brief.read_text(encoding="utf-8"), "PROJECT_BRIEF.md",
            "text/markdown", key="project_download_brief",
        )
