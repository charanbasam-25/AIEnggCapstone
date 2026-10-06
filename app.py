"""Polity practice with automatic verification before questions are released.

Run: .venv/bin/python -m streamlit run app.py

The sidebar contains Practice, My Practice and How it works. Verification is
part of the generation service; saved evaluations and diagnostics are optional
views under How it works.
"""

import streamlit as st

from src.ui.about import render_about_site
from src.ui.evaluations import render_evaluation_dashboard
from src.ui.guardrails import render_guardrails
from src.ui.project_details import render_error_handling, render_privacy, render_system_design
from src.ui.practice import (
    render_generation_workspace, render_latency_workspace, render_my_practice,
)
from src.ui.theme import apply_style


st.set_page_config(
    page_title="UPSC Practice | Polity MCQs",
    page_icon="⚖️",
    layout="wide",
)


def preserve_page_inputs() -> None:
    """Keep page choices and in-progress answers when their widgets are hidden."""
    keys = {
        "about_section", "practice_topic", "practice_count", "practice_format",
        "practice_difficulty", "practice_mistakes_only", "practice_latency_run",
        "benchmark_split",
        "eval_question_set", "eval_focus_system", "eval_breakdown", "eval_result_system",
        "eval_result_outcome", "eval_inspect_result", "eval_judged_question",
        "eval_generation_arm", "eval_generation_topic", "eval_unstable_only",
        "eval_saved_report", "eval_rag_run", "eval_rag_benchmark_query",
    }
    session = st.session_state.get("practice_session") or {}
    quiz_prefix = f"quiz_choice_{session['id']}_" if session.get("id") else None
    for key in list(st.session_state):
        if (key in keys or key.startswith(("eval_rag_slot_", "eval_rag_search_"))
                or (key.startswith("rag_") and key.endswith("_candidate"))
                or (quiz_prefix and key.startswith(quiz_prefix))):
            # Reassigning detaches the value from Streamlit's hidden-widget
            # cleanup, without executing a request or changing its value.
            st.session_state[key] = st.session_state[key]


def main() -> None:
    apply_style()
    preserve_page_inputs()
    pages = ["Practice", "My Practice", "How it works"]
    if st.session_state.get("workspace") not in {None, *pages}:
        st.session_state["workspace"] = "Practice"
    sections = [
        "About this site", "Privacy & PII", "Guardrails", "System design", "Evals",
        "Error handling", "Cost & latency",
    ]
    if st.session_state.get("about_section") == "Latency & Usage":
        st.session_state["about_section"] = "Cost & latency"
    elif st.session_state.get("about_section") not in {None, *sections}:
        st.session_state["about_section"] = "About this site"

    with st.sidebar:
        st.markdown('<div class="studio-brand">UPSC Practice</div>', unsafe_allow_html=True)
        st.caption("PRELIMS · TOPIC-WISE PRACTICE")
        st.divider()
        with st.container(key="workspace_navigation"):
            page = st.radio(
                "Workspace", pages, key="workspace", label_visibility="collapsed",
            )
        st.divider()
        st.markdown("**Available now**")
        st.write("Polity")
        st.caption("Geography, History, Economy and Environment are coming soon.")
        st.divider()
        st.caption("Sources, question checks and project metrics are in How it works.")
        st.caption("AI Engineering Capstone")

    if page == "My Practice":
        st.markdown(
            '<div class="studio-hero"><div class="studio-eyebrow">YOUR PRACTICE / PROGRESS</div>'
            '<h1>Learn from your last attempt.</h1>'
            '<p>Review your results, revisit missed questions, and strengthen the concepts you’re studying.</p></div>',
            unsafe_allow_html=True,
        )
        render_my_practice()
    elif page == "How it works":
        st.markdown(
            '<div class="studio-hero"><div class="studio-eyebrow">ABOUT / UPSC PRACTICE</div>'
            '<h1>See how your practice is prepared.</h1>'
            '<p>Understand the problem, scope, privacy and architecture, then inspect the checks and measured results.</p></div>',
            unsafe_allow_html=True,
        )
        with st.container(key="about_navigation"):
            section = st.radio(
                "Explore the project", sections,
                horizontal=True, key="about_section", label_visibility="collapsed",
            )
        if section == "Privacy & PII":
            render_privacy()
        elif section == "Guardrails":
            render_guardrails()
        elif section == "System design":
            render_system_design()
        elif section == "Evals":
            render_evaluation_dashboard()
        elif section == "Error handling":
            render_error_handling()
        elif section == "Cost & latency":
            render_latency_workspace()
        else:
            render_about_site()
    else:
        st.markdown(
            '<div class="studio-hero"><div class="studio-eyebrow">UPSC PRELIMS / PRACTICE</div>'
            '<h1>Practice the topic you’re studying.</h1>'
            '<p>Choose a Polity topic, attempt source-checked questions, and learn from explanations for every option.</p></div>',
            unsafe_allow_html=True,
        )
        render_generation_workspace()

    st.divider()
    st.caption("UPSC Practice · AI-generated preparation material with source references.")


if __name__ == "__main__":
    main()
