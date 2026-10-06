"""Standalone UI for the project's saved evaluation results.

Run: .venv/bin/python -m streamlit run evals_app.py --server.port 8502
"""

import streamlit as st

from src.ui.evaluations import render_evaluation_dashboard
from src.ui.theme import apply_style


st.set_page_config(page_title="Polity Studio | Evals", page_icon="📊", layout="wide")


def main() -> None:
    apply_style()
    with st.sidebar:
        st.markdown('<div class="studio-brand">Polity Studio</div>', unsafe_allow_html=True)
        st.caption("EVALUATION WORKSPACE")
        st.divider()
        st.markdown("**Explore your experiments**")
        st.write("Compare systems, inspect explanations and follow the verification workflow.")
        st.caption("The dashboard reads saved reports from your project. Refresh after an evaluation finishes.")
        st.divider()
        st.markdown("**Question sets**")
        st.caption("75 official PYQs · development and test")
        st.caption("13 earlier regression questions")
        st.caption("Generation and repeated-verdict experiments")
        st.divider()
        st.caption("AI Engineering Capstone · Indian Polity")
    st.markdown('''
    <div class="studio-hero">
      <div class="studio-eyebrow">POLITY STUDIO / EVALS</div>
      <h1>Evaluation Studio</h1>
      <p>Understand how well the system answers, follows its evidence and improves questions through revision.</p>
    </div>
    ''', unsafe_allow_html=True)
    render_evaluation_dashboard()
    st.divider()
    st.caption("Polity Studio · Scores describe the saved experiments and their recorded question sets.")


if __name__ == "__main__":
    main()
