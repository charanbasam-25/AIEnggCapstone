"""Read-only explanation of the implemented practice publication guardrails."""

import streamlit as st


PUBLICATION_CHECKS = (
    {
        "name": "Sources — start from reference material",
        "rule": "Retrieve passages from the approved Constitution and NCERT library before drafting. "
                "Complete source pages preserve qualifications and footnotes for verification. "
                "Named Articles also locate their provision pages for review.",
        "enforcement": "Python requires generation evidence; model prompts require source-based reasoning.",
        "example": "If no generation evidence is available, the draft cannot be released.",
    },
    {
        "name": "Format — require a usable MCQ",
        "rule": "Require four nonempty, distinct options and the selected question format. "
                "Check statement structure and reject exact duplicate stems after text normalization.",
        "enforcement": "Python parsing, duplicate checks and structured-data validation.",
        "example": "If options A and B contain the same text, the format check fails.",
    },
    {
        "name": "Answer — settle one answer with evidence",
        "rule": "For statement questions, resolve every claim and map its truth pattern to the options. "
                "For direct questions, require one supported answer and three evidence-based exclusions. "
                "A separate model review must confirm the answer, and Python checks the key.",
        "enforcement": "Python binds selected quotation references to source words; separate model review and Python answer mapping validate the result.",
        "example": "If statement 2 remains INSUFFICIENT, the question is revised or withheld. "
                   "A CONTRADICTED statement is allowed when its falsehood is resolved and the options have one valid answer.",
    },
    {
        "name": "Quality — check clarity and exam style",
        "rule": "Assess ambiguity, a single best answer, plausible distractors, appropriate wording "
                "and relevance to the selected topic. All five flags must pass, with no reported issues.",
        "enforcement": "A model audits the question; Python checks every flag, the verdict and the issue list.",
        "example": "An auditor returning PASS while its unambiguous flag is false still fails this check.",
    },
    {
        "name": "Explanations — review every option and quotation",
        "rule": "Write fresh explanations for the answer and all alternatives. Bind citation references "
                "to actual source excerpts, then review the summary and each option against its own cited evidence.",
        "enforcement": "Python validates quotations and option coverage; a separate model review assesses support and consistency.",
        "example": "A quotation missing from the cited source, an unexplained option, or a failed explanation review blocks release.",
    },
)


def render_guardrails() -> None:
    st.subheader("Guardrails")
    st.write("These checks control which generated questions enter your practice set and how their answers and explanations are reviewed.")
    st.caption("Implemented checking rules · Viewing this page makes no model calls.")

    with st.container(border=True):
        st.markdown("**All five required checks must pass before a question is released.**")
        st.write("Python rechecks the publication conditions before saving a question. Missing, failed or skipped checks block release.")

    st.markdown("### Five required checks")
    for number, check in enumerate(PUBLICATION_CHECKS, 1):
        with st.expander(f"{number}. {check['name']}", expanded=number == 1):
            st.write(check["rule"])
            st.caption(f"How it is checked: {check['enforcement']}")
            st.markdown("**Example**")
            st.write(check["example"])

    st.markdown("### What happens when a check fails?")
    outcomes = (
        ("All checks pass", "The accepted question is saved and becomes available for practice."),
        ("A check fails", "The workflow can revise the MCQ at most twice. Every changed draft is checked again; previous approvals are cleared."),
        ("Checks remain unresolved", "The draft is withheld. You may receive a smaller practice set containing only accepted questions, or no set."),
    )
    for column, (title, description) in zip(st.columns(3), outcomes):
        with column, st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(description)

    st.markdown("### Additional safeguards")
    question_column, session_column = st.columns(2, gap="medium")
    with question_column, st.container(border=True):
        st.markdown("**Question bank & request limits**")
        st.markdown(
            "- Reuse requires the same source snapshot and checking-policy version.\n"
            "- Saved content is validated again when loaded.\n"
            "- A source change during preparation retires the resulting set.\n"
            "- Requests are limited to available Polity topics and 1–10 questions."
        )
    with session_column, st.container(border=True):
        st.markdown("**Practice & service handling**")
        st.markdown(
            "- Answer keys and learner explanations unlock after submission.\n"
            "- Active-quiz RAG passages stay hidden until submission.\n"
            "- Failed model calls or invalid responses cannot approve a draft.\n"
            "- Learner error messages exclude provider details and credentials."
        )

    with st.expander("How supplied text is handled"):
        st.write("Model prompts tell reviewers to treat questions, options and source passages as data rather than instructions, and to use the supplied evidence.")
        st.caption("This is an instruction-level defense. Dedicated prompt-injection attack testing is still needed.")

    st.markdown("### What these checks can establish")
    st.write("Python can confirm that a quotation exists and that every required review passed. Whether the quotation proves the claim still depends on model judgment. Separate automated reviews can agree and still make a mistake.")
    st.write("The guardrails reduce risk; they cannot guarantee zero wrong answers. Expert review of generated questions and broader evaluation remain necessary.")
    st.caption("Evals contains the saved experiments and their dataset scope. Grounding, faithfulness and stability currently cover the earlier 13-question regression set, not all 75 benchmark questions.")
