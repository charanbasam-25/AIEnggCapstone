"""Read-only explanations of implemented boundaries and engineering choices."""

from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parents[2]


def render_privacy() -> None:
    st.subheader("Privacy & PII")
    st.write(
        "Personally identifiable information (PII) includes details such as a name, "
        "email address, phone number or identity number. The practice screen asks for "
        "curated study choices and answer options; it has no profile fields, free-text "
        "question box or private-document upload. This reduces the personal data the app receives."
    )
    st.markdown("**What goes where**")
    st.table([
        {"Data": "Topic, style, level and question count", "Storage": "Session memory and local run report",
         "Sent to the model?": "Topic, style and level guide generation"},
        {"Data": "Your choices, score and practice history", "Storage": "Streamlit server memory for your session; latest 20 completed sets",
         "Sent to the model?": "No; scoring runs in Python"},
        {"Data": "Accepted MCQs, explanations and source quotations", "Storage": "Shared local SQLite question bank",
         "Sent to the model?": "Yes, during generation and review"},
        {"Data": "Search queries, source excerpts, rankings and review outcomes", "Storage": "Shared local SQLite run reports",
         "Sent to the model?": "Selected source pages and review feedback are used in prompts"},
        {"Data": "Model, tokens, timings and error class", "Storage": "Local run report; no SDK prompt body or provider error message",
         "Sent to the model?": "Diagnostics are recorded after calls"},
        {"Data": "API key", "Storage": "Process environment / local .env configuration",
         "Sent to the model?": "Used for API authentication; never added to a prompt or run report"},
    ], hide_index=True, width="stretch")
    st.caption(
        "Session memory is on the Streamlit server, not in browser storage. It is not "
        "a durable learner account. The SQLite bank and diagnostics outlive that session."
    )
    implemented, gaps = st.columns(2, gap="medium")
    with implemented, st.container(border=True):
        st.markdown("**Implemented today**")
        st.write(
            "Bounded input choices, public reference documents, Python scoring, "
            "session-scoped learner history, and error-class-only diagnostics. "
            "Learner answers and history are not included in generation prompts."
        )
    with gaps, st.container(border=True):
        st.markdown("**Not implemented yet**")
        st.write(
            "An automatic PII detector/redactor, login and access controls, encrypted "
            "database storage, user-separated diagnostics, and automatic retention or deletion. "
            "Public source documents can still contain personal names."
        )
    with st.expander("Before adding personal inputs or public hosting"):
        st.write(
            "Define which personal fields are necessary. Check and redact sensitive text "
            "before it reaches either a model or persistent logs, isolate users and diagnostics, "
            "and provide retention and deletion controls. Test those boundaries with synthetic "
            "names, emails, phone numbers and identity numbers. These are future requirements, "
            "not protections claimed by this local MVP."
        )
        st.write(
            "The table describes this app's storage and outgoing data. It does not establish "
            "the model provider's retention policy or a regulatory compliance guarantee."
        )


def render_system_design() -> None:
    st.subheader("System design")
    st.write(
        "The app retrieves evidence before drafting, then treats the generated answer as "
        "untrusted until every publication gate passes. LangGraph runs a fixed sequence "
        "with bounded revisions; the model roles do not browse the web or choose new tools."
    )
    diagram = ROOT / "docs/diagrams/practice_architecture.png"
    if diagram.is_file():
        st.image(str(diagram), width="stretch", caption="Current practice architecture · 7 October 2026")
        st.download_button(
            "Download architecture image", diagram.read_bytes(), "practice_architecture.png",
            "image/png", key="project_download_architecture",
        )
    else:
        st.info("The architecture image is not available. The component descriptions below show the current flow.")
    st.markdown("**Components and responsibilities**")
    st.table([
        {"Component": "Streamlit learner UI", "Responsibility": "Collect a bounded request, hide the key until submission, show option explanations and keep session history"},
        {"Component": "Source preparation", "Responsibility": "Extract public PDFs into page-aware chunks; retain document/page provenance and hash the source snapshot"},
        {"Component": "RAG retrieval", "Responsibility": "Semantic top 20 → cross-encoder top 5 → complete source pages; named Articles add their provision pages and footnotes"},
        {"Component": "Generator role", "Responsibility": "Propose a structured MCQ from topic and retrieved evidence; the proposed key is untrusted"},
        {"Component": "Verification roles + Python", "Responsibility": "Review statements or all four direct options; bind quotes to sources and resolve one defensible answer"},
        {"Component": "Quality and explanation roles", "Responsibility": "Audit ambiguity/distractors, write fresh option notes and review each note against its own evidence"},
        {"Component": "Publication boundary", "Responsibility": "Python rechecks all five gates; accept, revise up to twice, or withhold"},
        {"Component": "SQLite + diagnostics", "Responsibility": "Save accepted questions and versioned runs; Evals and Cost & latency read the saved evidence and measurements"},
    ], hide_index=True, width="stretch")
    st.caption(
        "Statement questions map resolved truth patterns to an option. Direct questions require "
        "one supported option, three ruled-out alternatives and an agreeing blind review. "
        "Bank reuse requires matching topic, style, requested level, source hash and policy."
    )
    st.subheader("Tradeoffs")
    st.table([
        {"Choice": "Small approved source library", "Benefit": "Traceable evidence and a manageable capstone scope", "Cost / limitation": "Limited coverage and no live current affairs; unchanged source hashes do not prove legal currency"},
        {"Choice": "Strict gates and withholding", "Benefit": "Unresolved answers stay out of student practice", "Cost / limitation": "Smaller or empty sets; automated checks can still miss factual errors"},
        {"Choice": "Top 20 → top 5, then full pages", "Benefit": "Relevant retrieval with context for exceptions and footnotes", "Cost / limitation": "Search can miss evidence; complete pages increase input tokens"},
        {"Choice": "Multiple model reviews", "Benefit": "Separate answer and explanation checks can expose disagreement", "Cost / limitation": "Extra cost and latency; reviewers from the same model family can share mistakes"},
        {"Choice": "Checked question bank", "Benefit": "Reuse avoids new retrieval and model calls", "Cost / limitation": "Source/policy changes invalidate reuse; exact-stem deduplication misses paraphrases"},
        {"Choice": "Local Streamlit + SQLite", "Benefit": "Simple to inspect, run and demonstrate", "Cost / limitation": "No production identity isolation, job queue or distributed serving"},
    ], hide_index=True, width="stretch")


def render_error_handling() -> None:
    st.subheader("Error handling")
    st.write(
        "A draft that lacks evidence is a withheld question. A timeout or missing API key "
        "is a service failure. The UI and reports keep these outcomes separate so a refusal "
        "is not mistaken for a correct answer or a completed evaluation."
    )
    st.table([
        {"Condition": "No useful source evidence", "System response": "Block ungrounded publication; revise within the limit or return no accepted question", "Student experience": "A smaller set, or a message to try another topic/level"},
        {"Condition": "Unresolved claim, ambiguous option or disagreeing review", "System response": "Reject the candidate's answer gate; never substitute a guessed key", "Student experience": "Only questions that pass every gate appear"},
        {"Condition": "Invalid structured output or source quotation", "System response": "Validation blocks publication; record rejection and continue other slots where possible", "Student experience": "The invalid draft stays hidden"},
        {"Condition": "Model API, timeout or connection failure", "System response": "Stop the batch, record only the error class and preserve already accepted questions when sources still match", "Student experience": "Use the accepted partial set, or see a generic preparation failure"},
        {"Condition": "Missing API key", "System response": "Eligible bank questions can still be reused; new generation stops", "Student experience": "Available checked questions or a preparation failure; setup details in Cost & latency"},
        {"Condition": "Source snapshot changes", "System response": "Retire the entire in-flight or active quiz and exclude older bank records", "Student experience": "Generate a freshly checked set"},
        {"Condition": "Corrupted stored question", "System response": "Validate on load; skip invalid bank records", "Student experience": "A malformed record does not enter a new quiz"},
        {"Condition": "Unexpected UI exception", "System response": "Store the error class in session state; do not render credentials, provider messages or partial drafts", "Student experience": "A generic retry message; a durable run report may not have been saved"},
    ], hide_index=True, width="stretch")
    st.markdown("**Bounded recovery**")
    st.write(
        "Each question slot allows one initial draft plus at most two revisions. "
        "Explanation writing has one local repair attempt. Model clients use a 45-second "
        "timeout and at most one SDK transport retry; this is separate from rewriting a question. "
        "The graph also has a recursion limit. These bounds prevent indefinite retry loops, "
        "but do not impose one fixed total latency on a complete practice set."
    )
    with st.container(border=True):
        st.markdown("**Example: asking for five questions**")
        st.write(
            "If three questions pass and the remaining drafts cannot be resolved, the quiz "
            "contains three questions. If the sources change before publication, the whole "
            "set is retired. Time and tokens spent on rejected drafts still appear in the run's measurements."
        )
    st.caption(
        "Selecting an answer, switching pages and reading saved diagnostics make no new model calls. "
        "Retrying Generate MCQs starts a new bounded preparation request."
    )
