"""Learner-first practice screens; diagnostics live under How it works."""

import json
from datetime import datetime, timezone
from html import escape
from uuid import uuid4

import streamlit as st

from src.orchestration.nodes import corpus_hash
from src.orchestration.telemetry import GENERATION_POLICY
from src.practice.catalog import SUBJECTS, TOPICS, DIFFICULTY_LABELS, FORMAT_LABELS
from src.practice.costs import PRICING_DATE, PRICING_SOURCE, estimate_run_cost
from src.practice.models import PracticeQuestion, PracticeRequest, score_answers
from src.practice.service import prepare_practice
from src.practice.store import PracticeStore


def get_practice_store() -> PracticeStore:
    return PracticeStore()


def _source_name(source: str) -> str:
    if "constitution" in source.casefold():
        return "Constitution of India"
    if "ncert" in source.casefold():
        return "NCERT"
    return source


def render_subjects() -> None:
    st.markdown("### Choose your subject")
    with st.container(key="subject_catalog"):
        for column, subject in zip(st.columns(len(SUBJECTS), gap="small"), SUBJECTS):
            with column, st.container(border=True, key=f"subject_{subject.name.lower()}"):
                status = "Available now" if subject.available else "Coming soon"
                tone = "available" if subject.available else "locked"
                st.markdown(
                    f'<div class="subject-icon {tone}">{escape(subject.icon)}</div>'
                    f'<div class="subject-title">{escape(subject.name)}</div>'
                    f'<div class="subject-description">{escape(subject.description)}</div>'
                    f'<div class="subject-status {tone}">{escape(status)}</div>',
                    unsafe_allow_html=True,
                )
                st.button(
                    "Polity selected" if subject.available else "🔒 Coming soon",
                    key=f"choose_{subject.name.lower()}", disabled=not subject.available,
                    width="stretch",
                )


def _checked_session() -> tuple[dict | None, list[PracticeQuestion]]:
    session = st.session_state.get("practice_session")
    if not session:
        return None, []
    try:
        questions = [PracticeQuestion.model_validate(item) for item in session["questions"]]
        current_hash = corpus_hash()
        if not questions or any(question.corpus_hash != current_hash for question in questions):
            raise ValueError("The source snapshot changed.")
    except (ValueError, KeyError, TypeError):
        st.session_state.pop("practice_session", None)
        st.info("This practice set needs fresh source checks. Generate a new set to continue.")
        return None, []
    return session, questions


def _source_quotes(question: PracticeQuestion) -> None:
    quoted = {}
    explanations = [question.notes.summary, *question.notes.options]
    for explanation in explanations:
        for citation in explanation.citations:
            evidence = question.evidence[citation.evidence_id - 1]
            key = (evidence["source"], evidence["page"], citation.quote)
            quoted.setdefault(key, evidence)
    with st.expander("View the source evidence"):
        for (source, page, quote), evidence in quoted.items():
            st.caption(f"{_source_name(source)} · PDF page {page}")
            st.text(quote)
        st.caption("Quotations refer to the source snapshot used when this question was checked.")


def _review_question(question: PracticeQuestion, answer: str | None, number: int) -> None:
    with st.container(border=True):
        st.caption(f"QUESTION {number} · {question.topic}")
        st.markdown(question.mcq.question)
        correct = question.mcq.correct_answer
        if answer == correct:
            st.success(f"Correct · {correct}. {getattr(question.mcq, f'option_{correct.lower()}')}")
        elif answer:
            st.warning(f"Your answer: {answer} · Correct answer: {correct}")
        else:
            st.info(f"Skipped · Correct answer: {correct}")
        st.write(question.notes.summary.text)
        st.markdown("**Understand the options**")
        for note in sorted(question.notes.options, key=lambda note: note.option):
            label = "Why this answer fits" if note.is_correct else "Why this option does not answer the question"
            st.markdown(
                f"**{note.option}. {getattr(question.mcq, f'option_{note.option.lower()}')}** "
                f"— {label}"
            )
            st.write(note.text)
        _source_quotes(question)
        st.caption("AI-generated practice · Automated evidence and explanation checks")


def _render_active_quiz() -> None:
    session, questions = _checked_session()
    if not session:
        with st.container(border=True):
            st.markdown("#### Your next practice session starts here")
            st.write("Choose a topic above and generate a set. Attempt the questions, then review the explanations and sources.")
            st.caption("Polity is available now. Other subjects will open as their question sources and checks are ready.")
        return
    st.divider()
    st.subheader("Your practice set")
    st.caption(
        f"{session['topic']} · {len(questions)} questions · "
        f"{DIFFICULTY_LABELS.get(session['difficulty'], session['difficulty'])}"
    )
    if session.get("submitted"):
        score = score_answers(questions, session["answers"])
        for column, label, value in zip(
            st.columns(4), ("Correct", "Incorrect", "Skipped", "Accuracy"),
            (score["correct"], score["wrong"], score["skipped"], f"{score['accuracy']:.0%}"),
        ):
            column.metric(label, value)
        st.write("Review each option and revisit the questions you missed in My Practice.")
        for index, question in enumerate(questions, 1):
            _review_question(question, session["answers"].get(question.id), index)
        return
    st.caption("Choose one answer for each question. You can leave a question unanswered.")
    choices = {}
    # Each choice reaches session state immediately so sidebar navigation can
    # preserve an unfinished quiz. Scoring and explanations still wait for Submit.
    for index, question in enumerate(questions, 1):
        with st.container(border=True):
            st.caption(f"QUESTION {index} OF {len(questions)}")
            st.markdown(question.mcq.question)
            choices[question.id] = st.radio(
                f"Your answer · Question {index}", list("ABCD"), index=None,
                format_func=lambda letter, mcq=question.mcq:
                    f"{letter}. {getattr(mcq, f'option_{letter.lower()}')}",
                key=f"quiz_choice_{session['id']}_{question.id}",
            )
    submitted = st.button("Submit answers", type="primary", width="stretch")
    if submitted:
        if any(question.corpus_hash != corpus_hash() for question in questions):
            st.session_state.pop("practice_session", None)
            st.warning("The sources changed during this session. Generate a freshly checked set.")
            return
        session["submitted"] = True
        session["answers"] = choices
        session["completed_at"] = datetime.now(timezone.utc).isoformat()
        session["score"] = score_answers(questions, choices)
        history = st.session_state.get("practice_history", [])
        history.append(dict(session))
        st.session_state["practice_history"] = history[-20:]
        seen = set(st.session_state.get("practice_seen_ids", []))
        seen.update(question.id for question in questions)
        st.session_state["practice_seen_ids"] = list(seen)
        st.rerun()


def render_generation_workspace() -> None:
    render_subjects()
    st.markdown("### Create your Polity practice set")
    with st.form("practice_setup"):
        topic_column, count_column = st.columns([3, 1])
        topic = topic_column.selectbox("Topic", list(TOPICS), key="practice_topic")
        count = count_column.selectbox(
            "Questions", [1, 5, 10], key="practice_count",
            help="Larger sets take longer to prepare. Start with one question or choose a longer set.",
        )
        format_column, difficulty_column = st.columns(2)
        question_format = format_column.selectbox(
            "Question style", list(FORMAT_LABELS), format_func=FORMAT_LABELS.get,
            key="practice_format",
        )
        difficulty = difficulty_column.selectbox(
            "Practice level", list(DIFFICULTY_LABELS), index=1,
            format_func=DIFFICULTY_LABELS.get, key="practice_difficulty",
            help="This is a requested level; difficulty has not been calibrated from learner performance.",
        )
        st.caption("Each question and its explanations must pass source checks before the set is ready.")
        generate = st.form_submit_button("Generate MCQs", type="primary", width="stretch")
    if generate:
        # Starting a request retires its previous active answers immediately.
        st.session_state.pop("practice_session", None)
        st.session_state.pop("practice_generation_error", None)
        request = PracticeRequest(
            topic=topic, count=count, question_format=question_format,
            difficulty=difficulty, exclude_ids=st.session_state.get("practice_seen_ids", []),
        )
        try:
            with st.status("Preparing your practice set…", expanded=True) as status:
                message = st.empty()
                progress = st.progress(0)
                message.write("Finding source material and checking suitable questions.")

                def update(slot, total, label):
                    message.write(f"Question {slot} of {total} · {label}")
                    progress.progress(min((slot - 1) / total, 1.0))

                result = prepare_practice(request, get_practice_store(), on_progress=update)
                st.session_state["practice_last_run"] = result.report
                progress.progress(1.0)
                if result.questions:
                    st.session_state["practice_session"] = {
                        "id": uuid4().hex, "topic": topic, "difficulty": difficulty,
                        "questions": [question.model_dump(mode="json") for question in result.questions],
                        "submitted": False, "answers": {}, "run_id": result.report["id"],
                    }
                    status.update(
                        label=f"Ready · {len(result.questions)} source-checked questions",
                        state="complete", expanded=False,
                    )
                    if len(result.questions) < count:
                        st.info(
                            f"{len(result.questions)} of {count} requested questions passed the checks. "
                            "You can practise this smaller set."
                        )
                else:
                    status.update(label="No practice set is ready yet", state="error", expanded=False)
                    if result.report["error_type"]:
                        st.error("Question preparation could not complete. Service details are available under How it works → Cost & latency.")
                    else:
                        st.info("No question passed every check for this request. Try another topic or practice level.")
        except Exception as exception:
            # Do not render provider errors, prompts, credentials or a partial draft.
            st.session_state["practice_generation_error"] = type(exception).__name__
            st.error("Question preparation could not complete. Please try again; no draft answers have been added.")
    _render_active_quiz()


def render_my_practice() -> None:
    st.subheader("My Practice")
    st.write("Review your recent attempts and return to the concepts you missed.")
    history = st.session_state.get("practice_history", [])
    if not history:
        st.info("Complete a practice set to see your results and mistake notebook here.")
        return
    totals = {
        key: sum(session["score"][key] for session in history)
        for key in ("total", "correct", "wrong", "skipped")
    }
    for column, label, value in zip(
        st.columns(3), ("Completed sets", "Questions attempted", "Correct answers"),
        (len(history), totals["total"] - totals["skipped"], totals["correct"]),
    ):
        column.metric(label, value)
    st.caption("The latest 20 completed sets are kept in Streamlit server memory for your current session.")
    st.dataframe([
        {
            "Topic": session["topic"], "Correct": session["score"]["correct"],
            "Incorrect": session["score"]["wrong"], "Skipped": session["score"]["skipped"],
            "Completed": session["completed_at"][:16].replace("T", " "),
        }
        for session in reversed(history)
    ], hide_index=True, width="stretch")
    st.subheader("Mistake notebook")
    only_mistakes = st.checkbox("Show incorrect and skipped questions only", value=True, key="practice_mistakes_only")
    displayed = 0
    for session in reversed(history):
        for raw in session["questions"]:
            try:
                question = PracticeQuestion.model_validate(raw)
            except ValueError:
                continue
            answer = session["answers"].get(question.id)
            if only_mistakes and answer == question.mcq.correct_answer:
                continue
            displayed += 1
            if question.corpus_hash != corpus_hash():
                st.info("This previous attempt uses an older source snapshot; generate a new set for current practice.")
                continue
            _review_question(question, answer, displayed)
    if not displayed:
        st.success("No mistakes in your saved practice sets.")


def render_latency_workspace() -> None:
    st.subheader("Cost & latency")
    st.write("Measured preparation time and token usage, with an estimated model cost for each saved run.")
    with st.expander("How cost and latency are calculated"):
        st.write(
            "Preparation time measures the service's timed preparation work, after initial request and store setup. "
            "It includes bank lookups, initial retrieval-model loading, source searches, model calls, "
            "validation and revisions. Final run-report storage and time spent answering the quiz are outside it. "
            "A bank-only run makes no new model calls; a fresh or revised question can take much longer."
        )
        st.markdown(
            f"**Pricing reference ({PRICING_DATE}, USD per million text tokens):** "
            f"gpt-4o-mini standard input $0.15, cached input $0.075, output $0.60. "
            f"[Official model pricing]({PRICING_SOURCE})."
        )
        st.code(
            "Estimated USD = ((input − cached input) × 0.15\n"
            "                 + cached input × 0.075\n"
            "                 + output × 0.60) / 1,000,000",
            language=None,
        )
        st.write(
            "The estimate prices recorded calls, including rejected drafts, using the dated rates above. "
            "Older traces without cache details use the uncached rate; missing tier details assume standard pricing. "
            "Unsupported models/tiers or missing usage leave the full estimate unavailable. "
            "This is not a provider invoice: infrastructure, taxes, account-specific pricing and "
            "unreported transport-retry usage are excluded."
        )
        st.caption(
            "These are individual preparation measurements. Production median/p95 latency and a service-level "
            "target have not been established. Gate acceptance is not a factual-accuracy score."
        )
    error = st.session_state.get("practice_generation_error")
    if error:
        st.warning(f"The last preparation request stopped: {error}. No draft was published.")
    store = get_practice_store()
    runs = store.load_runs()
    current = st.session_state.get("practice_last_run")
    if current and not any(run["id"] == current["id"] for run in runs):
        runs.insert(0, current)
    if not runs:
        st.info("Generate a practice set to record real latency, model calls, token usage and estimated cost.")
        st.caption("Viewing this screen does not run generation or evaluation.")
        return
    runs_by_id = {run["id"]: run for run in runs}
    current_id = current["id"] if current else None
    if current_id and current_id != st.session_state.get("practice_latency_current_id"):
        st.session_state["practice_latency_run"] = current_id
        st.session_state["practice_latency_current_id"] = current_id
    if st.session_state.get("practice_latency_run") not in runs_by_id:
        st.session_state["practice_latency_run"] = current_id or runs[0]["id"]
    selected = st.selectbox(
        "Practice run", list(runs_by_id), key="practice_latency_run",
        format_func=lambda run_id:
            f"{runs_by_id[run_id]['created_at'][:19].replace('T', ' ')} UTC · "
            f"{runs_by_id[run_id]['topic']} · {runs_by_id[run_id]['status']} · {run_id[:6]}",
    )
    run = runs_by_id[selected]
    estimate = estimate_run_cost(run)

    def dollars(value: float) -> str:
        if value == 0:
            return "$0.00"
        return f"${value:.5f}" if value >= 0.00001 else f"${value:.8f}"

    if run["generation_policy"] != GENERATION_POLICY:
        st.info("Historical run under earlier checks. Its questions need fresh review before reuse in the current practice bank.")
    for column, label, value in zip(
        st.columns(5), ("Preparation time", "Model calls", "Input tokens", "Output tokens", "Estimated model cost"),
        (f"{run['elapsed_seconds']:.1f}s", run["api_calls"],
         run["input_tokens"] if run["usage_recorded"] else "—",
         run["output_tokens"] if run["usage_recorded"] else "—",
         dollars(estimate.usd) if estimate.usd is not None else "—"),
    ):
        column.metric(label, value)
    st.caption(
        f"{run['accepted']} of {run['requested']} questions prepared · "
        f"{run['reused']} loaded from the checked bank · "
        f"{run['revisions']} candidate revisions · Policy: {run['generation_policy']}"
    )
    if estimate.usd is None:
        st.info("Full cost estimate unavailable. " + " ".join(estimate.issues))
        if estimate.priced_calls:
            st.caption(
                f"Known recorded subtotal: {dollars(estimate.recorded_usd)} USD across "
                f"{estimate.priced_calls} of {estimate.total_calls} calls. This is not the full run cost."
            )
    else:
        per_question = dollars(estimate.usd / run["accepted"]) if run["accepted"] else "undefined (no delivered questions)"
        st.caption(
            f"Estimated cost per delivered question: {per_question} · Includes revised/rejected work. "
            f"Model pricing as of {PRICING_DATE}; infrastructure and unreported retries excluded."
        )
    if estimate.uncached_assumptions:
        st.caption(
            f"Cached-token details are missing for {estimate.uncached_assumptions} priced calls; "
            "their input uses the uncached rate."
        )
    if estimate.standard_tier_assumptions:
        st.caption(f"Standard service-tier pricing is assumed for {estimate.standard_tier_assumptions} priced calls.")
    if run.get("error_type"):
        st.warning(f"Run ended with {run['error_type']}. This is a service/validation outcome, not an answer-quality score.")
        if run["error_type"] == "MissingAPIKey":
            st.code("Set OPENAI_API_KEY in the project's .env file, then restart Streamlit.", language=None)
    if run["stages"]:
        rows = [
            {"Question slot": item["question"], "Stage": item["stage"].replace("_", " "),
             "Seconds": item["seconds"], "Completed": item["success"]}
            for item in run["stages"]
        ]
        st.markdown("**Where the time went**")
        st.dataframe(rows, hide_index=True, width="stretch")
    if run["calls"]:
        with st.expander("Model calls and recorded usage"):
            st.dataframe(run["calls"], hide_index=True, width="stretch")
    with st.expander("Preparation outcomes"):
        st.json({
            "status": run["status"], "corpus_hash": run["corpus_hash"],
            "candidates": run["candidates"],
        })
    st.download_button(
        "Download measured run", json.dumps({**run, "cost_estimate": estimate.report()}, indent=2), "practice_run.json",
        "application/json", key="practice_download_run",
    )
    st.caption("Model-call counts refer to SDK calls; transport retries are not counted separately. Cold retrieval-model loading is included in preparation time. Gate acceptance is not a measurement of factual accuracy.")
