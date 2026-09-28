"""
Streamlit demo for the source-verified UPSC Polity MCQ verifier.

Run it with:

    .venv/Scripts/python.exe -m streamlit run app.py

Design notes
------------

1. Nothing is reimplemented. Claim construction, retrieval, verification,
   option mapping and the plain-text question parser are all imported from
   the modules the evaluation harness and the CLI already use, so a number
   shown here and a number in data/evaluation/verified_pyq_results.json
   come from the same code. This is not tidiness: evaluate_verified_pyqs.py
   used to carry its own copy of the mapping logic, and the copy was the
   buggy one. A UI that re-derived "which option do these verdicts imply"
   would recreate that defect in the most visible place in the project.

2. Verdicts are computed once, cached in session state, and the abstention
   policy is re-applied on every rerun from those cached verdicts. That
   mirrors the architecture exactly - the LLM judges statements, plain
   Python maps them onto an option - and it has a useful consequence for a
   demo: after one verification run you can switch between strict,
   closed_world and elimination and watch the answer change instantly,
   with no further API calls. The coverage-versus-error trade-off stops
   being a table and becomes a radio button.

3. Model and index objects are held in @st.cache_resource, the UI-layer
   counterpart of the lru_cache(maxsize=1) factories in
   src/orchestration/nodes.py. Streamlit re-executes this whole script on
   every interaction, and building the retriever embeds all 1,149 chunks
   with bge-small-en-v1.5 - about 70 seconds. Without the cache that cost
   would be paid per click.

4. Abstention is presented as a result, not an error. When the pipeline
   declines, the UI names the typed reason and the statements that blocked
   it. A demo that hid this would be demonstrating the opposite of what
   the system is for.
"""

import json
import re
from pathlib import Path

import streamlit as st

from src.evaluation.answer_mapping import (
    CLOSED_WORLD,
    ELIMINATION,
    STRICT,
    determine_statement_statuses,
    map_answer,
)
from src.evaluation.evaluate_verified_pyqs import (
    ANSWER_TOP_K,
    CHUNKS_PATH,
    CLAIM_MODE,
    CLAIM_TOP_K,
    DATASET_PATH,
    build_pyq_claims,
    pyq_to_mcq,
)
from src.verify_cli import parse_plain_text

st.set_page_config(
    page_title="Polity MCQ Verifier",
    page_icon="⚖️",
    layout="wide",
)

OPTION_LETTERS = ("A", "B", "C", "D")

VERDICT_ICON = {
    "SUPPORTED": "✅",
    "CONTRADICTED": "❌",
    "INSUFFICIENT": "⚠️",
}

POLICY_HELP = {
    STRICT: (
        "Abstain if any statement is unsettled. The shipped default, and "
        "the setting behind the headline result."
    ),
    CLOSED_WORLD: (
        "Answer anyway, treating unsettled statements as false. Measured "
        "the highest accuracy in this project and was rejected: the error "
        "rate goes from 0% to 30.77%."
    ),
    ELIMINATION: (
        "Answer only when exactly one option survives. Measured "
        "identically to strict on this data - an exact null result."
    ),
}

ABSTENTION_WORDING = {
    "policy_insufficient_evidence": (
        "at least one statement could not be settled from the corpus, and "
        "this policy declines rather than guesses"
    ),
    "no_option_matched": (
        "the verdicts are consistent but no option describes them, which "
        "usually means a statement was misread or the official key is "
        "disputable"
    ),
    "ambiguous_match": (
        "more than one option matches, so answering would mean picking "
        "arbitrarily"
    ),
    "multiple_options_possible": (
        "elimination narrowed the field but more than one option is still "
        "viable"
    ),
}


# ---------------------------------------------------------------------------
# Cached loaders
#
# cache_resource, not cache_data: these hold live model objects that must
# not be pickled or copied. load_chunks is the same function the harness
# uses, so the corpus is identical to the evaluated one.
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading corpus…")
def get_chunks() -> list[dict]:
    from src.retrieval.semantic_reranker import load_chunks

    return load_chunks(CHUNKS_PATH)


@st.cache_resource(
    show_spinner="Embedding 1,149 chunks — about 70s, once per session…"
)
def get_retriever():
    from src.verification.claim_retriever import ClaimRetriever

    return ClaimRetriever(get_chunks())


@st.cache_resource(show_spinner="Loading verifier…")
def get_fact_verifier():
    from src.verification.fact_verifier import FactVerifier

    # Constructed WITH chunks. A bare FactVerifier() silently disables
    # absence-claim handling, an omission that has already shipped twice
    # in this codebase.
    return FactVerifier(get_chunks())


@st.cache_resource(show_spinner="Loading answer-key verifier…")
def get_answer_key_verifier():
    from src.verification.answer_key_verifier import AnswerKeyVerifier

    return AnswerKeyVerifier()


@st.cache_data
def get_dataset() -> dict:
    return json.loads(Path(DATASET_PATH).read_text(encoding="utf-8"))


@st.cache_data
def get_report(path: str):
    target = Path(path)

    if not target.exists():
        return None

    return json.loads(target.read_text(encoding="utf-8"))


# Display-only cleanup. The stored questions carry stray control bytes -
# a BEL, 0x07 - left behind by PDF text extraction, and they render as
# mojibake. They are deliberately NOT fixed in the dataset: every stored
# result in data/evaluation/ was produced from exactly these strings, so
# rewriting them would silently invalidate the numbers. Stripping them at
# the last moment keeps the demo readable without touching the input the
# measurements were taken on.
CONTROL_BYTES = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean(text: str) -> str:
    return CONTROL_BYTES.sub(" ", text).strip()


def option_block(options: dict) -> None:

    columns = st.columns(4)

    for column, letter in zip(columns, OPTION_LETTERS):
        column.markdown(f"**{letter}.** {options[letter]}")


# ---------------------------------------------------------------------------
# Tab 1 — verify a question
# ---------------------------------------------------------------------------

PASTE_EXAMPLE = """Consider the following statements regarding the \
Vice-President of India:
I. The Vice-President is elected by the members of both Houses of \
Parliament.
II. The Vice-President is the ex-officio Chairman of the Rajya Sabha.
Which of the statements given above is/are correct?
(a) I only
(b) II only
(c) Both I and II
(d) Neither I nor II"""


def read_question_from_ui() -> "dict | None":
    """Return a question dict, or None if the input is not yet usable."""

    mode = st.radio(
        "Question source",
        ["Stored 2025 Prelims question", "Paste your own"],
        horizontal=True,
    )

    if mode == "Stored 2025 Prelims question":

        questions = get_dataset()["questions"]

        index = st.selectbox(
            "Question",
            range(len(questions)),
            format_func=lambda i: (
                f"Q{questions[i]['q_number']} — "
                f"{clean(questions[i]['question_text'])[:80]}…"
            ),
        )

        return questions[index]

    text = st.text_area(
        "Paste the question and its four options",
        value=PASTE_EXAMPLE,
        height=220,
        help=(
            "Everything above the first option line is the question. "
            "Option labels may be written (a), a) or A. — roman-numeral "
            "statement labels are never mistaken for them."
        ),
    )

    if not text.strip():
        return None

    try:
        # The same parser the CLI uses, so a question that works in one
        # works in the other.
        return parse_plain_text(text)

    except SystemExit as failure:
        st.error(str(failure))

        return None


def run_verification(question: dict) -> None:
    """Verify every statement once and store the verdicts."""

    try:
        claims = build_pyq_claims(question["question_text"])

    except ValueError:
        # question_parser raises rather than returning an empty list when
        # there are no numbered items, so this has to be caught, not
        # tested for. Guarding on `not claims` alone let a raw traceback
        # reach the page.
        claims = []

    if not claims:
        st.error(
            "**No numbered statements found**, so there is nothing to "
            "decompose."
        )

        st.markdown(
            "This verifier works on the *Consider the following "
            "statements: I… II…* form, where each statement can be "
            "checked against the corpus on its own. A single-fact "
            "question such as *Who appoints the Governor?* has no "
            "independent statements to verify, and the design has no "
            "honest way to answer it — so it declines rather than "
            "falling back on the model's memory.\n\n"
            "That is a real boundary of the architecture, not a parsing "
            "bug."
        )

        return

    retriever = get_retriever()
    verifier = get_fact_verifier()

    records = []
    fact_verifications = {}

    progress = st.progress(0.0, text="Verifying statements…")

    for position, claim in enumerate(claims, start=1):

        evidence = retriever.retrieve(claim.claim, top_k=CLAIM_TOP_K)
        result = verifier.verify(claim.claim, evidence)

        fact_verifications[claim.claim_id] = result

        records.append(
            {
                "claim_id": claim.claim_id,
                "statement_number": claim.statement_number,
                "claim": claim.claim,
                "verdict": result.verdict,
                "reasoning": result.reasoning,
                "pages": list(result.supporting_pages or []),
                "evidence": [
                    {
                        "source": chunk["source"],
                        "page": chunk["page"],
                        "text": " ".join(chunk["text"].split()),
                    }
                    for chunk in evidence
                ],
            }
        )

        progress.progress(
            position / len(claims),
            text=f"Verified {position} of {len(claims)} statements",
        )

    progress.empty()

    st.session_state["verified"] = {
        "question": question,
        "claims": claims,
        "fact_verifications": fact_verifications,
        "records": records,
        "claim_mode": CLAIM_MODE,
    }

    # Cleared because it belongs to the previous question.
    st.session_state.pop("single_call", None)


def run_single_call_comparison(question: dict) -> None:
    """The rejected architecture: one LLM call for the whole question."""

    mcq = pyq_to_mcq(
        {
            **question,
            # Placeholder for an unlabelled pasted question. Nothing
            # downstream reads it; the verifier reports which options it
            # finds supported on its own.
            "official_answer": question.get("official_answer") or "A",
        }
    )

    evidence = get_retriever().retrieve(mcq.question, top_k=ANSWER_TOP_K)

    st.session_state["single_call"] = get_answer_key_verifier().verify(
        mcq,
        evidence,
    )


def render_verdicts(records: list[dict], show_evidence: bool) -> None:

    st.markdown("#### Per-statement verdicts")

    st.caption(
        f"Each statement was judged on its own against the top "
        f"{CLAIM_TOP_K} retrieved passages. The model saw one claim at a "
        f"time and never saw the options."
    )

    for record in records:

        label = (
            f"Statement {record['statement_number']}"
            if record["statement_number"]
            else record["claim_id"]
        )

        icon = VERDICT_ICON.get(record["verdict"], "•")

        with st.expander(
            f"{icon} {label} — {record['verdict']}",
            expanded=True,
        ):
            st.markdown(f"**Claim.** {clean(record['claim'])}")
            st.markdown(f"**Reasoning.** {record['reasoning']}")

            if record["pages"]:
                pages = ", ".join(str(page) for page in record["pages"])
                st.markdown(f"**Pages cited.** {pages}")

            else:
                # Worth naming rather than showing an empty list: no
                # cited page is how INSUFFICIENT is supposed to look, and
                # also how a SUPPORTED verdict looks when it should not.
                st.markdown("**Pages cited.** none cited")

            if show_evidence:

                st.caption("Evidence retrieved for this claim")

                for chunk in record["evidence"]:
                    st.text(
                        f"[{chunk['source']} p.{chunk['page']}] "
                        f"{chunk['text'][:320]}…"
                    )


def render_decision(policy: str) -> None:

    stored = st.session_state["verified"]
    question = stored["question"]

    # Recomputed on every rerun from the cached verdicts. Pure Python, no
    # API calls - which is why switching policy is instant, and is the
    # architectural claim made visible.
    statuses = determine_statement_statuses(
        stored["claims"],
        stored["fact_verifications"],
    )

    predicted, debug = map_answer(
        statement_statuses=statuses,
        options=question["options"],
        question_text=question["question_text"],
        policy=policy,
    )

    st.markdown("#### Decision")

    if statuses:

        columns = st.columns(len(statuses))

        for column, statement in zip(columns, sorted(statuses)):
            column.metric(
                f"Statement {statement}",
                statuses[statement].title(),
            )

    if debug["question_asks_for_incorrect"]:
        st.caption(
            "This question asks which statements are **not** correct, so "
            "the target set is the contradicted ones, not the supported "
            "ones."
        )

    official = question.get("official_answer")

    if predicted is None:

        reason = debug["abstention_reason"]

        st.warning("**Abstained — no answer given.**")

        st.markdown(f"**Why.** {ABSTENTION_WORDING.get(reason, reason)}")

        blocking = debug["insufficient_statements"]

        if blocking:
            st.markdown(
                f"**Blocked by.** statement(s) {', '.join(blocking)}"
            )

        st.caption(
            "A designed outcome, not a failure. Across the 13 audited "
            "questions this pipeline states a false answer 0 times; the "
            "unverified RAG baseline does so 7 times."
        )

    elif official and predicted == official:
        st.success(
            f"**Answer: {predicted}. {question['options'][predicted]}** "
            f"— matches the official key."
        )

    elif official:
        st.error(
            f"**Answer: {predicted}. {question['options'][predicted]}** "
            f"— the official key is {official}."
        )

    else:
        st.info(
            f"**Answer: {predicted}. {question['options'][predicted]}** "
            f"— a pasted question has no official key, so this is "
            f"unscored."
        )

    with st.expander("Mapping trace"):
        st.caption(
            "Produced by src/evaluation/answer_mapping.py. No LLM is "
            "involved past this point."
        )
        st.json(debug)


def render_verify_tab() -> None:

    st.subheader("Verify a question")

    st.caption(
        "The measured flow. Every numbered statement is decided "
        "independently against the Constitution and NCERT Polity, then "
        "plain Python — not the model — maps the verdicts onto an option "
        "or abstains."
    )

    question = read_question_from_ui()

    if question is None:
        return

    st.markdown("**Question**")
    st.info(clean(question["question_text"]))
    option_block(question["options"])

    left, middle, right = st.columns([1, 1, 2])

    with left:
        verify_clicked = st.button("Verify", type="primary")

    with middle:
        show_evidence = st.checkbox("Show retrieved evidence")

    with right:
        compare = st.checkbox(
            "Also run the single-call verifier",
            help=(
                "The architecture this project measured and rejected: "
                "one LLM call judging the whole question at once. Costs "
                "one extra call."
            ),
        )

    if verify_clicked:

        run_verification(question)

        if compare and "verified" in st.session_state:
            with st.spinner("Running the single-call verifier…"):
                run_single_call_comparison(question)

    stored = st.session_state.get("verified")

    if not stored:
        return

    if stored["question"]["question_text"] != question["question_text"]:
        st.info(
            "Showing the previous result. Press **Verify** to run this "
            "question."
        )

    st.markdown("---")

    render_verdicts(stored["records"], show_evidence)

    st.markdown("---")

    policy = st.radio(
        "Abstention policy",
        [STRICT, CLOSED_WORLD, ELIMINATION],
        horizontal=True,
        help=(
            "Switching this re-maps the verdicts already computed above. "
            "No further API calls — the mapping is deterministic Python."
        ),
    )

    st.caption(POLICY_HELP[policy])

    render_decision(policy)

    single_call = st.session_state.get("single_call")

    if single_call is not None:

        st.markdown("---")
        st.markdown("#### The rejected architecture, for comparison")

        st.caption(
            "One LLM call, whole question, options visible. This is what "
            "the per-statement design replaced."
        )

        columns = st.columns(2)

        columns[0].metric("Verdict", single_call.verdict)
        columns[1].metric(
            "Options it found supported",
            ", ".join(single_call.supported_options) or "none",
        )

        st.markdown(f"**Reasoning.** {single_call.reasoning}")

        if len(single_call.supported_options) > 1:
            st.caption(
                "More than one supported option means this call did not "
                "actually decide anything — the failure mode that "
                "motivated splitting the question into statements."
            )


# ---------------------------------------------------------------------------
# Tab 2 — generate and gate
# ---------------------------------------------------------------------------


def render_generate_tab() -> None:

    from src.generation.mcq_generator import (
        FORMAT_SIMPLE,
        FORMAT_STATEMENTS,
    )
    from src.orchestration.graph import build_graph

    st.subheader("Generate and gate")

    st.caption(
        "The second flow. A generator proposes an MCQ; three independent "
        "gates — fact verification, answer-key verification and a quality "
        "audit — return ACCEPT, REVISE or REJECT, and every failure "
        "reason is fed back into the next attempt."
    )

    left, middle, right = st.columns(3)

    with left:
        topic = st.text_input("Topic", value="Fundamental Rights")

    with middle:
        question_format = st.selectbox(
            "Question format",
            [FORMAT_SIMPLE, FORMAT_STATEMENTS],
            help=(
                "simple is what the loop shipped with and decomposes to "
                "about 1 claim. statements is the real UPSC "
                "multi-statement form and decomposes to about 3 — the "
                "only setting that exercises the claim machinery."
            ),
        )

    with right:
        max_retries = st.number_input(
            "Max retries",
            min_value=0,
            max_value=4,
            value=2,
            help="Attempts = retries + 1.",
        )

    st.caption(
        f"Up to {int(max_retries) + 1} attempts at roughly 6–10 API calls "
        f"each, so expect one to three minutes."
    )

    if not st.button("Generate", type="primary"):
        return

    # Warmed first so the ~70s embedding cost is not mistaken for the
    # generator being slow.
    get_retriever()
    get_fact_verifier()

    workflow = build_graph()

    box = None

    # stream, not invoke: generate_mcq clears failure_reasons at the start
    # of every attempt and the graph returns only terminal state, so the
    # per-attempt history that makes a revision loop worth watching exists
    # only in the stream.
    for update in workflow.stream(
        {
            "topic": topic,
            "difficulty": "medium",
            "question_format": question_format,
            "retry_count": 0,
            "max_retries": int(max_retries),
        },
        config={"recursion_limit": 50},
        stream_mode="updates",
    ):
        for node, payload in update.items():

            if node == "generate_mcq":

                mcq = payload["mcq"]

                box = st.container(border=True)

                box.markdown(f"#### Attempt {payload['retry_count']}")
                box.markdown(mcq.question)

                for letter in ("a", "b", "c", "d"):
                    box.markdown(
                        f"**{letter.upper()}.** "
                        f"{getattr(mcq, f'option_{letter}')}"
                    )

                box.caption(f"Declared answer: {mcq.correct_answer}")

                continue

            if box is None:
                continue

            if node == "verify_claims":

                verdicts = payload["fact_verifications"]

                box.markdown(
                    f"**Gate 1 — fact verification** "
                    f"({len(verdicts)} claim(s))"
                )

                for claim_id, result in verdicts.items():
                    box.markdown(
                        f"- {VERDICT_ICON.get(result.verdict, '•')} "
                        f"`{claim_id}` {result.verdict} · pages "
                        f"{result.supporting_pages or '—'}"
                    )

            elif node == "verify_answer_key":

                result = payload["answer_verification"]
                passed = result.verdict == "VALID"

                box.markdown(
                    f"**Gate 2 — answer key** "
                    f"{'✅' if passed else '❌'} {result.verdict}"
                )

                if not passed:
                    box.caption(result.reasoning)

            elif node == "audit_quality":

                result = payload["quality_audit"]
                passed = result.overall_quality == "PASS"

                box.markdown(
                    f"**Gate 3 — quality audit** "
                    f"{'✅' if passed else '❌'} {result.overall_quality}"
                )

                for issue in result.issues:
                    box.caption(f"· {issue}")

            elif node == "decide":

                decision = payload["decision"]

                if decision == "ACCEPT":
                    box.success("**ACCEPT** — every gate passed.")

                elif decision == "REVISE":
                    box.warning(
                        "**REVISE** — regenerating with these reasons fed "
                        "back into the prompt."
                    )

                else:
                    box.error(
                        "**REJECT** — retry budget exhausted. The "
                        "question is discarded rather than shipped."
                    )

                for reason in payload["failure_reasons"]:
                    box.caption(f"· {reason}")


# ---------------------------------------------------------------------------
# Tab 3 — the measurement record
# ---------------------------------------------------------------------------

GENERATION_ARMS = [
    ("simple", "data/evaluation/generation_loop_results.json"),
    (
        "statements",
        "data/evaluation/generation_loop_results.statements.json",
    ),
]


def render_selective_metrics() -> None:

    st.markdown("#### Answering the 13 PYQs — baseline vs verified")

    report = get_report("data/evaluation/selective_metrics.json")

    if report is None:
        st.info(
            "selective_metrics.json not found. Produce it with "
            "`python -m src.evaluation.run_all --only selective_metrics`."
        )

        return

    st.json(report, expanded=False)

    st.caption(
        "The headline is the **error rate**, not the accuracy. Read "
        "quickly, the baseline looks better because it answers "
        "everything — which is the trap. For exam preparation a "
        "confidently wrong answer is worse than no answer, so all four "
        "metrics are reported together: any single one flatters one "
        "system and hides the trade-off."
    )


def render_generation_arms() -> None:

    st.markdown("#### Gating generated MCQs — two format arms")

    reports = {name: get_report(path) for name, path in GENERATION_ARMS}

    if not any(reports.values()):
        st.info(
            "No generation-loop results yet. Run "
            "`python -m src.evaluation.evaluate_generation_loop "
            "--format simple`, then again with `--format statements`."
        )

        return

    columns = st.columns(len(GENERATION_ARMS))

    for column, (name, _) in zip(columns, GENERATION_ARMS):

        report = reports[name]

        with column:

            st.markdown(f"**`--format {name}`**")

            if report is None:
                st.info("not run yet")
                continue

            summary = report["summary"]

            st.caption(
                f"n = {summary['topics_completed']} topics "
                f"({summary['topics_errored']} errored)"
            )

            st.metric(
                "ACCEPT rate",
                f"{summary['terminal_decisions']['accept_rate']}%",
            )

            st.metric(
                "Attempt-1 defect rate",
                f"{summary['unverified_baseline']['defect_rate']}%",
                help=(
                    "Attempt 1 is a free baseline: what a generator with "
                    "no gate in front of it would have shipped."
                ),
            )

            st.metric(
                "Mean claims per attempt",
                summary["claims"]["mean_claims_per_attempt"],
            )

            st.metric(
                "Repair rate",
                f"{summary['repair']['repair_rate']}%",
                help=(
                    "Of topics blocked on attempt 1, the fraction "
                    "revision eventually got to ACCEPT."
                ),
            )

            st.caption("Verdicts across all attempts")

            st.write(summary["claims"]["verdict_distribution"])

            st.caption("Gate attribution — blocked / sole blocker")

            blocked = summary["gate_attribution"]["blocked_by"]
            sole = summary["gate_attribution"]["sole_blocker"]

            for gate in ("fact", "answer_key", "quality"):
                st.text(
                    f"{gate:<11} "
                    f"{blocked.get(gate, 0):>3} / {sole.get(gate, 0)}"
                )

    st.caption(
        "The contrast between the arms is the finding. A simple generated "
        "question decomposes to about one claim, so gating it exercises "
        "almost none of this project's machinery and the fact gate "
        "produces no CONTRADICTED verdicts at all. The multi-statement "
        "arm decomposes to roughly three, and every topic fails. Both "
        "arms say the same thing about ungated output: nearly all of it "
        "carries a detectable defect."
    )


def render_evaluation_tab() -> None:

    st.subheader("Evaluation")

    st.caption(
        "Read straight from `data/evaluation/`. No API calls and no "
        "recomputation — this is the record on disk."
    )

    render_selective_metrics()

    st.markdown("---")

    render_generation_arms()

    st.markdown("---")

    with st.expander(
        "Verdict stability — temperature 0 is not determinism"
    ):
        report = get_report("data/evaluation/verdict_stability.json")

        if report is None:
            st.info("verdict_stability.json not found.")
        else:
            st.json(report, expanded=False)

        st.caption(
            "Pinning temperature raises stable claims from 28/37 to 35/37 "
            "and never reaches 37. Both counts are lower bounds at 5 "
            "repeats. Every arm-to-arm difference in the retrieval "
            "ablation sits inside the resulting noise band, which is why "
            "those are reported as null results rather than improvements."
        )

    with st.expander("LLM-as-a-judge — blind, slot-swapped"):
        report = get_report("data/evaluation/judge_results.json")

        if report is None:
            st.info("judge_results.json not found.")
        else:
            st.json(report, expanded=False)


# ---------------------------------------------------------------------------


def main() -> None:

    st.title("Source-Verified UPSC Polity MCQ Verifier")

    st.markdown(
        "Decides Indian Polity statements against the **text of the "
        "Constitution** and **NCERT Polity** — 422 pages, 1,149 chunks — "
        "and refuses to answer when the corpus does not settle it. The "
        "model judges one claim at a time; Python picks the option."
    )

    verify, generate, evaluation = st.tabs(
        ["Verify a question", "Generate and gate", "Evaluation"]
    )

    with verify:
        render_verify_tab()

    with generate:
        render_generate_tab()

    with evaluation:
        render_evaluation_tab()

    st.sidebar.markdown("### The governing rule")

    st.sidebar.markdown(
        "Generation and verification are separate responsibilities, and "
        "**the LLM never selects the answer.** It only ever judges one "
        "small statement against retrieved text. The option is chosen by "
        "deterministic Python, which is why an abstention can name the "
        "statement that caused it."
    )

    st.sidebar.markdown("### Retrieval")

    st.sidebar.markdown(
        "Six retrievers were measured on 16 gold queries before one was "
        "chosen (Recall@5):\n\n"
        "- BM25 — 75.00%\n"
        "- Hybrid RRF — 81.25%\n"
        "- Semantic — 87.50%\n"
        "- Parent-child + reranker — 87.50%\n"
        "- **Semantic + cross-encoder — 93.75%**"
    )

    st.sidebar.markdown("### Same code, no UI")

    st.sidebar.code(
        "python -m src.verify_cli --pyq 54\n"
        "python -m src.verify_cli --file q.txt",
        language="bash",
    )

    st.sidebar.markdown("### Docs")

    st.sidebar.markdown(
        "- `DESIGN.md` — the design doc\n"
        "- `EVALUATION.md` — every arm, including the null results\n"
        "- `docs/SYSTEM_DESIGN.md` — components and contracts"
    )


if __name__ == "__main__":
    main()
