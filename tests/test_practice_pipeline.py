"""Test publication boundaries with synthetic sources and offline model calls."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from openai import OpenAIError
from pydantic import ValidationError

from src.generation.mcq_generator import MCQ
from src.evaluation.evaluate_generation_loop import gates_fired, run_topic
from src.orchestration import nodes
from src.orchestration.graph import build_graph
from src.orchestration.telemetry import MeasuredClient
from src.practice.models import PracticeRequest
from src.practice.service import prepare_practice
from src.practice.store import PracticeStore
from src.verification.direct_mcq_verifier import (
    DirectAnswerAnalysis, DirectMCQVerifier, EvidenceCitation, OptionAssessment,
    ReferencedDirectAnalysis, ReferencedOptionAssessment,
)
from src.verification.fact_verifier import (
    FactEvidenceCitation, FactVerificationResult, FactVerifier, ReferencedFactEvidenceAssessment,
)
from src.verification.learning_notes import (
    CitedExplanation, ExplanationItemReview, ExplanationReview, LearningNotes, OptionExplanation,
    ExplanationDraft, LearningNotesDraft, OptionExplanationDraft, QuoteSelection,
    article_lookup_notes,
)
from src.verification.mcq_quality_auditor import QualityAuditResult


QUOTES = [
    "Article 14 guarantees equality before the law.",
    "Article 19 protects specified freedoms.",
    "Article 21 protects life and personal liberty.",
    "Article 32 provides constitutional remedies.",
]
EVIDENCE = [{
    "id": 1, "source": "synthetic_test_source", "document": "test.txt",
    "page": 1, "chunk_index": 0, "text": " ".join(QUOTES),
}]


def candidate(form="simple", answer="A"):
    return MCQ(
        question=(
            "Which article guarantees equality before the law?"
            if form == "simple" else
            "Consider the following statements:\n"
            "I. Article 14 is the provision for equality before the law.\n"
            "II. Article 19 is the provision for equality before the law.\n"
            "Which of the statements given above is/are correct?"
        ),
        option_a="Article 14" if form == "simple" else "I only",
        option_b="Article 19" if form == "simple" else "II only",
        option_c="Article 21" if form == "simple" else "Both I and II",
        option_d="Article 32" if form == "simple" else "Neither I nor II",
        correct_answer=answer, explanation="UNREVIEWED_GENERATOR_EXPLANATION",
    )


def lesson():
    return LearningNotes(
        summary=CitedExplanation(
            text="REVIEWED_EXPLANATION_SENTINEL",
            citations=[EvidenceCitation(evidence_id=1, quote=QUOTES[0])],
        ),
        options=[
            OptionExplanation(
                option=letter, is_correct=letter == "A", text=f"Reviewed explanation for {letter}.",
                citations=[EvidenceCitation(evidence_id=1, quote=QUOTES[index])],
            )
            for index, letter in enumerate("ABCD")
        ],
    )


class OfflineModels:
    def __init__(self, form="simple"):
        self.form = form
        self.declared_answer = "A"
        self.unresolved_alternative = False
        self.unknown_answer_quote = False
        self.review_disagrees = False
        self.quality_pass = True
        self.quality_flags = True
        self.explanation_pass = True
        self.invalid_note_quote = False
        self.repeated_note = False
        self.wrong_note_flag = False
        self.item_grounding = True
        self.repair_first_note = False
        self.fail = False
        self.requests = []
        self.responses = SimpleNamespace(parse=self.parse)

    def parse(self, **kwargs):
        self.requests.append(kwargs)
        if self.fail:
            raise OpenAIError("PRIVATE_PROVIDER_INFORMATION")
        schema = kwargs["text_format"]
        if schema == MCQ:
            result = candidate(self.form, self.declared_answer)
        elif schema == ReferencedFactEvidenceAssessment:
            # Synthetic judgments exercise the actual fact-verifier protocol;
            # they do not measure a model's factual accuracy.
            claim = kwargs["input"].split("CLAIM:\n", 1)[1].split("SOURCE EVIDENCE:", 1)[0]
            contradicted = "Article 19" in claim
            result = ReferencedFactEvidenceAssessment(
                verdict="CONTRADICTED" if contradicted else "SUPPORTED",
                reasoning="Synthetic independent claim assessment.",
                citations=[QuoteSelection(quote_id=999 if self.unknown_answer_quote else
                                          2 if contradicted else 1)],
            )
        elif schema in (DirectAnswerAnalysis, ReferencedDirectAnalysis):
            is_review = "No prior assessments" in kwargs["input"]
            winner = "B" if is_review and self.review_disagrees else "A"
            assessments = []
            for index, letter in enumerate("ABCD"):
                unresolved = self.unresolved_alternative and letter == "D"
                assessments.append(OptionAssessment(
                    option=letter,
                    answer_fit="INSUFFICIENT" if unresolved else "SUPPORTED" if letter == winner else "RULED_OUT",
                    reasoning="Synthetic evidence comparison.",
                    citations=[] if unresolved else [EvidenceCitation(evidence_id=1, quote=QUOTES[index])],
                ))
            if schema == ReferencedDirectAnalysis:
                result = ReferencedDirectAnalysis(
                    **{f"option_{item.option.lower()}": ReferencedOptionAssessment(
                        answer_fit=item.answer_fit, reasoning=item.reasoning,
                        citations=[] if item.answer_fit == "INSUFFICIENT" else [QuoteSelection(
                            quote_id=999 if self.unknown_answer_quote and item.option == "D" else index + 1,
                        )],
                    ) for index, item in enumerate(assessments)},
                    reasoning="Synthetic independent answer analysis.",
                )
            else:
                result = DirectAnswerAnalysis(
                    option_assessments=assessments, reasoning="Synthetic independent answer analysis.",
                )
        elif schema == QualityAuditResult:
            result = QualityAuditResult(
                unambiguous=self.quality_flags, single_best_answer=True,
                plausible_distractors=True, appropriate_wording=True, topic_relevant=True,
                overall_quality="PASS" if self.quality_pass else "FAIL", issues=[],
            )
        elif schema == LearningNotesDraft:
            baseline = lesson()
            result = LearningNotesDraft(
                summary=ExplanationDraft(text=baseline.summary.text, citations=[QuoteSelection(quote_id=1)]),
                options=[OptionExplanationDraft(
                    option=note.option, is_correct=note.is_correct, text=note.text,
                    citations=[QuoteSelection(quote_id=index + 1)],
                ) for index, note in enumerate(baseline.options)],
            )
            first_notes = sum(call["text_format"] == LearningNotesDraft for call in self.requests) == 1
            if self.invalid_note_quote or self.repair_first_note and first_notes:
                result.options[2].citations[0].quote_id = 999
            if self.repeated_note:
                result.options[1].text = candidate(self.form).option_b
            if self.wrong_note_flag:
                result.options[1].is_correct = True
        elif schema == ExplanationReview:
            result = ExplanationReview(
                grounded=self.explanation_pass, answer_consistent=True,
                every_option_explained=True, no_unsupported_facts=True, issues=[],
                verdict="PASS",  # Conflicting boolean flags must still block publication.
                items=[ExplanationItemReview(
                    item=label, facts_supported=self.item_grounding or label != "D",
                    quotations_justify_text=True, answer_consistent=True,
                    explains_option=True, issues=[],
                ) for label in ("Summary", "A", "B", "C", "D")],
            )
        else:
            raise AssertionError(f"Unexpected model schema {schema}")
        return SimpleNamespace(
            output_parsed=result, model="synthetic-offline-model",
            usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        )


class OfflineRetriever:
    def __init__(self):
        self.empty = False
        self.queries = []

    def retrieve(self, query, top_k):
        self.queries.append(query)
        return [] if self.empty else EVIDENCE


class OfflineFactVerifier:
    def verify(self, claim, evidence):
        verdict = "CONTRADICTED" if "Article 19" in claim else "SUPPORTED"
        return FactVerificationResult(
            verdict=verdict, reasoning="Resolved synthetic statement.",
            supporting_pages=[1], independently_reviewed=True,
            verification_method="quoted_llm_review", checked_evidence=EVIDENCE,
            citations=[FactEvidenceCitation(evidence_id=1, quote=QUOTES[1] if verdict == "CONTRADICTED" else QUOTES[0])],
        )


@pytest.fixture
def services(monkeypatch, tmp_path):
    import src.generation.mcq_generator as generation
    import src.verification.learning_notes as explanations
    import src.verification.mcq_quality_auditor as quality

    models = OfflineModels()
    client = MeasuredClient(models)
    retriever = OfflineRetriever()
    monkeypatch.setattr(generation, "create_model_client", lambda: client)
    monkeypatch.setattr(explanations, "create_model_client", lambda: client)
    monkeypatch.setattr(quality, "create_model_client", lambda: client)
    monkeypatch.setattr(nodes, "get_chunks", lambda: tuple(EVIDENCE))
    monkeypatch.setattr(nodes, "get_retriever", lambda: retriever)
    monkeypatch.setattr(nodes, "get_fact_verifier", lambda: OfflineFactVerifier())
    monkeypatch.setattr(nodes, "get_direct_verifier", lambda: DirectMCQVerifier(
        chunks=EVIDENCE, client=client, reference_quotes=True,
    ))
    return SimpleNamespace(
        models=models, retriever=retriever, store=PracticeStore(tmp_path / "practice.sqlite3"),
        workflow=build_graph(), source_hash="synthetic-source-hash",
    )


def prepare(services, **kwargs):
    return prepare_practice(
        PracticeRequest(count=1, max_retries=0, **kwargs),
        services.store, workflow=services.workflow,
        source_hash_fn=lambda: services.source_hash,
    )


def test_source_grounded_direct_question_reaches_quiz_only_after_every_review(services):
    result = prepare(services)
    assert len(result.questions) == 1
    question = result.questions[0]
    assert question.mcq.correct_answer == "A"
    assert question.mcq.explanation == "REVIEWED_EXPLANATION_SENTINEL"
    assert "UNREVIEWED" not in question.mcq.explanation
    assert question.notes.options[3].citations
    assert result.report["api_calls"] == 6
    assert result.report["input_tokens"] == 600
    assert result.report["output_tokens"] == 120
    assert result.report["stages"]
    generation_prompt = services.models.requests[0]["input"]
    assert QUOTES[0] in generation_prompt
    option_calls = [
        call for call in services.models.requests if call["text_format"] == ReferencedDirectAnalysis
    ]
    assert len(option_calls) == 2
    assert all("UNREVIEWED_GENERATOR_EXPLANATION" not in call["input"] for call in option_calls)
    assert all("DECLARED ANSWER" not in call["input"] for call in option_calls)
    assert services.retriever.queries[0].startswith("Fundamental Rights.")
    assert len(services.retriever.queries) == 6  # Topic, question and each option.


def test_resolved_false_statement_can_be_part_of_a_valid_mcq(services):
    services.models.form = "statements"
    result = prepare(services, question_format="statements")
    assert len(result.questions) == 1
    assert result.questions[0].mcq.correct_answer == "A"
    assert result.questions[0].mcq.option_a == "I only"


@pytest.mark.parametrize("invalid_reference", [False, True])
def test_action_statements_use_real_quote_bound_fact_verification_before_publication(
    services, monkeypatch, invalid_reference,
):
    baseline_candidate = candidate
    def action_candidate(form="simple", answer="A"):
        mcq = baseline_candidate(form, answer)
        return mcq.model_copy(update={"question": (
            "Consider the following statements:\n"
            "I. Article 14 guarantees equality before the law.\n"
            "II. Article 19 guarantees equality before the law.\n"
            "Which of the statements given above is/are correct?"
        )})
    monkeypatch.setattr(f"{__name__}.candidate", action_candidate)
    services.models.form = "statements"
    services.models.unknown_answer_quote = invalid_reference
    monkeypatch.setattr(nodes, "get_fact_verifier", lambda: FactVerifier(
        chunks=EVIDENCE, client=MeasuredClient(services.models), reference_quotes=True,
    ))
    result = prepare(services, question_format="statements")
    calls = [request for request in services.models.requests
             if request["text_format"] == ReferencedFactEvidenceAssessment]
    assert len(calls) == (2 if invalid_reference else 4)
    assert bool(result.questions) is not invalid_reference
    if invalid_reference:
        assert not services.store.load_questions("Fundamental Rights", "medium", "statements", services.source_hash)
    else:
        assert result.questions[0].mcq.correct_answer == "A"
        assert result.questions[0].gates["answer"] == "PASS"


@pytest.mark.parametrize("setting", [
    "unresolved_alternative", "review_disagrees", "invalid_note_quote",
    "repeated_note", "wrong_note_flag", "unknown_answer_quote",
])
def test_unresolved_disagreement_or_invented_quotes_never_reach_practice(services, setting):
    setattr(services.models, setting, True)
    result = prepare(services)
    assert not result.questions
    assert result.report["status"] == "empty"
    assert not services.store.load_questions("Fundamental Rights", "medium", "simple", services.source_hash)


@pytest.mark.parametrize("setting", ["quality_pass", "quality_flags", "explanation_pass", "item_grounding"])
def test_fail_empty_issues_and_conflicting_pass_flags_cannot_bypass_gates(services, setting):
    setattr(services.models, setting, False)
    result = prepare(services)
    assert not result.questions
    assert result.report["candidates"][0]["decision"] == "REJECT"


def test_answer_key_must_match_independent_verification(services):
    services.models.declared_answer = "B"
    result = prepare(services)
    assert not result.questions
    assert not any(call["text_format"] == LearningNotesDraft for call in services.models.requests)


def test_explanation_repair_rechecks_notes_without_reusing_a_failed_approval(services):
    services.models.repair_first_note = True
    result = prepare(services)
    assert len(result.questions) == 1
    assert result.report["generated_candidates"] == 1
    assert result.report["api_calls"] == 7
    writes = [call for call in services.models.requests if call["text_format"] == LearningNotesDraft]
    assert len(writes) == 2
    assert "outside the approved source catalog" in writes[1]["input"]


def test_literal_article_explanation_uses_the_matching_provision_without_other_legal_claims():
    rule = "The State shall not deny to any person equality before the law or the equal protection of the laws within the territory of India."
    mcq = candidate().model_copy(update={"question": f"Which Article explicitly states that {rule}"})
    evidence = [{**EVIDENCE[0], "text": f"14. Equality before law.—{rule} 15. A different provision."}]
    notes = article_lookup_notes(mcq, evidence)
    assert notes is not None
    assert notes.summary.citations[0].quote == f"14. Equality before law.—{rule}"
    assert [note.is_correct for note in notes.options] == [True, False, False, False]
    assert "does not identify the Article requested" in notes.options[1].text


def test_literal_article_explanation_does_not_match_only_shared_keywords():
    evidence = [{**EVIDENCE[0], "text": "14. Equality before law.—The State shall follow a different rule entirely."}]
    assert article_lookup_notes(candidate(), evidence) is None


def test_candidate_revision_budget_is_bounded_and_drafts_stay_hidden(services):
    services.models.declared_answer = "B"
    result = prepare_practice(
        PracticeRequest(count=1, max_retries=2), services.store,
        workflow=services.workflow, source_hash_fn=lambda: services.source_hash,
    )
    assert not result.questions
    assert result.report["generated_candidates"] == 3
    assert result.report["revisions"] == 2
    assert result.report["api_calls"] == 9
    assert "UNREVIEWED_GENERATOR_EXPLANATION" not in json.dumps(result.report)


def test_missing_sources_cannot_trigger_a_model_call(services):
    services.retriever.empty = True
    result = prepare(services)
    assert not result.questions
    assert not services.models.requests
    assert result.report["api_calls"] == 0


def test_bank_reuse_runs_no_models_and_current_sources_are_required(services):
    first = prepare(services)
    services.models.fail = True
    second = prepare(services)
    assert second.questions[0].id == first.questions[0].id
    assert second.report["api_calls"] == 0
    assert second.report["reused"] == 1
    assert not services.store.load_questions("Fundamental Rights", "medium", "simple", "changed-source")


def test_forcing_fresh_generation_cannot_publish_a_duplicate(services):
    assert prepare(services).questions
    second = prepare(services, reuse_checked=False)
    assert not second.questions
    assert second.report["api_calls"] == 1  # Structural duplicate check stops paid verification.


def test_source_changes_retire_inflight_questions_before_publication(services):
    hashes = iter(["old", "changed", "changed"])
    result = prepare_practice(
        PracticeRequest(count=1, max_retries=0), services.store,
        workflow=services.workflow, source_hash_fn=lambda: next(hashes, "changed"),
    )
    assert not result.questions
    assert result.report["error_type"] == "CorpusChanged"
    assert not services.store.load_questions("Fundamental Rights", "medium", "simple", "old")


def test_provider_failures_are_recorded_without_private_details_or_draft_answers(services):
    services.models.fail = True
    result = prepare(services)
    assert not result.questions
    assert result.report["error_type"] == "OpenAIError"
    assert result.report["failed_api_calls"] == 1
    assert "PRIVATE_PROVIDER_INFORMATION" not in json.dumps(result.report)


def test_new_attempt_clears_previous_question_approvals(services):
    old = {
        "topic": "Fundamental Rights", "generation_evidence": EVIDENCE,
        "quality_audit": object(), "learning_notes": object(),
        "direct_verification": object(), "answer_verification": object(),
    }
    update = nodes.generate_mcq(old)
    assert update["answer_verification"] is None
    assert update["quality_audit"] is None
    assert update["learning_notes"] is None
    assert update["direct_verification"] is None


def test_corrupted_saved_question_is_not_reused(services):
    first = prepare(services)
    question = first.questions[0]
    payload = question.model_dump(mode="json")
    payload["mcq"]["correct_answer"] = "D"
    with services.store._connection() as connection:
        connection.execute("UPDATE questions SET payload=? WHERE id=?", (json.dumps(payload), question.id))
        connection.commit()
    assert not services.store.load_questions("Fundamental Rights", "medium", "simple", services.source_hash)


def test_locked_subjects_and_unlisted_topics_are_rejected_before_workflow_execution():
    with pytest.raises(ValidationError):
        PracticeRequest(subject="History")
    with pytest.raises(ValidationError):
        PracticeRequest(topic="An arbitrary unsupported topic")


def test_generation_report_allows_resolved_false_statements(services):
    services.models.form = "statements"
    report = run_topic(services.workflow, "Fundamental Rights", "statements")
    assert report["terminal_decision"] == "ACCEPT"
    attempt = report["attempts"][0]
    assert "CONTRADICTED" in attempt["fact_verdicts"].values()
    assert not any(gates_fired(attempt).values())


def test_generation_report_blocks_false_quality_flags_and_skipped_explanations(services):
    services.models.quality_flags = False
    report = run_topic(services.workflow, "Fundamental Rights")
    assert report["terminal_decision"] == "REJECT"
    assert all(gates_fired(attempt)["quality"] for attempt in report["attempts"])
    assert all(gates_fired(attempt)["explanations"] for attempt in report["attempts"])


@pytest.fixture
def practice_app(services, monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    import src.ui.practice as ui

    services.source_hash = nodes.corpus_hash()
    monkeypatch.setattr(ui, "get_practice_store", lambda: services.store)

    def offline_prepare(request, store, on_progress=None):
        return prepare_practice(
            request, store, workflow=services.workflow, on_progress=on_progress,
            source_hash_fn=lambda: services.source_hash,
        )

    monkeypatch.setattr(ui, "prepare_practice", offline_prepare)
    st.cache_resource.clear()
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=60).run()
    yield app
    st.cache_resource.clear()


def generate_in_ui(app):
    app.selectbox(key="practice_count").set_value(1).run()
    next(button for button in app.button if button.label == "Generate MCQs").click().run()
    assert not app.exception, [item.message for item in app.exception]
    return app


def test_main_ui_starts_with_practice_and_keeps_measurements_in_sidebar_page(practice_app, services):
    assert not practice_app.exception
    assert not practice_app.tabs
    assert practice_app.radio(key="workspace").options == [
        "Practice", "My Practice", "How it works",
    ]
    assert not practice_app.metric
    subject_buttons = [
        button for button in practice_app.button
        if button.key and button.key.startswith("choose_")
    ]
    assert len(subject_buttons) == 5
    assert not subject_buttons[0].disabled
    assert all(button.disabled for button in subject_buttons[1:])
    practice_app.selectbox(key="practice_count").set_value(5).run()
    practice_app.radio(key="workspace").set_value("How it works").run()
    assert not practice_app.exception
    assert not practice_app.tabs
    practice_app.radio(key="workspace").set_value("Practice").run()
    assert not practice_app.exception
    assert practice_app.selectbox(key="practice_count").value == 5
    assert not services.models.requests  # Viewing the UI makes no model calls.


def test_project_details_are_read_only_and_migrate_the_old_usage_route(practice_app, services):
    app = practice_app
    app.radio(key="workspace").set_value("How it works").run()
    assert app.radio(key="about_section").options == [
        "About this site", "Privacy & PII", "Guardrails", "System design", "Evals",
        "Error handling", "Cost & latency",
    ]
    for section in ("Privacy & PII", "System design", "Error handling", "Cost & latency"):
        app.radio(key="about_section").set_value(section).run()
        assert not app.exception, [item.message for item in app.exception]
        assert any(item.value == section for item in app.subheader)
    # Inject a saved legacy value while its widget is absent. AppTest cannot
    # serialize an out-of-options value into an already rendered new radio.
    app.radio(key="workspace").set_value("Practice").run()
    app.session_state["about_section"] = "Latency & Usage"
    app.radio(key="workspace").set_value("How it works").run()
    assert not app.exception
    assert app.radio(key="about_section").value == "Cost & latency"
    assert not services.models.requests
    assert "practice_session" not in app.session_state


def test_ui_hides_answers_until_submission_and_keeps_mistakes_without_reverification(practice_app, services):
    app = generate_in_ui(practice_app)
    assert "practice_session" in app.session_state
    assert not any("REVIEWED_EXPLANATION_SENTINEL" in item.value for item in app.markdown)
    answer = next(item for item in app.radio if item.label == "Your answer · Question 1")
    assert answer.value is None
    answer.set_value("B").run()
    app.radio(key="workspace").set_value("How it works").run()
    assert not app.exception
    assert len(services.models.requests) == 6
    next(button for button in app.button if button.label == "Explore Evals").click().run()
    assert not app.exception
    assert any("Submit your active practice set" in item.value for item in app.info)
    assert not any("REVIEWED_EXPLANATION_SENTINEL" in item.value for item in app.markdown)
    app.radio(key="workspace").set_value("My Practice").run()
    assert not app.exception
    assert not app.session_state["practice_session"]["submitted"]
    app.radio(key="workspace").set_value("Practice").run()
    assert not app.exception
    assert next(item for item in app.radio if item.label == "Your answer · Question 1").value == "B"
    assert not any("REVIEWED_EXPLANATION_SENTINEL" in item.value for item in app.markdown)
    next(button for button in app.button if button.label == "Submit answers").click().run()
    assert not app.exception, [item.message for item in app.exception]
    assert any("REVIEWED_EXPLANATION_SENTINEL" in item.value for item in app.markdown)
    assert app.session_state["practice_session"]["score"]["wrong"] == 1
    assert len(app.session_state["practice_history"]) == 1
    assert len(services.models.requests) == 6
    app.radio(key="workspace").set_value("How it works").run()
    assert not app.exception, [item.message for item in app.exception]
    assert not any("Submit your active practice set" in item.value for item in app.info)
    app.radio(key="about_section").set_value("Cost & latency").run()
    assert not app.exception
    app.radio(key="about_section").set_value("Evals").run()
    assert not app.exception, [item.message for item in app.exception]
    app.run()
    assert not app.exception
    assert len(services.models.requests) == 6


def test_ui_latency_shows_the_latest_reused_run_and_preserves_history(practice_app, services):
    app = generate_in_ui(practice_app)
    generated_id = app.session_state["practice_last_run"]["id"]
    app.radio(key="workspace").set_value("How it works").run()
    next(button for button in app.button if button.label == "View cost & latency").click().run()
    assert not app.exception
    assert next(metric for metric in app.metric if metric.label == "Model calls").value == "6"
    assert next(metric for metric in app.metric if metric.label == "Estimated model cost").value == "—"
    app.radio(key="workspace").set_value("Practice").run()
    next(button for button in app.button if button.label == "Generate MCQs").click().run()
    assert not app.exception
    assert app.session_state["practice_last_run"]["reused"] == 1
    assert not any(metric.label == "Model calls" for metric in app.metric)
    app.radio(key="workspace").set_value("How it works").run()
    assert not app.exception
    assert next(metric for metric in app.metric if metric.label == "Model calls").value == "0"
    assert next(metric for metric in app.metric if metric.label == "Estimated model cost").value == "$0.00"
    app.selectbox(key="practice_latency_run").set_value(generated_id).run()
    assert next(metric for metric in app.metric if metric.label == "Model calls").value == "6"
    app.run()
    assert next(metric for metric in app.metric if metric.label == "Model calls").value == "6"
    app.radio(key="workspace").set_value("Practice").run()
    next(button for button in app.button if button.label == "Generate MCQs").click().run()
    assert not app.exception
    app.radio(key="workspace").set_value("How it works").run()
    assert not app.exception
    assert next(metric for metric in app.metric if metric.label == "Model calls").value == "0"
    assert len(services.models.requests) == 6


def test_ui_failed_new_generation_clears_old_quiz_and_private_provider_details(practice_app, services):
    app = generate_in_ui(practice_app)
    services.models.fail = True
    app.selectbox(key="practice_difficulty").set_value("easy").run()
    next(button for button in app.button if button.label == "Generate MCQs").click().run()
    assert not app.exception
    assert "practice_session" not in app.session_state
    assert not any("PRIVATE_PROVIDER_INFORMATION" in item.value for item in app.error)
    assert not any(item.label.startswith("Your answer ·") for item in app.radio)


def test_ui_retires_a_quiz_when_its_source_snapshot_changes(practice_app, monkeypatch):
    import src.ui.practice as ui
    app = generate_in_ui(practice_app)
    monkeypatch.setattr(ui, "corpus_hash", lambda: "changed")
    app.run()
    assert not app.exception
    assert "practice_session" not in app.session_state
    assert any("fresh source checks" in item.value for item in app.info)


def test_practice_records_each_stage_search_and_bank_reuse_makes_no_new_searches(services, monkeypatch):
    from src.orchestration.telemetry import RETRIEVAL_TRACE_VERSION, record_retrieval

    def retrieve(query, top_k):
        evidence = services.retriever.retrieve(query, top_k)
        reranked = [{**item, "reranker_score": 1.0} for item in evidence]
        record_retrieval(
            query=query, candidates=evidence, reranked=reranked, top_k=top_k,
            candidate_k=20, semantic_seconds=.01, rerank_seconds=.02, elapsed_seconds=.03,
            embedding_model="offline", reranker_model="offline",
        )
        return reranked

    monkeypatch.setattr(nodes, "get_retriever", lambda: SimpleNamespace(retrieve=retrieve))
    result = prepare(services)
    assert result.report["accepted"] == 1
    assert result.report["retrieval_trace_version"] == RETRIEVAL_TRACE_VERSION
    assert len(result.report["retrievals"]) == 6
    first, *verification = result.report["retrievals"]
    assert first["stage"] == "retrieve_sources" and first["attempt"] == 0
    assert all(trace["stage"] == "verify_answer_key" and trace["attempt"] == 1 for trace in verification)
    assert all(trace["question"] == 1 for trace in result.report["retrievals"])
    assert result.report["question_records"][0]["question_id"] == result.questions[0].id
    assert result.report["candidates"][0]["attempt_history"][0]["decision"] == "ACCEPT"
    packets = result.report["evidence_packets"]
    assert len(packets) == 2
    assert packets[0]["retrieval_ids"] == [first["id"]]
    assert packets[1]["retrieval_ids"] == [trace["id"] for trace in verification]
    reused = prepare(services)
    assert reused.report["reused"] == 1
    assert reused.report["retrievals"] == [] and reused.report["api_calls"] == 0
    assert reused.report["question_records"][0]["origin"] == "checked_bank"
    assert len(services.models.requests) == 6
