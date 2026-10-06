"""Prepare a bounded batch; rejected candidates never become practice items."""

import os
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from dotenv import load_dotenv
from openai import OpenAIError

from src.orchestration.graph import build_graph
from src.orchestration.nodes import corpus_hash, decide
from src.orchestration.telemetry import GENERATION_POLICY, RETRIEVAL_TRACE_VERSION, capture_telemetry
from src.practice.catalog import topic_focus
from src.practice.models import (
    REQUIRED_GATES, PracticeQuestion, PracticeRequest, PreparationResult,
    question_fingerprint, record_id,
)
from src.practice.store import PracticeStore


PROGRESS_LABELS = {
    "retrieve_sources": "Finding source passages",
    "generate_mcq": "Writing a practice question",
    "extract_claims": "Checking the question format",
    "verify_claims": "Checking the statements against sources",
    "verify_answer_key": "Checking the answer and alternatives",
    "audit_quality": "Reviewing clarity and exam style",
    "explain_question": "Checking explanations for every option",
    "decide": "Completing the evidence checks",
}


def question_from_state(state: dict, request: PracticeRequest, source_hash: str) -> PracticeQuestion:
    # The final trust boundary rechecks all mandatory gates. A terminal
    # string or a structured model response by itself cannot publish an item.
    if state.get("decision") != "ACCEPT" or decide(state)["decision"] != "ACCEPT":
        raise ValueError("The candidate has not passed every publication gate.")
    lesson = state["learning_notes"].notes
    mcq = state["mcq"].model_copy(update={"explanation": lesson.summary.text})
    payload = {
        "topic": request.topic, "subject": "Polity",
        "question_format": state["question_format"], "difficulty": request.difficulty,
        "mcq": mcq.model_dump(mode="json"), "notes": lesson.model_dump(mode="json"),
        "evidence": state["answer_evidence"], "corpus_hash": source_hash,
        "generation_policy": GENERATION_POLICY,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gates": {gate: "PASS" for gate in sorted(REQUIRED_GATES)},
        "review_type": "automated_evidence_review",
    }
    payload["id"] = record_id(payload)
    return PracticeQuestion.model_validate(payload)


def prepare_practice(
    request: PracticeRequest, store: PracticeStore | None = None,
    workflow=None, on_progress=None, source_hash_fn=corpus_hash,
) -> PreparationResult:
    request = PracticeRequest.model_validate(request.model_dump())
    store = store or PracticeStore()
    started = perf_counter()
    source_hash = source_hash_fn()
    questions, candidate_records, question_records = [], [], []
    excluded = set(request.exclude_ids)
    avoid_questions = store.used_stems(request.topic, source_hash)
    focus_offset = len(avoid_questions)
    used_fingerprints = set()
    bank = {
        form: store.load_questions(request.topic, request.difficulty, form, source_hash, excluded)
        if request.reuse_checked else []
        for form in ("simple", "statements")
    }
    reused, generated, revisions, error_type = 0, 0, 0, None
    load_dotenv()

    with capture_telemetry() as telemetry:
        for index in range(request.count):
            telemetry.question_number = index + 1
            form = request.question_format
            if form == "mixed":
                form = "simple" if index % 2 == 0 else "statements"
            while bank[form] and (
                bank[form][0].id in excluded
                or question_fingerprint(bank[form][0].mcq.question) in used_fingerprints
            ):
                bank[form].pop(0)
            if bank[form]:
                question = bank[form].pop(0)
                questions.append(question)
                excluded.add(question.id)
                used_fingerprints.add(question_fingerprint(question.mcq.question))
                reused += 1
                question_records.append({
                    "slot": index + 1, "question_id": question.id, "origin": "checked_bank",
                    "format": form,
                })
                if on_progress:
                    on_progress(index + 1, request.count, "Loaded a source-checked question")
                continue
            if workflow is None and not os.getenv("OPENAI_API_KEY"):
                error_type = "MissingAPIKey"
                break
            if workflow is None:
                workflow = build_graph()
            state = {
                "topic": request.topic, "focus": topic_focus(request.topic, focus_offset + index),
                "difficulty": request.difficulty, "question_format": form,
                "retry_count": 0, "max_retries": request.max_retries,
                "avoid_questions": avoid_questions,
            }
            terminal = None
            attempts = []
            try:
                for update in workflow.stream(
                    state, config={"recursion_limit": 50}, stream_mode="updates",
                ):
                    for node, payload in update.items():
                        state.update(payload)
                        if node == "generate_mcq":
                            generated += 1
                        if node == "decide":
                            terminal = payload["decision"]
                            revisions += int(terminal == "REVISE")
                            attempts.append({
                                "attempt": state.get("retry_count", 0), "decision": terminal,
                                "answer_verdict": getattr(state.get("answer_verification"), "verdict", "Not checked"),
                                "quality_verdict": getattr(state.get("quality_audit"), "overall_quality", "Not checked"),
                                "explanation_verdict": getattr(state.get("learning_notes"), "verdict", "Not checked"),
                                "failure_reasons": list(state.get("failure_reasons", [])),
                            })
                        if on_progress:
                            on_progress(index + 1, request.count, PROGRESS_LABELS.get(node, "Checking the question"))
                record = {
                    "slot": index + 1, "format": form, "decision": terminal or "ERROR",
                    "attempts": state.get("retry_count", 0),
                    "failure_reasons": state.get("failure_reasons", []),
                    "attempt_history": attempts,
                }
                if terminal == "ACCEPT":
                    question = question_from_state(state, request, source_hash)
                    fingerprint = question_fingerprint(question.mcq.question)
                    if fingerprint in used_fingerprints:
                        record["decision"] = "DUPLICATE"
                    elif source_hash_fn() != source_hash:
                        error_type = "CorpusChanged"
                        break
                    elif store.save_question(question):
                        questions.append(question)
                        record["question_id"] = question.id
                        question_records.append({
                            "slot": index + 1, "question_id": question.id,
                            "origin": "generated", "format": form,
                        })
                        used_fingerprints.add(fingerprint)
                        avoid_questions.append(question.mcq.question)
                    else:
                        record["decision"] = "DUPLICATE"
                candidate_records.append(record)
            except (OpenAIError, TimeoutError, ConnectionError) as exception:
                error_type = type(exception).__name__
                candidate_records.append({
                    "slot": index + 1, "decision": "ERROR", "error_type": error_type,
                    "attempts": state.get("retry_count", 0),
                })
                break
            except ValueError:
                candidate_records.append({
                    "slot": index + 1, "decision": "REJECT",
                    "failure_reasons": ["Structured output or publication validation failed."],
                    "attempts": state.get("retry_count", 0),
                })
        if source_hash_fn() != source_hash:
            # Retire an in-flight session when its source snapshot changes.
            questions = []
            error_type = "CorpusChanged"
        report = {
            "id": uuid4().hex, "created_at": datetime.now(timezone.utc).isoformat(),
            "generation_policy": GENERATION_POLICY, "corpus_hash": source_hash,
            "topic": request.topic, "format": request.question_format,
            "difficulty": request.difficulty, "requested": request.count,
            "accepted": len(questions), "reused": reused, "generated_candidates": generated,
            "revisions": revisions, "error_type": error_type,
            "status": "complete" if len(questions) == request.count else
                      "partial" if questions else "error" if error_type else "empty",
            "elapsed_seconds": round(perf_counter() - started, 4),
            **telemetry.summary(), "stages": telemetry.stages,
            "calls": telemetry.calls, "candidates": candidate_records,
            "retrieval_trace_version": RETRIEVAL_TRACE_VERSION,
            "retrievals": telemetry.retrievals, "evidence_packets": telemetry.evidence_packets,
            "question_records": question_records,
        }
    store.save_run(report)
    return PreparationResult(questions=questions, report=report)
