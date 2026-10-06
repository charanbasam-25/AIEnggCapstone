"""Measure the workflow and retain approved-source retrieval diagnostics."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from math import isfinite
from time import perf_counter

from openai import OpenAI


GENERATION_POLICY = "source-grounded-practice-v4"
RETRIEVAL_TRACE_VERSION = "retrieval-trace-v1"
MODEL_NAME = "gpt-4o-mini"
_active_run: ContextVar = ContextVar("practice_run", default=None)
_active_stage: ContextVar = ContextVar("practice_stage", default="Model call")
_active_attempt: ContextVar = ContextVar("practice_attempt", default=0)


@dataclass
class RunTelemetry:
    stages: list[dict] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)
    retrievals: list[dict] = field(default_factory=list)
    evidence_packets: list[dict] = field(default_factory=list)
    question_number: int = 0

    def summary(self) -> dict:
        return {
            "api_calls": len(self.calls),
            "failed_api_calls": sum(not call["success"] for call in self.calls),
            "input_tokens": sum(call["input_tokens"] for call in self.calls),
            "output_tokens": sum(call["output_tokens"] for call in self.calls),
            "usage_recorded": all(call["usage_recorded"] for call in self.calls),
        }


@contextmanager
def capture_telemetry():
    run = RunTelemetry()
    token = _active_run.set(run)
    try:
        yield run
    finally:
        _active_run.reset(token)


@contextmanager
def measured_stage(name: str, attempt: int = 0):
    token = _active_stage.set(name)
    attempt_token = _active_attempt.set(attempt)
    started = perf_counter()
    success = False
    try:
        yield
        success = True
    finally:
        run = _active_run.get()
        if run is not None:
            run.stages.append({
                "stage": name,
                "question": run.question_number,
                "attempt": attempt,
                "seconds": round(perf_counter() - started, 4),
                "success": success,
            })
        _active_stage.reset(token)
        _active_attempt.reset(attempt_token)


def measured_node(function):
    @wraps(function)
    def wrapped(state):
        attempt = state.get("retry_count", 0) + int(function.__name__ == "generate_mcq")
        with measured_stage(function.__name__, attempt=attempt):
            return function(state)
    return wrapped


def _chunk_key(chunk: dict) -> tuple:
    return tuple(str(chunk.get(field, "")) for field in
                 ("id", "source", "document", "page", "chunk_index", "text"))


def _finite_score(value):
    try:
        value = float(value)
        return value if isfinite(value) else None
    except (TypeError, ValueError):
        return None


def record_retrieval(
    *, query: str, candidates: list[dict], reranked: list[dict], top_k: int,
    candidate_k: int, semantic_seconds: float | None, rerank_seconds: float | None,
    elapsed_seconds: float, embedding_model: str, reranker_model: str,
    error_type: str | None = None,
) -> None:
    """Observe an actual search without changing its output or making new calls."""
    run = _active_run.get()
    if run is None:
        return
    ranked = {_chunk_key(item): (rank, item) for rank, item in enumerate(reranked, 1)}
    semantic_ranks = {_chunk_key(item): rank for rank, item in enumerate(candidates, 1)}

    def row(item, semantic_rank, reranker_rank=None, reranker_score=None):
        return {
            **{field: item.get(field) for field in
               ("id", "source", "document", "page", "chunk_index", "text")},
            "semantic_rank": semantic_rank, "semantic_score": _finite_score(item.get("score")),
            "reranker_rank": reranker_rank, "reranker_score": _finite_score(reranker_score),
            "selected": reranker_rank is not None and reranker_rank <= top_k,
        }

    candidate_rows = []
    for rank, item in enumerate(candidates, 1):
        reranker_rank, reranked_item = ranked.get(_chunk_key(item), (None, {}))
        candidate_rows.append(row(item, rank, reranker_rank, reranked_item.get("reranker_score")))
    run.retrievals.append({
        "id": f"retrieval-{len(run.retrievals) + 1}",
        "question": run.question_number, "attempt": _active_attempt.get(),
        "stage": _active_stage.get(), "query": query,
        "candidate_k": candidate_k, "top_k": top_k,
        "embedding_model": embedding_model, "reranker_model": reranker_model,
        "semantic_seconds": semantic_seconds, "rerank_seconds": rerank_seconds,
        "elapsed_seconds": elapsed_seconds, "success": error_type is None,
        "error_type": error_type, "candidates": candidate_rows,
        "selected": [row(item, semantic_ranks[_chunk_key(item)], rank, item.get("reranker_score"))
                     for rank, item in enumerate(reranked[:top_k], 1)],
    })


def record_evidence_packet(label: str, evidence: list[dict], **metadata) -> None:
    """Record the actual expanded pages passed to generation or verification."""
    run = _active_run.get()
    if run is None:
        return
    pages = {(item.get("source"), item.get("document"), item.get("page")) for item in evidence}
    trace_ids = [trace["id"] for trace in run.retrievals
                 if trace["question"] == run.question_number
                 and trace["attempt"] == _active_attempt.get()
                 and any((item.get("source"), item.get("document"), item.get("page")) in pages
                         for item in trace["selected"])]
    run.evidence_packets.append({
        "label": label, "question": run.question_number, "attempt": _active_attempt.get(),
        "stage": _active_stage.get(), "retrieval_ids": trace_ids,
        "evidence": [dict(item) for item in evidence], **metadata,
    })


class _MeasuredResponses:
    def __init__(self, responses):
        self.responses = responses

    def parse(self, **kwargs):
        started = perf_counter()
        response, error = None, None
        try:
            response = self.responses.parse(**kwargs)
            return response
        except Exception as exception:
            error = type(exception).__name__
            raise
        finally:
            run = _active_run.get()
            if run is not None:
                usage = getattr(response, "usage", None)
                input_details = getattr(usage, "input_tokens_details", None)
                run.calls.append({
                    "stage": _active_stage.get(),
                    "question": run.question_number,
                    "attempt": _active_attempt.get(),
                    "model": getattr(response, "model", kwargs.get("model")),
                    "service_tier": getattr(response, "service_tier", None),
                    "seconds": round(perf_counter() - started, 4),
                    "success": error is None,
                    "error_type": error,
                    "usage_recorded": usage is not None,
                    "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
                    # Missing historical/provider details remain unknown; a
                    # missing cache count must not invent a cache discount.
                    "cached_input_tokens": getattr(input_details, "cached_tokens", None),
                    "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
                })


class MeasuredClient:
    def __init__(self, client):
        self.responses = _MeasuredResponses(client.responses)


def create_model_client():
    # SDK transport retries are bounded separately from candidate revisions.
    # OpenAI reads OPENAI_API_KEY; it is never added to the telemetry.
    return MeasuredClient(OpenAI(timeout=45.0, max_retries=1))
