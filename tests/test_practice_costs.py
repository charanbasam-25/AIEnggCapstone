"""Prevent incomplete or unsupported billing evidence from becoming zero cost."""

import json
from types import SimpleNamespace

import pytest

from src.orchestration.telemetry import MeasuredClient, capture_telemetry
from src.practice.costs import estimate_run_cost


def call(**changes):
    return {
        "model": "gpt-4o-mini", "service_tier": "default", "usage_recorded": True,
        "input_tokens": 1_000_000, "cached_input_tokens": 400_000,
        "output_tokens": 1_000_000, **changes,
    }


def run(*calls, **changes):
    return {"api_calls": len(calls), "usage_recorded": True, "calls": list(calls), **changes}


def test_prices_actual_cached_usage_and_the_supported_snapshot():
    estimate = estimate_run_cost(run(call(), call(model="gpt-4o-mini-2024-07-18")))
    assert estimate.usd == pytest.approx(1.44)
    assert estimate.priced_calls == 2
    assert not estimate.issues and not estimate.uncached_assumptions
    assert not estimate.standard_tier_assumptions
    assert estimate.report()["currency"] == "USD"


def test_historical_cache_and_tier_gaps_are_conservative_explicit_assumptions():
    old_call = call()
    del old_call["cached_input_tokens"]
    del old_call["service_tier"]
    estimate = estimate_run_cost(run(old_call))
    assert estimate.usd == pytest.approx(0.75)
    assert estimate.uncached_assumptions == estimate.standard_tier_assumptions == 1


def test_checked_bank_reuse_has_zero_new_model_cost():
    estimate = estimate_run_cost(run())
    assert estimate.usd == 0
    assert estimate.total_calls == estimate.priced_calls == 0


@pytest.mark.parametrize("changes", [
    {"usage_recorded": False, "input_tokens": 0, "output_tokens": 0},
    {"model": "unknown-model"},
    {"model": ["malformed-model"]},
    {"service_tier": "priority"},
    {"service_tier": "flex"},
    {"input_tokens": -1},
    {"input_tokens": True},
    {"output_tokens": None},
    {"cached_input_tokens": 1_000_001},
    {"cached_input_tokens": -1},
    {"cached_input_tokens": 1.5},
])
def test_missing_invalid_or_unsupported_call_details_cannot_be_priced_as_zero(changes):
    estimate = estimate_run_cost(run(call(**changes)))
    assert estimate.usd is None
    assert estimate.issues and estimate.priced_calls == 0


@pytest.mark.parametrize("changes", [
    {"api_calls": 2}, {"api_calls": True}, {"calls": None}, {"usage_recorded": False},
])
def test_inconsistent_run_metadata_cannot_yield_a_complete_estimate(changes):
    assert estimate_run_cost(run(call(), **changes)).usd is None


def test_failed_call_leaves_a_labeled_known_subtotal_instead_of_a_total():
    estimate = estimate_run_cost(run(call(), call(usage_recorded=False, input_tokens=0, output_tokens=0)))
    assert estimate.usd is None
    assert estimate.recorded_usd == pytest.approx(0.72)
    assert (estimate.priced_calls, estimate.total_calls) == (1, 2)


def test_telemetry_captures_cache_and_tier_without_prompt_or_credentials():
    response = SimpleNamespace(
        model="gpt-4o-mini-2024-07-18", service_tier="default",
        usage=SimpleNamespace(
            input_tokens=100, output_tokens=20,
            input_tokens_details=SimpleNamespace(cached_tokens=40),
        ),
    )
    raw_client = SimpleNamespace(
        api_key="SECRET_API_KEY", responses=SimpleNamespace(parse=lambda **kwargs: response),
    )
    with capture_telemetry() as telemetry:
        assert MeasuredClient(raw_client).responses.parse(
            model="gpt-4o-mini", input="PRIVATE_PROMPT student@example.test",
        ) is response
    recorded = telemetry.calls[0]
    assert recorded["cached_input_tokens"] == 40
    assert recorded["service_tier"] == "default"
    assert recorded["model"] == response.model
    text = json.dumps(recorded)
    assert "PRIVATE_PROMPT" not in text and "student@example.test" not in text
    assert "SECRET_API_KEY" not in text


def test_failed_telemetry_retains_only_error_class_and_unknown_usage():
    def fail(**kwargs):
        raise ConnectionError("PRIVATE_PROVIDER_MESSAGE")
    with capture_telemetry() as telemetry:
        with pytest.raises(ConnectionError):
            MeasuredClient(SimpleNamespace(responses=SimpleNamespace(parse=fail))).responses.parse(
                model="gpt-4o-mini", input="PRIVATE_PROMPT",
            )
    recorded = telemetry.calls[0]
    assert recorded["error_type"] == "ConnectionError"
    assert recorded["cached_input_tokens"] is None and not recorded["usage_recorded"]
    assert "PRIVATE_PROVIDER_MESSAGE" not in json.dumps(recorded)
    assert estimate_run_cost({**telemetry.summary(), "calls": telemetry.calls}).usd is None
