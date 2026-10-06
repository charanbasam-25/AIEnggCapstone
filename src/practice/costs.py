"""Dated estimates for recorded standard-tier text calls, never invoice totals."""

from dataclasses import asdict, dataclass


PRICING_DATE = "2026-10-07"
PRICING_SOURCE = "https://developers.openai.com/api/docs/models/gpt-4o-mini"
# USD per million tokens, verified against the official model page above.
MODEL_RATES = {
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-4o-mini-2024-07-18": (0.15, 0.075, 0.60),
}


@dataclass(frozen=True)
class CostEstimate:
    usd: float | None
    recorded_usd: float
    priced_calls: int
    total_calls: int
    uncached_assumptions: int
    standard_tier_assumptions: int
    issues: tuple[str, ...]

    def report(self) -> dict:
        return {
            **asdict(self), "currency": "USD", "pricing_as_of": PRICING_DATE,
            "pricing_source": PRICING_SOURCE,
            "scope": "Recorded model usage; excludes infrastructure and unreported retry usage.",
        }


def _token_count(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def estimate_run_cost(run: dict) -> CostEstimate:
    """Price complete recorded usage; expose known subtotals without guessing gaps.

    Older traces lack cached-token and service-tier fields. Their input is
    conservatively priced as uncached, with a stated standard-tier assumption.
    Unknown models, tiers or token counts cannot become a zero-cost result.
    """
    issues = []
    calls = run.get("calls")
    if not isinstance(calls, list):
        calls = []
        issues.append("Per-call usage is missing.")
    expected = run.get("api_calls")
    if not _token_count(expected) or expected != len(calls):
        issues.append("The call count does not match the saved call trace.")
    if run.get("usage_recorded") is False:
        issues.append("The run has incomplete token usage.")
    total, priced, uncached, tier_assumed = 0.0, 0, 0, 0
    for call in calls:
        if not isinstance(call, dict) or call.get("usage_recorded") is not True:
            issues.append("A model call has no recorded token usage.")
            continue
        model = call.get("model")
        rates = MODEL_RATES.get(model) if isinstance(model, str) else None
        if rates is None:
            issues.append("A recorded model has no supported price in this estimator.")
            continue
        tier = call.get("service_tier")
        if tier not in (None, "auto", "default", "standard"):
            issues.append("A recorded service tier has no supported price in this estimator.")
            continue
        input_tokens, output_tokens = call.get("input_tokens"), call.get("output_tokens")
        cached = call.get("cached_input_tokens")
        if not _token_count(input_tokens) or not _token_count(output_tokens):
            issues.append("A model call has invalid token counts.")
            continue
        if cached is not None and (not _token_count(cached) or cached > input_tokens):
            issues.append("A model call has an invalid cached-input count.")
            continue
        if cached is None:
            cached = 0
            uncached += 1
        tier_assumed += tier in (None, "auto")
        input_rate, cached_rate, output_rate = rates
        total += ((input_tokens - cached) * input_rate + cached * cached_rate
                  + output_tokens * output_rate) / 1_000_000
        priced += 1
    return CostEstimate(
        usd=None if issues else total, recorded_usd=total, priced_calls=priced,
        total_calls=expected if _token_count(expected) else len(calls),
        uncached_assumptions=uncached, standard_tier_assumptions=tier_assumed,
        issues=tuple(dict.fromkeys(issues)),
    )
