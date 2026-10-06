"""Exercise verification CLI routing with fake retrieval and model responses."""

import json
import sys
from types import SimpleNamespace

import pytest

from src.verification.direct_mcq_verifier import (
    DirectAnswerAnalysis,
    DirectMCQVerifier,
    EvidenceCitation,
    OptionAssessment,
)
from src.verification.fact_verifier import FactVerificationResult, FactVerifier


QUOTE = "A constitution defines and limits the powers of government."
EVIDENCE = [{"source": "test_source", "page": 7, "text": QUOTE}]


class FakeRetriever:
    def __init__(self, chunks):
        pass

    def retrieve(self, query, top_k):
        return EVIDENCE


@pytest.fixture
def offline_services(monkeypatch):
    import streamlit as st
    import src.verification.claim_retriever as retrieval

    st.cache_resource.clear()
    requests = []
    analysis = DirectAnswerAnalysis(
        option_assessments=[OptionAssessment(
            option=letter, answer_fit="SUPPORTED" if letter == "C" else "RULED_OUT",
            reasoning="Synthetic option comparison.",
            citations=[EvidenceCitation(evidence_id=1, quote=QUOTE)],
        ) for letter in "ABCD"],
        reasoning="Synthetic evidence establishes one answer.",
    )
    def parse(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(output_parsed=analysis)
    fake_client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
    monkeypatch.setattr(retrieval, "ClaimRetriever", FakeRetriever)
    monkeypatch.setattr(DirectMCQVerifier, "client", property(lambda self: fake_client))
    monkeypatch.setattr(FactVerifier, "__init__", lambda self, chunks=None: None)
    monkeypatch.setattr(FactVerifier, "verify", lambda self, claim, evidence: FactVerificationResult(
        verdict="SUPPORTED", reasoning="Synthetic statement check.", supporting_pages=[7],
    ))
    yield requests
    st.cache_resource.clear()


def test_cli_direct_example_saves_citations_without_answer_key_leakage(
    offline_services, monkeypatch, tmp_path, capsys,
):
    import src.verify_cli as cli

    monkeypatch.setattr(cli, "ClaimRetriever", FakeRetriever)
    output = tmp_path / "direct_result.json"
    monkeypatch.setattr(sys, "argv", [
        "verify_cli", "--file", "data/examples/direct_constitution_2023.txt",
        "--save", str(output), "--show-evidence",
    ])
    cli.main()
    result = json.loads(output.read_text())
    assert result["predicted_answer"] == "C"
    assert result["question_type"] == "best_answer_mcq"
    assert result["evidence"][0]["source"] == "test_source"
    assert result["option_assessments"][2]["citations"][0]["quote"] == QUOTE
    assert "ANSWER: C." in capsys.readouterr().out
    assert len(offline_services) == 2


def test_cli_rejects_statement_only_policy_for_direct_mcq(offline_services, monkeypatch):
    import src.verify_cli as cli

    monkeypatch.setattr(sys, "argv", [
        "verify_cli", "--file", "data/examples/direct_constitution_2023.txt",
        "--policy", "closed_world",
    ])
    with pytest.raises(SystemExit, match="numbered statements only"):
        cli.main()
    assert not offline_services
