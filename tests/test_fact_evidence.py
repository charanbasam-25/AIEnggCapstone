"""Evidence and legal-status regressions, without live model calls."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.verification.evidence_context import PageEvidenceContext
from src.verification.fact_verifier import (
    FactEvidenceAssessment,
    FactEvidenceCitation,
    FactVerifier,
    ReferencedFactEvidenceAssessment,
    resolve_fact_assessment,
    resolve_fact_references,
    verify_constructed_claim,
)
from src.verification.source_quotes import QuoteSelection, build_quote_catalog


PASSAGE = {
    "id": 1, "source": "handbook", "document": "Example handbook", "page": 12,
    "chunk_index": 0, "text": "The committee has exactly seven members.",
}


def assessment(verdict="SUPPORTED", evidence_id=1, quote=PASSAGE["text"]):
    return FactEvidenceAssessment(
        verdict=verdict, reasoning="The quoted rule establishes the member count.",
        citations=[FactEvidenceCitation(evidence_id=evidence_id, quote=quote)],
    )


def fake_client(result):
    requests = []
    def parse(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(output_parsed=result)
    return SimpleNamespace(responses=SimpleNamespace(parse=parse)), requests


def test_pages_are_derived_from_actual_evidence_ids():
    result = resolve_fact_assessment(assessment(), [PASSAGE])
    assert result.verdict == "SUPPORTED"
    assert result.supporting_pages == [12]
    assert not result.validation_issues


@pytest.mark.parametrize("result", [
    assessment(evidence_id=2),
    assessment(evidence_id=0),
    assessment(quote="The committee has exactly eleven members."),
    assessment(quote="    "),
    FactEvidenceAssessment(verdict="SUPPORTED", reasoning="An unsupported assertion.", citations=[]),
    FactEvidenceAssessment(verdict="CONTRADICTED", reasoning="An unsupported contradiction.", citations=[]),
])
def test_unverifiable_committed_judgments_become_insufficient(result):
    checked = resolve_fact_assessment(result, [PASSAGE])
    assert checked.verdict == "INSUFFICIENT"
    assert checked.validation_issues


def test_one_invalid_quote_prevents_committing_even_with_another_valid_quote():
    result = assessment()
    result.citations.append(FactEvidenceCitation(evidence_id=1, quote="There are nine members."))
    checked = resolve_fact_assessment(result, [PASSAGE])
    assert checked.verdict == "INSUFFICIENT"
    assert len(checked.citations) == 1


def test_whitespace_normalization_does_not_allow_changed_words_or_negation():
    checked = resolve_fact_assessment(
        assessment(quote="The committee\n has exactly seven members."), [PASSAGE]
    )
    assert checked.verdict == "SUPPORTED"
    changed = resolve_fact_assessment(
        assessment(quote="The committee does not have exactly seven members."), [PASSAGE]
    )
    assert changed.verdict == "INSUFFICIENT"


def test_insufficient_judgment_needs_no_supporting_quotation():
    result = resolve_fact_assessment(FactEvidenceAssessment(
        verdict="INSUFFICIENT", reasoning="The term of office is not established.", citations=[]
    ), [PASSAGE])
    assert result.verdict == "INSUFFICIENT"
    assert not result.validation_issues


def test_source_and_document_identity_prevent_page_number_collisions():
    others = [
        {**PASSAGE, "id": 2, "source": "another_source", "text": "An unrelated rule."},
        {**PASSAGE, "id": 3, "document": "Another handbook", "text": "A conflicting rule."},
    ]
    context = PageEvidenceContext([PASSAGE, *others]).expand([PASSAGE])
    assert context[0]["text"] == PASSAGE["text"]
    assert context[0]["context_chunk_ids"] == [1]


def test_overlapping_chunks_are_restored_without_duplicating_the_boundary():
    overlap = "The continuing explanation of this specific provision. "
    first = {**PASSAGE, "text": "Start of the rule. " + overlap}
    second = {**PASSAGE, "id": 2, "chunk_index": 1, "text": overlap + "The status note is here."}
    context = PageEvidenceContext([second, first]).expand([first, second])
    assert len(context) == 1
    assert context[0]["text"].count(overlap) == 1
    assert context[0]["text"].endswith("The status note is here.")
    assert context[0]["retrieved_chunk_ids"] == [1, 2]


def test_actual_constitution_page_restores_invalidation_footnote():
    corpus = Path(__file__).resolve().parents[1] / "data/processed/chunks.jsonl"
    chunks = [json.loads(line) for line in corpus.read_text().splitlines() if line.strip()]
    truncated = next(chunk for chunk in chunks if chunk["id"] == 733)
    assert "Minerva Mills" not in truncated["text"]
    checked = PageEvidenceContext(chunks).expand([truncated])
    assert "declared invalid by the Supreme Court in Minerva Mills" in checked[0]["text"]
    assert checked[0]["context_chunk_ids"] == [733, 734]
    assert checked[0]["page"] == 260


def test_fact_verifier_checks_the_status_note_and_records_the_actual_model_context():
    first = {**PASSAGE, "text": "The committee's decisions cannot be reviewed by a court."}
    note = {**PASSAGE, "id": 2, "chunk_index": 1, "text": "This restriction was declared invalid by the court."}
    answer = FactEvidenceAssessment(
        verdict="CONTRADICTED", reasoning="The no-review restriction was invalidated.",
        citations=[FactEvidenceCitation(evidence_id=1, quote=note["text"])],
    )
    client, requests = fake_client(answer)
    result = FactVerifier([first, note], client=client).verify(first["text"], [first])
    assert note["text"] in requests[0]["input"]
    assert "CURRENT LEGAL STATUS" in requests[0]["input"]
    assert requests[0]["text_format"] is FactEvidenceAssessment
    assert result.verdict == "CONTRADICTED"
    assert result.supporting_pages == [12]
    assert result.checked_evidence[0]["context_kind"] == "full_page"
    assert result.checked_evidence[0]["context_chunk_ids"] == [1, 2]


def test_empty_evidence_abstains_without_an_api_call():
    client, requests = fake_client(assessment())
    result = FactVerifier(client=client).verify("The committee has seven members.", [])
    assert result.verdict == "INSUFFICIENT"
    assert requests == []


def test_literal_absence_hit_still_uses_the_full_corpus_without_a_model_call():
    client, requests = fake_client(assessment())
    result = FactVerifier([PASSAGE], client=client).verify(
        "There is no mention of the word 'committee' in the handbook.", []
    )
    assert result.verdict == "CONTRADICTED"
    assert result.verification_method == "lexical_scan"
    assert result.supporting_pages == [12]
    assert requests == []


def sequential_client(results):
    requests = []
    iterator = iter(results)
    def parse(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(output_parsed=next(iterator))
    return SimpleNamespace(responses=SimpleNamespace(parse=parse)), requests


def referenced_assessment(verdict="SUPPORTED", quote_id=1):
    return ReferencedFactEvidenceAssessment(
        verdict=verdict, reasoning="The source excerpt establishes this exact proposition.",
        citations=[QuoteSelection(quote_id=quote_id)],
    )


def test_reference_mode_copies_source_words_and_requires_blind_review():
    first = referenced_assessment()
    first.reasoning = "FIRST_REFERENCE_REASON_SENTINEL"
    client, requests = sequential_client([first, referenced_assessment()])
    result = FactVerifier(client=client, reference_quotes=True).verify(
        "The committee has seven members.", [PASSAGE],
    )
    assert result.verdict == "SUPPORTED"
    assert result.independently_reviewed
    assert result.verification_method == "quoted_reference_llm_review"
    assert result.citations == [FactEvidenceCitation(evidence_id=1, quote=PASSAGE["text"])]
    assert result.supporting_pages == [12]
    assert len(requests) == 2
    assert all(request["text_format"] is ReferencedFactEvidenceAssessment for request in requests)
    assert "FIRST_REFERENCE_REASON_SENTINEL" not in requests[1]["input"]
    assert result.model_assessment.reasoning == first.reasoning
    assert result.review_assessment is not None


@pytest.mark.parametrize("review", [
    referenced_assessment(verdict="CONTRADICTED"),
    referenced_assessment(quote_id=999),
    ReferencedFactEvidenceAssessment(verdict="INSUFFICIENT", reasoning="Scope is unresolved.", citations=[]),
])
def test_reference_review_disagreement_and_unknown_ids_still_block(review):
    client, requests = sequential_client([referenced_assessment(), review])
    result = FactVerifier(client=client, reference_quotes=True).verify(
        "The committee has seven members.", [PASSAGE],
    )
    assert len(requests) == 2
    assert result.verdict == "INSUFFICIENT"
    assert not result.independently_reviewed
    assert result.validation_issues


def test_unknown_initial_reference_abstains_without_an_additional_model_call():
    client, requests = sequential_client([referenced_assessment(quote_id=999)])
    result = FactVerifier(client=client, reference_quotes=True).verify(
        "The committee has seven members.", [PASSAGE],
    )
    assert result.verdict == "INSUFFICIENT"
    assert result.validation_issues
    assert len(requests) == 1


def test_reference_selection_still_requires_committed_citations_and_provenance():
    empty = ReferencedFactEvidenceAssessment(verdict="SUPPORTED", reasoning="An assertion.", citations=[])
    result = resolve_fact_references(empty, [PASSAGE], build_quote_catalog([PASSAGE]))
    assert result.verdict == "INSUFFICIENT"
    invalid = {**PASSAGE, "page": 0}
    result = resolve_fact_references(referenced_assessment(), [invalid], build_quote_catalog([invalid]))
    assert result.verdict == "INSUFFICIENT"


def test_article_heading_reference_remains_an_exact_span_on_its_own_page():
    first = {**PASSAGE, "text": "15. Prohibition of discrimination.—The stated rule. Another sentence."}
    second = {**PASSAGE, "page": 13, "text": "(2) A continuation with a qualification."}
    catalog = build_quote_catalog([first, second])
    assert catalog[0]["quote"].startswith("15. Prohibition of discrimination")
    assert all(row["quote"] != "15." for row in catalog)
    for row in catalog:
        assert row["quote"] in [first, second][row["evidence_id"] - 1]["text"]
        assert row["page"] == [first, second][row["evidence_id"] - 1]["page"]


def test_committed_fact_requires_blind_review_agreement():
    first = assessment()
    first.reasoning = "FIRST_ASSESSMENT_REASON_SENTINEL"
    client, requests = sequential_client([first, assessment()])
    result = FactVerifier(client=client).verify("The committee has seven members.", [PASSAGE])
    assert result.verdict == "SUPPORTED"
    assert result.independently_reviewed
    assert len(requests) == 2
    assert "FIRST_ASSESSMENT_REASON_SENTINEL" not in requests[1]["input"]
    assert result.model_assessment is first
    assert result.review_assessment is not None


@pytest.mark.parametrize("review", [
    assessment(verdict="CONTRADICTED"),
    FactEvidenceAssessment(verdict="INSUFFICIENT", reasoning="The scope is not established.", citations=[]),
    assessment(quote="A quotation absent from the checked evidence."),
])
def test_disagreement_or_invalid_review_blocks_the_initial_fact(review):
    client, _ = sequential_client([assessment(), review])
    result = FactVerifier(client=client).verify("The committee has seven members.", [PASSAGE])
    assert result.verdict == "INSUFFICIENT"
    assert not result.independently_reviewed
    assert result.validation_issues
    assert result.model_assessment.verdict == "SUPPORTED"
    assert result.review_assessment is review


def test_missing_review_response_fails_instead_of_publishing_the_first_verdict():
    client, _ = sequential_client([assessment(), None])
    with pytest.raises(ValueError, match="Independent fact review"):
        FactVerifier(client=client).verify("The committee has seven members.", [PASSAGE])


def test_uncommitted_first_judgment_does_not_run_an_extra_review():
    initial = FactEvidenceAssessment(verdict="INSUFFICIENT", reasoning="No term of office.", citations=[])
    client, requests = sequential_client([initial])
    result = FactVerifier(client=client).verify("The committee serves for two years.", [PASSAGE])
    assert result.verdict == "INSUFFICIENT"
    assert len(requests) == 1
    assert result.review_assessment is None


def test_invalid_proposed_quotation_is_retained_for_audit_without_claiming_acceptance():
    proposed = assessment(quote="The committee has eleven members.")
    result = resolve_fact_assessment(proposed, [PASSAGE])
    assert result.model_assessment is proposed
    assert result.citations == []
    assert "was not accepted" in result.reasoning


@pytest.mark.parametrize("propositional, introduced", [(False, []), (True, ["invented_predicate"])])
def test_failed_claim_construction_is_not_sent_to_a_model(propositional, introduced):
    claim = SimpleNamespace(claim="Police.", is_propositional=propositional, unsupported_tokens=introduced)
    def unexpected(*args):
        pytest.fail("Failed claim construction must not reach the fact model.")
    result = verify_constructed_claim(claim, [PASSAGE], SimpleNamespace(verify=unexpected))
    assert result.verdict == "INSUFFICIENT"
    assert result.verification_method == "claim_guard"
