"""Direct-question decisions tested with synthetic, cited model responses.

These tests do not establish model accuracy and make no API calls.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.verification.direct_mcq_verifier import (
    DirectAnswerAnalysis,
    DirectMCQVerifier,
    EvidenceCitation,
    OptionAssessment,
    resolve_direct_answer,
    retrieve_direct_evidence,
    verify_direct_question,
)
from src.verification.question_type import (
    BEST_ANSWER_MCQ,
    DIRECT_MCQ,
    STATEMENT_MCQ,
    classify_question,
    has_labelled_statement_pair,
)


QUOTE = "A constitution defines and limits the powers of government."
PASSAGE = QUOTE + " Creating offices and setting policy goals are other functions."
EVIDENCE = [{"id": 1, "source": "test_source", "page": 7, "text": PASSAGE}]
OPTIONS = {
    "A": "It determines objectives for laws.",
    "B": "It creates political offices.",
    "C": "It defines and limits government powers.",
    "D": "It secures social justice.",
}
QUESTION = "Which option best reflects the chief purpose of a Constitution?"


def analysis_for(statuses=("RULED_OUT", "RULED_OUT", "SUPPORTED", "RULED_OUT")):
    return DirectAnswerAnalysis(
        option_assessments=[OptionAssessment(
            option=letter, answer_fit=status,
            reasoning="Synthetic comparison of the requested chief purpose.",
            citations=[] if status == "INSUFFICIENT" else [
                EvidenceCitation(evidence_id=1, quote=QUOTE)
            ],
        ) for letter, status in zip("ABCD", statuses)],
        reasoning="The options are assessed for answer fit, not isolated truth.",
    )


def client_for(analysis, requests):
    def parse(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(output_parsed=analysis)
    return SimpleNamespace(responses=SimpleNamespace(parse=parse))


@pytest.mark.parametrize("text, expected", [
    (QUESTION, BEST_ANSWER_MCQ),
    ("Who appoints the Governor of a State?", DIRECT_MCQ),
    ("Which of the following is NOT correct?", DIRECT_MCQ),
    ("Who appoints the Chief Minister?", DIRECT_MCQ),
    ("Which institution provides primary education?", DIRECT_MCQ),
    ("Which option describes World War I. Choose one answer.", DIRECT_MCQ),
    ("What is the primary function of a Constitution?", BEST_ANSWER_MCQ),
    ("Consider the following: I. One claim. II. Another claim. Which is correct?", STATEMENT_MCQ),
    ("Consider: I. One claim. II. Another claim. Which is most appropriate?", STATEMENT_MCQ),
])
def test_question_routing(text, expected):
    assert classify_question(text) == expected


def test_all_existing_pyqs_keep_the_statement_path():
    dataset = json.loads(Path("data/evaluation/pyq_2025_polity.json").read_text())
    assert len(dataset["questions"]) == 13
    assert all(classify_question(item["question_text"]) == STATEMENT_MCQ
               for item in dataset["questions"])


@pytest.mark.parametrize("question", [
    "Consider:\nStatement-I:\nThe festival occurs annually.\nStatement-II:\nThe committee was founded on that date.",
    "statement i: One complete claim.\nstatement ii: Another complete claim.",
    "Statement–1 — A dated event.\nStatement—2 — A different event.",
    "Assertion (A): The event occurs annually.\nReason (R): The committee was formed then.",
    "Assertion: One complete claim.\nReason: Another complete claim.",
    "Statement-I\nOne complete claim.\nStatement-II\nAnother complete claim.",
    "Consider Statement-I: One claim. Statement-II: Another claim.",
    "Statement-Ⅰ: One complete claim.\nStatement-Ⅱ: Another complete claim.",
])
def test_unhandled_paired_statement_labels_abstain_without_model_calls(question):
    requests = []
    result = DirectMCQVerifier(client_for(analysis_for(), requests)).verify(
        question, OPTIONS, EVIDENCE
    )
    assert has_labelled_statement_pair(question)
    assert result.status == "ABSTAINED"
    assert result.predicted_answer is None
    assert result.verification_method == "unsupported_statement_format"
    assert result.model_analysis is None
    assert not requests


@pytest.mark.parametrize("question", [
    QUESTION,
    "What is a reason for adopting a constitution?",
    "Consider: I. One claim. II. Another claim. Which is correct?",
    "Statement-I is discussed in this sentence. Choose its purpose.",
])
def test_ordinary_questions_are_not_mistaken_for_paired_labels(question):
    assert not has_labelled_statement_pair(question)


def test_paired_statement_guard_skips_retrieval_and_needs_no_services():
    question = {
        "question_text": "Statement-I: One claim.\nStatement-II: Another claim.",
        "options": OPTIONS,
    }
    result = verify_direct_question(question, None, None)
    assert result.status == "ABSTAINED"
    assert not result.evidence


def test_labelled_statements_with_nested_list_keep_the_guarded_route():
    question = "Statement-I:\nI. A nested item.\nII. Another nested item.\nStatement-II:\nAnother claim."
    assert classify_question(question) == DIRECT_MCQ


def test_unique_supported_answer_is_resolved_in_python():
    result = resolve_direct_answer(BEST_ANSWER_MCQ, analysis_for(), EVIDENCE)
    assert result.status == "ANSWERED"
    assert result.predicted_answer == "C"


@pytest.mark.parametrize("statuses", [
    ("SUPPORTED", "RULED_OUT", "SUPPORTED", "RULED_OUT"),
    ("RULED_OUT", "INSUFFICIENT", "SUPPORTED", "RULED_OUT"),
    ("RULED_OUT", "RULED_OUT", "RULED_OUT", "RULED_OUT"),
    ("INSUFFICIENT", "INSUFFICIENT", "INSUFFICIENT", "INSUFFICIENT"),
])
def test_unresolved_or_competing_options_abstain(statuses):
    result = resolve_direct_answer(BEST_ANSWER_MCQ, analysis_for(statuses), EVIDENCE)
    assert result.status == "ABSTAINED"
    assert result.predicted_answer is None
    assert result.abstention_reason


@pytest.mark.parametrize("mode", ["missing", "duplicate"])
def test_missing_or_duplicate_assessments_cannot_select_an_answer(mode):
    analysis = analysis_for()
    if mode == "missing":
        analysis.option_assessments.pop()
    else:
        analysis.option_assessments[-1] = analysis.option_assessments[0]
    result = resolve_direct_answer(BEST_ANSWER_MCQ, analysis, EVIDENCE)
    assert result.predicted_answer is None
    assert "exactly once" in result.abstention_reason


@pytest.mark.parametrize("citations", [
    [],
    [EvidenceCitation(evidence_id=99, quote=QUOTE)],
    [EvidenceCitation(evidence_id=1, quote="A fabricated quotation.")],
    [EvidenceCitation(evidence_id=1, quote="   ")],
    [EvidenceCitation(evidence_id=1, quote=QUOTE),
     EvidenceCitation(evidence_id=1, quote="A fabricated quotation.")],
])
def test_unsupported_citation_downgrades_the_judgment(citations):
    analysis = analysis_for()
    analysis.option_assessments[2].citations = citations
    result = resolve_direct_answer(BEST_ANSWER_MCQ, analysis, EVIDENCE)
    assert result.predicted_answer is None
    assert result.option_assessments[2].answer_fit == "INSUFFICIENT"


def test_citation_matching_allows_pdf_whitespace_changes():
    evidence = [{**EVIDENCE[0], "text": PASSAGE.replace(" ", "\n")}]
    result = resolve_direct_answer(BEST_ANSWER_MCQ, analysis_for(), evidence)
    assert result.predicted_answer == "C"


def test_best_answer_prompt_preserves_qualifiers_and_omits_the_key():
    requests = []
    verifier = DirectMCQVerifier(client_for(analysis_for(), requests))
    question = {
        "question_text": QUESTION, "options": OPTIONS,
        "official_answer": "D", "explanation": "ANSWER_KEY_LEAK_SENTINEL",
    }
    retriever = SimpleNamespace(retrieve=lambda query, top_k: EVIDENCE)
    result = verify_direct_question(question, retriever, verifier)
    assert result.predicted_answer == "C"
    prompt = requests[0]["input"]
    assert "BEST-ANSWER" in prompt
    assert "true secondary function" in prompt
    assert "ANSWER_KEY_LEAK_SENTINEL" not in prompt
    assert "DECLARED ANSWER" not in prompt
    assert requests[0]["temperature"] == 0


def test_negative_direct_question_keeps_its_instruction():
    requests = []
    verifier = DirectMCQVerifier(client_for(analysis_for(), requests))
    verifier.verify("Which option is NOT correct?", OPTIONS, EVIDENCE)
    assert "Which option is NOT correct?" in requests[0]["input"]
    assert "NOT, incorrect, exception" in requests[0]["input"]


def test_no_evidence_abstains_without_a_model_call():
    requests = []
    result = DirectMCQVerifier(client_for(analysis_for(), requests)).verify(QUESTION, OPTIONS, [])
    assert result.predicted_answer is None
    assert result.status == "ABSTAINED"
    assert not requests


def test_numbered_questions_cannot_use_the_comparison_verifier():
    verifier = DirectMCQVerifier()
    with pytest.raises(ValueError, match="statement-verification"):
        verifier.verify("Consider: I. One claim. II. Another claim.", OPTIONS, EVIDENCE)


def test_missing_structured_model_response_fails_clearly():
    verifier = DirectMCQVerifier(client_for(None, []))
    with pytest.raises(ValueError, match="structured option assessment"):
        verifier.verify(QUESTION, OPTIONS, EVIDENCE)


@pytest.mark.parametrize("options", [
    {"A": "one", "B": "two", "C": "three"},
    {**OPTIONS, "D": ""},
    {**OPTIONS, "D": None},
])
def test_invalid_options_fail_before_retrieval(options):
    def unexpected_retrieval(*args, **kwargs):
        pytest.fail("Invalid options must not trigger retrieval.")
    with pytest.raises(ValueError):
        retrieve_direct_evidence(QUESTION, options, SimpleNamespace(retrieve=unexpected_retrieval))


def test_question_and_option_searches_are_combined_without_duplicate_chunks():
    calls = []
    def retrieve(query, top_k):
        calls.append((query, top_k))
        return EVIDENCE
    passages = retrieve_direct_evidence(QUESTION, OPTIONS, SimpleNamespace(retrieve=retrieve))
    assert len(calls) == 5
    assert all(k == 5 for _, k in calls)
    assert calls[0][0] == QUESTION
    assert all(OPTIONS[letter] in calls[index][0] for index, letter in enumerate("ABCD", 1))
    assert len(passages) == 1
    assert passages[0]["retrieved_for"] == ["question", "option_A", "option_B", "option_C", "option_D"]


def test_same_text_from_different_sources_preserves_provenance():
    retriever = SimpleNamespace(retrieve=lambda query, top_k: [
        EVIDENCE[0], {**EVIDENCE[0], "source": "another_source"},
    ])
    passages = retrieve_direct_evidence(QUESTION, OPTIONS, retriever)
    assert len(passages) == 2


def test_sample_files_parse_as_direct_questions():
    from src.verify_cli import parse_plain_text

    for name, expected in [("direct_constitution_2023.txt", BEST_ANSWER_MCQ),
                           ("direct_governor.txt", DIRECT_MCQ)]:
        question = parse_plain_text(Path("data/examples", name).read_text())
        assert set(question["options"]) == set("ABCD")
        assert classify_question(question["question_text"]) == expected


def sequential_client(analyses, requests):
    iterator = iter(analyses)
    def parse(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(output_parsed=next(iterator))
    return SimpleNamespace(responses=SimpleNamespace(parse=parse))


def test_direct_answer_requires_independent_review_and_withholds_first_assessment():
    first = analysis_for()
    first.reasoning = "FIRST_DIRECT_REASON_SENTINEL"
    requests = []
    result = DirectMCQVerifier(sequential_client([first, analysis_for()], requests)).verify(QUESTION, OPTIONS, EVIDENCE)
    assert result.predicted_answer == "C"
    assert result.independently_reviewed
    assert len(requests) == 2
    assert "FIRST_DIRECT_REASON_SENTINEL" not in requests[1]["input"]
    assert result.review_analysis is not None


@pytest.mark.parametrize("review", [
    analysis_for(("SUPPORTED", "RULED_OUT", "RULED_OUT", "RULED_OUT")),
    analysis_for(("RULED_OUT", "INSUFFICIENT", "SUPPORTED", "RULED_OUT")),
])
def test_direct_review_disagreement_or_uncertainty_abstains(review):
    requests = []
    result = DirectMCQVerifier(sequential_client([analysis_for(), review], requests)).verify(QUESTION, OPTIONS, EVIDENCE)
    assert result.status == "ABSTAINED"
    assert result.predicted_answer is None
    assert "Independent evidence review" in result.abstention_reason


def test_same_direct_winner_with_invalid_review_quotes_still_abstains():
    review = analysis_for()
    review.option_assessments[2].citations[0].quote = "A fabricated review quotation."
    result = DirectMCQVerifier(sequential_client([analysis_for(), review], [])).verify(QUESTION, OPTIONS, EVIDENCE)
    assert result.status == "ABSTAINED"
    assert result.predicted_answer is None


def test_direct_review_response_failure_does_not_publish_initial_answer():
    with pytest.raises(ValueError, match="Independent direct review"):
        DirectMCQVerifier(sequential_client([analysis_for(), None], [])).verify(QUESTION, OPTIONS, EVIDENCE)


def test_unresolved_direct_analysis_skips_review():
    requests = []
    unresolved = analysis_for(("RULED_OUT", "INSUFFICIENT", "SUPPORTED", "RULED_OUT"))
    result = DirectMCQVerifier(sequential_client([unresolved], requests)).verify(QUESTION, OPTIONS, EVIDENCE)
    assert result.status == "ABSTAINED"
    assert len(requests) == 1


@pytest.mark.parametrize("reference_quotes", [False, True])
def test_exact_unrelated_quotations_cannot_establish_a_direct_answer(reference_quotes):
    from src.verification.direct_mcq_verifier import (
        ReferencedDirectAnalysis, ReferencedOptionAssessment,
    )
    from src.verification.source_quotes import QuoteSelection, build_quote_catalog

    question = "The Aurora Innovation Mission is established under which institution?"
    options = {
        "A": "Research Office", "B": "Employment Bureau",
        "C": "National Planning Agency", "D": "Skills Ministry",
    }
    quote = "The boundary follows a river near a numbered pillar."
    evidence = [{"source": "synthetic", "page": 4, "text": quote}]
    if reference_quotes:
        citation = QuoteSelection(quote_id=build_quote_catalog(evidence)[0]["quote_id"])
        proposed = ReferencedDirectAnalysis(**{
            f"option_{letter.lower()}": ReferencedOptionAssessment(
                answer_fit="SUPPORTED" if letter == "C" else "RULED_OUT",
                reasoning="A remembered answer with an unrelated but authentic quote.",
                citations=[citation],
            ) for letter in "ABCD"
        }, reasoning="The source does not discuss this mission.")
    else:
        proposed = analysis_for()
        for item in proposed.option_assessments:
            item.citations = [EvidenceCitation(evidence_id=1, quote=quote)]
    requests = []
    result = DirectMCQVerifier(
        client_for(proposed, requests), reference_quotes=reference_quotes,
    ).verify(question, options, evidence)
    assert result.status == "ABSTAINED"
    assert result.predicted_answer is None
    assert all(item.answer_fit == "INSUFFICIENT" for item in result.option_assessments)
    assert "no informative terms" in result.option_assessments[2].reasoning
    assert len(requests) == 1  # No answer candidate remains to send for review.


def test_agreeing_winner_with_unrelated_review_quotations_still_abstains():
    unrelated = "The boundary follows a river near a numbered pillar."
    evidence = [*EVIDENCE, {"source": "synthetic", "page": 8, "text": unrelated}]
    review = analysis_for()
    for item in review.option_assessments:
        item.citations = [EvidenceCitation(evidence_id=2, quote=unrelated)]
    requests = []
    result = DirectMCQVerifier(sequential_client([analysis_for(), review], requests)).verify(
        QUESTION, OPTIONS, evidence,
    )
    assert len(requests) == 2
    assert result.status == "ABSTAINED"
    assert result.predicted_answer is None
    assert not result.independently_reviewed


def test_numeric_option_is_not_discarded_by_the_coarse_relevance_screen():
    quote = "Its establishment was in 2041."
    evidence = [{"source": "synthetic", "page": 2, "text": quote}]
    options = dict(zip("ABCD", ("2031", "2038", "2041", "2044")))
    proposed = analysis_for()
    for item in proposed.option_assessments:
        item.citations = [EvidenceCitation(evidence_id=1, quote=quote)]
    requests = []
    result = DirectMCQVerifier(client_for(proposed, requests)).verify(
        "In which year was the Aurora Mission established?", options, evidence,
    )
    assert result.predicted_answer == "C"
    assert result.independently_reviewed
    assert len(requests) == 2
