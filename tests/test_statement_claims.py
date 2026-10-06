"""Complete action-verb statements must reach verification without rebinding."""

import pytest

from src.verification.claim_builder import build_claims_for_question, self_check as builder_self_check
from src.verification.question_parser import is_propositional


@pytest.mark.parametrize("statement", [
    "Article 15 prohibits discrimination on grounds only of religion, race, caste, sex or place of birth.",
    "Article 14 guarantees equality before the law.",
    "Article 21 protects life and personal liberty.",
    "Article 55 ensures uniformity in the scale of representation of different States.",
    "Article 15 allows special provisions for women and children.",
    "Article 324 confers control over elections on the Election Commission.",
    "The Constitution recognises the stated institution.",
    "The amendment introduced a restriction on parliamentary power.",
    "The Act exempts several posts from disqualification.",
    "The Constitution classifies institutions into four categories.",
])
def test_complete_action_statement_is_retained_as_an_independent_claim(statement):
    question = (
        "Consider the following statements regarding the Constitution:\n"
        f"I. {statement}\n"
        "II. Article 15 is the provision under consideration.\n"
        "Which of the statements given above is/are correct?"
    )
    _, claims = build_claims_for_question(question)
    assert len(claims) == 2
    claim = claims[0]
    assert claim.is_propositional
    assert not claim.needed_binding
    assert claim.claim == statement
    assert claim.unsupported == []


@pytest.mark.parametrize("heading", [
    "Constitutional guarantees", "Guarantees of rights", "Fundamental Rights",
    "Protection of life and personal liberty", "The Election Commission",
    "Introduced Articles", "Recently introduced Articles",
])
def test_noun_headings_still_have_no_truth_value(heading):
    assert not is_propositional(heading)


def test_existing_stem_pair_conditional_and_source_word_checks_still_pass():
    builder_self_check()
