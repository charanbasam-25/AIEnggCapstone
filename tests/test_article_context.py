"""Named Articles supplement search with source context, without judging truth."""

import json
from pathlib import Path

from src.practice.catalog import TOPICS
from src.verification.article_context import (
    ArticleEvidenceContext, MAX_ARTICLE_PAGES, anchor_metadata, article_references,
)


def page(number, text, **changes):
    return {"id": number, "source": "constitution", "document": "Test Constitution",
            "page": number, "chunk_index": 0, "text": text, **changes}


def test_references_normalize_suffixes_and_groups_without_matching_other_numbers():
    assert article_references("Articles 54 and 55; Article 21-A; Article 51A; page 15") == [
        "54", "55", "21A", "51A",
    ]
    assert article_references("Articles 1, 2, 3, 4, 5, 6, 7, 8 and 9") == []


def test_source_lookup_restores_continuation_and_footnotes_after_semantic_miss():
    chunks = [
        page(1, "14. Equality before law.—All persons have the stated protection.\n"
                "15. Prohibition of discrimination.—(1) The State shall not discriminate."),
        page(2, "(2) The rule has the following qualification.\n1. A legal-status footnote."),
        page(3, "Explanation.—The qualification applies within this Article.\n"
                "16. Equality of opportunity.—A different provision."),
        page(9, "An unrelated passage ranked first by semantic search."),
    ]
    context = ArticleEvidenceContext(chunks)
    result = context.supplement("Article 15 prohibits discrimination.", [chunks[-1]])
    assert [row["page"] for row in result] == [9, 1, 2, 3]
    assert result[0]["retrieved_chunk_ids"] == [9]
    assert all(not row["retrieved_chunk_ids"] for row in result[1:])
    assert "legal-status footnote" in result[2]["text"]
    assert [row["page"] for row in anchor_metadata(result)] == [1, 2, 3]
    assert all(row["articles"] == ["15"] for row in anchor_metadata(result))
    assert all("verdict" not in row for row in result)


def test_toc_and_schedules_do_not_replace_operative_pages():
    chunks = [
        page(1, "14.\nEquality before law.\n15.\nProhibition of discrimination.\n"
                "[16.\nAn old title.—Omitted.]"),
        page(10, "14. Equality before law.—The operative text.\n"
                 "15. Prohibition of discrimination.—The operative rule."),
        page(11, "16. Equality of opportunity.—Another provision."),
        page(12, "395. Repeals.—The repealing provision."),
        page(13, "SCHEDULE\n15. A schedule item.—An unrelated provision."),
    ]
    result = ArticleEvidenceContext(chunks).lookup("Article 15")
    assert [row["page"] for row in result] == [10, 11]


def test_pdf_footnote_prefix_cannot_skip_the_next_hundred_articles():
    chunks = [
        page(1, "32. Remedies.—The rule.\n"
                "132A. [An omitted provision.].—Omitted.\n"
                "2[33. Power of Parliament, etc.—The following rule."),
        page(2, "34. Restrictions.—Another provision."),
    ]
    context = ArticleEvidenceContext(chunks)
    assert context.lookup("Article 32A")
    assert context.lookup("Article 33")
    assert context.lookup("Article 132A") == []


def test_source_document_identity_and_ranked_chunk_ids_are_preserved():
    original = page(1, "15. Prohibition of discrimination.—The correct document.")
    unrelated = page(1, "15. Another rule.—A different source.", id=99, source="ncert")
    other_document = page(1, "Unrelated text.", id=88, document="Other document")
    result = ArticleEvidenceContext([original, unrelated, other_document]).supplement(
        "Article 15", [original],
    )
    assert len(result) == 1
    assert result[0]["context_chunk_ids"] == [1]
    assert result[0]["retrieved_chunk_ids"] == [1]
    assert result[0]["anchor_articles"] == ["15"]


def test_missing_boundary_does_not_include_an_unbounded_document():
    chunks = [page(1, "15. Prohibition of discrimination.—The rule.")]
    chunks.extend(page(index, "A continuation without the next heading.")
                  for index in range(2, MAX_ARTICLE_PAGES + 2))
    assert ArticleEvidenceContext(chunks).lookup("Article 15") == []


def test_actual_sources_cover_topic_references_and_preserve_status_notes():
    corpus = Path(__file__).resolve().parents[1] / "data/processed/chunks.jsonl"
    chunks = [json.loads(line) for line in corpus.read_text().splitlines() if line.strip()]
    context = ArticleEvidenceContext(chunks)
    for focuses in TOPICS.values():
        for focus in focuses:
            for number in article_references(focus):
                assert context.lookup(f"Article {number}"), focus
    assert [row["page"] for row in context.lookup("Article 15")] == [37, 38, 39]
    assert [row["page"] for row in context.lookup("Articles 54 and 55")] == [57, 58]
    amendment = context.lookup("Article 368")
    assert [row["page"] for row in amendment] == [259, 260, 261]
    assert any("declared invalid by the Supreme Court in Minerva Mills" in row["text"]
               for row in amendment)
