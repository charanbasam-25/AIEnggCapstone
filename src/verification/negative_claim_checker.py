"""
Decide explicit lexical absence claims by scanning the whole corpus.

Why not retrieval
-----------------

This checker used to ask a BM25 retriever for the top 5 chunks matching
the target term and then look for the term in those 5. That is the wrong
instrument for the question being asked. An absence claim quantifies over
the entire corpus, so the only sound answer comes from looking at the
entire corpus; a ranked top-k can miss an occurrence that exists, and
then "not in the top 5" gets read as "not in the Constitution".

On the evaluation set this bug was latent rather than realised: for the
one absence claim present -- "there is no mention of the word 'political
party'" -- the term did happen to appear in BM25's top 5, so the verdict
was already correct. It was correct by luck. The term occurs on 5 pages
out of 1149 chunks, and nothing about the ranking guaranteed one of them
would surface.

A full scan is also cheap here: one pass of a substring test over ~1149
chunks, no embeddings, no model call. Dropping the retriever is what
makes this module importable and testable without rank_bm25 installed.

What a null result licenses
---------------------------

Because the scan is exhaustive, finding nothing is now evidence, and the
checker returns SUPPORTED rather than INSUFFICIENT. Two assumptions ride
on that, and both are stated in the reasoning the verifier emits rather
than left implicit:

  1. the corpus is complete for the claim's scope, and
  2. PDF extraction preserved the term.

Neither is checkable from inside this module. What is checkable is that
the search covered every chunk, and `chunks_scanned` reports it so the
conclusion can be audited instead of trusted.

Only explicitly quoted terms are handled. Conceptual absence ("the
Constitution does not provide for judicial review") is out of scope,
because no string search can settle it.

Standard library only.
"""

import re


class NegativeClaimChecker:
    """
    Handles explicit lexical absence claims such as:

    "There is no mention of the word 'political party'..."
    "The Constitution does not mention 'political party'..."

    This component only handles claims where the target term
    is explicitly quoted. It does not attempt to reason about
    conceptual absence.
    """

    ABSENCE_PATTERNS = [
        r"no mention of (?:the word|the term)?\s*[\"'‘“]([^\"'’”]+)[\"'’”]",
        r"does not mention (?:the word|the term)?\s*[\"'‘“]([^\"'’”]+)[\"'’”]",
        r"does not contain (?:the word|the term)?\s*[\"'‘“]([^\"'’”]+)[\"'’”]",
        r"not mentioned (?:as|by)?\s*[\"'‘“]([^\"'’”]+)[\"'’”]",
    ]

    def __init__(
        self,
        chunks: list[dict],
    ):
        self.chunks = chunks

    def extract_target_term(
        self,
        claim: str,
    ) -> str | None:

        for pattern in self.ABSENCE_PATTERNS:

            match = re.search(
                pattern,
                claim,
                flags=re.IGNORECASE,
            )

            if match:
                return match.group(1).strip()

        return None

    @staticmethod
    def normalise(text: str) -> str:
        """
        Collapse whitespace and case before comparing.

        The PDF extractor breaks phrases across lines, so "political
        party" can arrive as "political\\nparty". Comparing raw text
        would miss those occurrences and reintroduce the false absence
        this module exists to prevent.
        """

        return " ".join(text.split()).lower()

    def find_occurrences(
        self,
        target_term: str,
    ) -> list[dict]:
        """Every chunk containing the term, in corpus order."""

        needle = self.normalise(target_term)

        if not needle:
            return []

        return [
            chunk
            for chunk in self.chunks
            if needle in self.normalise(chunk["text"])
        ]

    def check(
        self,
        claim: str,
        max_evidence: int = 5,
    ) -> dict:

        target_term = self.extract_target_term(
            claim
        )

        if not target_term:
            return {
                "is_absence_claim": False,
                "target_term": None,
                "found": False,
                "occurrence_count": 0,
                "pages": [],
                "chunks_scanned": 0,
                "evidence": [],
            }

        matching_evidence = self.find_occurrences(
            target_term
        )

        # The full occurrence count and page list are reported even
        # though only `max_evidence` chunks are handed on, so a reader
        # can tell "found once" from "found on five pages" without
        # inferring it from a truncated list.
        return {
            "is_absence_claim": True,
            "target_term": target_term,
            "found": bool(matching_evidence),
            "occurrence_count": len(matching_evidence),
            "pages": sorted(
                {item["page"] for item in matching_evidence}
            ),
            "chunks_scanned": len(self.chunks),
            "evidence": matching_evidence[:max_evidence],
        }


def self_check() -> None:
    """Assertions covering extraction, the scan, and both verdicts."""

    chunks = [
        {"source": "coi", "page": 1, "text": "The President of India."},
        {
            "source": "coi",
            "page": 377,
            "text": "a member of a political\nparty shall be",
        },
        {"source": "coi", "page": 378, "text": "POLITICAL PARTY means"},
    ]

    checker = NegativeClaimChecker(chunks)

    # A quoted term must be extracted from each supported phrasing.
    for claim in (
        "There is no mention of the word 'political party' anywhere.",
        "The Constitution does not mention the term 'political party'.",
        "The Constitution does not contain 'political party'.",
    ):
        assert (
            checker.extract_target_term(claim) == "political party"
        ), claim

    # An unquoted claim is out of scope and must not be treated as one.
    assert (
        checker.extract_target_term(
            "The Constitution does not mention political parties."
        )
        is None
    )

    result = checker.check(
        "There is no mention of the word 'political party' in the "
        "Constitution of India."
    )

    assert result["is_absence_claim"]
    assert result["found"]

    # The line-broken occurrence on p.377 is the one a raw substring
    # test would miss, and the upper-case one on p.378 is the one a
    # case-sensitive test would miss. Both must be found.
    assert result["occurrence_count"] == 2, result
    assert result["pages"] == [377, 378], result
    assert result["chunks_scanned"] == 3

    # A term genuinely absent from every chunk.
    absent = checker.check(
        "There is no mention of the word 'referendum' in the "
        "Constitution of India."
    )

    assert absent["is_absence_claim"]
    assert not absent["found"]
    assert absent["occurrence_count"] == 0
    assert absent["chunks_scanned"] == 3


if __name__ == "__main__":

    self_check()

    print("negative claim checker self-check passed")

    from src.retrieval.lexical_index import load_chunks

    chunks = load_chunks(
        "data/processed/chunks.jsonl"
    )

    checker = NegativeClaimChecker(
        chunks
    )

    claim = (
        "There is no mention of the word "
        "'political party' in the Constitution of India."
    )

    result = checker.check(
        claim
    )

    print(
        "\n=== NEGATIVE CLAIM CHECK ===\n"
    )

    print(
        f"Absence claim: "
        f"{result['is_absence_claim']}"
    )

    print(
        f"Target term: "
        f"{result['target_term']}"
    )

    print(
        f"Found in corpus: "
        f"{result['found']}"
    )

    print(
        f"Occurrences: "
        f"{result['occurrence_count']} "
        f"across {result['chunks_scanned']} chunks scanned"
    )

    print(
        f"Pages: {result['pages']}"
    )

    for item in result["evidence"]:

        print(
            f"\n--- {item['source']} "
            f"p.{item['page']} ---"
        )

        print(
            item["text"][:500]
        )
