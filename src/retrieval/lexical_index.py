"""
BM25 retrieval over the chunk corpus using only the standard library.

Why this exists
---------------

This was written when no package could be installed: the configured index
resolved no candidates, so `rank_bm25`, `sentence-transformers` and
`openai` were all unavailable. That blocked every existing retriever, and
with it any attempt to re-measure whether a pipeline fix actually changed
a verdict. Without a runnable retriever the fixes could only be argued
for, not demonstrated.

That block is gone. The install failure was a TLS-interception problem
misreported by `huggingface_hub` as a network outage; see
`src/tls_trust.py`. The real packages are installed and every headline
number now comes from the shipped semantic + cross-encoder retriever, so
this module is no longer the fallback it was built to be.

It is kept for three reasons that outlive the outage: it is the BM25 arm
of the retrieval benchmark (75.00% Recall@5, EVALUATION.md section 1), it
is what `load_chunks` is imported from by several evaluation modules, and
it makes the deterministic parts of the pipeline runnable with no
third-party dependency at all. The superseded BM25 ablation it produced
is preserved at `data/evaluation/system_c_results.bm25.json`, because
three conclusions drawn from it did not survive the retriever swap and
the retractions in EVALUATION.md section 4 should be checkable.

It is a faithful
reimplementation of `rank_bm25.BM25Okapi`, not an approximation, because
an approximation would make its numbers incomparable to the retrieval
benchmark that was run earlier with the real package. Specifically it
reproduces:

    idf(q)  = log((N - df + 0.5) / (df + 0.5))
    score   = sum_q idf(q) * f(q,D)(k1+1)
                          / (f(q,D) + k1(1 - b + b|D|/avgdl))

with k1=1.5, b=0.75, and BM25Okapi's IDF floor: any term whose raw idf is
negative (a term appearing in more than about half the corpus) is
replaced by `epsilon * average_idf` rather than being allowed to push
scores down. Omitting that floor is the usual way a hand-rolled BM25
silently disagrees with the library, so it is implemented here and
asserted in `self_check`.

Interface matches `BM25Retriever.retrieve`, so this is a drop-in for
measurement purposes.
"""

import json
import math
import re
from collections import Counter
from pathlib import Path

# Pure constants, no third-party imports, so the zero-dependency property
# this module exists for is preserved.
from src.retrieval.retrieval_config import RETRIEVAL_TOP_K


K1 = 1.5
B = 0.75

# BM25Okapi's floor for non-discriminative terms, as a multiple of the
# mean IDF. Matching the library's default is what keeps scores
# comparable with the earlier benchmark.
EPSILON = 0.25

CHUNKS_PATH = "data/processed/chunks.jsonl"


def tokenize(text: str) -> "list[str]":
    """
    Tokenise exactly as the existing BM25 retriever does.

    Duplicated from bm25_retriever rather than imported, because that
    module imports rank_bm25 at module scope and would fail here. The
    pattern is kept byte-identical so the two tokenisations cannot drift.
    """

    return re.findall(r"\b\w+\b", text.lower())


def load_chunks(file_path: str = CHUNKS_PATH) -> "list[dict]":
    """Load chunk records from JSONL."""

    with Path(file_path).open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


class LexicalIndex:
    """BM25 over the chunk corpus, with an inverted index for scoring."""

    def __init__(self, chunks: "list[dict]") -> None:
        self.chunks = chunks

        term_frequencies = [
            Counter(tokenize(chunk["text"])) for chunk in chunks
        ]

        self.lengths = [
            sum(counts.values()) for counts in term_frequencies
        ]

        self.average_length = (
            sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        )

        # Postings map each term to the documents containing it, so
        # scoring touches only candidate documents instead of all 1,149.
        self.postings: "dict[str, list[tuple[int, int]]]" = {}

        for index, counts in enumerate(term_frequencies):
            for term, frequency in counts.items():
                self.postings.setdefault(term, []).append(
                    (index, frequency)
                )

        self.idf = self._build_idf(len(chunks))

    def _build_idf(self, total_documents: int) -> "dict[str, float]":
        """
        Compute IDF with BM25Okapi's floor for non-discriminative terms.

        The raw formula goes negative once a term appears in more than
        roughly half the corpus. BM25Okapi replaces those with a small
        positive constant derived from the mean IDF, so a very common
        term contributes almost nothing instead of actively penalising a
        document that contains it.
        """

        raw = {
            term: math.log(
                (total_documents - len(documents) + 0.5)
                / (len(documents) + 0.5)
            )
            for term, documents in self.postings.items()
        }

        if not raw:
            return {}

        average = sum(raw.values()) / len(raw)
        floor = EPSILON * average

        return {
            term: (value if value > 0 else floor)
            for term, value in raw.items()
        }

    def scores(self, query: str) -> "dict[int, float]":
        """Accumulate BM25 scores for documents matching the query."""

        totals: "dict[int, float]" = {}

        for term in tokenize(query):

            postings = self.postings.get(term)

            if postings is None:
                continue

            weight = self.idf[term]

            for index, frequency in postings:

                normalisation = K1 * (
                    1
                    - B
                    + B * self.lengths[index] / self.average_length
                )

                totals[index] = totals.get(index, 0.0) + weight * (
                    frequency * (K1 + 1) / (frequency + normalisation)
                )

        return totals

    def retrieve(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> "list[dict]":
        """
        Return the top_k highest scoring chunks, each with its score.

        Ties break on chunk order so a run is reproducible; without that
        an ablation could show a difference that came from dictionary
        ordering rather than from the change under test.
        """

        totals = self.scores(query)

        ranked = sorted(
            totals.items(), key=lambda item: (-item[1], item[0])
        )[:top_k]

        results = []

        for index, score in ranked:
            result = dict(self.chunks[index])
            result["score"] = float(score)
            results.append(result)

        return results

    def count_matches(self, phrase: str) -> int:
        """
        Count chunks containing a phrase, for answerability probing.

        Phrase-level rather than term-level, because a probe for
        "tenth schedule" must not be satisfied by a chunk that happens to
        contain "tenth" and "schedule" in unrelated places.
        """

        needle = " ".join(tokenize(phrase))

        if not needle:
            return 0

        return sum(
            1
            for chunk in self.chunks
            if needle in " ".join(tokenize(chunk["text"]))
        )


def expected_score(
    index: "LexicalIndex",
    document_index: int,
    query: str,
) -> float:
    """
    Recompute one BM25 score straight from the formula.

    Written independently of `scores`, which walks an inverted index, so
    that agreement between the two is real evidence the postings-based
    accumulation is correct rather than a restatement of it.
    """

    counts = Counter(tokenize(index.chunks[document_index]["text"]))
    length = index.lengths[document_index]

    total = 0.0

    for term in tokenize(query):

        frequency = counts.get(term, 0)

        if frequency == 0:
            continue

        denominator = frequency + K1 * (
            1 - B + B * length / index.average_length
        )

        total += index.idf[term] * frequency * (K1 + 1) / denominator

    return total


def self_check(index: "LexicalIndex") -> None:
    """
    Assertions that guard the properties this index must have.

    The IDF floor is checked against an independently recomputed raw IDF,
    because dropping that floor is the standard way a hand-written BM25
    silently diverges from the library it stands in for, and the
    divergence is invisible in ranking order on most queries.
    """

    total_documents = len(index.chunks)

    assert index.average_length > 0

    # Every IDF must be positive once the floor is applied.
    assert all(value > 0 for value in index.idf.values()), (
        "IDF floor not applied"
    )

    raw = {
        term: math.log(
            (total_documents - len(documents) + 0.5)
            / (len(documents) + 0.5)
        )
        for term, documents in index.postings.items()
    }

    floored = [term for term, value in raw.items() if value <= 0]

    # The corpus must actually exercise the floor, otherwise this check
    # would pass on an implementation that omits it entirely.
    assert floored, "no term is common enough to test the floor"

    floor = EPSILON * (sum(raw.values()) / len(raw))

    assert all(
        math.isclose(index.idf[term], floor) for term in floored
    ), "floored terms must share exactly epsilon * mean(raw idf)"

    # A discriminative term must outweigh a floored stopword.
    assert index.idf["ordinance"] > index.idf["the"]

    # Postings-based accumulation must agree with the direct formula.
    query = "ordinance promulgated by the President"
    results = index.retrieve(query, 5)

    assert len(results) == 5

    for document_index, score in sorted(
        index.scores(query).items(), key=lambda item: -item[1]
    )[:5]:
        assert math.isclose(
            score, expected_score(index, document_index, query)
        ), document_index

    # Ranking must be descending and reproducible.
    assert all(
        results[i]["score"] >= results[i + 1]["score"]
        for i in range(len(results) - 1)
    )
    assert [
        result["id"] for result in index.retrieve(query, 5)
    ] == [result["id"] for result in results]

    # An unmatched query must return nothing rather than arbitrary
    # chunks, so a probe cannot mistake noise for evidence.
    assert index.retrieve("zzzqqq nonexistentterm", 5) == []

    # Phrase counting must not be satisfied by scattered terms.
    assert index.count_matches("tenth schedule") > 0
    assert index.count_matches("zzzqqq") == 0


if __name__ == "__main__":

    chunks = load_chunks()
    index = LexicalIndex(chunks)

    self_check(index)

    print(f"indexed {len(chunks)} chunks")
    print(f"vocabulary {len(index.postings)} terms")
    print(f"mean chunk length {index.average_length:.1f} tokens")
    print("self-check passed\n")

    query = "An Ordinance can amend any Central Act"

    for rank, result in enumerate(index.retrieve(query, 5), start=1):
        snippet = " ".join(result["text"].split())[:150]
        print(
            f"{rank}. p.{result['page']} score={result['score']:.3f}  "
            f"{snippet.encode('ascii', 'replace').decode('ascii')}"
        )
