"""Supplement ranked evidence with source pages for explicitly named Articles.

An Article number is a source-location hint, never a truth label. Only actual
operative/omitted headings in the approved Constitution are indexed. Complete
pages retain qualifications and footnotes; reviewers still decide entailment.
"""

from collections import defaultdict
import re

from src.verification.evidence_context import PageEvidenceContext


MAX_ARTICLE_PAGES = 12
MAX_ARTICLE_REFERENCES = 8
_NUMBER = r"[1-9]\d{0,2}[A-Z]{0,2}"
_HEADER = re.compile(
    r"^[ \t]*(?:\d+\[)?(?P<number>[1-9]\d{0,4}[A-Z]{0,2})\.\s+"
    r"(?:(?P<title_prefix>\d+\[|\[)?(?P<title>[A-Z](?:[^.?!—–]|\.(?=,)){1,450})"
    r"\.?\]?\.?[—–]"
    r"|(?P<omitted>\[Omitted\.\]))",
    re.MULTILINE,
)
_REFERENCE_NUMBER = r"[1-9]\d{0,2}(?:[-–]?[A-Za-z]{1,2})?"
_REFERENCES = re.compile(
    rf"\barticles?\s+(?P<numbers>{_REFERENCE_NUMBER}"
    rf"(?:\s*(?:,|and|&)\s*{_REFERENCE_NUMBER})*)\b",
    re.IGNORECASE,
)


def article_references(text: str) -> list[str]:
    result = []
    for match in _REFERENCES.finditer(text):
        for number in re.findall(_REFERENCE_NUMBER, match["numbers"]):
            number = re.sub(r"[-–]", "", number).upper()
            if number not in result:
                result.append(number)
    # Avoid turning an unbounded list into a large model context.
    return result if len(result) <= MAX_ARTICLE_REFERENCES else []


def _order(number: str) -> tuple:
    match = re.fullmatch(r"(\d+)([A-Z]*)", number)
    return int(match[1]), match[2]


class ArticleEvidenceContext:
    def __init__(self, chunks: list[dict]):
        self.page_context = PageEvidenceContext(chunks)
        documents = defaultdict(list)
        for key, page in self.page_context.pages.items():
            source, document, number = key
            if source != "constitution" or not isinstance(number, int) or isinstance(number, bool) or number < 1:
                continue
            documents[(source, document)].append((key, page))
        self.articles = defaultdict(list)
        for pages in documents.values():
            pages.sort(key=lambda row: row[0][2])
            headings, previous, ended = [], None, False
            for key, page in pages:
                for match in _HEADER.finditer(page["text"]):
                    number = match["number"]
                    order = _order(number)
                    # Extraction can join a superscript footnote to an omitted
                    # heading: "1" + "32A." becomes "132A.". Resolve that
                    # artifact only for a bracketed heading immediately beside
                    # its preceding Article, never for an arbitrary reference.
                    if previous is not None and match["title_prefix"] and order[0] > previous[0] + 1:
                        candidates = [number[offset:] for offset in (1, 2)
                                      if re.fullmatch(_NUMBER, number[offset:])]
                        adjacent = [candidate for candidate in candidates
                                    if previous < _order(candidate)
                                    and _order(candidate)[0] <= previous[0] + 1]
                        if len(adjacent) == 1:
                            number, order = adjacent[0], _order(adjacent[0])
                    if order[0] > 395 or (previous is not None and order <= previous):
                        continue
                    # TOC entries have no operative heading dash. An omitted
                    # TOC entry cannot start the article-body index on its own.
                    if previous is None and match["omitted"]:
                        continue
                    headings.append((number, key[2]))
                    previous = order
                    if number == "395":
                        ended = True
                        break
                if ended:
                    break
            for index, (number, start) in enumerate(headings):
                end = (headings[index + 1][1] if index + 1 < len(headings) else
                       start if number == "395" else pages[-1][0][2])
                selected = [(key, page) for key, page in pages if start <= key[2] <= end]
                if len(selected) > MAX_ARTICLE_PAGES:
                    # A missing boundary is not permission to include the rest
                    # of a long document or claim it is complete.
                    continue
                self.articles[number].extend(selected)

    def lookup(self, text: str) -> list[dict]:
        selected = {}
        for number in article_references(text):
            for key, page in self.articles.get(number, []):
                if key not in selected:
                    selected[key] = {
                        "source": key[0], "document": key[1], "page": key[2],
                        **page, "context_kind": "full_page", "retrieved_chunk_ids": [],
                        "anchor_articles": [],
                    }
                selected[key]["anchor_articles"].append(number)
        return list(selected.values())

    def supplement(self, text: str, evidence: list[dict]) -> list[dict]:
        anchors = self.lookup(text)
        checked = self.page_context.expand([*evidence, *anchors])
        by_page = {(row["source"], row.get("document"), row["page"]): row for row in anchors}
        for row in checked:
            anchor = by_page.get((row.get("source"), row.get("document"), row.get("page")))
            if anchor:
                row["anchor_articles"] = anchor["anchor_articles"]
        return checked


def anchor_metadata(evidence: list[dict]) -> list[dict]:
    return [
        {"source": row["source"], "document": row.get("document"), "page": row["page"],
         "articles": row["anchor_articles"]}
        for row in evidence if row.get("anchor_articles")
    ]
