"""Restore page context after retrieval without changing retrieval ranking.

Constitution chunks can separate a provision from the footnote that marks it
omitted or invalidated. Verification therefore checks the complete retrieved
pages. Source, document and page jointly identify a page; unrelated documents
with the same page number cannot be merged.
"""

from collections import defaultdict


EVIDENCE_CONTEXT_VERSION = "retrieved-pages-v1"
FACT_VERIFICATION_POLICY = "quoted-evidence-review-v3"


def normalize_quote(text: str) -> str:
    """Permit extraction whitespace differences, while retaining actual words."""
    return " ".join(text.split())


def _page_key(chunk: dict) -> tuple:
    return (chunk.get("source"), chunk.get("document"), chunk.get("page"))


def _join_chunks(chunks: list[dict]) -> str:
    text = ""
    for chunk in sorted(chunks, key=lambda item: item.get("chunk_index", 0)):
        following = chunk["text"]
        if not text:
            text = following
            continue
        # Remove the exact overlap introduced by chunking. A substantial match
        # avoids treating a coincidental word ending as a chunk boundary.
        overlap = 0
        for size in range(min(len(text), len(following), 1000), 39, -1):
            if text[-size:] == following[:size]:
                overlap = size
                break
        text += following[overlap:] if overlap else "\n\n" + following
    return text


class PageEvidenceContext:
    def __init__(self, chunks: list[dict] | None = None):
        grouped: dict[tuple, list[dict]] = defaultdict(list)
        for chunk in chunks or []:
            grouped[_page_key(chunk)].append(chunk)
        self.pages = {
            key: {
                "text": _join_chunks(children),
                "context_chunk_ids": [item["id"] for item in children if "id" in item],
            }
            for key, children in grouped.items()
        }

    def expand(self, evidence: list[dict]) -> list[dict]:
        """Return checked passages and trace both retrieved and context chunks."""
        checked, positions = [], {}
        for chunk in evidence:
            key = _page_key(chunk)
            parent = self.pages.get(key)
            if parent and normalize_quote(chunk["text"]) in normalize_quote(parent["text"]):
                if key in positions:
                    target = checked[positions[key]]["retrieved_chunk_ids"]
                    if "id" in chunk and chunk["id"] not in target:
                        target.append(chunk["id"])
                    continue
                expanded = {
                    **chunk,
                    **parent,
                    "retrieved_chunk_ids": [chunk["id"]] if "id" in chunk else [],
                    "context_kind": "full_page",
                }
                positions[key] = len(checked)
            else:
                expanded = {**chunk, "context_kind": "retrieved_passage"}
            checked.append(expanded)
        return checked
