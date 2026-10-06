"""Bind model-selected references to exact excerpts in checked source pages."""

import re

from pydantic import BaseModel, Field

from src.verification.evidence_context import normalize_quote


class QuoteSelection(BaseModel):
    quote_id: int = Field(ge=1)


def build_quote_catalog(evidence: list[dict]) -> list[dict]:
    catalog = []
    for evidence_id, passage in enumerate(evidence, 1):
        excerpts = re.split(r"(?<=[.;:])\s+", normalize_quote(passage["text"]))
        bound = []
        for excerpt in excerpts:
            # Keep "15." with the following source proposition, rather than
            # making the Article identifier an isolated, unusable quotation.
            # Joining these adjacent pieces preserves an exact source span.
            if bound and re.fullmatch(r"(?:\d+\[)?\d{1,3}[A-Z]{0,2}\.", bound[-1]):
                bound[-1] += " " + excerpt
            else:
                bound.append(excerpt)
        for excerpt in bound:
            if excerpt.strip():
                catalog.append({
                    "quote_id": len(catalog) + 1, "evidence_id": evidence_id,
                    "source": passage["source"], "page": passage["page"], "quote": excerpt,
                })
    return catalog


def bind_quote_references(references, catalog: list[dict]) -> list[dict]:
    by_id = {item["quote_id"]: item for item in catalog}
    citations = []
    for reference in references:
        if reference.quote_id not in by_id:
            raise ValueError("The selected quotation is outside the checked source catalog.")
        item = by_id[reference.quote_id]
        citations.append({"evidence_id": item["evidence_id"], "quote": item["quote"]})
    return citations
