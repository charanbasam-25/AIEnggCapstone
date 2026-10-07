"""Export the documented implementation as PNG, SVG, PDF and Mermaid flows.

Run: .venv/bin/python docs/diagrams/render_diagrams.py
Uses the existing PyMuPDF dependency. No network or model calls.
Definitions below supply both the vector layouts and the editable flows.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from html import escape
from math import hypot
from pathlib import Path
import shutil

import pymupdf


HERE = Path(__file__).resolve().parent
INK, MUTED, LINE, BACKGROUND = "#17243B", "#50627B", "#DAE2EE", "#F5F7FB"
STYLES = {
    "source": ("#0B806A", "#EDF8F4"),
    "control": ("#2457D6", "#EFF4FE"),
    "model": ("#7650B6", "#F4F0FB"),
    "decision": ("#A56B13", "#FFF7E8"),
    "store": ("#52647B", "#FFFFFF"),
}


@dataclass
class Box:
    id: str
    x: float
    y: float
    w: float
    h: float
    title: str
    body: tuple[str, ...] = ()
    style: str = "control"
    size: int = 25
    title_size: int = 29


@dataclass
class Link:
    source: str
    target: str
    points: tuple[tuple[float, float], ...]
    label: str = ""
    label_at: tuple[float, float] | None = None
    style: str = "control"
    dashed: bool = False
    both: bool = False


@dataclass
class Diagram:
    name: str
    title: str
    subtitle: str
    width: int
    height: int
    notes: tuple[str, ...]
    direction: str = "TB"
    boxes: list[Box] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)
    captions: list[tuple[float, float, str]] = field(default_factory=list)

    def box(self, id, x, y, w, h, title, *body, style="control", size=25, title_size=29):
        self.boxes.append(Box(id, x, y, w, h, title, body, style, size, title_size))

    def link(self, source, target, points, label="", label_at=None, **kwargs):
        self.links.append(Link(source, target, tuple(points), label, label_at, **kwargs))


def wrap(value, width, size, bold=False):
    result, current = [], ""
    for word in value.split():
        candidate = f"{current} {word}".strip()
        if pymupdf.get_text_length(candidate, fontname="hebo" if bold else "helv", fontsize=size) > width:
            if not current:
                raise ValueError(f"Unbreakable text is too wide: {word}")
            result.append(current)
            current = word
        else:
            current = candidate
    if current:
        result.append(current)
    return result


def vector(diagram):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{diagram.width}" height="{diagram.height}" '
        f'viewBox="0 0 {diagram.width} {diagram.height}" role="img">',
        f"<title>{escape(diagram.title)}</title>",
        f"<desc>{escape(diagram.subtitle)} {escape(' '.join(diagram.notes))}</desc>",
        f'<rect width="{diagram.width}" height="{diagram.height}" fill="{BACKGROUND}"/>',
    ]

    def text(x, y, value, size=25, color=INK, bold=False):
        parts.append(
            f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
            f'font-weight="{700 if bold else 400}" fill="{color}">{escape(value)}</text>'
        )

    def rect(x, y, w, h, fill, stroke=LINE, radius=16):
        parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.7"/>'
        )

    def head(start, end, color):
        x1, y1 = start
        x2, y2 = end
        length = hypot(x2 - x1, y2 - y1)
        if not length:
            raise ValueError("Zero-length arrow segment.")
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        bx, by = x2 - 13 * ux, y2 - 13 * uy
        points = f"{x2},{y2} {bx - 6 * uy},{by + 6 * ux} {bx + 6 * uy},{by - 6 * ux}"
        parts.append(f'<polygon points="{points}" fill="{color}"/>')

    text(60, 43, "UPSC PRACTICE / SYSTEM DESIGN", 19, STYLES["control"][0], True)
    text(diagram.width - 205, 43, "07 OCT 2026", 19, MUTED)
    text(60, 104, diagram.title, 43, bold=True)
    subtitle = wrap(diagram.subtitle, diagram.width - 120, 25)
    if len(subtitle) > 2:
        raise ValueError(f"Long subtitle: {diagram.name}")
    for i, line in enumerate(subtitle):
        text(60, 142 + i * 30, line, 25, MUTED)
    ids = {box.id for box in diagram.boxes}
    if len(ids) != len(diagram.boxes):
        raise ValueError(f"Duplicate box ID: {diagram.name}")
    for link in diagram.links:
        if link.source not in ids or link.target not in ids:
            raise ValueError(f"Unknown endpoint: {link}")
        color = STYLES[link.style][0]
        path = "M " + " L ".join(f"{x},{y}" for x, y in link.points)
        dash = ' stroke-dasharray="9 7"' if link.dashed else ""
        parts.append(
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.8" '
            f'stroke-linecap="round" stroke-linejoin="round"{dash}/>'
        )
        head(link.points[-2], link.points[-1], color)
        if link.both:
            head(link.points[1], link.points[0], color)
    for box in diagram.boxes:
        if min(box.x, box.y) < 0 or box.x + box.w > diagram.width or box.y + box.h > diagram.height - 95:
            raise ValueError(f"Box outside drawing area: {box.title}")
        color, fill = STYLES[box.style]
        rect(box.x, box.y, box.w, box.h, fill)
        rect(box.x, box.y, 5, box.h, color, color, 2)
        cursor = box.y + 38
        for line in wrap(box.title, box.w - 48, box.title_size, True):
            text(box.x + 24, cursor, line, box.title_size, bold=True)
            cursor += box.title_size + 5
        cursor += 7
        for paragraph in box.body:
            for line in wrap(paragraph, box.w - 48, box.size):
                text(box.x + 24, cursor, line, box.size, MUTED)
                cursor += box.size + 6
        if cursor - (box.size + 6) > box.y + box.h - 13:
            raise ValueError(f"Card text overflows: {box.title}")
    for link in diagram.links:
        if link.label and link.label_at:
            x, y = link.label_at
            width = pymupdf.get_text_length(link.label, fontname="helv", fontsize=21)
            rect(x - 7, y - 23, width + 14, 31, BACKGROUND, BACKGROUND, 4)
            text(x, y, link.label, 21, STYLES[link.style][0])
    for x, y, value in diagram.captions:
        text(x, y, value, 20, MUTED, True)
    cursor = diagram.height - 76
    for note in diagram.notes:
        for line in wrap(note, diagram.width - 120, 22):
            text(60, cursor, line, 22, MUTED)
            cursor += 28
    if cursor > diagram.height + 3:
        raise ValueError(f"Footnote overflows: {diagram.name}")
    parts.append("</svg>")
    return "".join(parts)


def mermaid(diagram):
    lines = [
        f"%% {diagram.title}",
        "%% Current implementation reviewed 7 October 2026.",
        "%% Generated by render_diagrams.py; edit definitions there for matching image exports.",
        f"flowchart {diagram.direction}",
    ]
    for box in diagram.boxes:
        label = "<br/>".join((box.title, *box.body)).replace('"', "&quot;")
        lines.append(f'    {box.id}["{label}"]')
    lines.append("")
    for link in diagram.links:
        connector = "<-->" if link.both else "-.->" if link.dashed else "-->"
        label = f'|"{link.label.replace(chr(34), "&quot;")}"|' if link.label else ""
        lines.append(f"    {link.source} {connector}{label} {link.target}")
    lines.append("")
    for style, (stroke, fill) in STYLES.items():
        lines.append(f"    classDef {style} fill:{fill},stroke:{stroke},color:{INK};")
        members = ",".join(box.id for box in diagram.boxes if box.style == style)
        if members:
            lines.append(f"    class {members} {style};")
    lines.extend(f"%% {note}" for note in diagram.notes)
    return "\n".join(lines) + "\n"


def architecture():
    d = Diagram("practice_architecture", "How the practice app fits together",
                "A local Streamlit app prepares questions from approved sources and records the work behind each set.",
                1720, 1300, (
                    "Green: source data. Blue: Python and UI. Purple: model tasks. Grey: local storage.",
                    "Automated reviews reduce risk. Accuracy of generated questions still needs independent expert evaluation.",
                ))
    d.box("pdf", 60, 215, 480, 170, "Public source PDFs",
          "Constitution of India", "Selected NCERT Grade 7 chapter", "Extract pages and retain provenance", style="source")
    d.box("corpus", 620, 215, 480, 170, "Page-aware corpus",
          "1,149 chunks / chunks.jsonl", "Source, document, page and chunk ID", "SHA-256 identifies the snapshot", style="source", size=24)
    d.box("index", 1180, 215, 480, 170, "Local retrieval",
          "BGE embeddings / NumPy index", "Semantic 20 -> reranked 5 per query", "Full pages and named Article context", style="source", size=24)
    d.box("ui", 60, 495, 480, 175, "Streamlit learner UI",
          "Practice / My Practice / How it works", "Topic, style, level and set size", "Keys and notes appear on Submit")
    d.box("service", 620, 495, 480, 175, "Practice service",
          "Validate the bounded request", "Reuse eligible records or fill slots", "Return accepted items or a partial set")
    d.box("graph", 1180, 495, 480, 175, "Bounded LangGraph workflow",
          "Draft -> verify -> audit -> explain", "Five required publication gates", "At most two MCQ revisions", size=24)
    d.box("session", 60, 785, 480, 190, "Session history",
          "Answers and scores in server memory", "Latest 20 completed sets", "Python scoring; no model call",
          "No persistent learner accounts", style="store", size=24)
    d.box("sqlite", 620, 785, 480, 190, "Local SQLite store",
          "Accepted MCQs and source-bound notes", "Runs, retrieval traces and usage", "Source + policy must match for reuse",
          "Shared store; no account isolation", style="store", size=22)
    d.box("api", 1180, 785, 480, 190, "OpenAI model tasks",
          "gpt-4o-mini / structured responses", "Generator and separate reviewers", "45-second timeout / one SDK retry",
          "Reviews can share model errors", style="model", size=24)
    d.box("diagnostics", 60, 1090, 1600, 110, "How it works: saved diagnostics",
          "Evaluation JSON + practice run reports -> Evals, RAG, guardrails, cost and latency. Viewing makes no model calls.",
          style="store", size=24)
    d.link("pdf", "corpus", [(540, 300), (620, 300)], style="source")
    d.link("corpus", "index", [(1100, 300), (1180, 300)], style="source")
    d.link("index", "graph", [(1420, 385), (1420, 495)], "Evidence", (1434, 450), style="source")
    d.link("ui", "service", [(540, 582), (620, 582)], both=True)
    d.link("service", "graph", [(1100, 582), (1180, 582)], both=True)
    d.link("ui", "session", [(300, 670), (300, 785)])
    d.link("service", "sqlite", [(860, 670), (860, 785)], "Reuse / save", (879, 738), style="store", both=True)
    d.link("graph", "api", [(1420, 670), (1420, 785)], "Structured calls", (1435, 738), style="model", dashed=True, both=True)
    d.link("sqlite", "diagnostics", [(860, 975), (860, 1090)], "Saved reports", (879, 1040), style="store")
    return d


def learner_flow():
    d = Diagram("practice_flow", "The learner's practice flow",
                "Preparation checks happen before the quiz. Study results and project diagnostics have separate views.",
                1500, 1300, (
                    "Changing sources retires the active set. Sidebar navigation preserves an unfinished quiz.",
                    "Difficulty is a requested prompt setting. Learner history lasts for the current server session.",
                ))
    d.captions = [(60, 190, "LEARNER"), (860, 190, "PREPARATION SERVICE")]
    d.box("choose", 60, 220, 600, 150, "Choose a practice set",
          "Polity topic / direct, statements or mixed", "Foundation, Standard or Challenging / 1, 5 or 10")
    d.box("bank", 860, 220, 580, 150, "Look for eligible checked items",
          "Match topic, style, level, source hash and policy", "Validate the saved payload before reuse", style="store", size=24)
    d.box("quiz", 60, 515, 600, 160, "Attempt the accepted questions",
          "Choose A-D or leave an item unanswered", "No answer key or option notes before submission")
    d.box("prepare", 860, 515, 580, 160, "Prepare the remaining slots",
          "Retrieve sources, draft and run every gate", "Revise within the limit; withhold failed drafts", size=24)
    d.box("submit", 60, 790, 600, 140, "Submit the quiz",
          "Python compares choices with the checked key", "Correct / incorrect / skipped")
    d.box("partial", 860, 790, 580, 150, "Show the preparation outcome",
          "Partial set: use only accepted questions", "Empty set: explain why no quiz is available", style="decision", size=24)
    d.box("review", 60, 1050, 600, 150, "Review answers and sources",
          "Summary and explanations for every option", "Exact quotations with source PDF pages", style="source")
    d.box("history", 860, 1050, 580, 150, "My Practice",
          "Recent completed sets and a mistake notebook", "Revisit incorrect or skipped items", style="store")
    d.link("choose", "bank", [(660, 295), (860, 295)])
    d.link("bank", "prepare", [(1150, 370), (1150, 515)], "Missing slots", (1168, 450))
    d.link("bank", "quiz", [(860, 330), (760, 330), (760, 440), (360, 440), (360, 515)],
           "Eligible reuse", (398, 432), style="store")
    d.link("prepare", "quiz", [(860, 595), (660, 595)], "Accepted items", (682, 578), style="source")
    d.link("prepare", "partial", [(1150, 675), (1150, 790)], style="decision")
    d.link("quiz", "submit", [(360, 675), (360, 790)])
    d.link("submit", "review", [(360, 930), (360, 1050)], style="source")
    d.link("review", "history", [(660, 1125), (860, 1125)], style="store")
    return d


def retrieval_flow():
    d = Diagram("retrieval_flow", "From public PDFs to checked evidence",
                "Retrieval locates source text. Full-page context and exact quotations make that text inspectable.",
                1500, 1240, (
                    "Article context adds source pages beside the ranking; it is used for generation and statement checks.",
                    "Retrieval scores are ranking signals. A passage hit or an exact quote does not establish an answer.",
                ))
    d.captions = [(60, 190, "OFFLINE PREPARATION"), (60, 485, "PER QUERY")]
    d.box("pdf", 60, 220, 420, 180, "Extract source pages",
          "PyMuPDF / public PDFs", "Clean text; retain source and page", "PDF page numbers are one-based", style="source", size=24)
    d.box("chunks", 540, 220, 420, 180, "Make overlapping chunks",
          "1,000 characters / 150 overlap", "1,107 Constitution + 42 NCERT", "Persist as JSONL with provenance", style="source", size=24)
    d.box("index", 1020, 220, 420, 180, "Build a local index",
          "BAAI/bge-small-en-v1.5", "Normalized chunk embeddings", "NumPy similarity search", style="source", size=24)
    d.box("query", 60, 520, 420, 185, "Build the query",
          "Generation: topic + curated focus", "Statements: each complete claim", "Direct: stem and all four options", size=24)
    d.box("semantic", 540, 520, 420, 185, "Semantic search",
          "Embed the query", "Search the complete loaded index", "Select the top 20 candidates", style="source", size=24)
    d.box("rerank", 1020, 520, 420, 185, "Cross-encoder reranking",
          "Score each query-passage pair", "MS MARCO MiniLM-L-6-v2", "Keep the top 5 per query", style="source", size=24)
    d.box("article", 60, 900, 420, 180, "Named Article lookup",
          "Find the actual provision heading", "Include continuation and notes", "Approved Constitution source only", style="source", size=24)
    d.box("context", 540, 900, 420, 180, "Restore source context",
          "Reassemble retrieved full pages", "Retain qualifications and footnotes", "Record added Article pages separately", style="source", size=23)
    d.box("quotes", 1020, 900, 420, 180, "Quoted evidence packet",
          "Numbered source excerpts", "Models select quotation references", "Python binds text, source and page", size=21)
    d.link("pdf", "chunks", [(480, 310), (540, 310)], style="source")
    d.link("chunks", "index", [(960, 310), (1020, 310)], style="source")
    d.link("index", "semantic", [(1230, 400), (1230, 455), (750, 455), (750, 520)], style="source")
    d.link("query", "semantic", [(480, 612), (540, 612)])
    d.link("semantic", "rerank", [(960, 612), (1020, 612)], style="source")
    d.link("query", "article", [(270, 705), (270, 900)], "Named references", (287, 820), style="source")
    d.link("rerank", "context", [(1230, 705), (1230, 805), (750, 805), (750, 900)],
           "Ranked selections", (838, 797), style="source")
    d.link("article", "context", [(480, 990), (540, 990)], style="source")
    d.link("context", "quotes", [(960, 990), (1020, 990)], style="source")
    return d


def generation_flow():
    d = Diagram("generation_flow", "A bounded draft, review and publication loop",
                "Each changed candidate starts with cleared gate results. Only an accepted record reaches the learner.",
                1680, 1410, (
                    "Failed format checks skip answer work. Invalid answers skip quality calls; failed quality skips notes.",
                    "One initial draft + two MCQ revisions. Explanation writing has one local repair; SDK retries are separate.",
                ))
    d.box("sources", 70, 220, 650, 150, "1  Retrieve generation sources",
          "Topic and curated focus -> top 20 -> reranked 5", "Full pages, named Article context and source provenance", style="source", size=22, title_size=26)
    d.box("draft", 70, 465, 650, 150, "2  Generate a structured candidate",
          "Question, four options and a proposed key", "Use source packet, requested style and revision feedback", style="model", size=22, title_size=26)
    d.box("format", 70, 710, 650, 150, "3  Check format and construct claims",
          "Nonempty, distinct options; supported question structure", "Exact-stem duplicate check; safe deterministic binding", size=22, title_size=26)
    d.box("answer", 70, 955, 650, 175, "4  Verify claims and the answer",
          "Fresh claim retrieval OR direct stem-plus-options retrieval", "Quote validation and agreeing blind review",
          "Python resolves a unique key and compares the proposed key", size=22, title_size=26)
    d.box("quality", 960, 220, 650, 150, "5  Audit question quality",
          "Clarity, single best answer, distractors, wording, topic", "Every flag true, PASS and no issues", style="model")
    d.box("notes", 960, 465, 650, 175, "6  Write and review learning notes",
          "Replace generator prose with fresh summary and A-D notes",
          "Bind source excerpts; review every item against its citations",
          "All item and global explanation checks must pass", style="model", size=20, title_size=25)
    d.box("gate", 960, 735, 650, 175, "7  Python publication decision",
          "Sources + format + answer + quality + explanations", "All current-attempt gates must PASS",
          "ACCEPT / REVISE / REJECT", style="decision")
    d.box("accept", 960, 1050, 300, 155, "ACCEPT",
          "Recheck at service boundary", "Validate and store the item", style="source", size=20, title_size=24)
    d.box("reject", 1310, 1050, 300, 155, "REJECT",
          "No failed draft in the quiz", "Partial set or empty result", style="decision", size=20, title_size=24)
    d.link("sources", "draft", [(395, 370), (395, 465)], style="source")
    d.link("draft", "format", [(395, 615), (395, 710)], style="model")
    d.link("format", "answer", [(395, 860), (395, 955)])
    d.link("answer", "quality", [(720, 1042), (840, 1042), (840, 295), (960, 295)])
    d.link("quality", "notes", [(1285, 370), (1285, 465)], style="model")
    d.link("notes", "gate", [(1285, 640), (1285, 735)], style="model")
    d.link("gate", "accept", [(1110, 910), (1110, 1050)], "All pass", (1128, 998), style="source")
    d.link("gate", "reject", [(1460, 910), (1460, 1050)], "No budget", (1476, 998), style="decision")
    d.link("gate", "draft", [(1610, 822), (1640, 822), (1640, 1270), (28, 1270), (28, 540), (70, 540)],
           "REVISE: at most twice; keep generation evidence, retrieve verification evidence afresh",
           (385, 1262), style="decision", dashed=True)
    return d


def verification_flow():
    d = Diagram("verification_flow", "How an answer is established or withheld",
                "Models assess evidence. Python validates citations and applies the final answer rule.",
                1690, 1540, (
                    "INSUFFICIENT is unresolved evidence; it is never converted to a false statement in practice.",
                    "Blind review hides the earlier verdict, but uses the same model family. Agreement can still be wrong.",
                ))
    d.box("input", 505, 215, 665, 175, "Question text and A-D options",
          "Python routes direct or numbered questions", "Gold keys and earlier verdicts are withheld from reviewers")
    d.box("paired", 1260, 215, 370, 145, "Unsupported paired format",
          "Statement-I/II or assertion/reason", "Abstain before model retrieval",
          style="decision", size=19, title_size=23)
    d.box("parse", 60, 490, 700, 160, "Numbered statements: build complete claims",
          "Parse stem, items, closing question and coded options",
          "Bind the stem predicate; reject failed or introduced wording")
    d.box("prepare", 930, 490, 700, 160, "Direct questions: prepare comparison evidence",
          "Search the stem and each alternative",
          "Deduplicate up to five selections per query; restore full pages", size=22, title_size=26)
    d.box("facts", 60, 780, 700, 190, "Assess each complete claim",
          "Fresh retrieval plus full pages and named Article context",
          "SUPPORTED / CONTRADICTED / INSUFFICIENT",
          "Validate quote references; blind review must agree", style="model")
    d.box("options", 930, 780, 700, 190, "Assess the exact answer fit of A-D",
          "SUPPORTED / RULED_OUT / INSUFFICIENT", "Validate quotes and screen wholly unrelated citations",
          "Blind option review must resolve the same unique answer", style="model")
    d.box("map", 60, 1100, 700, 160, "Python strict statement mapping",
          "Resolved true and false statements form a truth pattern",
          "Map the pattern to one coded option; unresolved -> abstain")
    d.box("resolve", 930, 1100, 700, 160, "Python strict direct decision",
          "Exactly one SUPPORTED and three RULED_OUT", "Missing evidence cannot rule out an alternative")
    d.box("result", 505, 1360, 665, 85, "A-D with checked evidence, or ABSTAIN",
          style="decision", title_size=27)
    d.link("input", "paired", [(1170, 287), (1260, 287)], style="decision")
    d.link("input", "parse", [(720, 360), (720, 415), (410, 415), (410, 490)])
    d.link("input", "prepare", [(990, 360), (990, 415), (1280, 415), (1280, 490)])
    d.link("parse", "facts", [(410, 650), (410, 780)], "For each claim", (426, 724), style="model")
    d.link("prepare", "options", [(1280, 650), (1280, 780)], "Compare all options", (1296, 724), style="model")
    d.link("facts", "map", [(410, 970), (410, 1100)])
    d.link("options", "resolve", [(1280, 970), (1280, 1100)])
    d.link("map", "result", [(410, 1260), (410, 1308), (710, 1308), (710, 1360)])
    d.link("resolve", "result", [(1280, 1260), (1280, 1308), (1000, 1308), (1000, 1360)])
    d.link("paired", "result", [(1630, 287), (1660, 287), (1660, 1402), (1170, 1402)],
           style="decision", dashed=True)
    return d


def absence_flow():
    d = Diagram("absence_flow", "The special case: literal absence of a term",
                "A limited Python scan handles recognised quoted-term patterns. Other propositions use semantic verification.",
                1500, 1260, (
                    "The scan covers the combined loaded corpus, uses substrings and normalizes case/whitespace. Scope is a limitation.",
                    "Practice also requires cited blind review. A lexical-scan result lacks that marker and is currently withheld.",
                ))
    d.box("claim", 60, 220, 620, 160, "A complete constructed claim",
          "Example: The Constitution does not mention", 'the term "political party".')
    d.box("pattern", 860, 220, 580, 160, "Recognised quoted absence pattern?",
          "Only explicit forms implemented in Python",
          "A concept or ordinary statement is not a lexical absence test", size=24)
    d.box("semantic", 60, 600, 620, 185, "Normal evidence-based verification",
          "Retrieve and interpret the complete proposition", "Validate source quotes and require blind review",
          "A word match does not establish the claim", style="model")
    d.box("scan", 860, 600, 580, 185, "Scan every loaded corpus chunk",
          "No model call inside the scan branch", "Record matching chunks, pages and chunks scanned",
          "Counts can include overlapping chunk matches", style="source", size=24)
    d.box("found", 780, 1000, 305, 155, "Term found",
          "Literal absence contradicted", "Return matching source chunks", style="decision", size=19, title_size=23)
    d.box("missing", 1135, 1000, 305, 155, "No match",
          "Conditional corpus-only support", "Assumes complete, faithful text", style="decision", size=16, title_size=21)
    d.link("claim", "pattern", [(680, 300), (860, 300)])
    d.link("pattern", "semantic", [(860, 340), (765, 340), (765, 475), (370, 475), (370, 600)],
           "No: semantic claim", (392, 467), style="model")
    d.link("pattern", "scan", [(1150, 380), (1150, 600)],
           "Yes: literal quoted term", (1166, 496), style="source")
    d.link("scan", "found", [(1030, 785), (1030, 893), (932, 893), (932, 1000)], style="decision")
    d.link("scan", "missing", [(1270, 785), (1270, 893), (1287, 893), (1287, 1000)], style="decision")
    return d


def storage_flow():
    d = Diagram("data_model", "What is stored, and where",
                "SQLite stores reusable questions and operational reports. Learner answers stay in session memory.",
                1650, 1230, (
                    "No learner table or SQL foreign key between questions and runs. Question references are inside JSON payloads.",
                    "Source and policy govern reuse. Content hashes detect accidental edits, not malicious tampering.",
                ))
    d.captions = [(70, 198, "SQLITE: data/practice/practice.sqlite3"), (70, 825, "OUTSIDE THE PRACTICE DATABASE")]
    d.box("questions", 70, 235, 700, 370, "questions",
          "id: TEXT PRIMARY KEY", "fingerprint / topic / difficulty / format",
          "corpus_hash / policy / created_at", "payload: complete PracticeQuestion JSON",
          "UNIQUE(fingerprint, corpus_hash, policy)", "Accepted MCQ, summary, A-D notes, quotes and evidence",
          style="store", size=25)
    d.box("runs", 910, 235, 670, 370, "runs",
          "id: TEXT PRIMARY KEY", "created_at: TEXT / payload: JSON text",
          "Request, accepted/reused counts and error class", "Stages, calls, tokens and elapsed seconds",
          "Retrieval rankings and checked evidence packets", "question_records: slot, origin and question_id",
          style="store", size=24)
    d.box("session", 70, 865, 700, 225, "Streamlit server session",
          "Active set and learner choices", "Python score and latest 20 completed sets",
          "No learner choices sent to the model", "Browser refresh/restart can lose history", style="control")
    d.box("reports", 910, 865, 670, 225, "Frozen datasets and evaluation JSON",
          "data/benchmarks: questions, keys and source metadata",
          "data/evaluation: predictions, errors and metrics",
          "Gold keys are grading inputs only", "Separate task reports read by Evaluation Studio",
          style="store", size=24)
    d.link("runs", "questions", [(1245, 605), (1245, 718), (550, 718), (550, 605)],
           "Application reference: question_records[].question_id", (635, 710), style="store", dashed=True)
    d.link("questions", "session", [(70, 430), (28, 430), (28, 977), (70, 977)])
    return d


def evaluation_flow():
    d = Diagram("evaluation_flow", "How evaluation results reach the dashboard",
                "Answering and grading are separate. Every saved result belongs to a specific task and protocol.",
                1640, 1320, (
                    "75 mixed regression questions, 13 legacy cases and 16 retrieval queries have different units and references.",
                    "Generated-MCQ gate yield is measurable; expert-reviewed key and explanation accuracy is still unmeasured.",
                ))
    d.box("dataset", 60, 225, 650, 155, "Frozen question dataset",
          "Question text, options and source provenance", "Official keys retained for Python grading only", style="store")
    d.box("protocol", 950, 225, 630, 155, "Recorded run configuration",
          "Dataset, corpus and code hashes", "Model, temperature, split, systems and retrieval depths",
          style="store", size=24)
    d.box("input", 60, 525, 650, 145, "Answering input",
          "Stem and options only", "No official key, label audit or diagnostic worksheet")
    d.box("systems", 950, 525, 630, 145, "Run the selected answering systems",
          "Strict verifier and/or vanilla RAG", "Save evidence, predicted letter or abstention", size=24)
    d.box("checkpoint", 60, 820, 650, 155, "Write a resumable JSON checkpoint",
          "Save after each system/question result", "Resume only when configuration matches", style="store")
    d.box("grade", 950, 820, 630, 155, "Grade predictions in Python",
          "Correct / wrong / abstained / service error", "Precision, coverage, accuracy and breakdowns", size=24)
    d.box("ui", 60, 1110, 650, 115, "How it works -> Evals",
          "Validate and read saved reports; inspect or export", size=24)
    d.box("other", 950, 1110, 630, 115, "Other task-specific reports",
          "Retrieval, legacy judge/stability and generation yield", style="store", size=24)
    d.link("dataset", "input", [(385, 380), (385, 525)], "Questions only", (404, 464))
    d.link("input", "systems", [(710, 597), (950, 597)])
    d.link("protocol", "systems", [(1265, 380), (1265, 525)], style="store")
    d.link("systems", "grade", [(1265, 670), (1265, 820)])
    d.link("dataset", "grade", [(710, 287), (828, 287), (828, 858), (950, 858)],
           "Keys: grading only", (847, 758), style="decision", dashed=True)
    d.link("grade", "checkpoint", [(950, 925), (710, 925)], style="store")
    d.link("checkpoint", "ui", [(385, 975), (385, 1110)], "Saved data", (405, 1057), style="store")
    d.link("other", "ui", [(950, 1167), (710, 1167)], style="store")
    return d


def baseline_flow():
    d = Diagram("baseline_flow", "The vanilla RAG comparison",
                "One retrieval and one answer call provide a reference for the cost and behaviour of verification.",
                1500, 860, (
                    "The historical baseline uses retrieved chunks and asks for a letter; it has no blind publication review.",
                    "Saved keys grade correctness afterwards. Citing a page does not establish faithful reasoning.",
                ), direction="LR")
    d.box("question", 60, 230, 420, 190, "Question and A-D options",
          "Combine the stem and alternatives", "Keep the official answer out of the prompt")
    d.box("search", 540, 230, 420, 165, "Retrieve evidence",
          "Semantic top 20", "Cross-encoder rerank to top 5", style="source")
    d.box("answer", 1020, 230, 420, 165, "One model answer call",
          "gpt-4o-mini / temperature 0", "Choose an option and explain it", style="model")
    d.box("saved", 1020, 565, 420, 180, "Record the prediction",
          "Letter, explanation and retrieved text", "No separate statement or option review", style="store", size=20, title_size=25)
    d.box("grader", 540, 565, 420, 155, "Grade against saved keys",
          "Correct or wrong, with errors separate", "Same evaluation dataset", size=24)
    d.box("report", 60, 565, 420, 155, "Compare saved outcomes",
          "Accuracy together with coverage", "Reported as its own protocol", style="store", size=24)
    d.link("question", "search", [(480, 312), (540, 312)])
    d.link("search", "answer", [(960, 312), (1020, 312)], style="source")
    d.link("answer", "saved", [(1230, 395), (1230, 565)], style="model")
    d.link("saved", "grader", [(1020, 642), (960, 642)])
    d.link("grader", "report", [(540, 642), (480, 642)], style="store")
    return d


BUILDERS = (
    architecture, learner_flow, retrieval_flow, generation_flow, verification_flow,
    absence_flow, storage_flow, evaluation_flow, baseline_flow,
)


def export(diagram):
    svg = vector(diagram)
    (HERE / f"{diagram.name}.svg").write_text(svg, encoding="utf-8")
    (HERE / f"{diagram.name}.mmd").write_text(mermaid(diagram), encoding="utf-8")
    with pymupdf.open(stream=svg.encode(), filetype="svg") as document:
        document[0].get_pixmap(matrix=pymupdf.Matrix(1.4, 1.4), alpha=False).save(
            HERE / f"{diagram.name}.png"
        )
        pdf = document.convert_to_pdf()
    with pymupdf.open(stream=pdf, filetype="pdf") as document:
        document.set_metadata({"title": diagram.title, "subject": diagram.subtitle})
        document.save(HERE / f"{diagram.name}.pdf", garbage=4, deflate=True)
    if diagram.name == "practice_architecture":
        for extension in ("png", "svg", "pdf"):
            shutil.copyfile(HERE / f"{diagram.name}.{extension}", HERE / f"system-design.{extension}")
        shutil.copyfile(HERE / f"{diagram.name}.mmd", HERE / "overall-system.mmd")
    print(f"{diagram.name}: PNG, SVG, PDF and Mermaid")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=[builder().name for builder in BUILDERS], nargs="+")
    args = parser.parse_args(argv)
    for builder in BUILDERS:
        diagram = builder()
        if args.only is None or diagram.name in args.only:
            export(diagram)


if __name__ == "__main__":
    main()
