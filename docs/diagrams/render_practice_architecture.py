"""Export the current practice architecture as a local SVG and PNG.

Run: .venv/bin/python docs/diagrams/render_practice_architecture.py
Uses the project's existing PyMuPDF dependency; no network or model calls.
"""

from html import escape
from math import hypot
from pathlib import Path

import pymupdf


HERE = Path(__file__).resolve().parent
WIDTH, HEIGHT = 1600, 1810
INK, MUTED = "#172b46", "#52647b"
BLUE, GREEN, PURPLE, AMBER = "#2563eb", "#087f6b", "#7c3aed", "#a16207"


def render() -> str:
    parts = []

    def rect(x, y, w, h, fill="white", stroke="#d7e1ed", radius=18):
        parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        )

    def text(x, y, value, size=24, color=INK, bold=False, rotate=None):
        transform = f' transform="rotate({rotate} {x} {y})"' if rotate else ""
        parts.append(
            f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="{size}" font-weight="{700 if bold else 400}" '
            f'fill="{color}"{transform}>{escape(value)}</text>'
        )

    def lines(value, max_width, size, bold=False):
        result, current = [], ""
        for word in value.split():
            candidate = f"{current} {word}".strip()
            if current and pymupdf.get_text_length(
                candidate, fontname="hebo" if bold else "helv", fontsize=size,
            ) > max_width:
                result.append(current)
                current = word
            else:
                current = candidate
        if current:
            result.append(current)
        return result

    def card(x, y, w, h, title, paragraphs, color=BLUE, fill="white", size=24, title_size=30):
        rect(x, y, w, h, fill)
        rect(x, y, 5, h, color, color, 2)
        cursor = y + 39
        for line in lines(title, w - 46, title_size, True):
            text(x + 23, cursor, line, title_size, bold=True)
            cursor += title_size + 5
        cursor += 5
        for paragraph in paragraphs:
            for line in lines(paragraph, w - 46, size):
                text(x + 23, cursor, line, size, MUTED)
                cursor += size + 7
        if cursor - size - 7 > y + h - 12:
            raise ValueError(f"Card text overflows: {title}")

    def arrow(points, color=BLUE, dashed=False):
        path = "M " + " L ".join(f"{x},{y}" for x, y in points)
        dash = ' stroke-dasharray="10 8"' if dashed else ""
        parts.append(
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3" '
            f'stroke-linecap="round" stroke-linejoin="round"{dash}/>'
        )
        (x1, y1), (x2, y2) = points[-2:]
        length = hypot(x2 - x1, y2 - y1)
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        bx, by = x2 - 14 * ux, y2 - 14 * uy
        triangle = f"{x2},{y2} {bx - 6 * uy},{by + 6 * ux} {bx + 6 * uy},{by - 6 * ux}"
        parts.append(f'<polygon points="{triangle}" fill="{color}"/>')

    rect(0, 0, WIDTH, HEIGHT, "#f8fafc", "#f8fafc", 0)
    text(60, 48, "SYSTEM DESIGN / CURRENT PRACTICE MVP", 20, BLUE, True)
    text(1320, 48, "07 OCT 2026", 20, MUTED)
    text(60, 108, "UPSC Practice", 48, bold=True)
    text(60, 146, "Evidence first. Checked keys and option explanations. Bounded publication.", 25, MUTED)

    card(60, 180, 430, 160, "Public reference PDFs", [
        "Constitution of India",
        "Selected NCERT Grade 7 chapter",
    ], GREEN)
    card(570, 180, 430, 160, "Page-aware corpus", [
        "Extracted text / chunks.jsonl",
        "Source, page and SHA-256",
    ], GREEN)
    card(1080, 180, 435, 160, "Local retrieval index", [
        "BGE semantic embeddings",
        "Cross-encoder reranking",
    ], GREEN)
    arrow([(490, 260), (570, 260)], GREEN)
    arrow([(1000, 260), (1080, 260)], GREEN)
    arrow([(1300, 340), (1300, 430)], GREEN)

    text(60, 402, "LEARNER REQUEST", 20, BLUE, True)
    text(650, 402, "BOUNDED LANGGRAPH WORKFLOW", 20, PURPLE, True)
    card(60, 430, 440, 145, "Streamlit practice UI", [
        "Curated topic, style, level and count",
        "Polity open; other subjects locked",
    ], BLUE)
    arrow([(280, 575), (280, 640)], BLUE)
    card(60, 640, 440, 145, "Checked bank lookup", [
        "Validate record, source and policy",
        "Match topic, style and level",
    ], BLUE)
    arrow([(500, 690), (580, 690), (580, 502), (650, 502)], BLUE)
    text(512, 476, "Fresh", 19, BLUE)
    arrow([(280, 785), (280, 810), (530, 810), (530, 1620), (650, 1620)], GREEN)
    text(360, 803, "Eligible reuse", 19, GREEN)

    card(650, 430, 865, 145, "Source-first RAG", [
        "Semantic top 20 / cross-encoder top 5",
        "Full pages + named Article lookup + footnotes",
    ], GREEN, "#eefaf6")
    arrow([(1080, 575), (1080, 640)], PURPLE)
    card(650, 640, 865, 145, "Generate a candidate", [
        "Structured MCQ; proposed key remains untrusted",
        "Python format checks and exact-stem deduplication",
    ], PURPLE, "#f6f2ff")
    arrow([(1080, 785), (1080, 828), (865, 828), (865, 870)], PURPLE)
    arrow([(1080, 785), (1080, 828), (1300, 828), (1300, 870)], PURPLE)
    card(650, 870, 415, 210, "Statement questions", [
        "Retrieve and review each claim",
        "Resolved truth pattern",
        "Python maps pattern to A-D",
        "Unresolved claim blocks release",
    ], BLUE, size=22, title_size=27)
    card(1100, 870, 415, 210, "Direct questions", [
        "Compare all four options",
        "One supported; three ruled out",
        "Blind review must agree",
        "Python binds quotes and key",
    ], BLUE, size=22, title_size=27)
    arrow([(865, 1080), (865, 1120), (1080, 1120), (1080, 1160)], PURPLE)
    arrow([(1300, 1080), (1300, 1120), (1080, 1120)], PURPLE)
    card(650, 1160, 865, 145, "Quality audit + fresh explanation review", [
        "Clarity, unique answer, distractors, wording and topic fit",
        "Summary + A-D notes; source-bound quotes; per-item review",
    ], PURPLE, "#f6f2ff")
    arrow([(1080, 1305), (1080, 1370)], AMBER)
    card(650, 1370, 865, 110, "Python publication gate", [
        "Sources + format + answer + quality + explanations: all PASS",
    ], AMBER, "#fffbeb", size=23)
    arrow([(1410, 1370), (1538, 1335), (1538, 712), (1515, 712)], PURPLE, True)
    text(1573, 1240, "REVISE / AT MOST TWICE", 19, PURPLE, True, rotate=-90)
    arrow([(865, 1480), (865, 1545)], GREEN)
    arrow([(1300, 1480), (1300, 1545)], AMBER)
    card(650, 1545, 415, 155, "ACCEPT / quiz", [
        "SQLite holds accepted items",
        "Key and notes unlock on Submit",
        "Python scoring; session history",
    ], GREEN, "#eefaf6", size=22, title_size=27)
    card(1100, 1545, 415, 155, "WITHHOLD / no draft", [
        "Partial set or no accepted set",
        "Show a preparation outcome",
        "No guessed key fills the gap",
    ], AMBER, "#fffbeb", size=22, title_size=27)

    card(60, 875, 440, 210, "Bounded model services", [
        "gpt-4o-mini / structured outputs",
        "Generator and separate reviewers",
        "45s timeout / one SDK retry",
        "Same-family reviews can share errors",
    ], PURPLE, size=22, title_size=27)
    card(60, 1160, 440, 310, "Saved diagnostics", [
        "SQLite: accepted items and runs",
        "Actual retrieval and evidence traces",
        "Evals: saved task-specific reports",
        "Cost & latency: tokens and timings",
        "Viewing makes no new model calls",
        "Shared local store; no account isolation",
    ], BLUE, size=22, title_size=27)
    text(60, 1750, "Automated gates reduce risk; they do not guarantee zero factual errors.", 23, INK, True)
    text(60, 1785, "Current corpus scope and policy govern reuse. Generated-MCQ expert accuracy is not yet measured.", 21, MUTED)
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">' + "".join(parts) + "</svg>"


def export() -> None:
    svg = render()
    (HERE / "practice_architecture.svg").write_text(svg, encoding="utf-8")
    with pymupdf.open(stream=svg.encode(), filetype="svg") as document:
        document[0].get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False).save(
            HERE / "practice_architecture.png",
        )


if __name__ == "__main__":
    export()
    print("Saved docs/diagrams/practice_architecture.svg and practice_architecture.png")
