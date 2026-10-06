"""Render the project architecture locally as SVG, PNG and PDF.

Run from the repository root with:
    .venv/bin/python docs/diagrams/render_system_design.py
"""

from html import escape
from math import hypot
from pathlib import Path

import pymupdf


HERE = Path(__file__).resolve().parent
WIDTH, HEIGHT = 2800, 2020
PARTS: list[str] = []
INK = "#16243b"
MUTED = "#52647b"
BLUE = "#2563eb"
GREEN = "#059669"
PURPLE = "#7c3aed"
AMBER = "#b77910"


def rect(x, y, w, h, fill="white", stroke="#d9e2ee", radius=18, width=2):
    PARTS.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'
    )


def text(x, y, value, size=24, color=INK, bold=False, anchor="start"):
    PARTS.append(
        f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{700 if bold else 400}" '
        f'fill="{color}" text-anchor="{anchor}">{escape(value)}</text>'
    )


def wrap(value, max_width, size=24, bold=False):
    lines, current = [], ""
    for word in value.split():
        candidate = f"{current} {word}".strip()
        length = pymupdf.get_text_length(
            candidate, fontname="hebo" if bold else "helv", fontsize=size
        )
        if current and length > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def card(x, y, w, h, title, lines, color=BLUE, fill="white", size=24, title_size=28):
    rect(x, y, w, h, fill=fill, stroke="#d9e2ee", radius=16)
    rect(x, y, 6, h, fill=color, stroke=color, radius=3, width=0)
    cursor = y + 34
    for line in wrap(title, w - 46, title_size, bold=True):
        text(x + 23, cursor, line, title_size, bold=True)
        cursor += title_size + 4
    cursor += 6
    for paragraph in lines:
        for line in wrap(paragraph, w - 46, size):
            text(x + 23, cursor, line, size, color=MUTED)
            cursor += size + 5
    if cursor - (size + 5) > y + h - 12:
        raise ValueError(f"Card text overflows: {title}")


def arrow(points, color=MUTED, dashed=False, width=3, marker=True):
    segments = []
    if dashed:
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            distance = hypot(x2 - x1, y2 - y1)
            if not distance:
                continue
            ux, uy = (x2 - x1) / distance, (y2 - y1) / distance
            offset = 0
            while offset < distance:
                end = min(offset + 11, distance)
                segments.append([(x1 + ux * offset, y1 + uy * offset), (x1 + ux * end, y1 + uy * end)])
                offset += 19
    else:
        segments = [points]
    for segment in segments:
        d = "M " + " L ".join(f"{x},{y}" for x, y in segment)
        PARTS.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>')
    if marker:
        (x1, y1), (x2, y2) = points[-2:]
        distance = hypot(x2 - x1, y2 - y1)
        ux, uy = (x2 - x1) / distance, (y2 - y1) / distance
        bx, by = x2 - 15 * ux, y2 - 15 * uy
        triangle = [(x2, y2), (bx - 7 * uy, by + 7 * ux), (bx + 7 * uy, by - 7 * ux)]
        shape = " ".join(f"{x},{y}" for x, y in triangle)
        PARTS.append(f'<polygon points="{shape}" fill="{color}"/>')


def label(x, y, value, color=MUTED, size=21):
    w = pymupdf.get_text_length(value, fontsize=size, fontname="helv") + 22
    rect(x - w / 2, y - size, w, size + 12, fill="#f8fafc", stroke="none", radius=5, width=0)
    text(x, y + 1, value, size, color, anchor="middle")


def panel(x, y, w, h, number, title, subtitle, color, fill):
    rect(x, y, w, h, fill=fill, stroke="#d9e2ee", radius=22)
    rect(x + 24, y + 23, 48, 42, fill=color, stroke=color, radius=10)
    text(x + 48, y + 53, number, 25, "white", True, "middle")
    text(x + 89, y + 53, title, 31, bold=True)
    text(x + 26, y + 91, subtitle, 23, MUTED)


def render():
    PARTS.clear()
    rect(0, 0, WIDTH, HEIGHT, fill="#f8fafc", stroke="none", radius=0, width=0)
    text(80, 55, "SYSTEM ARCHITECTURE / CURRENT IMPLEMENTATION", 21, BLUE, True)
    text(2720, 55, "03 OCT 2026", 20, MUTED, anchor="end")
    text(80, 126, "UPSC Polity AI Tutor", 58, bold=True)
    text(80, 176, "Source verification, practice generation and measurable evaluation", 29, MUTED)

    text(80, 222, "01  PREPARE KNOWLEDGE & FIND EVIDENCE", 21, GREEN, True)
    knowledge = [
        (80, "Source PDFs", ["Constitution of India", "NCERT Polity"], 25),
        (616, "Extract & clean", ["PyMuPDF, page-aware text", "Keep source + document + page"], 23),
        (1152, "Chunk documents", ["1,000 characters per chunk", "150-character overlap"], 24),
        (1688, "Persist the corpus", ["chunks.jsonl: 1,149 chunks", "Local files with page provenance"], 23),
        (2224, "Shared retrieval", ["BGE embeddings + NumPy index", "Semantic top 20; cross-encoder top 5"], 22),
    ]
    for x, title, lines, size in knowledge:
        card(x, 247, 496, 170, title, lines, GREEN, size=size)
    for x in [80, 616, 1152, 1688]:
        arrow([(x + 496, 330), (x + 536, 330)], GREEN)

    card(80, 505, 760, 150, "Student interfaces", [
        "Streamlit tutor: verify questions or generate practice",
        "Verification CLI; shared and standalone Evaluation Studio",
    ], BLUE, size=23, title_size=30)
    card(920, 505, 940, 150, "OpenAI API + structured responses", [
        "gpt-4o-mini: runtime generation, verification and baseline",
        "Pydantic output schemas; earlier explanation judge: gpt-4o",
    ], PURPLE, size=24, title_size=30)
    card(1940, 505, 780, 150, "Special case: lexical absence", [
        "Python scans all loaded chunks for recognized quoted terms",
        "Used by FactVerifier for explicit absence claims",
    ], AMBER, size=23, title_size=29)
    arrow([(1936, 417), (1936, 464), (2320, 464), (2320, 505)], AMBER)
    label(2190, 460, "Entire corpus", AMBER)

    # Shared services use separated horizontal lanes above the workflows.
    arrow([(460, 655), (460, 700), (1420, 700)], BLUE, marker=False)
    arrow([(460, 700), (360, 700), (360, 875)], BLUE)
    arrow([(1260, 700), (1260, 875)], BLUE)
    label(1050, 696, "Verify / practice requests", BLUE)
    arrow([(2720, 330), (2750, 330), (2750, 742), (500, 742)], GREEN, marker=False)
    for x in [500, 1420, 2240]:
        arrow([(x, 742), (x, 875)], GREEN)
    label(1990, 738, "Shared evidence retrieval", GREEN)
    arrow([(1390, 655), (1390, 784), (700, 784)], PURPLE, dashed=True, marker=False)
    arrow([(1390, 784), (2580, 784)], PURPLE, dashed=True, marker=False)
    for x in [700, 1630, 2580]:
        arrow([(x, 784), (x, 875)], PURPLE, dashed=True)
    label(1170, 780, "Structured LLM calls", PURPLE)
    arrow([(2320, 655), (2320, 828), (885, 828)], AMBER, marker=False)
    for x in [885, 1790]:
        arrow([(x, 828), (x, 875)], AMBER)
    label(2010, 824, "Quoted absence claims only", AMBER)

    panel(80, 875, 840, 605, "02", "Question verification", "Full-page evidence + footnotes; official key withheld", BLUE, "#eff6ff")
    panel(980, 875, 840, 605, "03", "Practice generation", "LangGraph workflow with shared verification services", PURPLE, "#faf5ff")
    panel(1880, 875, 840, 605, "A", "Vanilla RAG baseline", "Comparison system used by the evaluation harness", AMBER, "#fffbeb")

    card(108, 992, 380, 265, "Numbered statements", [
        "Parse + validate each claim",
        "Top 5 -> full pages + footnotes",
        "LLM: SUPPORTED, CONTRADICTED",
        "or INSUFFICIENT",
        "Python validates quotes + pages",
        "Second evidence review must agree",
    ], BLUE, size=20, title_size=25)
    card(512, 992, 380, 265, "Direct / best answer", [
        "Paired statement labels -> abstain",
        "Question + four option queries",
        "Top 5 each; full pages (max 25)",
        "LLM: SUPPORTED, RULED_OUT",
        "or INSUFFICIENT",
        "Python validates quoted evidence",
        "Second review must agree on A-D",
    ], BLUE, size=20, title_size=25)
    arrow([(298, 1257), (298, 1287), (500, 1287), (500, 1320)], BLUE)
    arrow([(702, 1257), (702, 1287), (500, 1287)], BLUE, marker=False)
    card(108, 1320, 784, 112, "Python resolves the answer", [
        "Strict statement mapping / direct-option resolution",
        "Answer A-D or ABSTAIN; show reasons and sources",
    ], BLUE, size=23, title_size=27)

    card(1008, 992, 724, 108, "Generate a candidate MCQ", [
        "LLM uses topic, difficulty and question format",
    ], PURPLE, size=24, title_size=28)
    arrow([(1370, 1100), (1370, 1137)], PURPLE)
    card(1008, 1137, 724, 151, "Extract claims + run three gates", [
        "Facts + generated answer key + question quality",
        "Fact judgments: full pages + blind evidence review",
        "Deterministic claims first; generated key withheld",
    ], PURPLE, size=22, title_size=27)
    arrow([(1370, 1288), (1370, 1320)], PURPLE)
    card(1008, 1320, 724, 112, "Python: ACCEPT / REVISE / REJECT", [
        "Initial candidate + up to two revisions",
        "Accepted MCQ or visible rejection reasons",
    ], PURPLE, size=23, title_size=27)
    arrow([(1732, 1376), (1770, 1376), (1770, 1046), (1732, 1046)], PURPLE, dashed=True)
    label(1769, 1123, "REVISE", PURPLE, 16)

    card(1908, 992, 784, 112, "Retrieve the whole question", [
        "One full-question query; same corpus and retrieval depth",
    ], AMBER, size=24, title_size=28)
    arrow([(2300, 1104), (2300, 1140)], AMBER)
    card(1908, 1140, 784, 119, "LLM chooses A-D", [
        "gpt-4o-mini produces a structured answer",
        "A selected option plus explanation",
    ], AMBER, size=24, title_size=28)
    arrow([(2300, 1259), (2300, 1320)], AMBER)
    card(1908, 1320, 784, 112, "Record the baseline prediction", [
        "Compare outcomes with the source verifier",
        "The baseline selects a letter for every completed answer",
    ], AMBER, size=23, title_size=27)

    arrow([(500, 1432), (500, 1530), (1330, 1530)], INK, marker=False)
    arrow([(2300, 1432), (2300, 1530), (1330, 1530)], INK, marker=False)
    arrow([(1330, 1530), (1330, 1703)], INK)
    label(1560, 1526, "Predictions + evidence traces", INK)

    panel(80, 1628, 2640, 287, "04", "Evaluation framework & saved reports", "75 official PYQs: development 33 / regression test 42; resumable checkpoints", AMBER, "#fffbeb")
    # Cards sit below the shared header; their text sizes suit the landscape export.
    card(110, 1740, 434, 109, "Official benchmark", ["Frozen questions + matched keys"], AMBER, size=22, title_size=26)
    card(584, 1740, 434, 109, "Custom Python harness", ["Runs verifier + vanilla RAG"], AMBER, size=22, title_size=26)
    card(1058, 1740, 544, 109, "Python scoring", ["Correct / wrong / abstained / errors", "Accuracy, coverage and precision"], AMBER, size=21, title_size=26)
    card(1642, 1740, 434, 109, "JSON reports", ["Metrics, configuration and traces"], AMBER, size=22, title_size=26)
    card(2116, 1740, 574, 109, "Evaluation Studio", ["Inspect results; export CSV / JSON"], AMBER, size=23, title_size=26)
    # Move the prediction arrow endpoint to the top of the scoring card.
    arrow([(1330, 1530), (1330, 1740)], INK)
    for left, right in [(544, 584), (1018, 1058), (1602, 1642), (2076, 2116)]:
        arrow([(left, 1794), (right, 1794)], AMBER)
    arrow([(327, 1849), (327, 1877), (1330, 1877), (1330, 1849)], AMBER)
    label(1040, 1892, "Official keys are used only for scoring", AMBER, 19)

    text(80, 1963, "Earlier diagnostics: 13-question regression, citation presence, explanation judge, verdict stability, retrieval and generation experiments.", 24, MUTED)
    text(80, 1996, "75-question splits: regression after debugging. A fresh set is needed for independent evaluation. Dashed lines: LLM calls / revision feedback.", 21, MUTED)

    markers = "".join(
        f'<marker id="{color[1:]}" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="9" markerHeight="9" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{color}"/></marker>'
        for color in [MUTED, BLUE, GREEN, PURPLE, AMBER, INK]
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img">'
        '<title>UPSC Polity AI Tutor - overall system design</title>'
        '<desc>Knowledge preparation and retrieval, question verification, LangGraph practice generation, vanilla RAG baseline, and evaluation reports.</desc>'
        f'<defs>{markers}</defs>' + "".join(PARTS) + '</svg>'
    )
    svg_path = HERE / "system-design.svg"
    svg_path.write_text(svg, encoding="utf-8")
    with pymupdf.open(stream=svg.encode(), filetype="svg") as document:
        pixmap = document[0].get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
        pixmap.save(HERE / "system-design.png")
        pdf_data = document.convert_to_pdf()
        (HERE / "system-design.pdf").write_bytes(pdf_data)
        print(f"Saved PNG: {pixmap.width} x {pixmap.height}")
    for filename in ["system-design.png", "system-design.svg", "system-design.pdf"]:
        path = HERE / filename
        print(f"{path.relative_to(HERE.parent.parent)} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    render()
