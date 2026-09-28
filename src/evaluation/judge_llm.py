"""
LLM-as-a-judge comparison of System A and System B explanations.

Accuracy against the official UPSC key already tells us which system
picked the right letter. It does not tell us *why* a system was right,
and on a four-option MCQ a correct letter can come from invalid
reasoning. The capstone's research question is about factual and
evidence-grounding quality, so it needs a judge that reads the reasoning
rather than only the answer.

What the judge is, and is not
-----------------------------

The judge is an evaluator, never ground truth.

    Answer correctness   -> official UPSC answer key, always.
    Reasoning quality    -> this judge.

The judge is told the official answer. It is explicitly forbidden from
overriding it, and forbidden from rewarding a confident wrong answer for
sounding authoritative.

Bias controls
-------------

Three known LLM-judge failure modes are controlled for:

position bias
    The two responses are presented as "Response 1" and "Response 2".
    Which system occupies which slot alternates on question parity, so
    neither system sits in the favoured slot throughout. The mapping is
    recorded per question so the aggregation can undo it, and the run
    reports how often each *slot* won as a diagnostic.

self-enhancement bias
    Both systems run on gpt-4o-mini. A judge sharing that model tends to
    prefer its own outputs, so the judge defaults to a different, stronger
    model and records which model actually scored the run.

verbosity bias
    System B's output is structurally longer than System A's. The judge
    is instructed that length, citation count and confident tone are not
    quality, and that a justified abstention outranks an unsupported
    commitment.

Abstention
----------

System B may decline to answer. An abstention is scored on whether it was
*justified by the supplied evidence*, not treated as an automatic loss.
That is the behaviour the verification pipeline exists to produce, so
scoring it as a wrong answer would beg the research question.

Dependencies
------------

This module deliberately uses only the standard library. The rest of the
project calls OpenAI through the `openai` SDK with `pydantic` schemas,
which is the better pattern where those packages are installed. They are
not installable in the evaluation environment used for the final run
(the package index is unreachable), so the judge speaks HTTP directly and
declares its response schema as literal JSON. The API contract is the
same; only the transport differs.

Run:

    python -m src.evaluation.judge_llm
"""

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


# Both systems generate with gpt-4o-mini. Judging with a different,
# stronger model reduces self-enhancement bias. If the preferred judge is
# unavailable on the account the run falls back and records the fact
# rather than failing.
PREFERRED_JUDGE_MODEL = "gpt-4o"
FALLBACK_JUDGE_MODEL = "gpt-4o-mini"

API_URL = "https://api.openai.com/v1/chat/completions"
ENV_PATH = ".env"

VERIFIED_PATH = "data/evaluation/verified_pyq_results.json"
VANILLA_PATH = "data/evaluation/vanilla_rag_results.json"
OUTPUT_PATH = "data/evaluation/judge_results.json"

# Evidence passages are truncated so that a single question's prompt stays
# within a sensible size. The judge needs enough of each passage to check
# a citation, not the whole chunk.
EVIDENCE_CHAR_LIMIT = 1200

SCORE_FIELDS = (
    "factual_correctness",
    "evidence_faithfulness",
    "reasoning_validity",
    "epistemic_honesty",
)

MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 4


# ==========================================================
# Response schema
# ==========================================================

# Scores are declared as an enum rather than an integer with bounds
# because OpenAI's strict structured-output mode does not honour
# `minimum`/`maximum`. The enum makes the range part of the contract
# instead of something to validate afterwards.
SCORE_SCHEMA = {
    "type": "integer",
    "enum": [1, 2, 3, 4, 5],
}

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        **{field: SCORE_SCHEMA for field in SCORE_FIELDS},
        "hallucinated_claims": {
            "type": "array",
            "items": {"type": "string"},
        },
        "justification": {"type": "string"},
    },
    "required": [
        *SCORE_FIELDS,
        "hallucinated_claims",
        "justification",
    ],
    "additionalProperties": False,
}

PAIR_SCHEMA = {
    "type": "object",
    "properties": {
        "response_1": RESPONSE_SCHEMA,
        "response_2": RESPONSE_SCHEMA,
        "better_response": {
            "type": "string",
            "enum": ["1", "2", "TIE"],
        },
        "better_response_reason": {"type": "string"},
    },
    "required": [
        "response_1",
        "response_2",
        "better_response",
        "better_response_reason",
    ],
    "additionalProperties": False,
}


RUBRIC = """
You are an evaluator for a UPSC Indian Polity tutoring system. Two
systems answered the same official UPSC Prelims question. Score the
QUALITY OF THEIR REASONING.

GROUND TRUTH

The official UPSC answer key is given to you and is final. You must not
override it, argue with it, or substitute your own answer for it.

SCORE EACH RESPONSE 1-5 ON FOUR DIMENSIONS

factual_correctness
  Are the constitutional facts asserted actually correct, and does the
  response reach the official answer for defensible reasons?
  5 = correct answer reached by correct reasoning.
  3 = correct answer but the reasoning is partly wrong, or a wrong answer
      that rests on an understandable misreading.
  1 = confidently asserts false constitutional facts.
  A response that lands the correct letter through invalid reasoning must
  NOT score above 3.

evidence_faithfulness
  Every factual assertion must be traceable to the SOURCE EVIDENCE shown
  below. Penalise any claim drawn from model background knowledge rather
  than the supplied passages, and penalise page citations that do not
  support what they are attached to.
  5 = every assertion is grounded in the supplied evidence and cited
      accurately.
  1 = substantially ungrounded, or citations misrepresent the passages.

reasoning_validity
  Does the conclusion actually follow from the stated premises? Look for
  the specific failure where a system establishes that the corpus
  MENTIONS a topic and then treats that as establishing a CLAIM about
  that topic. Mentioning is not establishing.
  5 = the inference is valid and each step is stated.
  1 = a non sequitur, or the conclusion is asserted rather than derived.

epistemic_honesty
  Does the response represent its own certainty accurately given the
  evidence it was shown?
  5 = states what the evidence does and does not establish; where the
      evidence is genuinely incomplete, says so, and declining to answer
      on genuinely insufficient evidence scores HIGH here.
  1 = presents an unsupported conclusion with unwarranted confidence.

CRITICAL SCORING RULES

1. Length is not quality. A longer, more structured, more heavily cited
   response is not better for those reasons.
2. Confident tone is not quality. Do not reward authoritative phrasing.
3. A response that declines to answer, where the supplied evidence really
   is insufficient, is BETTER than one that commits to an unsupported
   answer. Judge the abstention on whether the evidence justifies it.
4. But an abstention where the supplied evidence WAS sufficient is a real
   failure. Say so.
5. Do not reward a correct letter that was reached by invalid reasoning.
6. hallucinated_claims must list the specific assertions you could not
   trace to the supplied evidence. Empty list if there are none.

better_response
  "1", "2", or "TIE" - which response is better overall as tutoring
  content for a student. Explain which dimension decided it.
"""


# ==========================================================
# Minimal OpenAI transport
# ==========================================================


def load_api_key() -> str:
    """
    Read the API key from the environment, falling back to .env.

    python-dotenv is unavailable in the evaluation environment, so the
    handful of lines needed to parse a KEY=value file are inlined rather
    than adding a dependency the run cannot install.
    """

    key = os.getenv("OPENAI_API_KEY")

    if key:
        return key

    env_file = Path(ENV_PATH)

    if env_file.exists():
        for line in env_file.read_text(
            encoding="utf-8"
        ).splitlines():

            stripped = line.strip()

            if not stripped or stripped.startswith("#"):
                continue

            name, separator, value = stripped.partition("=")

            if separator and name.strip() == "OPENAI_API_KEY":
                return value.strip().strip("\"'")

    raise RuntimeError(
        "OPENAI_API_KEY not found in the environment or in .env"
    )


def post_chat_completion(
    api_key: str,
    model: str,
    prompt: str,
    schema_name: str,
    schema: dict,
) -> dict:
    """
    Call the chat-completions endpoint and return the parsed JSON object.

    Raises urllib.error.HTTPError for API-level failures so the caller can
    distinguish "this model is not available to the account" from a
    transient network problem.
    """

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        },
    }

    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(
        request, timeout=180
    ) as response:
        body = json.loads(response.read().decode("utf-8"))

    content = body["choices"][0]["message"]["content"]

    return json.loads(content)


# ==========================================================
# Rendering
# ==========================================================


def truncate(text: str) -> str:
    """Shorten an evidence passage, marking it when cut."""

    collapsed = " ".join(text.split())

    if len(collapsed) <= EVIDENCE_CHAR_LIMIT:
        return collapsed

    return collapsed[:EVIDENCE_CHAR_LIMIT] + " [...]"


def collect_evidence(verified_record: dict) -> list[dict]:
    """
    Gather every passage System B retrieved for this question.

    System A stores only page-level stubs for its citations, not the
    passage text, but both systems draw on the same corpus through the
    same retriever, so System B's passages supply the text the judge needs
    to check either system's citations. Where System A cites a page that
    is absent here, the page number is still shown to the judge as an
    uncheckable citation.
    """

    items = list(verified_record.get("answer_evidence", []))

    for passages in verified_record.get(
        "claim_evidence", {}
    ).values():
        items.extend(passages)

    return items


def format_evidence(evidence_items: list[dict]) -> str:
    """Render the retrieved passages, deduplicated by page."""

    by_page: dict[int, dict] = {}

    for item in evidence_items:
        by_page.setdefault(item["page"], item)

    if not by_page:
        return "(no evidence was retrieved)"

    blocks = []

    for page in sorted(by_page):
        item = by_page[page]

        blocks.append(
            f"[{item['source']}, page {page}]\n"
            f"{truncate(item['text'])}"
        )

    return "\n\n".join(blocks)


def render_system_a(record: dict) -> str:
    """Render System A's answer as the judge will see it."""

    cited_pages = sorted(
        {source["page"] for source in record.get("sources", [])}
    )

    selected = record.get("selected_option") or "(none)"

    return (
        f"Answer given: {selected}\n"
        f"Pages cited by the system: {cited_pages}\n\n"
        f"Explanation:\n{record['rag_answer']}"
    )


def render_system_b(record: dict) -> str:
    """
    Render System B's verification trace as the judge will see it.

    System B does not produce prose. Its reasoning is the per-statement
    verdict set plus the mapping outcome, so that is what gets judged.
    """

    mapping = record.get("claim_mapping", {})
    statuses = mapping.get("statement_statuses", {})

    claims_by_id = {
        claim["claim_id"]: claim
        for claim in record.get("claims", [])
    }

    lines = []

    predicted = record.get("predicted_answer")

    if predicted is None:
        lines.append(
            "Answer given: DECLINED TO ANSWER "
            "(the pipeline judged the retrieved evidence "
            "insufficient to identify an option)"
        )
    else:
        lines.append(f"Answer given: {predicted}")

    lines.append("")
    lines.append("Per-statement verification:")

    for claim_id, verification in sorted(
        record.get("fact_verifications", {}).items()
    ):
        claim = claims_by_id.get(claim_id, {})

        statement = claim.get("statement_number") or claim_id

        lines.append("")
        lines.append(
            f"  Statement {statement}: {claim.get('claim', '')}"
        )
        lines.append(f"  Verdict: {verification['verdict']}")
        lines.append(
            f"  Pages cited: {verification['supporting_pages']}"
        )
        lines.append(f"  Reasoning: {verification['reasoning']}")

    if statuses:
        lines.append("")
        lines.append(f"Aggregated statement verdicts: {statuses}")

    matching = mapping.get("matching_options")

    if matching is not None:
        lines.append(
            f"Options consistent with those verdicts: "
            f"{matching or '(none)'}"
        )

    return "\n".join(lines)


def build_prompt(
    question_record: dict,
    evidence_text: str,
    response_1: str,
    response_2: str,
) -> str:
    """Assemble the judge prompt for one question."""

    options = "\n".join(
        f"  {letter}. {text}"
        for letter, text in sorted(
            question_record["options"].items()
        )
    )

    return f"""{RUBRIC}

========================================
QUESTION
========================================

{question_record["question"]}

Options:
{options}

OFFICIAL UPSC ANSWER KEY: {question_record["official_answer"]}

========================================
SOURCE EVIDENCE AVAILABLE TO BOTH SYSTEMS
========================================

{evidence_text}

========================================
RESPONSE 1
========================================

{response_1}

========================================
RESPONSE 2
========================================

{response_2}
"""


def slot_assignment(q_number: int) -> tuple[str, str]:
    """
    Decide which system occupies which slot for this question.

    Alternating on question parity is deterministic, so the run is
    reproducible, while still preventing either system from holding the
    same slot throughout.
    """

    if q_number % 2 == 0:
        return "A", "B"

    return "B", "A"


# ==========================================================
# Judge
# ==========================================================


class ExplanationJudge:
    """Scores both systems' reasoning on one question."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.model = PREFERRED_JUDGE_MODEL
        self.model_is_fallback = False

    def _call(self, prompt: str) -> dict:
        return post_chat_completion(
            api_key=self.api_key,
            model=self.model,
            prompt=prompt,
            schema_name="pair_judgement",
            schema=PAIR_SCHEMA,
        )

    def judge(self, prompt: str) -> dict:
        """
        Score one question, downgrading the model only on a 404.

        A 404 means the account cannot use the preferred judge model,
        which is permanent and worth recording. Anything else is treated
        as transient and retried on the same model, so a flaky connection
        does not silently change the judge mid-run.
        """

        last_error: Exception | None = None

        for attempt in range(1, MAX_ATTEMPTS + 1):

            try:
                return self._call(prompt)

            except urllib.error.HTTPError as error:

                model_unavailable = error.code in (403, 404)

                if (
                    model_unavailable
                    and not self.model_is_fallback
                ):
                    print(
                        f"  judge model {self.model} unavailable "
                        f"(HTTP {error.code}), falling back to "
                        f"{FALLBACK_JUDGE_MODEL}"
                    )

                    self.model = FALLBACK_JUDGE_MODEL
                    self.model_is_fallback = True

                    continue

                last_error = error

            except (
                urllib.error.URLError,
                TimeoutError,
                json.JSONDecodeError,
                KeyError,
            ) as error:
                last_error = error

            if attempt < MAX_ATTEMPTS:
                wait = RETRY_BACKOFF_SECONDS * attempt

                print(
                    f"  attempt {attempt} failed "
                    f"({type(last_error).__name__}), "
                    f"retrying in {wait}s"
                )

                time.sleep(wait)

        raise RuntimeError(
            f"Judge failed after {MAX_ATTEMPTS} attempts: "
            f"{last_error}"
        )


# ==========================================================
# Aggregation and reporting
# ==========================================================


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate(rows: list[dict]) -> dict:
    """
    Average each dimension per system and count head-to-head wins.

    The per-question slot swap is already undone by the time a row is
    built, so these averages are over systems, not slots.
    """

    summary: dict = {}

    for system in ("A", "B"):

        scores = {
            field: mean(
                [row["scores"][system][field] for row in rows]
            )
            for field in SCORE_FIELDS
        }

        dimension_means = [
            value
            for value in scores.values()
            if value is not None
        ]

        summary[system] = {
            **scores,
            "mean_across_dimensions": mean(dimension_means),
            "hallucinated_claim_count": sum(
                len(row["hallucinated_claims"][system])
                for row in rows
            ),
            "questions_with_hallucinations": sum(
                1
                for row in rows
                if row["hallucinated_claims"][system]
            ),
        }

    wins = {"A": 0, "B": 0, "TIE": 0}

    for row in rows:
        wins[row["preferred_system"]] += 1

    # If the judge overwhelmingly prefers one slot regardless of which
    # system occupies it, position bias is not being controlled and the
    # comparison should be treated with caution.
    slot_wins = {"1": 0, "2": 0, "TIE": 0}

    for row in rows:
        slot_wins[row["preferred_slot"]] += 1

    summary["head_to_head"] = wins
    summary["slot_preference_check"] = slot_wins

    return summary


def score(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def print_report(report: dict) -> None:
    summary = report["summary"]

    print()
    print("=== LLM-AS-A-JUDGE: REASONING QUALITY ===")
    print(f"Dataset:     {report['dataset']}")
    print(f"Judge model: {report['judge_model']}")

    if report["judge_model_is_fallback"]:
        print(
            "  NOTE: the judge shares the generator model, so "
            "self-enhancement bias is NOT controlled for."
        )

    print(f"Questions:   {report['questions_judged']}")
    print()

    header = (
        f"{'dimension (1-5)':<28}{'System A':>10}{'System B':>10}"
    )
    print(header)
    print("-" * len(header))

    for field in SCORE_FIELDS:
        print(
            f"{field.replace('_', ' '):<28}"
            f"{score(summary['A'][field]):>10}"
            f"{score(summary['B'][field]):>10}"
        )

    print("-" * len(header))
    print(
        f"{'mean across dimensions':<28}"
        f"{score(summary['A']['mean_across_dimensions']):>10}"
        f"{score(summary['B']['mean_across_dimensions']):>10}"
    )

    print()
    print(
        f"{'hallucinated claims':<28}"
        f"{summary['A']['hallucinated_claim_count']:>10}"
        f"{summary['B']['hallucinated_claim_count']:>10}"
    )
    print(
        f"{'questions affected':<28}"
        f"{summary['A']['questions_with_hallucinations']:>10}"
        f"{summary['B']['questions_with_hallucinations']:>10}"
    )

    wins = summary["head_to_head"]

    print()
    print("=== HEAD TO HEAD ===")
    print(f"System A preferred: {wins['A']}")
    print(f"System B preferred: {wins['B']}")
    print(f"Tie:                {wins['TIE']}")

    slots = summary["slot_preference_check"]

    print()
    print("=== POSITION BIAS CHECK ===")
    print(
        f"Slot 1 preferred: {slots['1']}   "
        f"Slot 2 preferred: {slots['2']}   "
        f"Tie: {slots['TIE']}"
    )
    print(
        "A large slot imbalance would indicate the judge is "
        "responding to position rather than content."
    )

    print()
    print("=== PER QUESTION ===")

    for row in report["rows"]:
        print(
            f"  Q{row['q_number']:<3} "
            f"A={score(row['mean']['A'])} "
            f"B={score(row['mean']['B'])}  "
            f"preferred={row['preferred_system']:<4} "
            f"(A said {row['answers']['A']}, "
            f"B said {row['answers']['B']}, "
            f"official {row['official_answer']})"
        )


def main() -> None:
    verified_records = {
        record["q_number"]: record
        for record in json.loads(
            Path(VERIFIED_PATH).read_text(encoding="utf-8")
        )
    }

    vanilla_records = {
        record["q_number"]: record
        for record in json.loads(
            Path(VANILLA_PATH).read_text(encoding="utf-8")
        )
    }

    shared = sorted(set(verified_records) & set(vanilla_records))

    judge = ExplanationJudge(load_api_key())

    rows = []

    for index, q_number in enumerate(shared, start=1):
        print(f"Judging {index}/{len(shared)}: Q{q_number}")

        vanilla_record = vanilla_records[q_number]
        verified_record = verified_records[q_number]

        rendered = {
            "A": render_system_a(vanilla_record),
            "B": render_system_b(verified_record),
        }

        slot_1_system, slot_2_system = slot_assignment(q_number)

        prompt = build_prompt(
            question_record=verified_record,
            evidence_text=format_evidence(
                collect_evidence(verified_record)
            ),
            response_1=rendered[slot_1_system],
            response_2=rendered[slot_2_system],
        )

        judgement = judge.judge(prompt)

        # Undo the slot swap so everything downstream is keyed by system.
        by_system = {
            slot_1_system: judgement["response_1"],
            slot_2_system: judgement["response_2"],
        }

        slot_to_system = {
            "1": slot_1_system,
            "2": slot_2_system,
            "TIE": "TIE",
        }

        rows.append(
            {
                "q_number": q_number,
                "official_answer": verified_record[
                    "official_answer"
                ],
                "answers": {
                    "A": vanilla_record.get("selected_option"),
                    "B": verified_record.get("predicted_answer"),
                },
                "slot_assignment": {
                    "1": slot_1_system,
                    "2": slot_2_system,
                },
                "scores": {
                    system: {
                        field: result[field]
                        for field in SCORE_FIELDS
                    }
                    for system, result in by_system.items()
                },
                "mean": {
                    system: mean(
                        [
                            result[field]
                            for field in SCORE_FIELDS
                        ]
                    )
                    for system, result in by_system.items()
                },
                "hallucinated_claims": {
                    system: result["hallucinated_claims"]
                    for system, result in by_system.items()
                },
                "justification": {
                    system: result["justification"]
                    for system, result in by_system.items()
                },
                "preferred_system": slot_to_system[
                    judgement["better_response"]
                ],
                "preferred_slot": judgement["better_response"],
                "preference_reason": judgement[
                    "better_response_reason"
                ],
            }
        )

    report = {
        "dataset": "UPSC 2025 Prelims Polity, Q54-Q66",
        "judge_model": judge.model,
        "judge_model_is_fallback": judge.model_is_fallback,
        "generator_model": "gpt-4o-mini",
        "questions_judged": len(rows),
        "systems": {
            "A": "Vanilla RAG (retrieve -> generate)",
            "B": (
                "RAG + independent claim verification "
                "(may abstain)"
            ),
        },
        "note": (
            "The judge evaluates reasoning quality only. Answer "
            "correctness is determined by the official UPSC answer "
            "key, never by the judge."
        ),
        "summary": aggregate(rows),
        "rows": rows,
    }

    print_report(report)

    Path(OUTPUT_PATH).write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print(f"Saved judge results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
