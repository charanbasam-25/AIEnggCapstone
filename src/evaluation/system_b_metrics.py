"""
Quantitative metrics for System B (RAG + independent claim verification).

Accuracy alone mis-describes this system. System B is allowed to abstain,
so a single accuracy number silently charges it for every question it
declined, and treats an honest "the corpus cannot settle this" as
identical to a wrong answer. That collapses the exact distinction the
architecture was built to make.

These metrics separate three questions that raw accuracy conflates:

    Did it answer?        -> coverage
    Was it right when it did?  -> precision when answered
    Why did it decline?   -> abstention taxonomy + corpus answerability

and then attribute each abstention to one of two very different causes:

    corpus limit       the evidence is genuinely not in the corpus, so
                       declining is CORRECT behaviour and not a defect
    pipeline defect    the evidence IS in the corpus but verification
                       did not establish it

Only the second is an engineering failure. Reporting them together
understates the system and hides what to fix.

Metrics computed
----------------

1. Selective prediction, every arm on one risk-coverage table.
2. Claim-level and statement-level verdict distributions.
3. Evidence grounding rate: do committed verdicts actually cite pages?
4. Claim construction quality: the non-propositional claim rate.
5. Corpus answerability probe, and answerability-adjusted accuracy.
6. Abstention taxonomy.
7. Answer-key verifier validity, with its contamination flag.

On metric 4
-----------

Verification can only be as good as the claim it is handed. Several
claims extracted from "consider the following subjects" and
match-the-pair questions are bare noun phrases, e.g.

    "List I-Union List, in the Seventh Schedule"

That is not a proposition: it has no truth value, so it cannot be
verified, and the verifier returned SUPPORTED merely because the corpus
mentions it. SUPPORTED silently degenerated into "this topic exists in
the corpus", which is a FALSE SUPPORTED and strictly worse than an
INSUFFICIENT, because the mapping layer then acts on it.

This metric measures how often that happened. It is a lexical proxy
(absence of a predicate marker), not a parser, so the flagged claims are
printed for inspection rather than reported as a settled count.

On metric 5
-----------

The probe is deliberately lexical and deterministic: it greps the chunk
corpus for the distinctive terms of each question. If a term appears in
zero of the 1,149 chunks, no retriever configuration could have found
it, so the limitation is the corpus, not retrieval.

The probe never rewrites a verdict. It only classifies an abstention
that already happened. Treating a zero-hit probe as proof that a claim
is FALSE would be the "absence of evidence is evidence of absence" error
the development log prohibits; the probe is used strictly to decide
whether the corpus could have answered, never whether a claim is true.

Dependencies: standard library only, so this runs in the evaluation
environment where the package index is unreachable.

Run:

    python -m src.evaluation.system_b_metrics
"""

import json
import re
from collections import Counter
from pathlib import Path


VERIFIED_PATH = "data/evaluation/verified_pyq_results.json"
VANILLA_PATH = "data/evaluation/vanilla_rag_results.json"
REPLAY_PATH = "data/evaluation/answer_mapping_replay.json"
DIRECT_PATH = "data/evaluation/direct_option_results.json"
CHUNKS_PATH = "data/processed/chunks.jsonl"
OUTPUT_PATH = "data/evaluation/system_b_metrics.json"

COMMITTED_VERDICTS = ("SUPPORTED", "CONTRADICTED")

# Tokens that signal a claim actually asserts something. A statement
# extracted without one of these is almost always a bare noun phrase
# rather than a proposition.
PREDICATE_MARKERS = frozenset(
    """
    is are was were be been being has have had shall will would can
    could may might must does do did requires require required
    provides provide provided includes include included states state
    stated appoints appoint appointed holds hold held consists consist
    consisted needs need needed becomes become became remains remain
    gives give given makes make made takes take taken cannot not no
    enjoys enjoy exercises exercise empowered entitled elected
    nominated prescribed declared deemed subject bound liable
    """.split()
)

# The load-bearing concepts of each question, each with its surface
# variants in the corpus.
#
# Probing single surface forms does not work, and getting this wrong
# inverts the conclusion. An early version of this table probed
# "Article 74" and "anti-defection", both of which return zero hits, and
# so classified Q58 and Q59 as unanswerable. The corpus in fact writes
# the same provisions as "Council of Ministers" (27 chunks), "Tenth
# Schedule" (10) and "Speaker" (51). Those questions are answerable and
# their abstentions are engineering defects.
#
# A concept therefore counts as absent only when EVERY variant scores
# zero, which is the only form of this probe that is decisive.
PROBE_CONCEPTS = {
    54: {
        "union list": ["union list"],
        "ratification by states": ["ratification", "ratified"],
    },
    55: {
        "lokpal institution": [
            "lokpal",
            "lokayukta",
            "ombudsman",
        ],
    },
    56: {
        "directive principles": ["directive principles"],
        "fundamental duties": ["fundamental duties"],
    },
    57: {
        "pardoning power": ["pardon", "reprieve", "remission"],
    },
    58: {
        "tenth schedule": ["tenth schedule", "10th schedule"],
        "council of ministers": ["council of ministers"],
    },
    59: {
        "speaker's office": ["speaker"],
        "dissolution of the house": ["dissolution"],
    },
    60: {
        "ordinance power": ["ordinance"],
    },
    61: {
        "governor's discretion": ["discretion"],
        "reservation of state bills": ["reserve", "reserved"],
    },
    62: {
        "governor's immunity": ["governor"],
    },
    63: {
        "panchayat": ["panchayat"],
        "age qualification": [
            "twenty-one years",
            "twenty one years",
            "21 years",
        ],
    },
    64: {
        "national parks": ["national park", "sanctuary"],
        "part c states": ["part c"],
        "north-eastern states": [
            "arunachal",
            "nagaland",
            "tripura",
        ],
    },
    65: {
        "fifth schedule": ["fifth schedule"],
        "scheduled areas": ["scheduled areas"],
    },
    66: {
        "petroleum regulator": [
            "petroleum and natural gas",
            "regulatory board",
        ],
        "crude oil": ["crude oil"],
    },
}


# ==========================================================
# Loading
# ==========================================================


def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_optional(path: str):
    file = Path(path)

    if not file.exists():
        return None

    return json.loads(file.read_text(encoding="utf-8"))


def load_chunk_texts() -> list[str]:
    """Load every chunk's text, lowercased, for the lexical probe."""

    texts = []

    with Path(CHUNKS_PATH).open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()

            if stripped:
                texts.append(json.loads(stripped)["text"].lower())

    return texts


# ==========================================================
# 1. Selective prediction
# ==========================================================


def selective_metrics(
    rows: list[tuple[str | None, str]],
) -> dict:
    """
    Compute coverage, precision-when-answered and overall accuracy.

    Each row is (predicted, official) with predicted None for an
    abstention. Separating these three is the whole point: an abstention
    reduces coverage without being counted as an error, which is what
    makes an abstaining system comparable with one that always guesses.
    """

    total = len(rows)

    answered = [
        (predicted, official)
        for predicted, official in rows
        if predicted is not None
    ]

    correct = sum(
        1
        for predicted, official in answered
        if predicted == official
    )

    wrong = len(answered) - correct

    return {
        "total": total,
        "answered": len(answered),
        "abstained": total - len(answered),
        "correct": correct,
        "wrong": wrong,
        "coverage": len(answered) / total if total else None,
        "precision_when_answered": (
            correct / len(answered) if answered else None
        ),
        "accuracy_overall": correct / total if total else None,
        # The rate at which the system states something false. This is
        # the cost an abstaining architecture is trying to buy down.
        "error_rate_overall": wrong / total if total else None,
    }


def collect_arms(
    verified: list[dict],
    vanilla: list[dict],
    replay: dict | None,
    direct: dict | None,
) -> dict:
    """Assemble one comparable selective-prediction row per arm."""

    arms: dict = {}

    arms["A_vanilla_rag"] = {
        "description": "Retrieve then generate. Always answers.",
        "metrics": selective_metrics(
            [
                (
                    record.get("selected_option"),
                    record["official_answer"],
                )
                for record in vanilla
            ]
        ),
    }

    arms["B_statement_mapping_strict"] = {
        "description": (
            "Verify each statement, map verdicts to an option, "
            "abstain if any statement is unresolved."
        ),
        "metrics": selective_metrics(
            [
                (
                    record.get("predicted_answer"),
                    record["official_answer"],
                )
                for record in verified
            ]
        ),
    }

    if replay:
        for policy in ("strict", "closed_world"):
            result = replay.get(policy)

            if not result:
                continue

            arms[f"B_mapping_fixed_{policy}"] = {
                "description": (
                    "Corrected mapping, verdicts held fixed, "
                    f"{policy} abstention policy."
                ),
                "metrics": selective_metrics(
                    [
                        (
                            row["predicted_answer"],
                            row["official_answer"],
                        )
                        for row in result["rows"]
                    ]
                ),
            }

    if direct:
        for condition, result in direct.get(
            "conditions", {}
        ).items():
            arms[f"B2_direct_option_{condition}"] = {
                "description": (
                    "Verify options directly from evidence, "
                    "declared answer withheld, evidence = "
                    f"{result['evidence']}."
                ),
                "metrics": selective_metrics(
                    [
                        (
                            row["predicted_answer"],
                            row["official_answer"],
                        )
                        for row in result["rows"]
                    ]
                ),
            }

    return arms


# ==========================================================
# 2-3. Verdict distributions
# ==========================================================


def verdict_distributions(verified: list[dict]) -> dict:
    """Count verdicts at claim level and statuses at statement level."""

    claim_verdicts = Counter()
    statement_statuses = Counter()

    for record in verified:
        for verification in record["fact_verifications"].values():
            claim_verdicts[verification["verdict"]] += 1

        statuses = record["claim_mapping"]["statement_statuses"]

        for status in statuses.values():
            statement_statuses[status] += 1

    claim_total = sum(claim_verdicts.values())
    statement_total = sum(statement_statuses.values())

    return {
        "claim_level": {
            "counts": dict(claim_verdicts),
            "total": claim_total,
            "rates": {
                verdict: count / claim_total
                for verdict, count in claim_verdicts.items()
            }
            if claim_total
            else {},
        },
        "statement_level": {
            "counts": dict(statement_statuses),
            "total": statement_total,
            "rates": {
                status: count / statement_total
                for status, count in statement_statuses.items()
            }
            if statement_total
            else {},
        },
    }


# ==========================================================
# 4. Evidence grounding
# ==========================================================


def grounding_metrics(verified: list[dict]) -> dict:
    """
    Check that committed verdicts actually cite evidence.

    A SUPPORTED or CONTRADICTED verdict with no cited page is
    ungrounded: the verifier asserted a conclusion it did not tie to a
    passage. This is the cheapest available check that the citation
    discipline in the prompt is being obeyed.
    """

    committed = 0
    grounded = 0
    pages_cited = []
    ungrounded_examples = []

    for record in verified:
        for claim_id, verification in record[
            "fact_verifications"
        ].items():

            if verification["verdict"] not in COMMITTED_VERDICTS:
                continue

            committed += 1

            pages = verification.get("supporting_pages") or []

            if pages:
                grounded += 1
                pages_cited.append(len(pages))
            else:
                ungrounded_examples.append(
                    {
                        "q_number": record["q_number"],
                        "claim_id": claim_id,
                        "verdict": verification["verdict"],
                    }
                )

    return {
        "committed_verdicts": committed,
        "grounded_verdicts": grounded,
        "evidence_support_rate": (
            grounded / committed if committed else None
        ),
        "mean_pages_per_grounded_verdict": (
            sum(pages_cited) / len(pages_cited)
            if pages_cited
            else None
        ),
        "ungrounded_verdicts": ungrounded_examples,
    }


# ==========================================================
# 5. Claim construction quality
# ==========================================================


def is_propositional(claim_text: str) -> bool:
    """
    True when the claim contains something that could be true or false.

    A lexical proxy: a claim with no predicate marker is almost always a
    bare noun phrase lifted from the question stem. Reported alongside
    the flagged claims so the heuristic can be checked rather than
    trusted.
    """

    tokens = re.findall(r"[a-z]+", claim_text.lower())

    return any(token in PREDICATE_MARKERS for token in tokens)


def claim_construction_metrics(verified: list[dict]) -> dict:
    """
    Measure how often a non-verifiable claim was nonetheless verified.

    The damaging case is a non-propositional claim that came back
    SUPPORTED: the verifier confirmed the corpus mentions a topic, and
    the mapping layer then treated that as the statement being true.
    """

    total = 0
    non_propositional = []

    for record in verified:
        claims_by_id = {
            claim["claim_id"]: claim
            for claim in record["claims"]
        }

        for claim_id, claim in claims_by_id.items():
            total += 1

            if is_propositional(claim["claim"]):
                continue

            verification = record["fact_verifications"].get(
                claim_id, {}
            )

            non_propositional.append(
                {
                    "q_number": record["q_number"],
                    "statement": claim.get("statement_number"),
                    "claim": claim["claim"],
                    "verdict": verification.get("verdict"),
                }
            )

    false_supported = [
        item
        for item in non_propositional
        if item["verdict"] == "SUPPORTED"
    ]

    return {
        "total_claims": total,
        "non_propositional_claims": len(non_propositional),
        "non_propositional_rate": (
            len(non_propositional) / total if total else None
        ),
        # The subset that actively corrupted a downstream decision.
        "false_supported_count": len(false_supported),
        "false_supported_rate": (
            len(false_supported) / total if total else None
        ),
        "questions_affected": sorted(
            {item["q_number"] for item in non_propositional}
        ),
        "flagged": non_propositional,
    }


# ==========================================================
# 6. Corpus answerability
# ==========================================================


def answerability_probe(
    verified: list[dict],
    chunk_texts: list[str],
) -> dict:
    """
    Decide, per question, whether the corpus could have answered at all.

    A term with zero hits across every chunk is unreachable by any
    retriever, so an abstention on that question is a corpus-scope limit
    and not a pipeline defect.
    """

    per_question = {}

    for record in verified:
        q_number = record["q_number"]

        concepts = PROBE_CONCEPTS.get(q_number, {})

        concept_hits = {
            concept: {
                variant: sum(
                    1
                    for text in chunk_texts
                    if variant in text
                )
                for variant in variants
            }
            for concept, variants in concepts.items()
        }

        absent = sorted(
            concept
            for concept, hits in concept_hits.items()
            if not any(hits.values())
        )

        # Every concept absent means the corpus has nothing to say about
        # the question. Some absent means it can settle part of it.
        if absent and len(absent) == len(concepts):
            classification = "OUT_OF_CORPUS"
        elif absent:
            classification = "PARTIAL"
        else:
            classification = "IN_CORPUS"

        per_question[q_number] = {
            "concept_hits": concept_hits,
            "absent_concepts": absent,
            "classification": classification,
            # Presence is only weak evidence of answerability: a term can
            # be mentioned without the passage settling the question.
            # Absence, in contrast, is decisive, so only OUT_OF_CORPUS is
            # treated as a hard corpus limit.
            "corpus_can_answer": classification != "OUT_OF_CORPUS",
        }

    def by_classification(label: str) -> list[int]:
        return sorted(
            q
            for q, info in per_question.items()
            if info["classification"] == label
        )

    return {
        "chunks_probed": len(chunk_texts),
        "method": (
            "Lexical probe over every chunk. A concept is absent only "
            "if all its surface variants score zero. Absence is "
            "decisive (no retriever could find it); presence is only "
            "weak evidence of answerability."
        ),
        "out_of_corpus_questions": by_classification(
            "OUT_OF_CORPUS"
        ),
        "partial_questions": by_classification("PARTIAL"),
        "in_corpus_questions": sorted(
            q
            for q, info in per_question.items()
            if info["corpus_can_answer"]
        ),
        "per_question": per_question,
    }


def answerability_adjusted(
    verified: list[dict],
    probe: dict,
) -> dict:
    """
    Re-score System B on only the questions the corpus can answer.

    Questions whose evidence is absent from the corpus measure the
    dataset, not the system. Reporting both figures keeps the headline
    honest while showing what the pipeline is worth on questions it was
    actually given the evidence for.
    """

    answerable = set(probe["in_corpus_questions"])

    rows = [
        (
            record.get("predicted_answer"),
            record["official_answer"],
        )
        for record in verified
        if record["q_number"] in answerable
    ]

    return {
        "questions_in_scope": sorted(answerable),
        "excluded_as_out_of_corpus": probe[
            "out_of_corpus_questions"
        ],
        "metrics": selective_metrics(rows),
    }


# ==========================================================
# 7. Abstention taxonomy
# ==========================================================


def abstention_taxonomy(
    verified: list[dict],
    probe: dict,
    replay: dict | None,
) -> dict:
    """
    Attribute every abstention to a corpus limit or a pipeline defect.

    This is the metric that changes what you would fix. An abstention
    caused by a corpus limit is correct behaviour; one caused by a
    pipeline defect is a bug with a known location.
    """

    reasons = Counter()

    replay_reasons = {}

    if replay and replay.get("strict"):
        replay_reasons = {
            row["q_number"]: row["abstention_reason"]
            for row in replay["strict"]["rows"]
        }

    per_question = []

    for record in verified:
        q_number = record["q_number"]

        if record.get("predicted_answer") is not None:
            continue

        corpus_limited = q_number in set(
            probe["out_of_corpus_questions"]
        )

        classification = (
            probe["per_question"]
            .get(q_number, {})
            .get("classification")
        )

        if corpus_limited:
            cause = "corpus_limit"
        elif classification == "PARTIAL":
            cause = "partial_corpus_limit"
        else:
            cause = "pipeline_defect"

        reasons[cause] += 1

        per_question.append(
            {
                "q_number": q_number,
                "cause": cause,
                "mapping_reason": replay_reasons.get(q_number),
                "absent_concepts": probe["per_question"]
                .get(q_number, {})
                .get("absent_concepts", []),
            }
        )

    total = sum(reasons.values())

    return {
        "abstentions": total,
        "causes": dict(reasons),
        "correct_abstention_share": (
            reasons["corpus_limit"] / total if total else None
        ),
        "per_question": per_question,
    }


# ==========================================================
# 8. Answer-key verifier
# ==========================================================


def answer_key_metrics(
    verified: list[dict],
    direct: dict | None,
) -> dict:
    """
    Score the answer-key verifier, and mark it as contaminated.

    This component's prompt contains DECLARED ANSWER, which the PYQ
    harness sets to the official key, so it is told the answer it is
    being scored on identifying. Its agreement rate is therefore an
    upper bound inflated by leakage and is NOT a result. It is reported
    only so the leak-free arm has something to be compared against.
    """

    verdicts = Counter()
    picked_official = 0

    for record in verified:
        verification = record.get("answer_verification") or {}

        verdicts[verification.get("verdict", "MISSING")] += 1

        supported = verification.get("supported_options") or []

        if supported == [record["official_answer"]]:
            picked_official += 1

    total = len(verified)

    leak_free = None

    if direct:
        leak_free = {
            condition: result["accuracy_overall"]
            for condition, result in direct.get(
                "conditions", {}
            ).items()
        }

    return {
        "CONTAMINATED": True,
        "contamination": (
            "The prompt in src/verification/answer_key_verifier.py "
            "includes the declared answer, which pyq_to_mcq sets to "
            "the official UPSC answer. Rule 7 instructs the model not "
            "to rely on it, but the correct letter is in the context "
            "window. These figures must not be reported as a result."
        ),
        "verdicts": dict(verdicts),
        "selected_exactly_the_official_option": picked_official,
        "apparent_agreement_rate": (
            picked_official / total if total else None
        ),
        "leak_free_accuracy": leak_free,
        "interpretation": (
            "Compare apparent_agreement_rate against "
            "leak_free_accuracy. The gap is the size of the leak, not "
            "a measure of verification quality."
        ),
    }


# ==========================================================
# Reporting
# ==========================================================


def ascii_safe(text: str) -> str:
    """
    Strip characters the Windows console cannot encode.

    The corpus contains mojibake from PDF extraction, so claim text can
    carry bytes that crash a cp1252 stdout. The JSON output keeps the
    original text; only the console view is reduced.
    """

    return text.encode("ascii", "replace").decode("ascii")


def percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def number(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def print_report(report: dict) -> None:
    print()
    print("=" * 72)
    print("SYSTEM B QUANTITATIVE METRICS")
    print("=" * 72)

    print()
    print("--- 1. SELECTIVE PREDICTION (all arms) ---")
    print()

    header = (
        f"{'arm':<34}{'cov':>9}{'prec':>9}"
        f"{'acc':>9}{'err':>9}"
    )
    print(header)
    print("-" * len(header))

    for name, arm in report["arms"].items():
        metrics = arm["metrics"]

        print(
            f"{name:<34}"
            f"{percent(metrics['coverage']):>9}"
            f"{percent(metrics['precision_when_answered']):>9}"
            f"{percent(metrics['accuracy_overall']):>9}"
            f"{percent(metrics['error_rate_overall']):>9}"
        )

    print()
    print(
        "cov = coverage, prec = precision when answered, "
        "acc = accuracy over all questions,"
    )
    print("err = rate of stating a wrong answer.")

    distributions = report["verdict_distributions"]

    print()
    print("--- 2. VERDICT DISTRIBUTION ---")
    print()
    print(
        f"claim level    (n={distributions['claim_level']['total']}): "
        f"{distributions['claim_level']['counts']}"
    )
    print(
        f"statement level(n="
        f"{distributions['statement_level']['total']}): "
        f"{distributions['statement_level']['counts']}"
    )

    grounding = report["grounding"]

    print()
    print("--- 3. EVIDENCE GROUNDING ---")
    print()
    print(
        f"committed verdicts (SUPPORTED/CONTRADICTED): "
        f"{grounding['committed_verdicts']}"
    )
    print(
        f"evidence support rate: "
        f"{percent(grounding['evidence_support_rate'])}"
    )
    print(
        f"mean pages cited per grounded verdict: "
        f"{number(grounding['mean_pages_per_grounded_verdict'])}"
    )
    print(
        f"ungrounded committed verdicts: "
        f"{len(grounding['ungrounded_verdicts'])}"
    )

    construction = report["claim_construction"]

    print()
    print("--- 4. CLAIM CONSTRUCTION QUALITY ---")
    print()
    print(f"total claims: {construction['total_claims']}")
    print(
        f"non-propositional claims: "
        f"{construction['non_propositional_claims']} "
        f"({percent(construction['non_propositional_rate'])})"
    )
    print(
        f"of those, returned SUPPORTED (false SUPPORTED): "
        f"{construction['false_supported_count']} "
        f"({percent(construction['false_supported_rate'])})"
    )
    print(f"questions affected: {construction['questions_affected']}")
    print()
    print("flagged claims (a bare noun phrase has no truth value):")

    for item in construction["flagged"]:
        print(
            f"  Q{item['q_number']} stmt {item['statement']}: "
            f"{item['verdict']}"
        )
        print(f"      {ascii_safe(item['claim'])[:96]}")

    probe = report["answerability"]

    print()
    print("--- 5. CORPUS ANSWERABILITY ---")
    print()
    print(f"chunks probed: {probe['chunks_probed']}")
    print(
        f"OUT_OF_CORPUS (no concept present): "
        f"{probe['out_of_corpus_questions']}"
    )
    print(
        f"PARTIAL (some concepts absent): "
        f"{probe['partial_questions']}"
    )
    print()

    for q_number, info in sorted(probe["per_question"].items()):

        if not info["absent_concepts"]:
            continue

        print(
            f"  Q{q_number} [{info['classification']}]: "
            f"zero hits for {info['absent_concepts']}"
        )

    adjusted = report["answerability_adjusted"]

    print()
    print("answerability-adjusted System B (strict policy):")
    print(
        f"  scope: {len(adjusted['questions_in_scope'])} questions "
        f"(excluded {adjusted['excluded_as_out_of_corpus']})"
    )
    print(
        f"  coverage {percent(adjusted['metrics']['coverage'])}   "
        f"precision "
        f"{percent(adjusted['metrics']['precision_when_answered'])}   "
        f"accuracy "
        f"{percent(adjusted['metrics']['accuracy_overall'])}"
    )

    taxonomy = report["abstention_taxonomy"]

    print()
    print("--- 6. ABSTENTION TAXONOMY ---")
    print()
    print(f"abstentions: {taxonomy['abstentions']}")
    print(f"causes: {taxonomy['causes']}")
    print(
        f"share that is CORRECT behaviour (corpus limit): "
        f"{percent(taxonomy['correct_abstention_share'])}"
    )
    print()

    for item in taxonomy["per_question"]:
        print(
            f"  Q{item['q_number']:<3} {item['cause']:<16} "
            f"mapping={item['mapping_reason']}"
        )

    answer_key = report["answer_key_verifier"]

    print()
    print("--- 7. ANSWER-KEY VERIFIER (CONTAMINATED) ---")
    print()
    print(f"verdicts: {answer_key['verdicts']}")
    print(
        f"apparent agreement with official key: "
        f"{percent(answer_key['apparent_agreement_rate'])}"
    )
    print(f"leak-free accuracy: {answer_key['leak_free_accuracy']}")
    print()
    print(
        "WARNING: the apparent rate is inflated by answer leakage "
        "and is not a result."
    )


def main() -> None:
    verified = load_json(VERIFIED_PATH)
    vanilla = load_json(VANILLA_PATH)
    replay = load_optional(REPLAY_PATH)
    direct = load_optional(DIRECT_PATH)

    chunk_texts = load_chunk_texts()

    probe = answerability_probe(verified, chunk_texts)

    report = {
        "dataset": "UPSC 2025 Prelims Polity, Q54-Q66",
        "questions": len(verified),
        "arms": collect_arms(verified, vanilla, replay, direct),
        "verdict_distributions": verdict_distributions(verified),
        "grounding": grounding_metrics(verified),
        "claim_construction": claim_construction_metrics(verified),
        "answerability": probe,
        "answerability_adjusted": answerability_adjusted(
            verified, probe
        ),
        "abstention_taxonomy": abstention_taxonomy(
            verified, probe, replay
        ),
        "answer_key_verifier": answer_key_metrics(
            verified, direct
        ),
    }

    print_report(report)

    Path(OUTPUT_PATH).write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print(f"Saved metrics to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
