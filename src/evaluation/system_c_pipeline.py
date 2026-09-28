"""
System C: re-run claim verification end to end and ablate each fix.

Purpose
-------

Three defects were diagnosed by inspection, and inspection is not
evidence. This module measures them. Each arm differs from its named
comparison arm in exactly one variable, so the resulting table attributes
a change in accuracy to a specific fix rather than to the combination:

    C0   stored claims, k=3, base prompt        compare: -
         The reference point. Uses the same claim text System B verified,
         so the only difference from System B is the retriever, which
         isolates and prices that substitution.

    C1   bound claims,  k=3, base prompt        compare: C0
         Predicate binding, at the depth the pipeline actually used.

    C1d  stored claims, k=5, base prompt        compare: C0
         Retrieval depth alone, with claim text held fixed. The retrieval
         benchmark optimised Recall@5, but claim verification queried at
         top_k=3, so the depth the retriever was tuned for was never
         actually used.

    C2   bound claims,  k=5, base prompt        compare: C1d
         Predicate binding again, now at depth 5. C1d exists so that
         binding and depth are never varied together; without it the
         C0-to-C2 gap would conflate the two and neither could be
         credited.

    C3   bound claims,  k=5, numeric prompt     compare: C2
         Two rules for numeric conflicts. Q63's item II claims "thirty
         years" where the corpus states twenty-one; that is an explicit
         conflict, so it should read CONTRADICTED, not INSUFFICIENT.

    C3d  stored claims, k=5, numeric prompt     compare: C1d
         The numeric rules without binding. Added after the first run
         showed binding costing precision: if binding is a regression,
         then the best configuration is depth plus numeric rules, and
         that arm has to be measured rather than assumed.

Every arm is then scored under all three abstention policies, because a
change in verdicts can move coverage and precision in opposite
directions and a single accuracy number would hide that.

Which retriever the arms run on
------------------------------

`RETRIEVER` selects the retrieval stage, and the choice changes what the
table can be compared against.

The first full ablation was measured on a standard-library BM25 index,
because `sentence-transformers` could not be installed at the time. That
made every arm-to-arm comparison sound - the retriever was held constant
across them - but it made the A-versus-C rows invalid, because System A
ran the real cross-encoder reranker. A one-question accuracy lead
measured across two different retrieval stages is not a result, and the
write-up had to carry that caveat.

The dependency is installable now (see `src/tls_trust.py` for what was
actually blocking it: TLS interception, not the package index), so the
default is `reranker`: the same semantic-plus-cross-encoder stage the
system ships and System B used. `bm25` stays reachable, with its own
cache and output file, so the earlier numbers remain reproducible rather
than being quietly replaced.

Note that the two runs are not expected to agree. Swapping the retriever
changes the evidence, which changes the verdicts, which changes the
answers. Whether the conclusions drawn from the BM25 run survive on the
real retriever is a question this harness is built to answer, not one to
assume either way.

What the answer turned out to be
--------------------------------

They did not survive. Three conclusions the BM25 run supported are
retracted in EVALUATION.md sections 4.2, 4.3 and 4.5:

  - Depth 3-to-5 was reported as the project's largest accuracy effect.
    On this retriever its sign depends on which label set you read, and
    the whole of it runs through one statement (Q58 item I).

  - The numeric rules were reported as fixing Q63 item II. Here C2-to-C3
    and C1d-to-C3d change no verdict at all. The rules stay in the prompt
    as harmless, not as measured.

  - C3d was reported as beating C3 on error rate, which made the arm not
    shipped look better. Here C3 and C3d are identical on every
    end-to-end metric and differ only in claim soundness.

Read this whole table as null results, for a reason that is not about the
retriever. verdict_stability.py resamples the verifier and finds an
accuracy spread of 15.38 points - two questions - across independent
repeats at temperature 0. Every arm-to-arm difference this harness
produces is 7.69 points or less, which is inside that band. The arms
cannot be distinguished end to end at n=13.

The columns that can be read are the claim-level ones: propositional rate
and false-SUPPORTED count, measured over 37 claims with no dependence on
the gold labels. Binding moves those from 70.27% and 8 to 100% and 0 at
both depths, which is the signal the shipped configuration rests on.

This harness pins temperature via post_chat_completion and always did, so
it is unaffected by the defect described in FactVerifier.verify - but it
is subject to the same residual instability, which is where the band
above comes from.

The absence check is deliberately different from the shipped one in both
modes. The original NegativeClaimChecker retrieved the top 5 BM25 hits
for the quoted term and then substring-matched within them, which can
report absence for a term that is present but ranked sixth. A lexical
absence check has no reason to be approximate when the whole corpus is in
memory, so this one scans every chunk.

Verdicts are cached on disk by (claim, prompt variant, evidence pages),
so arms that share a configuration never pay twice and a rerun after a
reporting change costs nothing.
"""

import hashlib
import json
import os
import re
import time
from pathlib import Path

from src.evaluation.answer_mapping import (
    CLOSED_WORLD,
    ELIMINATION,
    STRICT,
    map_answer,
)
from src.evaluation.judge_llm import load_api_key, post_chat_completion
from src.evaluation.label_audit import audit_status, corrected_label
from src.evaluation.system_b_metrics import selective_metrics
from src.retrieval.lexical_index import LexicalIndex, load_chunks
from src.verification.claim_builder import build_claims_for_question
from src.verification.question_parser import is_propositional


MODEL_NAME = "gpt-4o-mini"

# The record of what System B actually verified: verbatim statements at
# top_k=3. This is deliberately NOT verified_pyq_results.json.
#
# That file is the live output of evaluate_verified_pyqs, and re-running
# that module under the fixed defaults rewrote it with *bound* claims. The
# CLAIMS_STORED arms read their claim strings from here, so pointing this
# at the live file would silently make C0 identical to C1, C1d identical
# to C2 and C3d identical to C3 - six arms reporting as if they were six
# configurations while actually measuring three. The single-variable
# structure the whole table rests on would be gone, and nothing in the
# output would look wrong. `check_records` below asserts against exactly
# that, because a guard is worth more here than a comment.
RECORDS_PATH = "data/evaluation/verified_pyq_results.k3_verbatim.json"

VANILLA_PATH = "data/evaluation/vanilla_rag_scored.json"

# Which retrieval stage the arms run on. `reranker` is the one the system
# ships (semantic candidates, cross-encoder rerank) and is the default;
# `bm25` is the standard-library stand-in this harness was originally
# written against, kept reachable because the first full ablation was
# measured on it and those numbers must remain reproducible.
#
# Overridable from the environment so both can be produced without
# editing the file: SYSTEM_C_RETRIEVER=bm25 python -m ...
RETRIEVER_RERANKER = "reranker"
RETRIEVER_BM25 = "bm25"

RETRIEVER = os.getenv("SYSTEM_C_RETRIEVER", RETRIEVER_RERANKER)

if RETRIEVER not in (RETRIEVER_RERANKER, RETRIEVER_BM25):
    raise ValueError(
        f"SYSTEM_C_RETRIEVER must be "
        f"{RETRIEVER_RERANKER!r} or {RETRIEVER_BM25!r}, "
        f"got {RETRIEVER!r}"
    )

# Cache and output are keyed by retriever. Sharing one cache across
# retrievers would be safe, because the cache key already includes the
# evidence ids and different retrievers return different evidence, but
# sharing one *output* file would silently overwrite one run's table with
# the other's and make the caveat in the write-up unverifiable.
_SUFFIX = "" if RETRIEVER == RETRIEVER_RERANKER else f".{RETRIEVER}"

CACHE_PATH = (
    f"data/evaluation/system_c_verdict_cache{_SUFFIX}.json"
)
OUTPUT_PATH = f"data/evaluation/system_c_results{_SUFFIX}.json"

CLAIMS_STORED = "stored"
CLAIMS_BOUND = "bound"

PROMPT_BASE = "base"
PROMPT_NUMERIC = "numeric"

POLICIES = (STRICT, ELIMINATION, CLOSED_WORLD)

# The vanilla RAG baseline, scored under both label sets.
#
# This was a hard-coded 5/13 = 38.46%, which is the stored-label figure.
# Every arm below is reported under stored AND audited labels, so a single
# hard-coded baseline was only comparable to half the table, and the half
# it was not comparable to is the half quoted in the write-up. Under
# audited labels the baseline is 6/13 = 46.15%, because vanilla RAG
# answered D on Q58 and the audit makes D correct.
#
# This matters to the project's headline: against the correct baseline,
# the best verified arm leads by one question, not by two. Computing it
# from the same label function the arms use is the only way that stays
# true if the audit changes again.


def baseline_metrics() -> dict:
    """Selective-prediction metrics for vanilla RAG, per label set."""

    payload = json.loads(
        Path(VANILLA_PATH).read_text(encoding="utf-8")
    )

    rows = payload["results"]

    return {
        "stored_labels": selective_metrics(
            [
                (record["predicted_answer"], record["official_answer"])
                for record in rows
            ]
        ),
        "audited_labels": selective_metrics(
            [
                (
                    record["predicted_answer"],
                    corrected_label(
                        record["q_number"],
                        record["official_answer"],
                    ),
                )
                for record in rows
            ]
        ),
    }


MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 4

ARMS = (
    {
        "name": "C0",
        "label": "stored claims, k=3, base prompt",
        "claims": CLAIMS_STORED,
        "top_k": 3,
        "prompt": PROMPT_BASE,
        "isolates": "reference point; differs from System B only in retriever",
        "compare_to": None,
    },
    {
        "name": "C1",
        "label": "bound claims, k=3, base prompt",
        "claims": CLAIMS_BOUND,
        "top_k": 3,
        "prompt": PROMPT_BASE,
        "isolates": "predicate binding at k=3",
        "compare_to": "C0",
    },
    {
        "name": "C1d",
        "label": "stored claims, k=5, base prompt",
        "claims": CLAIMS_STORED,
        "top_k": 5,
        "prompt": PROMPT_BASE,
        "isolates": "retrieval depth 3 -> 5, claims held at stored",
        "compare_to": "C0",
    },
    {
        "name": "C2",
        "label": "bound claims, k=5, base prompt",
        "claims": CLAIMS_BOUND,
        "top_k": 5,
        "prompt": PROMPT_BASE,
        "isolates": "predicate binding at k=5 (compare against C1d)",
        "compare_to": "C1d",
    },
    {
        "name": "C3",
        "label": "bound claims, k=5, numeric prompt",
        "claims": CLAIMS_BOUND,
        "top_k": 5,
        "prompt": PROMPT_NUMERIC,
        "isolates": "numeric conflict detection",
        "compare_to": "C2",
    },
    {
        "name": "C3d",
        "label": "stored claims, k=5, numeric prompt",
        "claims": CLAIMS_STORED,
        "top_k": 5,
        "prompt": PROMPT_NUMERIC,
        "isolates": "numeric conflict detection without binding",
        "compare_to": "C1d",
    },
)

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT"],
        },
        "reasoning": {"type": "string"},
        "supporting_pages": {
            "type": "array",
            "items": {"type": "integer"},
        },
    },
    "required": ["verdict", "reasoning", "supporting_pages"],
    "additionalProperties": False,
}

# Copied verbatim from FactVerifier so that a verdict difference between
# System B and System C cannot be caused by a reworded prompt. The length
# is itself a finding: rules 21 to 24 were appended to suppress false
# CONTRADICTED verdicts, which is prompt accretion standing in for a
# scope check the pipeline never performed.
BASE_RULES = """
1. Use ONLY the provided source evidence.

2. Do not use outside knowledge.

3. Do not infer facts that are not supported by the evidence.

4. Do not assume that semantically related text proves the claim.

5. Return SUPPORTED only when the provided evidence explicitly
   establishes the complete claim.

6. Do not use background knowledge to connect separate facts.

7. Do not infer missing constitutional Articles, provisions,
   dates, relationships, authorities, or conclusions.

8. If the claim requires information that is not explicitly
   present in the evidence, return INSUFFICIENT.

9. Return CONTRADICTED only when the provided evidence explicitly
   establishes information that conflicts with the claim.

10. If the evidence is relevant but insufficient to establish
    the complete claim, return INSUFFICIENT.

11. Cite only pages that actually support your verdict.

12. A semantically related passage is not sufficient evidence.

13. Be especially careful with negative or absence claims.

14. If the claim says that something does not exist, is not
    mentioned, is absent, or is not provided for, failure to
    find that information in the supplied evidence does NOT
    prove the claim.

15. For an absence claim, return INSUFFICIENT unless the
    provided evidence explicitly establishes the absence.

16. Do not treat "the evidence does not mention X" as evidence
    that "the Constitution does not mention X".

17. Do not assume that the retrieved top-k evidence represents
    the entire Constitution or the entire knowledge base.

18. For multi-part claims, verify the entire claim. If only part
    of the claim is supported, return INSUFFICIENT unless the
    evidence explicitly contradicts the complete claim.
19. Do not use the wording of the claim itself as evidence.
20. Evaluate the claim within its exact scope and subject.
21. Evidence about a different constitutional institution, office,
    House, legislature, authority, or category must not be treated
    as contradictory merely because it states a different rule.
22. Do not construct a contradiction by comparing the claim with
    a different constitutional provision unless the evidence
    explicitly states that the claimed rule is not applicable.
23. When one passage directly supports the claim and another passage
    concerns a different scope or institution, treat the latter as
    irrelevant rather than contradictory.
24. For a claim about Parliament or the House of the People, do not
    use provisions concerning State Legislatures as contradictory
    evidence unless the claim itself covers both.
"""

# The numeric arm's only change. Phrased to require a *stated* conflicting
# value, so it can never turn a silence into a contradiction; that would
# reintroduce the closed-world assumption at the verdict level, where it
# would be far harder to see than in the mapping policy.
NUMERIC_RULES = """
25. If the claim states a specific number, age, duration, fraction,
    majority or threshold, and the evidence states a different value
    for the same provision and the same subject, return CONTRADICTED.
    The evidence must actually state a value; the absence of a value
    is INSUFFICIENT, never a contradiction.

26. When checking a number, confirm the evidence concerns the same
    office, body or provision as the claim before treating the values
    as comparable.
"""


def rules_for(variant: str) -> str:
    if variant == PROMPT_NUMERIC:
        return BASE_RULES + NUMERIC_RULES

    return BASE_RULES


def build_prompt(claim: str, evidence: "list[dict]", variant: str) -> str:
    """Render the verification prompt for one claim."""

    blocks = [
        f"\nEVIDENCE {rank}\n"
        f"Source: {item['source']}\n"
        f"Page: {item['page']}\n\n"
        f"{item['text']}\n"
        for rank, item in enumerate(evidence, start=1)
    ]

    return f"""
You are a factual verification component for a UPSC Indian
Polity MCQ system.

Your task is to determine whether the provided source evidence
supports, contradicts, or is insufficient to verify the claim.

IMPORTANT RULES:
{rules_for(variant)}
CLAIM:

{claim}

SOURCE EVIDENCE:

{"".join(blocks)}
"""


ABSENCE_PATTERNS = (
    r"no mention of (?:the word|the term)?\s*[\"'‘“]"
    r"([^\"'’”]+)[\"'’”]",
    r"does not mention (?:the word|the term)?\s*[\"'‘“]"
    r"([^\"'’”]+)[\"'’”]",
    r"does not contain (?:the word|the term)?\s*[\"'‘“]"
    r"([^\"'’”]+)[\"'’”]",
    r"not mentioned (?:as|by)?\s*[\"'‘“]"
    r"([^\"'’”]+)[\"'’”]",
)


class ExactAbsenceChecker:
    """
    Resolve explicit lexical absence claims against the whole corpus.

    Replaces NegativeClaimChecker's top-5 retrieval with a full scan.
    Retrieval depth is a relevance heuristic and has no business deciding
    a question of literal presence: a term ranked sixth is still in the
    corpus, so the original could report absence for a term that is
    demonstrably present. That is a false SUPPORTED on exactly the claim
    type the component was added to make safe.

    The patterns are copied from NegativeClaimChecker rather than
    imported. That started as a workaround - the module imports
    BM25Retriever at import time and rank_bm25 was not installable - and
    is kept deliberately now that it is: the copy is what guarantees the
    two definitions of "absence claim" stay identical while only the
    retrieval step differs, which is the whole point of the comparison.

    Takes anything exposing `chunks`, so it works unchanged whichever
    retrieval stage the arms are running on.
    """

    def __init__(self, index) -> None:
        self.index = index

    def target_term(self, claim: str) -> "str | None":

        for pattern in ABSENCE_PATTERNS:

            match = re.search(pattern, claim, flags=re.IGNORECASE)

            if match:
                return match.group(1).strip()

        return None

    def check(self, claim: str) -> "dict | None":
        """Return a verdict for an absence claim, or None if not one."""

        term = self.target_term(claim)

        if not term:
            return None

        needle = term.lower()

        pages = sorted(
            {
                chunk["page"]
                for chunk in self.index.chunks
                if needle in chunk["text"].lower()
            }
        )

        if pages:
            return {
                "verdict": "CONTRADICTED",
                "reasoning": (
                    f"The claim asserts that '{term}' is absent, but a "
                    f"full-corpus scan finds it on "
                    f"{len(pages)} page(s)."
                ),
                "supporting_pages": pages[:5],
                "resolved_by": "exact_absence_scan",
            }

        return {
            "verdict": "SUPPORTED",
            "reasoning": (
                f"A full-corpus scan finds no occurrence of '{term}', "
                f"so the absence is established over this corpus."
            ),
            "supporting_pages": [],
            "resolved_by": "exact_absence_scan",
        }


class RerankerIndex:
    """
    The shipped retrieval stage, behind LexicalIndex's interface.

    Exposes exactly what this harness uses - `chunks` for the absence
    scan and `retrieve(query, top_k)` for evidence - so the arms are
    retriever-agnostic and the only thing that changes between a BM25 run
    and a reranker run is which object is constructed.

    Loading is deferred to construction time rather than import time
    because importing SemanticReranker pulls in torch, and the BM25 arms
    should not pay a multi-second import to run.
    """

    def __init__(self, chunks: "list[dict]") -> None:
        from src.retrieval.semantic_reranker import SemanticReranker

        self.chunks = chunks
        self.retriever = SemanticReranker(chunks, candidate_k=20)

    def retrieve(self, query: str, top_k: int = 5) -> "list[dict]":
        return self.retriever.retrieve(query, top_k=top_k)


def build_index(chunks: "list[dict]"):
    """Construct the retrieval stage named by RETRIEVER."""

    if RETRIEVER == RETRIEVER_RERANKER:
        return RerankerIndex(chunks)

    return LexicalIndex(chunks)


# Recorded in the output so a reader of the JSON can tell which stage
# produced it without having to know the environment it ran in.
RETRIEVER_DESCRIPTIONS = {
    RETRIEVER_RERANKER: (
        "semantic (BAAI/bge-small-en-v1.5) -> cross-encoder rerank "
        "(cross-encoder/ms-marco-MiniLM-L-6-v2), candidate_k=20"
    ),
    RETRIEVER_BM25: "stdlib BM25 (LexicalIndex)",
}

RETRIEVER_CAVEATS = {
    RETRIEVER_RERANKER: (
        "This is the retrieval stage the system ships and the one System "
        "B used, so C0 is a genuine reference point for System B and the "
        "A-versus-C rows are retriever-matched."
    ),
    RETRIEVER_BM25: (
        "System B used a cross-encoder reranker over semantic "
        "candidates. This run substitutes a standard-library BM25 index, "
        "so C0 is a reference point, not a replication of System B; only "
        "C0-to-C3 comparisons hold the retriever constant."
    ),
}


class VerdictCache:
    """Disk cache so arms sharing a configuration never verify twice."""

    def __init__(self, path: str = CACHE_PATH) -> None:
        self.path = Path(path)

        self.entries: "dict[str, dict]" = (
            json.loads(self.path.read_text(encoding="utf-8"))
            if self.path.exists()
            else {}
        )

        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(claim: str, variant: str, evidence: "list[dict]") -> str:

        payload = json.dumps(
            {
                "claim": claim,
                "variant": variant,
                "evidence": [item["id"] for item in evidence],
            },
            sort_keys=True,
        )

        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def get(self, key: str) -> "dict | None":
        entry = self.entries.get(key)

        if entry is None:
            self.misses += 1
        else:
            self.hits += 1

        return entry

    def put(self, key: str, value: dict) -> None:
        self.entries[key] = value

    def save(self) -> None:
        self.path.write_text(
            json.dumps(self.entries, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def verify_claim(
    api_key: str,
    claim: str,
    evidence: "list[dict]",
    variant: str,
    cache: VerdictCache,
) -> dict:
    """
    Verify one claim, retrying transient failures on the same model.

    The model is never downgraded on failure. A silent switch would make
    one arm's verdicts incomparable with another's, which is precisely
    what this harness exists to compare.
    """

    key = cache.key(claim, variant, evidence)
    cached = cache.get(key)

    if cached is not None:
        return cached

    prompt = build_prompt(claim, evidence, variant)
    last_error: "Exception | None" = None

    for attempt in range(1, MAX_ATTEMPTS + 1):

        try:
            result = post_chat_completion(
                api_key,
                MODEL_NAME,
                prompt,
                "fact_verification",
                VERDICT_SCHEMA,
            )

            result["resolved_by"] = "llm"
            cache.put(key, result)

            return result

        except Exception as error:  # noqa: BLE001 - reported, not hidden
            last_error = error

            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    raise RuntimeError(
        f"Verification failed after {MAX_ATTEMPTS} attempts: {last_error}"
    )


def arm_claims(record: dict, mode: str) -> "list[dict]":
    """
    Produce the claims one arm will verify.

    CLAIMS_STORED replays the exact strings System B verified, which is
    what makes C0 a reference point rather than a fourth variant.
    """

    if mode == CLAIMS_STORED:
        return [
            {
                "statement": claim["statement_number"].upper(),
                "claim": claim["claim"],
                "binding": "stored",
            }
            for claim in record["claims"]
            if claim.get("statement_number")
        ]

    _, built = build_claims_for_question(record["question"])

    return [
        {
            "statement": claim.label,
            "claim": claim.claim,
            "binding": claim.binding,
        }
        for claim in built
    ]


def check_records(records: "list[dict]") -> None:
    """
    Fail loudly if the stored claims are not the verbatim ones.

    The stored and bound claim sets have to differ, or the arms that vary
    only claim mode are not varying anything. This is the failure mode
    RECORDS_PATH's comment describes: it produces a complete, plausible,
    internally consistent six-arm table in which three pairs of arms are
    the same experiment run twice. Nothing downstream can detect that, so
    it is detected here.
    """

    differing = sum(
        1
        for record in records
        if [row["claim"] for row in arm_claims(record, CLAIMS_STORED)]
        != [row["claim"] for row in arm_claims(record, CLAIMS_BOUND)]
    )

    assert differing, (
        f"stored and bound claims are identical for all "
        f"{len(records)} questions: {RECORDS_PATH} does not hold "
        f"verbatim claims, so the claim-mode arms measure nothing"
    )

    print(
        f"stored vs bound claims differ on {differing}/{len(records)} "
        f"questions"
    )


def statement_statuses(rows: "list[dict]") -> "dict[str, str]":
    """
    Aggregate claim verdicts into one status per numbered statement.

    Mirrors determine_statement_statuses: any CONTRADICTED makes the
    statement CONTRADICTED, all SUPPORTED makes it SUPPORTED, anything
    else is INSUFFICIENT.
    """

    grouped: "dict[str, list[str]]" = {}

    for row in rows:
        grouped.setdefault(row["statement"], []).append(row["verdict"])

    statuses = {}

    for statement, verdicts in grouped.items():

        if any(verdict == "CONTRADICTED" for verdict in verdicts):
            statuses[statement] = "CONTRADICTED"

        elif all(verdict == "SUPPORTED" for verdict in verdicts):
            statuses[statement] = "SUPPORTED"

        else:
            statuses[statement] = "INSUFFICIENT"

    return statuses


def run_arm(
    arm: dict,
    records: "list[dict]",
    index,
    absence: ExactAbsenceChecker,
    api_key: str,
    cache: VerdictCache,
) -> dict:
    """Verify every claim for one arm and score it under every policy."""

    questions = []

    for record in records:

        claim_rows = []

        for entry in arm_claims(record, arm["claims"]):

            resolved = absence.check(entry["claim"])

            if resolved is not None:
                evidence: "list[dict]" = []
                verdict = resolved

            else:
                evidence = index.retrieve(
                    entry["claim"], top_k=arm["top_k"]
                )
                verdict = verify_claim(
                    api_key,
                    entry["claim"],
                    evidence,
                    arm["prompt"],
                    cache,
                )

            claim_rows.append(
                {
                    **entry,
                    "verdict": verdict["verdict"],
                    "reasoning": verdict["reasoning"],
                    "supporting_pages": verdict["supporting_pages"],
                    "resolved_by": verdict.get("resolved_by", "llm"),
                    "evidence_pages": [
                        item["page"] for item in evidence
                    ],
                }
            )

        statuses = statement_statuses(claim_rows)

        predictions = {}

        for policy in POLICIES:

            predicted, debug = map_answer(
                statement_statuses=statuses,
                options=record["options"],
                question_text=record["question"],
                policy=policy,
            )

            predictions[policy] = {
                "predicted": predicted,
                "abstention_reason": debug["abstention_reason"],
            }

        questions.append(
            {
                "q_number": record["q_number"],
                "official_answer": record["official_answer"],
                "audited_answer": corrected_label(
                    record["q_number"], record["official_answer"]
                ),
                "label_audit_status": audit_status(record["q_number"]),
                "statement_statuses": statuses,
                "claims": claim_rows,
                "predictions": predictions,
            }
        )

    # Scored twice against the same predictions. The stored labels are
    # kept because every earlier number in this project was measured
    # against them and dropping them would make the history
    # incomparable; the audited labels are the ones to believe.
    policy_metrics = {
        label_set: {
            policy: selective_metrics(
                [
                    (
                        question["predictions"][policy]["predicted"],
                        question[key],
                    )
                    for question in questions
                ]
            )
            for policy in POLICIES
        }
        for label_set, key in (
            ("stored_labels", "official_answer"),
            ("audited_labels", "audited_answer"),
        )
    }

    return {
        **{
            k: arm[k]
            for k in ("name", "label", "isolates", "compare_to")
        },
        "claims_mode": arm["claims"],
        "top_k": arm["top_k"],
        "prompt": arm["prompt"],
        "verdict_counts": verdict_counts(questions),
        "claim_soundness": claim_soundness(questions),
        "policy_metrics": policy_metrics,
        "questions": questions,
    }


def claim_soundness(questions: "list[dict]") -> dict:
    """
    Measure claim quality directly, independently of the answer.

    Answer accuracy over 13 questions is a lagging indicator with almost
    no resolution: one question is 7.7 points, so a real improvement and
    a coin flip look identical. Claim soundness is measured over 37 claims
    and does not depend on the mapping layer or the answer key at all.

    The number that matters is `false_supported`: a claim with no truth
    value cannot be supported by anything, so a SUPPORTED verdict on one
    is wrong by construction. Unlike an ordinary wrong verdict this needs
    no gold label to identify, which makes it the one quality signal here
    that cannot be contaminated by a bad label.
    """

    total = 0
    non_propositional = 0
    false_supported = 0

    for question in questions:
        for claim in question["claims"]:

            total += 1

            if is_propositional(claim["claim"]):
                continue

            non_propositional += 1

            if claim["verdict"] == "SUPPORTED":
                false_supported += 1

    return {
        "claims": total,
        "non_propositional": non_propositional,
        "propositional_rate": (
            (total - non_propositional) / total if total else None
        ),
        "false_supported": false_supported,
    }


def verdict_counts(questions: "list[dict]") -> "dict[str, int]":
    """Claim-level verdict distribution for one arm."""

    counts: "dict[str, int]" = {}

    for question in questions:
        for claim in question["claims"]:
            counts[claim["verdict"]] = (
                counts.get(claim["verdict"], 0) + 1
            )

    return counts


def percent(value) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def print_arm(arm_result: dict) -> None:

    print(f"--- {arm_result['name']}: {arm_result['label']} ---")
    print(f"    isolates: {arm_result['isolates']}")
    print(f"    verdicts: {arm_result['verdict_counts']}")

    soundness = arm_result["claim_soundness"]
    print(
        f"    claims:   {soundness['claims']}  "
        f"propositional {percent(soundness['propositional_rate'])}  "
        f"false SUPPORTED {soundness['false_supported']}"
    )

    for label_set, policies in arm_result["policy_metrics"].items():

        print(f"    {label_set}:")

        for policy, metrics in policies.items():
            print(
                f"      {policy:<13} "
                f"coverage {percent(metrics['coverage']):>7}  "
                f"precision "
                f"{percent(metrics['precision_when_answered']):>7}"
                f"  accuracy {percent(metrics['accuracy_overall']):>7}"
                f"  wrong {percent(metrics['error_rate_overall']):>7}"
            )

    print()


def print_verdict_changes(
    baseline: dict,
    arm_result: dict,
) -> None:
    """
    Show which statements changed verdict between two arms.

    The pair compared is always one that differs in a single variable, so
    these lines are the evidence that a specific fix did something, as
    distinct from the aggregate accuracy moving for unrelated reasons. A
    fix that improves accuracy while changing no verdict would mean the
    gain came from somewhere else.
    """

    print(f"--- verdict changes, {baseline['name']} -> "
          f"{arm_result['name']} ---")

    changes = 0

    for before, after in zip(
        baseline["questions"], arm_result["questions"]
    ):

        for statement in sorted(before["statement_statuses"]):

            was = before["statement_statuses"][statement]
            now = after["statement_statuses"].get(statement)

            if now is None or was == now:
                continue

            changes += 1

            print(
                f"    Q{before['q_number']} {statement}: "
                f"{was} -> {now}"
            )

    if not changes:
        print("    no statement changed verdict")

    print()


def print_answer_table(results: "list[dict]") -> None:
    """
    Per-question answers across arms under the strict policy.

    Aggregates can improve while individual questions get worse, and a
    fix that trades one correct answer for another is not the same as a
    fix that adds one. Only a per-question view distinguishes them.
    """

    print("--- per-question answers (strict policy, audited labels) ---")

    print(
        "    Q     label     "
        + "  ".join(f"{arm['name']:<5}" for arm in results)
    )

    for position, question in enumerate(results[0]["questions"]):

        expected = question["audited_answer"]

        cells = []

        for arm in results:
            predicted = arm["questions"][position]["predictions"][
                STRICT
            ]["predicted"]

            cells.append(
                f"{predicted}*   "
                if predicted == expected
                else f"{'--' if predicted is None else predicted:<5}"
            )

        # A corrected label is flagged inline, so no row can silently be
        # scored against a label this project changed.
        note = (
            f" (was {question['official_answer']})"
            if expected != question["official_answer"]
            else ""
        )

        print(
            f"    Q{question['q_number']:<4} {expected:<9} "
            + "  ".join(cells)
            + note
        )

    print("    * = correct, -- = abstained\n")


def main() -> None:

    records = json.loads(
        Path(RECORDS_PATH).read_text(encoding="utf-8")
    )

    check_records(records)

    chunks = load_chunks()
    index = build_index(chunks)
    absence = ExactAbsenceChecker(index)

    api_key = load_api_key()
    cache = VerdictCache()

    vanilla_baseline = baseline_metrics()

    print(f"corpus {len(chunks)} chunks, {len(records)} questions")
    print(f"retriever: {RETRIEVER_DESCRIPTIONS[RETRIEVER]}")
    print(f"output:    {OUTPUT_PATH}")

    print("baseline vanilla RAG (answers every question):")

    for label_set, metrics in vanilla_baseline.items():

        print(
            f"  {label_set:<15} "
            f"accuracy {percent(metrics['accuracy_overall'])}"
            f"   wrong {percent(metrics['error_rate_overall'])}"
        )

    print()

    results = []

    try:
        for arm in ARMS:
            print(f"running {arm['name']} ...", flush=True)
            results.append(
                run_arm(arm, records, index, absence, api_key, cache)
            )

    finally:
        cache.save()
        print(
            f"\ncache: {cache.hits} hits, {cache.misses} misses "
            f"-> {CACHE_PATH}\n"
        )

    for arm_result in results:
        print_arm(arm_result)

    by_name = {arm["name"]: arm for arm in results}

    for arm_result in results:

        baseline = by_name.get(arm_result["compare_to"])

        if baseline is not None:
            print_verdict_changes(baseline, arm_result)

    print_answer_table(results)

    report = {
        "model": MODEL_NAME,
        "retriever": RETRIEVER_DESCRIPTIONS[RETRIEVER],
        "retriever_caveat": RETRIEVER_CAVEATS[RETRIEVER],
        "baseline_vanilla_rag": vanilla_baseline,
        "arms": results,
    }

    Path(OUTPUT_PATH).write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
