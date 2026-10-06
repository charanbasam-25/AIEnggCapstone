"""Evaluate both answering paths on a frozen split, with resumable checkpoints.

--check and --dry-run are entirely offline. Only --run loads models or calls
OpenAI. The old 13-question reports are never overwritten by this runner.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from src.evaluation.benchmark import (
    BENCHMARK_PATH,
    ROOT,
    inventory,
    load_benchmark,
    question_input,
    score_records,
    sha256_file,
)
from src.retrieval.retrieval_config import RETRIEVAL_TOP_K, RERANK_CANDIDATE_K
from src.verification.question_type import STATEMENT_MCQ, classify_question
from src.verification.evidence_context import EVIDENCE_CONTEXT_VERSION, FACT_VERIFICATION_POLICY


CHUNKS_PATH = ROOT / "data/processed/chunks.jsonl"
SYSTEMS = ("verified", "vanilla")
VERIFICATION_PIPELINE = "source-references-article-context-v2"


def format_query(question: dict) -> str:
    return question["question_text"] + "\n\n" + "\n".join(
        f"({letter}) {question['options'][letter]}" for letter in "ABCD"
    )


def verify_question(question: dict, retriever, fact_verifier, direct_verifier, *, article_context=None) -> dict:
    # These imports stay behind --run so inventory/validation does not load
    # sentence-transformers, model weights or an OpenAI client.
    from src.evaluation.evaluate_verified_pyqs import (
        build_pyq_claims,
        determine_answer_from_claims,
    )
    from src.verification.fact_verifier import (
        FactEvidenceAssessment, resolve_fact_assessment, verify_constructed_claim,
    )
    from src.verification.direct_mcq_verifier import verify_direct_question

    if classify_question(question["question_text"]) != STATEMENT_MCQ:
        return verify_direct_question(question, retriever, direct_verifier).model_dump()

    claims = build_pyq_claims(question["question_text"])
    if not claims:
        return {
            "status": "ABSTAINED", "predicted_answer": None,
            "abstention_reason": "Numbered statements could not be extracted.",
            "claims": [],
        }
    verdicts, records = {}, []
    for claim in claims:
        evidence = retriever.retrieve(claim.claim, top_k=RETRIEVAL_TOP_K)
        if article_context is not None:
            evidence = article_context.supplement(claim.claim, evidence)
        result = verify_constructed_claim(claim, evidence, fact_verifier)
        if result.verdict != "INSUFFICIENT" and result.verification_method != "lexical_scan":
            checked = resolve_fact_assessment(FactEvidenceAssessment(
                verdict=result.verdict, reasoning=result.reasoning, citations=result.citations,
            ), result.checked_evidence or evidence)
            issues = [*result.validation_issues, *checked.validation_issues]
            if not result.independently_reviewed:
                issues.append("A model-based benchmark verdict requires an agreeing blind evidence review.")
            if issues:
                result = result.model_copy(update={
                    "verdict": "INSUFFICIENT", "validation_issues": issues,
                    "reasoning": "The benchmark acceptance checks did not pass: " + " ".join(issues),
                })
        verdicts[claim.claim_id] = result
        records.append({
            **claim.model_dump(),
            "verification": result.model_dump(),
            "evidence": evidence,
        })
    prediction, mapping = determine_answer_from_claims(
        claims, verdicts, question["options"], question["question_text"], policy="strict"
    )
    return {
        "status": "ANSWERED" if prediction else "ABSTAINED",
        "predicted_answer": prediction,
        "claims": records,
        "mapping": mapping,
    }


def make_services(systems: list[str]) -> dict:
    from src.retrieval.semantic_reranker import SemanticReranker, load_chunks

    chunks = load_chunks(str(CHUNKS_PATH))
    # Both arms share one model/index instance; each uses its own normal
    # query construction. This is an end-to-end comparison, not an isolated
    # estimate of the verification layer's contribution.
    retriever = SemanticReranker(chunks, candidate_k=RERANK_CANDIDATE_K)
    services = {}
    if "verified" in systems:
        from src.orchestration.telemetry import create_model_client
        from src.verification.article_context import ArticleEvidenceContext
        from src.verification.direct_mcq_verifier import DirectMCQVerifier
        from src.verification.fact_verifier import FactVerifier

        client = create_model_client()
        facts = FactVerifier(chunks, client=client, reference_quotes=True)
        direct = DirectMCQVerifier(chunks=chunks, client=client, reference_quotes=True)
        context = ArticleEvidenceContext(chunks)
        services["verified"] = lambda question: verify_question(
            question, retriever, facts, direct, article_context=context,
        )
    if "vanilla" in systems:
        from src.rag.vanilla_rag import VanillaRAG

        baseline = VanillaRAG(chunks, retriever=retriever)

        def answer(question):
            result = baseline.answer(format_query(question), top_k=RETRIEVAL_TOP_K)
            return {**result, "status": "ANSWERED", "predicted_answer": result["selected_option"]}

        services["vanilla"] = answer
    return services


def code_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src").rglob("*.py")):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_config(dataset: dict, questions: list[dict], split: str, systems: list[str]) -> dict:
    return {
        "dataset_id": dataset["dataset_id"],
        "dataset_sha256": sha256_file(BENCHMARK_PATH),
        "corpus_sha256": sha256_file(CHUNKS_PATH),
        "code_sha256": code_fingerprint(),
        "split": split,
        "question_ids": [q["id"] for q in questions],
        "is_full_split": len(questions) == sum(q["split"] == split for q in dataset["questions"]),
        "systems": systems,
        "policy": "strict",
        "model": "gpt-4o-mini",
        "temperature": 0,
        "semantic_model": "BAAI/bge-small-en-v1.5",
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "candidate_k": RERANK_CANDIDATE_K,
        "top_k_per_query": RETRIEVAL_TOP_K,
        "verification_evidence_context": EVIDENCE_CONTEXT_VERSION,
        "fact_verification_policy": FACT_VERIFICATION_POLICY,
        "verification_pipeline": VERIFICATION_PIPELINE,
        "quotation_binding": "Numbered source excerpts; Python copies exact words and validates provenance.",
        "statement_evidence_context": "Reranked full pages plus named Article provision pages, continuations and footnotes.",
        "direct_citation_screen": "Reject quotations with no informative question/option term overlap; overlap is not proof of entailment.",
        "request_timeout_seconds": 45,
        "sdk_transport_retries": 1,
        "independent_review": "Required for every committed model-based claim and every direct answer candidate; reviewer receives no prior verdict or answer key.",
        "unsupported_formats": "Paired Statement-I/II and Assertion/Reason labels abstain until a separate statement-and-relationship verification path exists.",
        "evaluation_scope": "Regression validation: the frozen test split has now informed verifier debugging. A fresh untouched set is required for independent evaluation.",
        "comparison": "End-to-end systems; statement verification uses per-claim retrieval, direct verification combines five queries, vanilla RAG uses one full-question query.",
    }


def paired_summary(questions: list[dict], results: dict) -> dict | None:
    if not all(system in results for system in SYSTEMS):
        return None
    by_system = {system: {r["question_id"]: r for r in results[system]} for system in SYSTEMS}
    common = []
    wins = {"verified_only_correct": 0, "vanilla_only_correct": 0, "both_correct": 0, "neither_correct": 0}
    for question in questions:
        rows = [by_system[system].get(question["id"]) for system in SYSTEMS]
        if any(row is None or row["status"] != "ANSWERED" for row in rows):
            continue
        common.append(question["id"])
        correct = [row["predicted_answer"] == question["official_answer"] for row in rows]
        key = "both_correct" if all(correct) else (
            "verified_only_correct" if correct[0] else "vanilla_only_correct" if correct[1] else "neither_correct"
        )
        wins[key] += 1
    return {"both_answered": len(common), "question_ids": common, **wins}


def save_checkpoint(output: Path, config: dict, questions: list[dict], results: dict, started_at: str) -> dict:
    report = {
        "schema_version": 1,
        "config": config,
        "started_at": started_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "complete": all(len(results[system]) == len(questions) for system in config["systems"]),
        "results": results,
        "metrics": {system: score_records(questions, results[system]) for system in config["systems"]},
        "paired_answered_subset": paired_summary(questions, results),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    return report


def run_evaluation(questions: list[dict], config: dict, services: dict, output: Path, *, retry_errors: bool = False) -> dict:
    results = {system: [] for system in config["systems"]}
    started_at = datetime.now(timezone.utc).isoformat()
    if output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous["config"] != config:
            raise ValueError("Dataset, code, corpus or run settings changed. Use a new output file.")
        results, started_at = previous["results"], previous["started_at"]
        # Validate existing rows before trusting a checkpoint or making calls.
        for system in config["systems"]:
            score_records(questions, results[system])
    if retry_errors:
        results = {system: [row for row in rows if row["status"] != "ERROR"] for system, rows in results.items()}

    for index, question in enumerate(questions, 1):
        for system in config["systems"]:
            if any(row["question_id"] == question["id"] for row in results[system]):
                continue
            start = time.monotonic()
            try:
                # Never pass a gold label, topic, source file, answer key or
                # gold evidence to either answering system.
                result = services[system](question_input(question))
                prediction = result.get("predicted_answer")
                status = result["status"]
                if status not in ("ANSWERED", "ABSTAINED") or (
                    (status == "ANSWERED") != (prediction in set("ABCD"))
                ) or (status == "ABSTAINED" and prediction is not None):
                    raise ValueError("Malformed answering-system result")
                row = {"status": status, "predicted_answer": prediction, "details": result}
            except Exception as exc:
                # Errors are not an evidence-based abstention. Store a safe
                # class name; API error bodies can contain private inputs.
                row = {"status": "ERROR", "predicted_answer": None, "error_type": type(exc).__name__}
            row.update({"question_id": question["id"], "elapsed_seconds": round(time.monotonic() - start, 3)})
            results[system].append(row)
            report = save_checkpoint(output, config, questions, results, started_at)
            print(f"{index}/{len(questions)} {question['id']} {system}: {row['status']}", flush=True)
    return save_checkpoint(output, config, questions, results, started_at)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true", help="Validate the frozen data and source PDFs; no API calls")
    modes.add_argument("--dry-run", action="store_true", help="Preview the selected questions and run settings; no API calls (default)")
    modes.add_argument("--run", action="store_true", help="Execute the benchmark using the configured OpenAI API key")
    parser.add_argument("--split", choices=("development", "test"), default="development")
    parser.add_argument("--systems", choices=SYSTEMS, nargs="+", default=list(SYSTEMS))
    parser.add_argument("--limit", type=int, help="Development-only smoke run; never a headline benchmark result")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--retry-errors", action="store_true", help="On resume, retry service errors; completed answers/abstentions stay cached")
    args = parser.parse_args(argv)
    if args.limit is not None and (args.split == "test" or args.limit < 1):
        parser.error("--limit must be positive and is available only for development")
    try:
        dataset = load_benchmark(check_sources=args.check)
        if args.check:
            print(json.dumps(inventory(dataset), indent=2))
            print("PASS: frozen dataset, official labels, source hashes, routing and split integrity.")
            return
        questions = [q for q in dataset["questions"] if q["split"] == args.split]
        if args.limit is not None:
            questions = questions[:args.limit]
        systems = [system for system in SYSTEMS if system in args.systems]
        config = run_config(dataset, questions, args.split, systems)
        if not args.run:
            print(json.dumps(config, indent=2))
            print("Dry run complete. No answering models loaded and no API calls made. Add --run to execute.")
            return
        suffix = f"_first{len(questions)}" if not config["is_full_split"] else ""
        output = args.output or ROOT / f"data/evaluation/benchmark/{args.split}{suffix}_results.json"
        services = make_services(systems)
        report = run_evaluation(questions, config, services, output, retry_errors=args.retry_errors)
        for system, metrics in report["metrics"].items():
            print(system, json.dumps(metrics["overall"], indent=2))
        print(f"Saved: {output}")
        if any(metrics["overall"]["errors"] for metrics in report["metrics"].values()):
            raise SystemExit("The run contains service errors; inspect the report and resume with --retry-errors.")
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
