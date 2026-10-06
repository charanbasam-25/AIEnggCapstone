"""Save a current semantic-versus-reranked benchmark without LLM API calls."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

from src.evaluation.benchmark import sha256_file
from src.evaluation.retrieval_report import (
    ROOT, RETRIEVAL_REPORT_VERSION, load_retrieval_queries, summarize_retrieval,
)
from src.orchestration.telemetry import capture_telemetry, measured_stage
from src.retrieval.retrieval_config import RERANK_CANDIDATE_K, RETRIEVAL_TOP_K


def run_retrieval_benchmark(retriever, queries: list[dict], config: dict) -> dict:
    rows = []
    started = perf_counter()
    with capture_telemetry() as telemetry:
        for number, query in enumerate(queries, 1):
            telemetry.question_number = number
            before = len(telemetry.retrievals)
            with measured_stage("retrieval_benchmark"):
                retriever.retrieve(query["query"], top_k=config["top_k"])
            if len(telemetry.retrievals) != before + 1:
                raise ValueError("The retriever did not record its actual candidate search.")
            rows.append({**query, "trace": telemetry.retrievals[-1]})
    return {
        "created_at": datetime.now(timezone.utc).isoformat(), "config": config,
        "metrics": summarize_retrieval(rows, config["top_k"]), "results": rows,
        "elapsed_seconds": perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Save a separate smoke report for this many queries.")
    args = parser.parse_args()
    corpus_path = ROOT / "data/processed/chunks.jsonl"
    queries_path = ROOT / "src/evaluation/retrieval_queries_gold.json"
    queries = load_retrieval_queries(queries_path)
    if args.limit is not None and not 1 <= args.limit <= len(queries):
        parser.error(f"--limit must be between 1 and {len(queries)}.")
    corpus_hash, queries_hash = sha256_file(corpus_path), sha256_file(queries_path)
    selected = queries[:args.limit] if args.limit is not None else queries
    # Model imports and construction happen only when this command is run.
    from src.retrieval.semantic_reranker import (
        EMBEDDING_MODEL_NAME, MODEL_NAME, SemanticReranker, load_chunks,
    )

    loading = perf_counter()
    retriever = SemanticReranker(load_chunks(str(corpus_path)), candidate_k=RERANK_CANDIDATE_K)
    model_loading_seconds = perf_counter() - loading
    config = {
        "report_version": RETRIEVAL_REPORT_VERSION,
        "corpus_sha256": corpus_hash, "queries_sha256": queries_hash,
        "query_ids": [row["query_id"] for row in selected],
        "is_full_benchmark": len(selected) == len(queries),
        "top_k": RETRIEVAL_TOP_K, "candidate_k": RERANK_CANDIDATE_K,
        "embedding_model": EMBEDDING_MODEL_NAME, "reranker_model": MODEL_NAME,
        "corpus_changed_during_run": False, "queries_changed_during_run": False,
    }
    report = run_retrieval_benchmark(retriever, selected, config)
    report["model_loading_seconds"] = model_loading_seconds
    config["corpus_changed_during_run"] = sha256_file(corpus_path) != corpus_hash
    config["queries_changed_during_run"] = sha256_file(queries_path) != queries_hash
    filename = "retrieval_results.json" if config["is_full_benchmark"] else "retrieval_smoke_results.json"
    path = ROOT / "data/evaluation" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    for name, metrics in report["metrics"].items():
        print(f"{name}: {metrics['hits']}/{metrics['queries']} source-page hits in the top {RETRIEVAL_TOP_K} "
              f"({metrics['hit_rate']:.1%}).")
    print(f"Saved {path.relative_to(ROOT)}. No LLM calls were made.")
    if config["corpus_changed_during_run"] or config["queries_changed_during_run"]:
        print("Sources or labels changed during the run; this report cannot be shown as current.")


if __name__ == "__main__":
    main()
