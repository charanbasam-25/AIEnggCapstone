"""Contracts and scoring for source-page retrieval benchmarks; no model imports."""

import json
from pathlib import Path
from statistics import mean

from src.evaluation.benchmark import sha256_file
from src.retrieval.retrieval_config import RERANK_CANDIDATE_K, RETRIEVAL_TOP_K


ROOT = Path(__file__).resolve().parents[2]
QUERIES_PATH = ROOT / "src/evaluation/retrieval_queries_gold.json"
RETRIEVAL_REPORT_VERSION = "retrieval-benchmark-v1"
RETRIEVAL_SYSTEM_NAMES = {
    "semantic": "Semantic search", "reranked": "Semantic + cross-encoder",
}


def load_retrieval_queries(path: Path = QUERIES_PATH) -> list[dict]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("The retrieval query set is empty or unreadable.")
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("A retrieval query is malformed.")
        query_id, query, source, pages = (row.get(field) for field in
                                         ("query_id", "query", "expected_source", "expected_pages"))
        if (
            not all(isinstance(value, str) and value.strip() for value in (query_id, query, source))
            or query_id in seen or not isinstance(pages, list) or not pages
            or any(not isinstance(page, int) or isinstance(page, bool) or page < 1 for page in pages)
            or len(set(pages)) != len(pages)
        ):
            raise ValueError("Retrieval queries need unique IDs and valid source-page labels.")
        seen.add(query_id)
    return rows


def page_hit(passages: list[dict], expected: dict) -> bool:
    return any(item.get("source") == expected["expected_source"]
               and item.get("page") in expected["expected_pages"] for item in passages)


def summarize_retrieval(rows: list[dict], top_k: int) -> dict:
    """Keep the existing any-gold-page hit definition, not document-level accuracy."""
    metrics = {}
    for system in RETRIEVAL_SYSTEM_NAMES:
        hits = 0
        times = []
        for row in rows:
            trace = row["trace"]
            passages = trace["candidates"][:top_k] if system == "semantic" else trace["selected"]
            hits += int(page_hit(passages, row))
            seconds = trace["semantic_seconds"] if system == "semantic" else trace["elapsed_seconds"]
            if isinstance(seconds, (float, int)):
                times.append(seconds)
        metrics[system] = {
            "queries": len(rows), "hits": hits, "misses": len(rows) - hits,
            "hit_rate": hits / len(rows) if rows else None,
            "mean_search_seconds": mean(times) if times else None,
        }
    return metrics


def retrieval_report_issue(payload: dict, root: Path = ROOT) -> str | None:
    """Do not present a stale, partial or internally inconsistent run as current."""
    try:
        config, results = payload["config"], payload["results"]
        if config.get("report_version") != RETRIEVAL_REPORT_VERSION:
            return "This retrieval report uses an earlier report format. Run a fresh evaluation."
        if config.get("corpus_changed_during_run") or config.get("queries_changed_during_run"):
            return "The source corpus or query labels changed during this evaluation. Run it again."
        if config.get("corpus_sha256") != sha256_file(root / "data/processed/chunks.jsonl"):
            return "This retrieval report uses a different corpus. Run it against the current sources."
        query_path = root / "src/evaluation/retrieval_queries_gold.json"
        if config.get("queries_sha256") != sha256_file(query_path):
            return "This retrieval report uses a different query set or source-page labels."
        if config.get("top_k") != RETRIEVAL_TOP_K or config.get("candidate_k") != RERANK_CANDIDATE_K:
            return "The retrieval depths differ from the current practice configuration."
        queries = load_retrieval_queries(query_path)
        expected = {row["query_id"]: row for row in queries}
        ids = [row["query_id"] for row in results]
        if not config.get("is_full_benchmark") or set(ids) != set(expected):
            return "This is a partial retrieval run. Inspect its raw report under Reports."
        if len(set(ids)) != len(ids) or config.get("query_ids") != ids:
            return "The retrieval report contains duplicate or inconsistent query IDs."
        for row in results:
            original = expected[row["query_id"]]
            if any(row.get(field) != original[field] for field in
                   ("query", "expected_source", "expected_pages")):
                return "A saved query disagrees with its annotated source-page labels."
            trace = row["trace"]
            if not trace["success"] or trace["error_type"] or trace["query"] != row["query"]:
                return "A retrieval search failed or belongs to a different query."
            if trace["top_k"] != config["top_k"] or trace["candidate_k"] != config["candidate_k"]:
                return "A search used inconsistent retrieval depths."
            if (trace["embedding_model"] != config["embedding_model"]
                    or trace["reranker_model"] != config["reranker_model"]):
                return "The recorded search models disagree with the benchmark configuration."
            candidates, selected = trace["candidates"], trace["selected"]
            if (not candidates or len(candidates) > config["candidate_k"]
                    or len(selected) != min(config["top_k"], len(candidates))):
                return "A search has an incomplete candidate or selected-passage list."
            if [item["semantic_rank"] for item in candidates] != list(range(1, len(candidates) + 1)):
                return "The recorded semantic rankings are inconsistent."
            ranked = sorted(candidates, key=lambda item: item["reranker_rank"])
            if [item["reranker_rank"] for item in ranked] != list(range(1, len(candidates) + 1)):
                return "The recorded cross-encoder rankings are inconsistent."
            if selected != ranked[:config["top_k"]]:
                return "The selected passages disagree with the recorded reranking."
        if payload["metrics"] != summarize_retrieval(results, config["top_k"]):
            return "The saved retrieval metrics disagree with the recorded searches."
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return "The retrieval report or its query labels are incomplete or unreadable."
    return None
