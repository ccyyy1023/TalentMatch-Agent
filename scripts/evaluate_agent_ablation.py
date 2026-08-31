from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.schemas import AnalysisRequest  # noqa: E402
from app.services.analyzers import CandidateAnalyzer, JDAnalyzer  # noqa: E402
from app.services.baselines import keyword_coverage_ranking  # noqa: E402
from app.services.evaluation import ranking_metrics  # noqa: E402
from app.services.matcher import MatchingEngine  # noqa: E402
from app.services.ollama_client import OllamaClient  # noqa: E402
from app.services.reviewer import ConflictReviewer  # noqa: E402
from app.services.workflow import TalentMatchWorkflow  # noqa: E402


def metric(order: list[str], labels: dict[str, int]) -> dict:
    return ranking_metrics(order, labels)


def run_llm_pipeline(client: OllamaClient, request: AnalysisRequest) -> dict:
    started = perf_counter()
    job, job_trace = JDAnalyzer(client).analyze(request.job_description, "ollama")
    analyzer = CandidateAnalyzer(client)
    parsed_candidates = []
    analyzer_modes = []
    for item in request.candidates:
        parsed, used_mode = analyzer.analyze(item.id, item.name, item.text, "ollama")
        parsed_candidates.append(parsed)
        analyzer_modes.append(used_mode)
    engine = MatchingEngine()
    raw_results = [engine.match(job, candidate) for candidate in parsed_candidates]
    extraction_order = [item.candidate_id for item in sorted(raw_results, key=lambda item: (item.score, item.confidence), reverse=True)]
    reviewer = ConflictReviewer(client)
    reviewed_results = [
        reviewer.review(job, candidate, result.model_copy(deep=True), "ollama")
        for candidate, result in zip(parsed_candidates, raw_results)
    ]
    reviewed_order = [item.candidate_id for item in sorted(reviewed_results, key=lambda item: (item.score, item.confidence), reverse=True)]
    return {
        "job_trace": job_trace.model_dump(),
        "candidate_analyzer_modes": analyzer_modes,
        "extraction_order": extraction_order,
        "reviewed_order": reviewed_order,
        "matcher_findings": sum(len(item.findings) for item in raw_results),
        "reviewed_findings": sum(len(item.findings) for item in reviewed_results),
        "elapsed_seconds": round(perf_counter() - started, 3),
    }


def direct_llm_order(client: OllamaClient, request: AnalysisRequest) -> tuple[list[str], float]:
    started = perf_counter()
    candidates = [{"candidate_id": item.id, "text": item.text} for item in request.candidates]
    payload = client.generate_json(
        "你是单次人岗匹配基线。根据岗位和候选人原文直接排序。输出JSON字段ranking，"
        "每项只含candidate_id和score。不得创造候选人ID。该输出仅作实验对照，不用于实际决策。",
        json.dumps({"job_description": request.job_description, "candidates": candidates}, ensure_ascii=False),
        cache_namespace="single_llm_direct", prompt_version="direct-v1",
    )
    valid_ids = {item.id for item in request.candidates}
    scored = []
    for item in payload.get("ranking", []):
        candidate_id = str(item.get("candidate_id", ""))
        if candidate_id in valid_ids and candidate_id not in {row[0] for row in scored}:
            try:
                score = float(item.get("score", 0))
            except (TypeError, ValueError):
                score = 0.0
            scored.append((candidate_id, score))
    order = [item[0] for item in sorted(scored, key=lambda item: item[1], reverse=True)]
    order.extend(item.id for item in request.candidates if item.id not in order)
    return order, round(perf_counter() - started, 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    payload = json.loads((ROOT / "data" / "sample_dataset.json").read_text(encoding="utf-8"))
    payload["candidates"] = payload["candidates"][: args.limit]
    payload["mode"] = "rules"
    request = AnalysisRequest.model_validate(payload)
    labels = {item.id: item.relevance_label or 0 for item in request.candidates}
    client = OllamaClient()

    keyword_order = [item_id for item_id, _ in keyword_coverage_ranking(request)]
    rules_response = TalentMatchWorkflow(client).run(request)
    rules_order = [item.candidate_id for item in rules_response.ranking]
    direct_order, direct_seconds = direct_llm_order(client, request)
    first = run_llm_pipeline(client, request)
    warm = run_llm_pipeline(client, request)

    report = {
        "scope": {"job_queries": 1, "candidates": len(request.candidates), "labels": "fixed demo labels"},
        "ranking": {
            "keyword": metric(keyword_order, labels),
            "deterministic_evidence": metric(rules_order, labels),
            "single_llm_direct": metric(direct_order, labels),
            "llm_extraction_deterministic_matcher": metric(first["extraction_order"], labels),
            "full_with_conflict_reviewer": metric(first["reviewed_order"], labels),
        },
        "reviewer_effect": {
            "matcher_findings": first["matcher_findings"],
            "reviewed_findings": first["reviewed_findings"],
        },
        "latency_seconds": {
            "single_llm_direct": direct_seconds,
            "llm_pipeline_first_observed": first["elapsed_seconds"],
            "llm_pipeline_warm_cache": warm["elapsed_seconds"],
        },
        "cache": client.cache_status(),
        "first_run_modes": first["candidate_analyzer_modes"],
        "warm_run_modes": warm["candidate_analyzer_modes"],
        "limitations": [
            "This is the fixed eight-candidate engineering demo, not an independent test set.",
            "Ranking metrics are diagnostic only; the experiment primarily verifies real model calls and cache behavior.",
            "The reviewer can change findings, confidence and decision state but does not directly rewrite deterministic scores.",
        ],
    }
    output = ROOT / "data" / "derived" / "agent_ablation_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
