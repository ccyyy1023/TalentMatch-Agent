from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.hybrid_skill_extractor import JobBertDocumentSkillExtractor  # noqa: E402
from app.services.talentclef_benchmark import load_talentclef_task_a  # noqa: E402
from app.services.talentclef_cascade import tune_and_evaluate_cascade  # noqa: E402
from app.services.talentclef_extraction_ab import (  # noqa: E402
    build_stratified_sample,
    evaluate_raw_sample,
    raw_sample_rankings,
    run_model_extraction,
)


def _delta(right: dict, left: dict, key: str) -> dict[str, float]:
    return {
        metric: round(right[key][metric] - left[key][metric], 6)
        for metric in left[key]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen versus Qwen+JobBERT on TalentCLEF hard negatives.")
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--query-limit", type=int, default=2)
    parser.add_argument("--positives-per-query", type=int, default=2)
    parser.add_argument("--negatives-per-query", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument(
        "--data-dir", type=Path,
        default=ROOT / "data" / "external" / "talentclef2026" / "TaskA",
    )
    parser.add_argument(
        "--model-cache-dir", type=Path,
        default=ROOT / "data" / "external" / "skillspan_models",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data" / "derived" / "talentclef_hard_negative_ab.json",
    )
    args = parser.parse_args()
    dataset = load_talentclef_task_a(args.data_dir, split="development", language="en")
    sample = build_stratified_sample(
        dataset,
        query_limit=args.query_limit,
        positives_per_query=args.positives_per_query,
        negatives_per_query=args.negatives_per_query,
        seed=args.seed,
        negative_strategy="bm25_hard",
    )
    baseline = run_model_extraction(
        dataset, sample, model=args.model, cache_enabled=True, variant="qwen_only",
    )
    hybrid = run_model_extraction(
        dataset,
        sample,
        model=args.model,
        cache_enabled=True,
        skill_extractor=JobBertDocumentSkillExtractor(args.model_cache_dir),
        variant="qwen_plus_jobbert",
    )
    raw_rankings = raw_sample_rankings(dataset, sample)
    evidence_rankings = {
        query_id: [(str(item[0]), float(item[1])) for item in rows]
        for query_id, rows in baseline["deterministic_matching_rankings"].items()
    }
    report = {
        "benchmark": "TalentCLEF hard-negative Qwen and hybrid extraction A/B",
        "scope": {
            "language": dataset.language,
            "query_ids": list(sample.query_ids),
            "unique_candidates": len(sample.candidate_ids),
            "candidates_per_query": args.positives_per_query + args.negatives_per_query,
            "positive_per_query": args.positives_per_query,
            "negative_per_query": args.negatives_per_query,
            "negative_strategy": "bm25_hard",
            "seed": args.seed,
        },
        "raw_text_bm25": evaluate_raw_sample(dataset, sample),
        "raw_text_rankings": {
            query_id: [[candidate_id, round(score, 8)] for candidate_id, score in rows]
            for query_id, rows in raw_rankings.items()
        },
        "variants": [baseline, hybrid],
        "two_stage_cascade": tune_and_evaluate_cascade(
            raw_rankings, evidence_rankings, sample.labels, seed=args.seed,
        ) if len(sample.query_ids) >= 3 else None,
        "delta_hybrid_minus_qwen": {
            "extracted_text_bm25": _delta(hybrid, baseline, "sample_ranking_metrics"),
            "deterministic_matcher": _delta(hybrid, baseline, "deterministic_matching_metrics"),
        },
        "claim_gate": {
            "resume_metric_eligible": False,
            "reason": "Small development hard-negative diagnostic; use to choose architecture, not as a headline metric.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
