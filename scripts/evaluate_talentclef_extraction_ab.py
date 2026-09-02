from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.talentclef_benchmark import load_talentclef_task_a  # noqa: E402
from app.services.talentclef_extraction_ab import (  # noqa: E402
    build_stratified_sample,
    compare_model_reports,
    evaluate_raw_sample,
    run_model_extraction,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B test Qwen extraction on fixed TalentCLEF text.")
    parser.add_argument("--left-model", default="qwen2.5:7b")
    parser.add_argument("--right-model", default="qwen3:4b")
    parser.add_argument("--language", choices=("en", "es"), default="en")
    parser.add_argument("--query-limit", type=int, default=3)
    parser.add_argument("--positives-per-query", type=int, default=4)
    parser.add_argument("--negatives-per-query", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--negative-strategy", choices=("random", "bm25_hard"), default="bm25_hard")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "external" / "talentclef2026" / "TaskA",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "derived" / "talentclef_extraction_ab.json",
    )
    args = parser.parse_args()

    dataset = load_talentclef_task_a(args.data_dir, language=args.language)
    sample = build_stratified_sample(
        dataset,
        query_limit=args.query_limit,
        positives_per_query=args.positives_per_query,
        negatives_per_query=args.negatives_per_query,
        seed=args.seed,
        negative_strategy=args.negative_strategy,
    )
    report = {
        "benchmark": "TalentCLEF 2026 Task A extraction A/B",
        "dataset_version": "0.3.0",
        "scope": {
            "language": args.language,
            "query_ids": list(sample.query_ids),
            "unique_candidates": len(sample.candidate_ids),
            "candidates_per_query": args.positives_per_query + args.negatives_per_query,
            "positive_per_query": args.positives_per_query,
            "negative_per_query": args.negatives_per_query,
            "seed": args.seed,
            "negative_strategy": args.negative_strategy,
            "labels_used_for_training": False,
            "cache_enabled": False,
        },
        "raw_text_bm25_same_sample": evaluate_raw_sample(dataset, sample),
        "models": [],
        "comparison": None,
        "limitations": [
            "This fixed stratified development sample is a model-selection diagnostic, not a sealed test set.",
            "BM25 hard negatives deliberately stress lexical confounders and do not represent the natural candidate distribution.",
            "TalentCLEF texts are synthetic and privacy-preserving rather than production ATS documents.",
            "Source quote rates measure grounding after application validation; rejected raw model items are not counted.",
            "Extracted-text BM25 measures information retention, not final production matching quality.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for model in (args.left_model, args.right_model):
        model_report = run_model_extraction(dataset, sample, model=model)
        report["models"].append(model_report)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Completed {model}: {model_report['elapsed_seconds']} seconds")
    report["comparison"] = compare_model_reports(report["models"][0], report["models"][1])
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
