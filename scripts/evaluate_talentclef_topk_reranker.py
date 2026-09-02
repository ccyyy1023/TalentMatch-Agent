from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.evidence_reranker import leave_one_job_out_rerank  # noqa: E402
from app.services.talentclef_benchmark import load_talentclef_task_a  # noqa: E402
from app.services.talentclef_extraction_ab import build_stratified_sample  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate evidence-gated Top-K direct-agent reranking.")
    parser.add_argument("--query-limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--positives-per-query", type=int, default=2)
    parser.add_argument("--negatives-per-query", type=int, default=4)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "external" / "talentclef2026" / "TaskA")
    parser.add_argument("--evidence-report", type=Path, default=ROOT / "data" / "derived" / "talentclef_hard_negative_ab_q10.json")
    parser.add_argument("--direct-report", type=Path, default=ROOT / "data" / "derived" / "talentclef_agent_ablation_q10.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "derived" / "talentclef_topk_reranker_q10.json")
    args = parser.parse_args()

    dataset = load_talentclef_task_a(args.data_dir, split="development", language="en")
    sample = build_stratified_sample(
        dataset, query_limit=args.query_limit, positives_per_query=args.positives_per_query,
        negatives_per_query=args.negatives_per_query, seed=args.seed, negative_strategy="bm25_hard",
    )
    evidence = json.loads(args.evidence_report.read_text(encoding="utf-8"))
    direct = json.loads(args.direct_report.read_text(encoding="utf-8"))
    expected_ids = list(sample.query_ids)
    if evidence["scope"]["query_ids"] != expected_ids or direct["scope"]["query_ids"] != expected_ids:
        raise ValueError("report query scope does not match regenerated sample")

    qwen = next(item for item in evidence["variants"] if item["variant"] == "qwen_only")
    base_rankings = {
        query_id: [(str(item[0]), float(item[1])) for item in rows]
        for query_id, rows in qwen["deterministic_matching_rankings"].items()
    }
    direct_scores = {query_id: {} for query_id in sample.query_ids}
    valid_scores = {query_id: set() for query_id in sample.query_ids}
    for pair_key, row in direct["single_agent"]["details"].items():
        query_id, candidate_id = pair_key.split(":", 1)
        direct_scores[query_id][candidate_id] = float(row["score"])
        if row.get("job_quote_valid") and row.get("cv_quote_valid"):
            valid_scores[query_id].add(candidate_id)

    evaluation = leave_one_job_out_rerank(base_rankings, direct_scores, valid_scores, sample.labels)
    total = sum(len(rows) for rows in direct_scores.values())
    accepted = sum(len(rows) for rows in valid_scores.values())
    report = {
        "benchmark": "TalentCLEF evidence-gated Top-K direct-agent reranking",
        "scope": evidence["scope"],
        "method": {
            "first_stage": "Qwen evidence extraction plus deterministic matcher",
            "second_stage": "direct Qwen score accepted only with exact JD and CV quotes",
            "invalid_output_behavior": "retain first-stage score",
        },
        "grounding_gate": {
            "attempted_scores": total,
            "accepted_scores": accepted,
            "rejected_scores": total - accepted,
            "accepted_quote_validity": 1.0 if accepted else 0.0,
        },
        "evaluation": evaluation,
        "claim_gate": {
            "production_default_enabled": bool(evaluation["activation_gate"]["eligible"]),
            "resume_metric_eligible": False,
            "reason": "Ten development jobs with leave-one-job-out selection remain a development diagnostic, not a sealed test set.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
