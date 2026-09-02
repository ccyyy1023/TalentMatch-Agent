from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.talentclef_agent_ablation import run_direct_pair_agent  # noqa: E402
from app.services.talentclef_benchmark import load_talentclef_task_a  # noqa: E402
from app.services.talentclef_extraction_ab import build_stratified_sample  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a single direct LLM with the evidence workflow on hard negatives.")
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--query-limit", type=int, default=5)
    parser.add_argument("--positives-per-query", type=int, default=2)
    parser.add_argument("--negatives-per-query", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument(
        "--data-dir", type=Path,
        default=ROOT / "data" / "external" / "talentclef2026" / "TaskA",
    )
    parser.add_argument(
        "--evidence-report", type=Path,
        default=ROOT / "data" / "derived" / "talentclef_hard_negative_ab_q5.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data" / "derived" / "talentclef_agent_ablation_q5.json",
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
    evidence_report = json.loads(args.evidence_report.read_text(encoding="utf-8"))
    expected_scope = {
        "query_ids": list(sample.query_ids),
        "candidates_per_query": args.positives_per_query + args.negatives_per_query,
        "seed": args.seed,
    }
    for key, expected in expected_scope.items():
        if evidence_report["scope"].get(key) != expected:
            raise ValueError(f"evidence report scope mismatch for {key}")
    qwen_variant = next(item for item in evidence_report["variants"] if item["variant"] == "qwen_only")

    def progress(done: int, total: int) -> None:
        print(f"single-agent progress {done}/{total}", flush=True)

    direct = run_direct_pair_agent(dataset, sample, model=args.model, progress_callback=progress)
    evidence_metrics = qwen_variant["deterministic_matching_metrics"]
    report = {
        "benchmark": "TalentCLEF hard-negative single-agent versus evidence workflow ablation",
        "scope": evidence_report["scope"],
        "single_agent": direct,
        "evidence_workflow": {
            "method": "JD Agent + Candidate Evidence Agent + deterministic matcher",
            "model": qwen_variant["model"],
            "fallbacks": qwen_variant["fallbacks"],
            "ranking_metrics": evidence_metrics,
            "job_quote_valid_rate": qwen_variant["extraction"]["job_valid_quote_rate"],
            "candidate_quote_valid_rate": qwen_variant["extraction"]["candidate_valid_quote_rate"],
        },
        "delta_evidence_minus_single": {
            key: round(evidence_metrics[key] - direct["ranking_metrics"][key], 6)
            for key in evidence_metrics
        },
        "reviewer_boundary": (
            "Conflict Reviewer does not rewrite deterministic ranking scores; its separate controlled benchmark "
            "measures conflict recall, false positives and grounding rather than TalentCLEF relevance."
        ),
        "claim_gate": {
            "resume_metric_eligible": False,
            "reason": "Development hard-negative diagnostic with a small query count; report as an ablation, not production accuracy.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
