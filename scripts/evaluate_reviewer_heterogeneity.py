from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.reviewer_benchmark import compare_reviewer_models, run_reviewer_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare homogeneous and heterogeneous Conflict Reviewer models.")
    parser.add_argument("--extractor-model", default="qwen3:4b")
    parser.add_argument("--homogeneous-reviewer", default="qwen3:4b")
    parser.add_argument("--heterogeneous-reviewer", default="qwen2.5:7b")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data" / "derived" / "reviewer_heterogeneity_report.json",
    )
    args = parser.parse_args()
    report = {
        "benchmark": "Conflict Reviewer heterogeneous-model reliability",
        "extractor_model_context": args.extractor_model,
        "input": "24 frozen controlled cases; no extractor calls during this benchmark",
        "homogeneous": run_reviewer_model(args.homogeneous_reviewer),
        "heterogeneous": run_reviewer_model(args.heterogeneous_reviewer),
        "comparison": None,
        "limitations": [
            "The 24 cases are developer-authored boundary cases, not an independent human-labeled recruitment set.",
            "The benchmark isolates contradiction review and does not measure end-to-end ranking or hiring quality.",
            "A resume claim remains prohibited until a larger independent labeled set confirms improvement.",
        ],
    }
    report["comparison"] = compare_reviewer_models(report["homogeneous"], report["heterogeneous"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
