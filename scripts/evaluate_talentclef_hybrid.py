from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.hybrid_skill_extractor import JobBertDocumentSkillExtractor  # noqa: E402
from app.services.talentclef_benchmark import load_talentclef_task_a  # noqa: E402
from app.services.talentclef_hybrid_benchmark import run_hybrid_benchmark  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate JobBERT+catalog hybrid retrieval on TalentCLEF development.")
    parser.add_argument(
        "--data-dir", type=Path,
        default=ROOT / "data" / "external" / "talentclef2026" / "TaskA",
    )
    parser.add_argument(
        "--model-cache-dir", type=Path,
        default=ROOT / "data" / "external" / "skillspan_models",
    )
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data" / "derived" / "talentclef_hybrid_benchmark.json",
    )
    args = parser.parse_args()
    dataset = load_talentclef_task_a(args.data_dir, split="development", language="en")
    extractor = JobBertDocumentSkillExtractor(args.model_cache_dir)
    report = run_hybrid_benchmark(dataset, extractor, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
