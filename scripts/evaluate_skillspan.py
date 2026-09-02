from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.skillspan_benchmark import evaluate_skillspan, load_skillspan, stratified_sample  # noqa: E402


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Evaluate exact and overlap skill-span extraction on SkillSpan.")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--mode", choices=["catalog", "ollama", "jobbert"], default="catalog")
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--sample-size", type=int, default=0, help="0 evaluates the complete split.")
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "external" / "skillspan")
    parser.add_argument("--model-cache-dir", type=Path, default=ROOT / "data" / "external" / "skillspan_models")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = load_skillspan(args.data_root / "data" / "json" / f"{args.split}.json")
    records = stratified_sample(records, args.sample_size, args.seed)
    report = evaluate_skillspan(records, args.mode, model=args.model, model_cache_dir=args.model_cache_dir)
    report["split"] = args.split
    report["sample_seed"] = args.seed if args.sample_size else None
    output = args.output or ROOT / "data" / "derived" / f"skillspan_{args.split}_{args.mode}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {key: value for key, value in report.items() if key != "diagnostics"}
    summary["diagnostic_records_written"] = len(report["diagnostics"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
