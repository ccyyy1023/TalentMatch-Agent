from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.jth_benchmark import run_jth_benchmark  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate transparent ranking methods on JTH.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "external" / "jth")
    parser.add_argument("--cutoff", default="2024-01-01")
    parser.add_argument("--min-pool", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "derived" / "jth_benchmark_report.json")
    args = parser.parse_args()
    result = run_jth_benchmark(args.data_dir, cutoff=args.cutoff, min_pool=args.min_pool)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
