from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.talentclef_benchmark import (  # noqa: E402
    load_talentclef_task_a,
    run_bm25_benchmark,
    write_trec_run,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a BM25 baseline on TalentCLEF 2026 Task A.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "external" / "talentclef2026" / "TaskA",
    )
    parser.add_argument("--split", choices=("development", "dev", "test"), default="development")
    parser.add_argument("--language", choices=("en", "es"), default="en")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON report path; defaults to data/derived/talentclef_<split>_<language>_bm25.json",
    )
    parser.add_argument("--run-output", type=Path, default=None, help="Optional official-format TREC run path.")
    args = parser.parse_args()

    dataset = load_talentclef_task_a(args.data_dir, split=args.split, language=args.language)
    report, rankings = run_bm25_benchmark(dataset)
    output = args.output or ROOT / "data" / "derived" / (
        f"talentclef_{dataset.split}_{dataset.language}_bm25.json"
    )
    run_output = args.run_output or ROOT / "data" / "derived" / (
        f"talentclef_{dataset.split}_{dataset.language}_bm25.run"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_trec_run(run_output, rankings)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {output}")
    print(f"TREC run: {run_output}")


if __name__ == "__main__":
    main()
