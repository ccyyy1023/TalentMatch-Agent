from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.jth_ltr import train_and_evaluate_lambdamart  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select with rolling 2023/2024 validation and evaluate the 2025 JTH holdout once."
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "external" / "jth")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "derived" / "jth_lambdamart_report.json")
    parser.add_argument("--model-output", type=Path, default=ROOT / "data" / "derived" / "jth_lambdamart_model.txt")
    parser.add_argument("--min-pool", type=int, default=5)
    args = parser.parse_args()
    report, model = train_and_evaluate_lambdamart(args.data_dir, min_pool=args.min_pool)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    model.booster_.save_model(str(args.model_output))
    summary = {
        "selected_params": report["selected_params"],
        "selected_anchor_alpha": report["selected_anchor_alpha"],
        "test_fixed_structured": report["test_fixed_structured"],
        "test_anchored_lambdamart": report["test_lambdamart"],
        "delta": report["delta_lambdamart_minus_fixed"],
        "paired_bootstrap_95ci_delta": report["paired_bootstrap_95ci_delta"],
        "activation_gate": report["activation_gate"],
        "full_report": str(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
