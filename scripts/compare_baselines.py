from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.schemas import AnalysisRequest  # noqa: E402
from app.services.baselines import keyword_coverage_ranking  # noqa: E402
from app.services.evaluation import evaluate_run, ranking_metrics  # noqa: E402
from app.services.workflow import TalentMatchWorkflow  # noqa: E402


def main() -> None:
    payload = json.loads((ROOT / "data" / "sample_dataset.json").read_text(encoding="utf-8"))
    payload["mode"] = "rules"
    request = AnalysisRequest.model_validate(payload)
    labels = {item.id: item.relevance_label for item in request.candidates if item.relevance_label is not None}
    keyword = keyword_coverage_ranking(request)
    response = TalentMatchWorkflow().run(request)
    report = {
        "warning": "固定演示集结果，仅验证评测链路，不代表真实招聘效果。",
        "dataset": {"candidates": len(request.candidates), "labels": {"0": 3, "1": 3, "2": 2}},
        "keyword_coverage": {
            "ranking": keyword,
            "metrics": ranking_metrics([candidate_id for candidate_id, _ in keyword], labels),
        },
        "evidence_workflow": {
            "ranking": [(item.candidate_id, item.score) for item in response.ranking],
            "metrics": evaluate_run(response, labels).model_dump(),
        },
    }
    output = ROOT / "data" / "baseline_comparison.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
