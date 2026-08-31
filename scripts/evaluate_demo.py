from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.schemas import AnalysisRequest  # noqa: E402
from app.services.evaluation import evaluate_run  # noqa: E402
from app.services.workflow import TalentMatchWorkflow  # noqa: E402


def main() -> None:
    payload = json.loads((ROOT / "data" / "sample_dataset.json").read_text(encoding="utf-8"))
    payload["mode"] = "rules"
    request = AnalysisRequest.model_validate(payload)
    response = TalentMatchWorkflow().run(request)
    labels = {item.id: item.relevance_label for item in request.candidates if item.relevance_label is not None}
    metrics = evaluate_run(response, labels)
    output = {
        "run_id": response.run_id,
        "ranking": [{"candidate_id": item.candidate_id, "score": item.score, "recommendation": item.recommendation} for item in response.ranking],
        "metrics": metrics.model_dump(),
        "traces": [item.model_dump() for item in response.traces],
    }
    output_path = ROOT / "data" / "demo_evaluation.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
