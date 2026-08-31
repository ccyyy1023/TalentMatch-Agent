from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.schemas import AnalysisRequest  # noqa: E402
from app.services.workflow import TalentMatchWorkflow  # noqa: E402


def main() -> None:
    payload = json.loads((ROOT / "data" / "sample_dataset.json").read_text(encoding="utf-8"))
    payload["mode"] = "ollama"
    payload["enable_semantic_matching"] = False
    payload["candidates"] = payload["candidates"][:1]
    request = AnalysisRequest.model_validate(payload)
    response = TalentMatchWorkflow().run(request)
    report = {
        "run_id": response.run_id,
        "mode": response.mode,
        "model_info": response.model_info,
        "job_requirement_count": len(response.job.requirements),
        "candidate_score": response.ranking[0].score,
        "candidate_recommendation": response.ranking[0].recommendation,
        "criteria": [
            {
                "requirement": item.requirement_text,
                "priority": item.priority,
                "status": item.status,
                "score": item.score,
                "evidence_ids": item.evidence_ids,
                "explanation": item.explanation,
            }
            for item in response.ranking[0].criteria
        ],
        "findings": [item.model_dump() for item in response.ranking[0].findings],
        "trace_statuses": [{"node": trace.node, "status": trace.status, "detail": trace.detail} for trace in response.traces],
        "elapsed_ms": response.elapsed_ms,
    }
    output = ROOT / "data" / "ollama_smoke.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
