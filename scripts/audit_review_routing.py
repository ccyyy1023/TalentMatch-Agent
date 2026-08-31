from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.schemas import AnalysisRequest  # noqa: E402
from app.services.ollama_client import OllamaClient  # noqa: E402
from app.services.workflow import TalentMatchWorkflow  # noqa: E402


def main() -> None:
    payload = json.loads((ROOT / "data" / "sample_dataset.json").read_text(encoding="utf-8"))
    payload.update({"mode": "ollama", "enable_semantic_matching": False})
    response = TalentMatchWorkflow(OllamaClient()).run(AnalysisRequest.model_validate(payload))
    report = {
        "elapsed_seconds": round(response.elapsed_ms / 1000, 3),
        "trace_details": {item.node: item.detail for item in response.traces},
        "cache": response.model_info.get("cache", {}),
        "fallback_nodes": [item.node for item in response.traces if item.status == "fallback"],
    }
    output = ROOT / "data" / "derived" / "review_routing_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
