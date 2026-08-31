from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.schemas import AnalysisRequest  # noqa: E402
from app.config import settings  # noqa: E402
from app.services.evaluation import evaluate_run  # noqa: E402
from app.services.ollama_client import OllamaClient  # noqa: E402
from app.services.workflow import TalentMatchWorkflow  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("ollama", "adaptive"), default="ollama")
    args = parser.parse_args()
    payload = json.loads((ROOT / "data" / "sample_dataset.json").read_text(encoding="utf-8"))
    payload.update({"mode": args.mode, "enable_semantic_matching": False})
    request = AnalysisRequest.model_validate(payload)
    client = OllamaClient(cache_enabled=False)
    response = TalentMatchWorkflow(client).run(request)
    labels = {item.id: item.relevance_label for item in request.candidates if item.relevance_label is not None}
    metrics = evaluate_run(response, labels)
    report = {
        "scope": {"job_queries": 1, "candidates": len(request.candidates), "cache_enabled": False, "mode": args.mode},
        "configured_workers": settings.ollama_workers,
        "elapsed_seconds": round(response.elapsed_ms / 1000, 3),
        "trace_seconds": {trace.node: round(trace.elapsed_ms / 1000, 3) for trace in response.traces},
        "trace_details": {trace.node: trace.detail for trace in response.traces},
        "fallback_nodes": [trace.node for trace in response.traces if trace.status == "fallback"],
        "metrics": metrics.model_dump(),
        "ranking": [item.candidate_id for item in response.ranking],
        "sequential_reference_seconds": 265.08,
        "limitations": [
            "The sequential reference is the earlier observed run on the same machine and fixed demo, not repeated in the same process.",
            "Ollama may internally schedule requests; worker count two does not guarantee exact twofold speedup.",
            "This fixed demo measures engineering latency, not production concurrency capacity.",
        ],
    }
    report["observed_speedup_vs_reference"] = round(report["sequential_reference_seconds"] / report["elapsed_seconds"], 3)
    output = ROOT / "data" / "derived" / f"{args.mode}_cold_benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
