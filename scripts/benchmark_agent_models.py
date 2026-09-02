from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import settings  # noqa: E402
from app.schemas import AnalysisRequest  # noqa: E402
from app.services.evaluation import evaluate_run  # noqa: E402
from app.services.ollama_client import OllamaClient  # noqa: E402
from app.services.workflow import TalentMatchWorkflow  # noqa: E402


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare independently configured TalentMatch Agent models.")
    parser.add_argument("--jd-model", default=settings.jd_model)
    parser.add_argument("--candidate-model", default=settings.candidate_model)
    parser.add_argument("--reviewer-model", default=settings.reviewer_model)
    parser.add_argument("--mode", choices=("ollama", "adaptive"), default="adaptive")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = json.loads((ROOT / "data" / "sample_dataset.json").read_text(encoding="utf-8"))
    payload["candidates"] = payload["candidates"][: max(1, args.limit)]
    payload.update({"mode": args.mode, "enable_semantic_matching": False})
    request = AnalysisRequest.model_validate(payload)
    workflow = TalentMatchWorkflow(
        jd_ollama=OllamaClient(chat_model=args.jd_model, cache_enabled=False),
        candidate_ollama=OllamaClient(chat_model=args.candidate_model, cache_enabled=False),
        reviewer_ollama=OllamaClient(chat_model=args.reviewer_model, cache_enabled=False),
    )
    response = workflow.run(request)
    labels = {item.id: item.relevance_label for item in request.candidates if item.relevance_label is not None}
    metrics = evaluate_run(response, labels)
    report = {
        "scope": {
            "job_queries": 1,
            "candidates": len(request.candidates),
            "cache_enabled": False,
            "mode": args.mode,
            "labels": "fixed synthetic engineering demo",
        },
        "agent_models": workflow.agent_models,
        "elapsed_seconds": round(response.elapsed_ms / 1000, 3),
        "trace_seconds": {trace.node: round(trace.elapsed_ms / 1000, 3) for trace in response.traces},
        "fallback_nodes": [trace.node for trace in response.traces if trace.status == "fallback"],
        "metrics": metrics.model_dump(),
        "ranking": [item.candidate_id for item in response.ranking],
        "limitations": [
            "This is a fixed synthetic engineering demo, not an independent real-world test set.",
            "Different model names do not establish reviewer independence or quality improvement by themselves.",
            "Keep a heterogeneous configuration only after repeated evaluation improves reliability or ranking.",
        ],
    }
    if args.output is None:
        signature = "__".join(_safe_name(model) for model in workflow.agent_models.values())
        output = ROOT / "data" / "derived" / f"agent_models__{signature}.json"
    else:
        output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
