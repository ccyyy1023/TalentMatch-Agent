from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"


def load(name: str) -> dict:
    return json.loads((DERIVED / name).read_text(encoding="utf-8"))


def main() -> None:
    full = load("parallel_ollama_benchmark.json")
    adaptive = load("adaptive_cold_benchmark.json")
    sequential = float(full["sequential_reference_seconds"])
    parallel = float(full["elapsed_seconds"])
    adaptive_seconds = float(adaptive["elapsed_seconds"])
    report = {
        "fixed_demo_candidates": 8,
        "cache_enabled": False,
        "seconds": {
            "sequential_reference": sequential,
            "full_ollama_parallel_selective_review": parallel,
            "adaptive_agent": adaptive_seconds,
        },
        "speedup_vs_sequential": {
            "full_ollama_parallel_selective_review": round(sequential / parallel, 3),
            "adaptive_agent": round(sequential / adaptive_seconds, 3),
        },
        "latency_reduction_percent": {
            "full_ollama_vs_sequential": round((1 - parallel / sequential) * 100, 1),
            "adaptive_vs_sequential": round((1 - adaptive_seconds / sequential) * 100, 1),
            "adaptive_vs_full_ollama": round((1 - adaptive_seconds / parallel) * 100, 1),
        },
        "adaptive_routing": {
            "candidate_llm": "3/8",
            "reviewer_llm": "4/8",
            "fallback_nodes": adaptive["fallback_nodes"],
        },
        "quality_check": {
            "same_ranking_as_full_ollama": adaptive["ranking"] == full["ranking"],
            "adaptive_metrics": adaptive["metrics"],
        },
        "limitations": [
            "Sequential reference was observed in an earlier run rather than a randomized repeated timing study.",
            "The comparison uses one fixed engineering demo and does not establish production throughput.",
        ],
    }
    output = DERIVED / "performance_comparison.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
