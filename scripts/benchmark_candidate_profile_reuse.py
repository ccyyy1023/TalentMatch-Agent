from __future__ import annotations

import argparse
import json
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import settings  # noqa: E402
from app.schemas import CandidateInput, ParsedCandidate  # noqa: E402
from app.services.analyzers import CandidateAnalyzer  # noqa: E402
from app.services.model_cache import ModelResponseCache  # noqa: E402
from app.services.ollama_client import OllamaClient  # noqa: E402


JOB_TARGETS = {
    "agent_engineer": ["python", "fastapi", "langgraph", "rag", "docker", "postgresql"],
    "data_engineer": ["python", "sql", "postgresql", "redis", "docker"],
    "backend_engineer": ["python", "fastapi", "rest_api", "postgresql", "docker"],
}


def _build_profiles(
    analyzer: CandidateAnalyzer,
    candidates: list[CandidateInput],
    workers: int,
) -> tuple[list[ParsedCandidate], list[str], float]:
    started = perf_counter()

    def build(item: CandidateInput):
        return analyzer.analyze_profile(item.id, item.name, item.text, "ollama")

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="profile-benchmark") as executor:
        rows = list(executor.map(build, candidates))
    return [item[0] for item in rows], [item[1] for item in rows], perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure reusable candidate-profile behavior across multiple jobs.")
    parser.add_argument("--workers", type=int, default=settings.ollama_workers)
    args = parser.parse_args()
    payload = json.loads((ROOT / "data" / "sample_dataset.json").read_text(encoding="utf-8"))
    candidates = [CandidateInput.model_validate(item) for item in payload["candidates"]]

    with tempfile.TemporaryDirectory(prefix="talentmatch-profile-") as temporary:
        cache = ModelResponseCache(Path(temporary) / "profile-cache.db")
        client = OllamaClient(chat_model=settings.candidate_model, cache=cache)
        analyzer = CandidateAnalyzer(client)

        cold_profiles, cold_origins, cold_seconds = _build_profiles(analyzer, candidates, args.workers)
        evidence_before = sum(len(item.evidence) for item in cold_profiles)
        verified = {
            job: [
                analyzer.enrich_for_job(profile, source.text, targets)
                for profile, source in zip(cold_profiles, candidates)
            ]
            for job, targets in JOB_TARGETS.items()
        }
        evidence_after = sum(len(item.evidence) for rows in verified.values() for item in rows)
        warm_profiles, warm_origins, warm_seconds = _build_profiles(analyzer, candidates, args.workers)

        quotes = [
            (evidence.source_quote, source.text)
            for rows in verified.values()
            for item, source in zip(rows, candidates)
            for evidence in item.evidence
        ]
        pairwise_calls = len(JOB_TARGETS) * len(candidates)
        profile_calls = len(candidates)
        report = {
            "scope": {
                "jobs": len(JOB_TARGETS),
                "candidates": len(candidates),
                "job_candidate_pairs": pairwise_calls,
                "model": settings.candidate_model,
                "workers": args.workers,
            },
            "architecture": {
                "candidate_profile_scope": "job_independent",
                "job_conditioned_verification": "deterministic_exact_quote",
                "pairwise_reference_calls": pairwise_calls,
                "cold_profile_model_calls": client.cache_misses,
                "avoided_pairwise_model_calls": pairwise_calls - profile_calls,
                "model_call_reduction_ratio": round(1 - profile_calls / pairwise_calls, 6),
            },
            "cold_profile_run": {
                "elapsed_seconds": round(cold_seconds, 3),
                "origins": {origin: cold_origins.count(origin) for origin in sorted(set(cold_origins))},
            },
            "warm_profile_replay": {
                "elapsed_seconds": round(warm_seconds, 3),
                "origins": {origin: warm_origins.count(origin) for origin in sorted(set(warm_origins))},
                "persistent_profile_cache_hits": analyzer.profile_cache_hits,
                "same_profiles": [item.model_dump() for item in cold_profiles] == [item.model_dump() for item in warm_profiles],
            },
            "verification": {
                "base_evidence_count": evidence_before,
                "job_conditioned_evidence_count_across_jobs": evidence_after,
                "exact_quote_validity": round(sum(quote in text for quote, text in quotes) / len(quotes), 6) if quotes else 1.0,
            },
            "limitations": [
                "The call reduction compares reusable profile extraction with a per-job candidate extraction design; it is not an accuracy gain.",
                "The fixed eight-candidate demo is an engineering benchmark, not a production throughput claim.",
                "Job-conditioned verification is exact text matching and does not make a new LLM inference.",
            ],
        }
        cache.engine.dispose()

    output = ROOT / "data" / "derived" / "candidate_profile_reuse_benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
