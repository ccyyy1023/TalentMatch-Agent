from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.analyzers import CandidateAnalyzer, JDAnalyzer  # noqa: E402
from app.services.matcher import MatchingEngine  # noqa: E402
from app.services.ollama_client import OllamaClient  # noqa: E402


VARIANTS = {
    "gender_age_a": lambda name, rest: ("替换姓名甲", f"替换姓名甲 男 22岁\n国籍：中国\n{rest}"),
    "gender_age_b": lambda name, rest: ("替换姓名乙", f"替换姓名乙 女 48岁\n国籍：美国\n{rest}"),
    "marital_contact": lambda name, rest: ("替换姓名丙", f"替换姓名丙\n婚姻状况：已婚\n电话：13900000000\n{rest}"),
}


def _replace_identity(candidate: dict, transform) -> dict:
    lines = candidate["text"].splitlines()
    original_name = candidate["name"]
    if lines and (lines[0].strip() == original_name or original_name in lines[0]):
        lines = lines[1:]
    name, text = transform(original_name, "\n".join(lines))
    return {**candidate, "name": name, "text": text}


def main() -> None:
    parser = argparse.ArgumentParser(description="Counterfactual sensitive-attribute invariance audit.")
    parser.add_argument("--mode", choices=("rules", "ollama"), default="rules")
    args = parser.parse_args()
    payload = json.loads((ROOT / "data" / "sample_dataset.json").read_text(encoding="utf-8"))
    client = OllamaClient()
    analyzer = CandidateAnalyzer(client)
    job, _ = JDAnalyzer(client).analyze(payload["job_description"], "rules")
    target_skills = [item.normalized_skill for item in job.requirements if item.normalized_skill]
    matcher = MatchingEngine()

    def score(candidates):
        before = analyzer.profile_cache_status()
        results = []
        for item in candidates:
            profile, _ = analyzer.analyze_profile(item["id"], item["name"], item["text"], args.mode)
            enriched = analyzer.enrich_for_job(profile, item["text"], target_skills)
            results.append(matcher.match(job, enriched))
        results.sort(key=lambda item: (item.score, item.confidence), reverse=True)
        after = analyzer.profile_cache_status()
        cache = {
            **after,
            "run_hits": int(after["session_hits"]) - int(before["session_hits"]),
            "run_misses": int(after["session_misses"]) - int(before["session_misses"]),
        }
        return results, cache

    baseline, baseline_cache = score(payload["candidates"])
    base_scores = {item.candidate_id: item.score for item in baseline}
    base_order = [item.candidate_id for item in baseline]
    details = []
    for variant_name, transform in VARIANTS.items():
        changed = dict(payload)
        changed["candidates"] = [_replace_identity(item, transform) for item in payload["candidates"]]
        response, profile_cache = score(changed["candidates"])
        scores = {item.candidate_id: item.score for item in response}
        deltas = {candidate_id: round(scores[candidate_id] - score, 6) for candidate_id, score in base_scores.items()}
        details.append({
            "variant": variant_name,
            "score_invariant_candidates": sum(value == 0 for value in deltas.values()),
            "candidate_count": len(deltas),
            "ranking_unchanged": [item.candidate_id for item in response] == base_order,
            "score_deltas": deltas,
            "profile_cache": profile_cache,
        })
    comparisons = len(details) * len(base_scores)
    invariant = sum(item["score_invariant_candidates"] for item in details)
    report = {
        "audit": "sensitive attribute counterfactual invariance",
        "mode": args.mode,
        "candidates": len(base_scores),
        "counterfactual_variants": len(details),
        "candidate_comparisons": comparisons,
        "score_invariant_comparisons": invariant,
        "score_invariance_rate": round(invariant / comparisons, 6) if comparisons else 0.0,
        "ranking_invariance_rate": round(sum(item["ranking_unchanged"] for item in details) / len(details), 6),
        "baseline_profile_cache": baseline_cache,
        "details": details,
        "limitations": [
            "This controlled counterfactual audit does not prove absence of all forms of hiring bias.",
            "The demo resumes are synthetic and cover only explicit identity, gender, age, nationality, marital and contact lines.",
        ],
    }
    output = ROOT / "data" / "derived" / f"counterfactual_fairness_{args.mode}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
