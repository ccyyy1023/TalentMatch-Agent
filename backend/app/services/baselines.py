from __future__ import annotations

from app.schemas import AnalysisRequest
from app.services.skill_catalog import extract_skills


def keyword_coverage_ranking(request: AnalysisRequest) -> list[tuple[str, float]]:
    required = {skill for skill, _ in extract_skills(request.job_description)}
    results = []
    for candidate in request.candidates:
        candidate_skills = {skill for skill, _ in extract_skills(candidate.text)}
        coverage = len(required & candidate_skills) / len(required) if required else 0.0
        results.append((candidate.id, round(coverage * 100, 2)))
    return sorted(results, key=lambda item: item[1], reverse=True)
