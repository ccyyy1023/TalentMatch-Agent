from collections import Counter

from app.services.jth_benchmark import structured_features, structured_score
from app.services.jth_ltr import (
    FEATURE_NAMES, MONOTONIC_CONSTRAINTS, NEGATIVE_MONOTONIC_FEATURES,
    _blend_predictions, _canonical_skills, ranking_features,
)


def test_structured_feature_refactor_preserves_fixed_score():
    job = {"skills": "python;sql", "llm_required_lowest_diploma": "master"}
    candidate = {"skills": "python;sql", "llm_highest_diploma": "phd"}
    features = structured_features(job, candidate)
    assert set(features)
    assert 0 <= structured_score(job, candidate) <= 1


def test_skill_aliases_and_idf_coverage_are_job_aligned():
    job = {"skills": "postgres;js"}
    aligned = {"skills": "postgresql;javascript"}
    unrelated = {"skills": "excel"}
    frequency = Counter({"postgresql": 1, "javascript": 1, "excel": 1})
    left = ranking_features(job, aligned, frequency, 2)
    right = ranking_features(job, unrelated, frequency, 2)
    assert _canonical_skills("postgres;js") == {"postgresql", "javascript"}
    assert left["idf_skill_coverage"] > right["idf_skill_coverage"]
    assert left["matched_skill_count"] == 2


def test_required_experience_and_boolean_gaps_are_explicit_features():
    job = {
        "skills": "python", "llm_required_years_of_work_experience": "5 years",
        "llm_required_management_experience": "true",
    }
    candidate = {
        "skills": "python", "llm_years_of_work_experience": "2 years",
        "llm_management_experience": "false",
    }
    features = ranking_features(job, candidate, Counter({"python": 1}), 1)
    assert set(features) == set(FEATURE_NAMES)
    assert features["experience_shortfall"] == 3
    assert features["management_experience_match"] == 0


def test_protected_fields_are_not_features():
    forbidden = {"llm_sex", "llm_age_bucket", "llm_nationality", "actual_salary", "actual_daily_salary"}
    assert forbidden.isdisjoint(FEATURE_NAMES)


def test_monotonic_constraints_follow_feature_meaning():
    constraints = dict(zip(FEATURE_NAMES, MONOTONIC_CONSTRAINTS))
    assert all(constraints[name] == -1 for name in NEGATIVE_MONOTONIC_FEATURES)
    assert constraints["skill_coverage"] == 1
    assert constraints["experience_known"] == 0


def test_anchor_alpha_zero_preserves_fixed_order():
    from app.services.jth_ltr import RankingGroup

    group = RankingGroup("j", "2025-01-01", {}, ({}, {}, {}), ("a", "b", "c"), (0, 1, 2))
    blended = _blend_predictions([group], [0.1, 0.9, 0.4], [0.9, 0.1, 0.4], 0.0)
    assert sorted(range(3), key=lambda index: -blended[index]) == [1, 2, 0]
