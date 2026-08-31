from app.services.jth_benchmark import (
    EXCLUDED_MATCHING_FIELDS, STRUCTURED_WEIGHTS, QueryMetrics, _paired_bootstrap_ci,
    keyword_score, query_metrics, structured_score, surface_attribute_score,
)


def test_query_metrics_rewards_correct_ranking():
    labels = {"a": 4, "b": 2, "c": 0, "d": 0}
    good = query_metrics(["a", "b", "c", "d"], labels)
    bad = query_metrics(["c", "d", "b", "a"], labels)
    assert good.ndcg_at_5 > bad.ndcg_at_5
    assert good.mrr == 1.0


def test_structured_matcher_uses_multiple_relevant_attributes():
    job = {
        "skills": "python", "llm_hard_skills": "sql", "job_category": "data engineer",
        "expertise_area": "data", "contract_type": "permanent",
        "llm_required_years_of_work_experience": "3",
    }
    strong = {
        "skills": "python;sql", "job_category": "data engineer", "expertise_area": "data",
        "contract_type": "permanent", "llm_years_of_work_experience": "5-10",
    }
    weak = {"skills": "python", "job_category": "sales", "expertise_area": "sales"}
    assert structured_score(job, strong) > structured_score(job, weak)
    assert keyword_score(job, strong) == keyword_score(job, weak)


def test_surface_attribute_baseline_is_stronger_than_skill_only_keyword():
    job = {"skills": "python", "job_category": "data engineer", "expertise_area": "data"}
    aligned = {"skills": "python", "job_category": "data engineer", "expertise_area": "data"}
    unrelated = {"skills": "python", "job_category": "sales", "expertise_area": "sales"}
    assert keyword_score(job, aligned) == keyword_score(job, unrelated)
    assert surface_attribute_score(job, aligned) > surface_attribute_score(job, unrelated)


def test_structured_weights_are_normalized_and_use_extended_job_evidence():
    assert abs(sum(STRUCTURED_WEIGHTS.values()) - 1.0) < 1e-12
    job = {
        "skills": "python", "llm_industry_domains": "fintech",
        "llm_required_languages_spoken": "french", "llm_required_lowest_diploma": "master",
        "llm_seniority_level": "senior",
    }
    aligned = {
        "skills": "python", "llm_industry_domains": "fintech", "llm_languages_spoken": "french",
        "llm_highest_diploma": "phd", "llm_seniority_level": "lead",
    }
    weak = {
        "skills": "python", "llm_industry_domains": "retail", "llm_languages_spoken": "english",
        "llm_highest_diploma": "bachelor", "llm_seniority_level": "junior",
    }
    assert structured_score(job, aligned) > structured_score(job, weak)


def test_protected_attributes_cannot_change_scores():
    job = {"skills": "python", "job_category": "developer"}
    base = {"skills": "python", "job_category": "developer", "llm_sex": "Male", "llm_nationality": "French"}
    swapped = {**base, "llm_sex": "Female", "llm_nationality": "Chinese", "llm_age_bucket": "50-60"}
    assert keyword_score(job, base) == keyword_score(job, swapped)
    assert structured_score(job, base) == structured_score(job, swapped)
    assert {"llm_sex", "llm_nationality", "llm_age_bucket"} <= EXCLUDED_MATCHING_FIELDS


def test_paired_bootstrap_interval_is_reproducible_and_positive():
    baseline = [QueryMetrics(0.2, 0.2, 0.2, 0.2)] * 10
    structured = [QueryMetrics(0.4, 0.4, 0.4, 0.4)] * 10
    interval = _paired_bootstrap_ci(baseline, structured, samples=100, seed=7)
    assert interval["ndcg_at_5"] == [0.2, 0.2]
