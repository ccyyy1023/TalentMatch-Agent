from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


STAGE_RELEVANCE = {
    "Application Made": 0,
    "Qualification": 0,
    "Shortlist": 1,
    "Resume Sent": 2,
    "1st Interview": 3,
    "2nd Interview": 3,
    "3rd Interview": 3,
    "4th Interview": 3,
    "Offer Received": 4,
    "Offer Accepted": 4,
}

# These columns are deliberately unavailable to both scorers. They are listed
# explicitly so a future schema change cannot silently introduce them.
EXCLUDED_MATCHING_FIELDS = {
    "candidate_id", "create_date", "actual_salary", "actual_daily_salary",
    "zipcode", "source", "llm_sex", "llm_nationality", "llm_age_bucket",
    "llm_hobbies", "llm_values", "llm_has_linkedin",
    "llm_has_portfolio_or_github_or_website",
}

# Selected without touching the 2024+ test set: random search on 2022 jobs,
# then model selection on 2023 jobs. Bounds keep skills and job attributes
# dominant so the scorer remains interpretable rather than becoming an
# unconstrained weak-label optimizer.
STRUCTURED_WEIGHTS = {
    "skill_coverage": 0.3421465380525026,
    "skill_overlap": 0.11084738577146244,
    "category_match": 0.1046116649628787,
    "expertise_match": 0.05710898865589636,
    "contract_match": 0.06875143421090658,
    "experience_match": 0.06971594257754461,
    "industry_coverage": 0.06402773948426158,
    "soft_skill_coverage": 0.013926439075495239,
    "language_coverage": 0.01803293189242716,
    "certification_coverage": 0.02044010779994737,
    "seniority_match": 0.04823727740371755,
    "diploma_match": 0.08215355011295973,
}

SENIORITY_LEVELS = {
    "intern": 0, "entry": 1, "junior": 1, "mid": 2, "middle": 2,
    "senior": 3, "lead": 4, "manager": 4, "director": 5, "executive": 6,
}
DIPLOMA_LEVELS = {
    "none": 0, "high school": 1, "associate": 2, "bachelor": 3,
    "master": 4, "mba": 4, "phd": 5, "doctorate": 5,
}


@dataclass(frozen=True)
class QueryMetrics:
    ndcg_at_5: float
    precision_at_3: float
    recall_at_5: float
    mrr: float


def _tokens(*values: str) -> set[str]:
    result: set[str] = set()
    for value in values:
        for token in value.lower().split(";"):
            normalized = " ".join(token.strip().split())
            if normalized and normalized != "_rare_skill_":
                result.add(normalized)
    return result


def _coverage(required: set[str], available: set[str]) -> float:
    return len(required & available) / len(required) if required else 0.0


def _overlap(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left and right else 0.0


def _years_lower_bound(value: str) -> float | None:
    value = value.strip().lower()
    if not value:
        return None
    if value.startswith("+"):
        digits = "".join(ch for ch in value if ch.isdigit() or ch == ".")
        return float(digits) if digits else None
    pieces = value.replace("years", "").replace("year", "").strip().split("-")
    try:
        return float(pieces[0].strip())
    except (ValueError, IndexError):
        return None


def _token_coverage_or_neutral(required: set[str], available: set[str]) -> float:
    """Return neutral evidence when a job does not specify the attribute."""
    return _coverage(required, available) if required else 0.5


def _ordinal(value: str, levels: dict[str, int]) -> int | None:
    normalized = " ".join(value.lower().replace("'s", "").split())
    if not normalized:
        return None
    for label, rank in sorted(levels.items(), key=lambda item: -len(item[0])):
        if label in normalized:
            return rank
    return None


def _ordinal_match(required: str, available: str, levels: dict[str, int]) -> float:
    required_rank = _ordinal(required, levels)
    available_rank = _ordinal(available, levels)
    if required_rank is None or available_rank is None:
        return 0.5
    return 1.0 if available_rank >= required_rank else max(0.0, available_rank / max(required_rank, 1))


def keyword_score(job: dict[str, str], candidate: dict[str, str]) -> float:
    """Transparent surface-keyword baseline using only original skill fields."""
    return _coverage(_tokens(job.get("skills", "")), _tokens(candidate.get("skills", "")))


def surface_attribute_score(job: dict[str, str], candidate: dict[str, str]) -> float:
    """Stronger baseline using only the original, non-LLM attribute columns."""
    job_skills = _tokens(job.get("skills", ""))
    candidate_skills = _tokens(candidate.get("skills", ""))
    categories = _tokens(job.get("job_category", ""))
    candidate_categories = _tokens(candidate.get("job_category", ""))
    expertise = _tokens(job.get("expertise_area", ""))
    candidate_expertise = _tokens(candidate.get("expertise_area", ""))
    contract_match = float(
        bool(job.get("contract_type"))
        and job.get("contract_type", "").strip().lower() == candidate.get("contract_type", "").strip().lower()
    )
    job_years = _years_lower_bound(job.get("years_experience", ""))
    candidate_years = _years_lower_bound(candidate.get("years_experience", ""))
    experience_match = 0.5
    if job_years is not None and candidate_years is not None:
        experience_match = min(1.0, candidate_years / job_years) if job_years > 0 else 1.0
    return (
        0.45 * _coverage(job_skills, candidate_skills)
        + 0.10 * _overlap(job_skills, candidate_skills)
        + 0.20 * float(bool(categories & candidate_categories))
        + 0.10 * float(bool(expertise & candidate_expertise))
        + 0.05 * contract_match
        + 0.10 * experience_match
    )


def structured_score(job: dict[str, str], candidate: dict[str, str]) -> float:
    """Evidence-style attribute matcher; no interaction or protected fields."""
    job_skills = _tokens(
        job.get("skills", ""), job.get("llm_hard_skills", ""),
        job.get("llm_programming_languages", ""), job.get("llm_tools_technologies", ""),
    )
    candidate_skills = _tokens(
        candidate.get("skills", ""), candidate.get("llm_hard_skills", ""),
        candidate.get("llm_programming_languages", ""), candidate.get("llm_tools_technologies", ""),
    )
    skill_coverage = _coverage(job_skills, candidate_skills)
    skill_overlap = _overlap(job_skills, candidate_skills)

    job_categories = _tokens(job.get("job_category", ""), job.get("llm_job_category", ""))
    candidate_categories = _tokens(candidate.get("job_category", ""), candidate.get("llm_job_category", ""))
    category_match = 1.0 if job_categories & candidate_categories else 0.0

    job_expertise = _tokens(job.get("expertise_area", ""), job.get("llm_expertise_area", ""))
    candidate_expertise = _tokens(candidate.get("expertise_area", ""), candidate.get("llm_expertise_area", ""))
    expertise_match = 1.0 if job_expertise & candidate_expertise else 0.0

    contract_match = float(
        bool(job.get("contract_type"))
        and job.get("contract_type", "").strip().lower() == candidate.get("contract_type", "").strip().lower()
    )
    job_years = _years_lower_bound(job.get("llm_required_years_of_work_experience", "") or job.get("years_experience", ""))
    candidate_years = _years_lower_bound(candidate.get("llm_years_of_work_experience", "") or candidate.get("years_experience", ""))
    experience_match = 0.5
    if job_years is not None and candidate_years is not None:
        experience_match = min(1.0, candidate_years / job_years) if job_years > 0 else 1.0

    features = {
        "skill_coverage": skill_coverage,
        "skill_overlap": skill_overlap,
        "category_match": category_match,
        "expertise_match": expertise_match,
        "contract_match": contract_match,
        "experience_match": experience_match,
        "industry_coverage": _token_coverage_or_neutral(
            _tokens(job.get("llm_industry_domains", "")),
            _tokens(candidate.get("llm_industry_domains", "")),
        ),
        "soft_skill_coverage": _token_coverage_or_neutral(
            _tokens(job.get("llm_soft_skills", "")),
            _tokens(candidate.get("llm_soft_skills", "")),
        ),
        "language_coverage": _token_coverage_or_neutral(
            _tokens(job.get("llm_required_languages_spoken", "")),
            _tokens(candidate.get("llm_languages_spoken", "")),
        ),
        "certification_coverage": _token_coverage_or_neutral(
            _tokens(job.get("llm_certifications", "")),
            _tokens(candidate.get("llm_certifications", "")),
        ),
        "seniority_match": _ordinal_match(
            job.get("llm_seniority_level", ""), candidate.get("llm_seniority_level", ""), SENIORITY_LEVELS,
        ),
        "diploma_match": _ordinal_match(
            job.get("llm_required_lowest_diploma", ""), candidate.get("llm_highest_diploma", ""), DIPLOMA_LEVELS,
        ),
    }
    return sum(STRUCTURED_WEIGHTS[name] * value for name, value in features.items())


def _dcg(labels: list[int], k: int) -> float:
    return sum((2**label - 1) / math.log2(index + 2) for index, label in enumerate(labels[:k]))


def query_metrics(order: list[str], labels: dict[str, int], relevant_minimum: int = 2) -> QueryMetrics:
    ranked = [labels[item] for item in order if item in labels]
    ideal = sorted(labels.values(), reverse=True)
    ideal_dcg = _dcg(ideal, 5)
    relevant_total = sum(label >= relevant_minimum for label in labels.values())
    return QueryMetrics(
        ndcg_at_5=_dcg(ranked, 5) / ideal_dcg if ideal_dcg else 0.0,
        precision_at_3=sum(label >= relevant_minimum for label in ranked[:3]) / min(3, len(ranked)) if ranked else 0.0,
        recall_at_5=sum(label >= relevant_minimum for label in ranked[:5]) / relevant_total if relevant_total else 0.0,
        mrr=next((1 / (index + 1) for index, label in enumerate(ranked) if label >= relevant_minimum), 0.0),
    )


def _read_csv(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row[key]: row for row in csv.DictReader(handle)}


def _mean_metrics(rows: Iterable[QueryMetrics]) -> dict[str, float]:
    rows = list(rows)
    return {
        field: round(sum(getattr(row, field) for row in rows) / len(rows), 4)
        for field in QueryMetrics.__dataclass_fields__
    } if rows else {field: 0.0 for field in QueryMetrics.__dataclass_fields__}


def _paired_bootstrap_ci(
    baseline: list[QueryMetrics], structured: list[QueryMetrics],
    samples: int = 2000, seed: int = 20260829,
) -> dict[str, list[float]]:
    if not baseline:
        return {field: [0.0, 0.0] for field in QueryMetrics.__dataclass_fields__}
    rng = random.Random(seed)
    size = len(baseline)
    output: dict[str, list[float]] = {}
    for field in QueryMetrics.__dataclass_fields__:
        deltas = [getattr(right, field) - getattr(left, field) for left, right in zip(baseline, structured)]
        estimates = sorted(sum(deltas[rng.randrange(size)] for _ in range(size)) / size for _ in range(samples))
        lower = estimates[int(samples * 0.025)]
        upper = estimates[min(samples - 1, int(samples * 0.975))]
        output[field] = [round(lower, 4), round(upper, 4)]
    return output


def _ndcg_win_tie_loss(left: list[QueryMetrics], right: list[QueryMetrics]) -> dict[str, int]:
    wins = ties = losses = 0
    for baseline, candidate in zip(left, right):
        delta = candidate.ndcg_at_5 - baseline.ndcg_at_5
        if delta > 1e-12:
            wins += 1
        elif delta < -1e-12:
            losses += 1
        else:
            ties += 1
    return {"wins": wins, "ties": ties, "losses": losses}


def run_jth_benchmark(data_dir: Path, cutoff: str = "2024-01-01", min_pool: int = 5) -> dict:
    candidates = _read_csv(data_dir / "candidates.csv", "candidate_id")
    jobs = _read_csv(data_dir / "jobs.csv", "job_id")
    applications: dict[str, list[dict[str, str]]] = defaultdict(list)
    with (data_dir / "history.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["job_id"] in jobs and row["candidate_id"] in candidates:
                applications[row["job_id"]].append(row)

    baseline_rows: list[QueryMetrics] = []
    surface_rows: list[QueryMetrics] = []
    structured_rows: list[QueryMetrics] = []
    evaluated_pairs = 0
    stage_counts: dict[str, int] = defaultdict(int)

    for job_id, pool in applications.items():
        job = jobs[job_id]
        if job.get("create_date", "") < cutoff or len(pool) < min_pool:
            continue
        labels = {row["candidate_id"]: STAGE_RELEVANCE.get(row["last_stage_reached"], 0) for row in pool}
        if len(set(labels.values())) < 2 or not any(value >= 2 for value in labels.values()):
            continue
        for row in pool:
            stage_counts[row["last_stage_reached"]] += 1
        evaluated_pairs += len(pool)
        baseline_order = sorted(labels, key=lambda cid: (-keyword_score(job, candidates[cid]), cid))
        surface_order = sorted(labels, key=lambda cid: (-surface_attribute_score(job, candidates[cid]), cid))
        structured_order = sorted(labels, key=lambda cid: (-structured_score(job, candidates[cid]), cid))
        baseline = query_metrics(baseline_order, labels)
        surface = query_metrics(surface_order, labels)
        structured = query_metrics(structured_order, labels)
        baseline_rows.append(baseline)
        surface_rows.append(surface)
        structured_rows.append(structured)

    baseline_summary = _mean_metrics(baseline_rows)
    surface_summary = _mean_metrics(surface_rows)
    structured_summary = _mean_metrics(structured_rows)
    return {
        "benchmark": "JTH recruiter-history ranking",
        "label_type": "behavioral weak label from last_stage_reached; not ground-truth job fit",
        "split": {
            "job_create_date_gte": cutoff,
            "minimum_applicants": min_pool,
            "weight_tuning": "2022 jobs",
            "weight_selection": "2023 jobs",
            "sealed_test": "2024+ jobs",
        },
        "scope": {
            "queries": len(structured_rows), "candidate_job_pairs": evaluated_pairs,
            "relevant_stage_minimum": "Resume Sent", "stage_counts": dict(sorted(stage_counts.items())),
        },
        "protected_fields_used": False,
        "excluded_matching_fields": sorted(EXCLUDED_MATCHING_FIELDS),
        "keyword_baseline": baseline_summary,
        "surface_attribute_baseline": surface_summary,
        "structured_matcher": structured_summary,
        "delta_structured_minus_keyword": {
            key: round(structured_summary[key] - baseline_summary[key], 4) for key in structured_summary
        },
        "delta_structured_minus_surface_attribute": {
            key: round(structured_summary[key] - surface_summary[key], 4) for key in structured_summary
        },
        "paired_bootstrap_95ci_delta": _paired_bootstrap_ci(baseline_rows, structured_rows),
        "paired_bootstrap_95ci_delta_vs_surface_attribute": _paired_bootstrap_ci(surface_rows, structured_rows),
        "per_query_ndcg_comparison": _ndcg_win_tie_loss(baseline_rows, structured_rows),
        "per_query_ndcg_comparison_vs_surface_attribute": _ndcg_win_tie_loss(surface_rows, structured_rows),
        "structured_weights": {key: round(value, 6) for key, value in STRUCTURED_WEIGHTS.items()},
        "limitations": [
            "Only actual historical applicant pools are ranked; non-applicants are not sampled as negatives.",
            "Recruiter progression is affected by historical policy and bias and is only a weak relevance proxy.",
            "The benchmark contains structured/pseudonymized attributes, not original JD or resume text.",
            "JTH is France-specific and licensed CC BY-NC 4.0 for non-commercial use.",
        ],
    }
