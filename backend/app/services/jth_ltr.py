from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from app.services.jth_benchmark import (
    EXCLUDED_MATCHING_FIELDS,
    _mean_metrics,
    _ndcg_win_tie_loss,
    _ordinal,
    _paired_bootstrap_ci,
    _read_csv,
    _tokens,
    highest_stage_labels,
    query_metrics,
    structured_features,
    structured_score,
    DIPLOMA_LEVELS,
    SENIORITY_LEVELS,
)


SKILL_ALIASES = {
    "postgres": "postgresql", "postgre sql": "postgresql",
    "js": "javascript", "node js": "node.js", "nodejs": "node.js",
    "k8s": "kubernetes", "py torch": "pytorch", "tf": "tensorflow",
    "ml": "machine learning", "ai": "artificial intelligence",
    "nlp": "natural language processing", "restful api": "rest api",
}

BOOLEAN_PAIRS = {
    "management_experience_match": ("llm_required_management_experience", "llm_management_experience"),
    "startup_experience_match": ("llm_is_startup", "llm_startup_experience"),
    "large_company_experience_match": ("llm_is_large_company", "llm_large_company_experience"),
    "freelance_experience_match": ("llm_required_freelance_experience", "llm_freelance_experience"),
    "contract_experience_match": ("llm_required_contract_experience", "llm_contract_experience"),
    "international_experience_match": (
        "llm_required_international_work_experience", "llm_international_work_experience",
    ),
    "leadership_experience_match": ("llm_required_leadership_experience", "llm_leadership_experience"),
    "client_facing_match": ("llm_client_facing_role", "llm_client_facing_role"),
}

BASE_FEATURE_NAMES = tuple(structured_features({}, {}))
EXTRA_FEATURE_NAMES = (
    "fixed_structured_score", "idf_skill_coverage", "skill_precision", "skill_gap",
    "matched_skill_count", "experience_known", "experience_shortfall",
    "seniority_gap", "diploma_gap",
    *BOOLEAN_PAIRS,
)
FEATURE_NAMES = BASE_FEATURE_NAMES + EXTRA_FEATURE_NAMES
NEGATIVE_MONOTONIC_FEATURES = {
    "skill_gap", "experience_shortfall", "seniority_gap", "diploma_gap",
}
UNCONSTRAINED_FEATURES = {"experience_known"}
MONOTONIC_CONSTRAINTS = tuple(
    0 if name in UNCONSTRAINED_FEATURES else -1 if name in NEGATIVE_MONOTONIC_FEATURES else 1
    for name in FEATURE_NAMES
)


@dataclass(frozen=True)
class RankingGroup:
    job_id: str
    create_date: str
    job: dict[str, str]
    candidates: tuple[dict[str, str], ...]
    candidate_ids: tuple[str, ...]
    labels: tuple[int, ...]


def _canonical_skills(*values: str) -> set[str]:
    output = set()
    for value in _tokens(*values):
        collapsed = " ".join(value.replace("_", " ").replace("-", " ").split())
        output.add(SKILL_ALIASES.get(collapsed, collapsed))
    return output


def _number(value: str) -> float | None:
    digits = []
    seen_digit = False
    for char in value.strip().lower():
        if char.isdigit() or (char == "." and seen_digit):
            digits.append(char)
            seen_digit = seen_digit or char.isdigit()
        elif digits:
            break
    try:
        return float("".join(digits)) if digits else None
    except ValueError:
        return None


def _truth(value: str) -> bool | None:
    normalized = value.strip().casefold()
    if normalized in {"true", "yes", "1", "y", "oui"}:
        return True
    if normalized in {"false", "no", "0", "n", "non"}:
        return False
    return None


def _required_boolean_match(required: str, available: str) -> float:
    required_value = _truth(required)
    available_value = _truth(available)
    if required_value is not True:
        return 0.5
    if available_value is None:
        return 0.25
    return float(available_value)


def _ordinal_gap(required: str, available: str, levels: dict[str, int]) -> float:
    required_rank = _ordinal(required, levels)
    available_rank = _ordinal(available, levels)
    if required_rank is None or available_rank is None:
        return 0.0
    return float(max(0, required_rank - available_rank))


def ranking_features(
    job: dict[str, str], candidate: dict[str, str],
    document_frequency: Counter[str], pool_size: int,
) -> dict[str, float]:
    base = structured_features(job, candidate)
    required = _canonical_skills(
        job.get("skills", ""), job.get("llm_hard_skills", ""),
        job.get("llm_programming_languages", ""), job.get("llm_tools_technologies", ""),
    )
    available = _canonical_skills(
        candidate.get("skills", ""), candidate.get("llm_hard_skills", ""),
        candidate.get("llm_programming_languages", ""), candidate.get("llm_tools_technologies", ""),
    )
    matched = required & available
    weights = {skill: math.log((pool_size + 1) / (document_frequency.get(skill, 0) + 1)) + 1 for skill in required}
    denominator = sum(weights.values())
    idf_coverage = sum(weights[skill] for skill in matched) / denominator if denominator else 0.0

    required_years = _number(job.get("llm_required_years_of_work_experience", "") or job.get("years_experience", ""))
    available_years = _number(candidate.get("llm_years_of_work_experience", "") or candidate.get("years_experience", ""))
    years_known = float(required_years is not None and available_years is not None)
    shortfall = max(0.0, required_years - available_years) if years_known else 0.0
    extras = {
        "fixed_structured_score": structured_score(job, candidate),
        "idf_skill_coverage": idf_coverage,
        "skill_precision": len(matched) / len(available) if available else 0.0,
        "skill_gap": len(required - available) / len(required) if required else 0.0,
        "matched_skill_count": float(len(matched)),
        "experience_known": years_known,
        "experience_shortfall": shortfall,
        "seniority_gap": _ordinal_gap(
            job.get("llm_seniority_level", ""), candidate.get("llm_seniority_level", ""), SENIORITY_LEVELS,
        ),
        "diploma_gap": _ordinal_gap(
            job.get("llm_required_lowest_diploma", ""), candidate.get("llm_highest_diploma", ""), DIPLOMA_LEVELS,
        ),
    }
    extras.update({
        name: _required_boolean_match(job.get(job_field, ""), candidate.get(candidate_field, ""))
        for name, (job_field, candidate_field) in BOOLEAN_PAIRS.items()
    })
    return {**base, **extras}


def load_ranking_groups(data_dir: Path, min_pool: int = 5) -> list[RankingGroup]:
    candidates = _read_csv(data_dir / "candidates.csv", "candidate_id")
    jobs = _read_csv(data_dir / "jobs.csv", "job_id")
    applications: dict[str, list[dict[str, str]]] = defaultdict(list)
    with (data_dir / "history.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["job_id"] in jobs and row["candidate_id"] in candidates:
                applications[row["job_id"]].append(row)
    groups = []
    for job_id in sorted(applications):
        pool = applications[job_id]
        label_by_candidate = highest_stage_labels(pool)
        labels = tuple(label_by_candidate.values())
        if len(labels) < min_pool or len(set(labels)) < 2 or not any(value >= 2 for value in labels):
            continue
        candidate_ids = tuple(label_by_candidate)
        groups.append(RankingGroup(
            job_id=job_id, create_date=jobs[job_id].get("create_date", ""), job=jobs[job_id],
            candidates=tuple(candidates[candidate_id] for candidate_id in candidate_ids),
            candidate_ids=candidate_ids, labels=labels,
        ))
    return groups


def _year(group: RankingGroup) -> int:
    try:
        return int(group.create_date[:4])
    except ValueError:
        return 0


def rolling_splits(groups: list[RankingGroup]) -> list[tuple[str, list[RankingGroup], list[RankingGroup]]]:
    return [
        (
            "train_2019_2022_validate_2023",
            [group for group in groups if 2019 <= _year(group) <= 2022],
            [group for group in groups if _year(group) == 2023],
        ),
        (
            "train_2019_2023_validate_2024",
            [group for group in groups if 2019 <= _year(group) <= 2023],
            [group for group in groups if _year(group) == 2024],
        ),
    ]


def _matrix(groups: list[RankingGroup]) -> tuple[list[list[float]], list[int], list[int]]:
    rows = []
    labels = []
    sizes = []
    for group in groups:
        candidate_skill_sets = [
            _canonical_skills(
                item.get("skills", ""), item.get("llm_hard_skills", ""),
                item.get("llm_programming_languages", ""), item.get("llm_tools_technologies", ""),
            )
            for item in group.candidates
        ]
        frequency = Counter(skill for skills in candidate_skill_sets for skill in skills)
        for candidate, label in zip(group.candidates, group.labels):
            feature_map = ranking_features(group.job, candidate, frequency, len(group.candidates))
            rows.append([feature_map[name] for name in FEATURE_NAMES])
            labels.append(label)
        sizes.append(len(group.candidates))
    return rows, labels, sizes


def _evaluate_predictions(groups: list[RankingGroup], predictions: Sequence[float]):
    metrics = []
    offset = 0
    for group in groups:
        size = len(group.candidates)
        scores = predictions[offset:offset + size]
        order = [
            candidate_id for candidate_id, _ in sorted(
                zip(group.candidate_ids, scores), key=lambda row: (-float(row[1]), row[0]),
            )
        ]
        metrics.append(query_metrics(order, dict(zip(group.candidate_ids, group.labels))))
        offset += size
    return metrics


def _fixed_predictions(groups: list[RankingGroup]) -> list[float]:
    return [
        structured_score(group.job, candidate)
        for group in groups
        for candidate in group.candidates
    ]


def _blend_predictions(
    groups: list[RankingGroup], fixed: Sequence[float], learned: Sequence[float], alpha: float,
) -> list[float]:
    """Blend within-query percentiles so model score scale cannot dominate the rule anchor."""
    output = [0.0] * len(learned)
    offset = 0
    for group in groups:
        size = len(group.candidates)
        stop = offset + size
        fixed_slice = fixed[offset:stop]
        learned_slice = learned[offset:stop]
        fixed_order = [0] * size
        learned_order = [0] * size
        for rank, index in enumerate(sorted(range(size), key=lambda index: (fixed_slice[index], index))):
            fixed_order[index] = rank
        for rank, index in enumerate(sorted(range(size), key=lambda index: (learned_slice[index], index))):
            learned_order[index] = rank
        denominator = max(size - 1, 1)
        output[offset:stop] = [
            (1.0 - alpha) * fixed_rank / denominator + alpha * learned_rank / denominator
            for fixed_rank, learned_rank in zip(fixed_order, learned_order)
        ]
        offset = stop
    return output


def _fixed_metrics(groups: list[RankingGroup]):
    output = []
    for group in groups:
        scores = {
            candidate_id: structured_score(group.job, candidate)
            for candidate_id, candidate in zip(group.candidate_ids, group.candidates)
        }
        order = sorted(group.candidate_ids, key=lambda candidate_id: (-scores[candidate_id], candidate_id))
        output.append(query_metrics(order, dict(zip(group.candidate_ids, group.labels))))
    return output


def train_and_evaluate_lambdamart(data_dir: Path, min_pool: int = 5, seed: int = 20260902) -> tuple[dict, object]:
    try:
        from lightgbm import LGBMRanker
    except ImportError as exc:
        raise RuntimeError("Install requirements-ranking.txt before running LambdaMART evaluation") from exc

    groups = load_ranking_groups(data_dir, min_pool=min_pool)
    folds = rolling_splits(groups)
    test = [group for group in groups if _year(group) == 2025]
    if not test or any(not training or not validation for _, training, validation in folds):
        raise ValueError("JTH rolling temporal split produced an empty partition")
    fold_cache = []
    for fold_name, training, validation in folds:
        x_train, y_train, train_sizes = _matrix(training)
        x_validation, _, _ = _matrix(validation)
        fold_cache.append({
            "name": fold_name,
            "training": training,
            "validation": validation,
            "x_train": x_train,
            "y_train": y_train,
            "train_sizes": train_sizes,
            "x_validation": x_validation,
            "fixed_predictions": _fixed_predictions(validation),
            "baseline": _mean_metrics(_fixed_metrics(validation)),
        })

    trials = []
    best = None
    for num_leaves, learning_rate, min_child_samples, n_estimators, alpha in product(
        (7, 15), (0.02, 0.05), (30, 60), (80, 160), (0.10, 0.20, 0.35),
    ):
        params = {
            "objective": "lambdarank",
            "label_gain": [0, 1, 3, 7, 15], "num_leaves": num_leaves,
            "learning_rate": learning_rate, "min_child_samples": min_child_samples,
            "n_estimators": n_estimators, "reg_lambda": 1.0, "random_state": seed,
            "n_jobs": -1, "verbosity": -1, "deterministic": True, "force_col_wise": True,
            "monotone_constraints": list(MONOTONIC_CONSTRAINTS),
        }
        fold_results = []
        for fold in fold_cache:
            model = LGBMRanker(**params)
            model.fit(fold["x_train"], fold["y_train"], group=fold["train_sizes"])
            learned = model.booster_.predict(fold["x_validation"])
            blended = _blend_predictions(
                fold["validation"], fold["fixed_predictions"], learned, alpha,
            )
            summary = _mean_metrics(_evaluate_predictions(fold["validation"], blended))
            fold_results.append({
                "fold": fold["name"],
                "baseline": fold["baseline"],
                "blended": summary,
                "delta_ndcg_at_5": round(summary["ndcg_at_5"] - fold["baseline"]["ndcg_at_5"], 4),
            })
        mean_delta = sum(row["delta_ndcg_at_5"] for row in fold_results) / len(fold_results)
        worst_delta = min(row["delta_ndcg_at_5"] for row in fold_results)
        trials.append({
            "params": {key: params[key] for key in ("num_leaves", "learning_rate", "min_child_samples", "n_estimators")},
            "anchor_alpha": alpha,
            "folds": fold_results,
            "mean_delta_ndcg_at_5": round(mean_delta, 4),
            "worst_delta_ndcg_at_5": round(worst_delta, 4),
        })
        score = (worst_delta, mean_delta, -alpha, -num_leaves, -n_estimators)
        if best is None or score > best[0]:
            best = (score, params, alpha, fold_results)
    assert best is not None

    # Refit once on all historical jobs after locking hyperparameters on 2023/2024.
    pretest = [group for group in groups if 2019 <= _year(group) <= 2024]
    x_pretest, y_pretest, pretest_sizes = _matrix(pretest)
    x_test, _, _ = _matrix(test)
    test_baseline = _fixed_metrics(test)
    model = LGBMRanker(**best[1])
    model.fit(x_pretest, y_pretest, group=pretest_sizes)
    test_predictions = _blend_predictions(
        test, _fixed_predictions(test), model.booster_.predict(x_test), best[2],
    )
    test_rows = _evaluate_predictions(test, test_predictions)
    baseline_summary = _mean_metrics(test_baseline)
    test_summary = _mean_metrics(test_rows)
    delta = {key: round(test_summary[key] - baseline_summary[key], 4) for key in baseline_summary}
    confidence = _paired_bootstrap_ci(test_baseline, test_rows, seed=seed)
    eligible = (
        delta["ndcg_at_5"] >= 0.01
        and delta["precision_at_3"] >= 0
        and delta["mrr"] >= 0
        and confidence["ndcg_at_5"][0] > 0
    )
    importance = sorted(
        zip(FEATURE_NAMES, model.booster_.feature_importance(importance_type="gain")),
        key=lambda row: (-float(row[1]), row[0]),
    )
    report = {
        "benchmark": "JTH LambdaMART ranking on non-protected structured match features",
        "split": {
            "selection": "rolling temporal validation on 2023 and 2024", "test": "2025 jobs",
            "groups": {
                "final_training_2019_2024": len(pretest), "test_2025": len(test),
                **{name: {"training": len(training), "validation": len(validation)} for name, training, validation in folds},
            },
            "pairs": {
                "final_training_2019_2024": sum(len(group.candidates) for group in pretest),
                "test_2025": sum(len(group.candidates) for group in test),
            },
        },
        "feature_count": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "excluded_matching_fields": sorted(EXCLUDED_MATCHING_FIELDS),
        "selection_trials": trials,
        "monotonic_constraints": dict(zip(FEATURE_NAMES, MONOTONIC_CONSTRAINTS)),
        "selected_params": {key: best[1][key] for key in ("num_leaves", "learning_rate", "min_child_samples", "n_estimators")},
        "selected_anchor_alpha": best[2],
        "selected_rolling_folds": best[3],
        "test_fixed_structured": baseline_summary,
        "test_lambdamart": test_summary,
        "delta_lambdamart_minus_fixed": delta,
        "paired_bootstrap_95ci_delta": confidence,
        "per_query_ndcg_comparison": _ndcg_win_tie_loss(test_baseline, test_rows),
        "feature_importance_gain": [
            {"feature": name, "gain": round(float(gain), 4)} for name, gain in importance
        ],
        "activation_gate": {
            "eligible": eligible,
            "requirements": "NDCG@5 +0.01, no Precision@3/MRR regression, NDCG paired-bootstrap lower bound > 0",
        },
        "limitations": [
            "Recruiter progression is a behavioral weak label and may encode historical policy or bias.",
            "The benchmark uses structured/pseudonymized JTH attributes rather than original CV and JD text.",
            "The 2025 holdout contains only 50 eligible jobs, so confidence intervals are reported.",
            "The trained artifact is not distributed because JTH is CC BY-NC 4.0 and the online feature adapter is not yet equivalent.",
        ],
    }
    return report, model
