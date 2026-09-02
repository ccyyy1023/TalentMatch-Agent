from __future__ import annotations

from itertools import product

from app.services.talentclef_benchmark import evaluate_rankings


RERANK_TOP_K_GRID = (3, 4, 5, 6)
BASE_WEIGHT_GRID = tuple(round(value / 10, 1) for value in range(11))


def _normalize(values: dict[str, float]) -> dict[str, float]:
    minimum = min(values.values(), default=0.0)
    maximum = max(values.values(), default=0.0)
    width = maximum - minimum
    return {key: (value - minimum) / width if width else 0.0 for key, value in values.items()}


def rerank_top_k(
    base_rows: list[tuple[str, float]],
    direct_scores: dict[str, float],
    valid_scores: set[str],
    *,
    top_k: int,
    base_weight: float,
) -> list[tuple[str, float]]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not 0 <= base_weight <= 1:
        raise ValueError("base_weight must be between zero and one")
    head = base_rows[:top_k]
    tail = base_rows[top_k:]
    base = _normalize({item: float(score) for item, score in head})
    accepted = {item: float(direct_scores[item]) for item, _ in head if item in valid_scores and item in direct_scores}
    direct = _normalize(accepted)
    fused = []
    for candidate_id, _ in head:
        # Invalid or missing quotes cannot affect ranking; retain the deterministic base value.
        llm_value = direct.get(candidate_id, base[candidate_id])
        fused.append((candidate_id, base_weight * base[candidate_id] + (1 - base_weight) * llm_value))
    fused.sort(key=lambda row: (-row[1], row[0]))
    if not tail:
        return fused
    floor = min((score for _, score in fused), default=0.0) - 1.0
    return fused + [(candidate_id, floor - index) for index, (candidate_id, _) in enumerate(tail, start=1)]


def leave_one_job_out_rerank(
    base_rankings: dict[str, list[tuple[str, float]]],
    direct_scores: dict[str, dict[str, float]],
    valid_scores: dict[str, set[str]],
    qrels: dict[str, dict[str, int]],
) -> dict:
    query_ids = sorted(qrels)
    if len(query_ids) < 3:
        raise ValueError("leave-one-job-out evaluation requires at least three jobs")

    def metrics(rankings, selected):
        return evaluate_rankings(
            {query_id: rankings[query_id] for query_id in selected},
            {query_id: qrels[query_id] for query_id in selected},
        )[0]

    selected_by_fold = {}
    cross_validated = {}
    for holdout in query_ids:
        tuning = [query_id for query_id in query_ids if query_id != holdout]
        trials = []
        for top_k, base_weight in product(RERANK_TOP_K_GRID, BASE_WEIGHT_GRID):
            rankings = {
                query_id: rerank_top_k(
                    base_rankings[query_id], direct_scores[query_id], valid_scores[query_id],
                    top_k=top_k, base_weight=base_weight,
                )
                for query_id in tuning
            }
            trial_metrics = metrics(rankings, tuning)
            trials.append((trial_metrics["ndcg"], trial_metrics["map"], base_weight, top_k))
        _, _, base_weight, top_k = max(trials, key=lambda row: (row[0], row[1], row[2], -row[3]))
        selected_by_fold[holdout] = {"top_k": top_k, "base_weight": base_weight}
        cross_validated[holdout] = rerank_top_k(
            base_rankings[holdout], direct_scores[holdout], valid_scores[holdout],
            top_k=top_k, base_weight=base_weight,
        )

    baseline_metrics = metrics(base_rankings, query_ids)
    reranked_metrics, per_query = evaluate_rankings(cross_validated, qrels)
    delta = {key: round(reranked_metrics[key] - baseline_metrics[key], 6) for key in baseline_metrics}
    eligible = delta["ndcg"] >= 0.01 and delta["map"] >= 0
    return {
        "protocol": "leave_one_job_out_parameter_selection",
        "selected_by_fold": selected_by_fold,
        "baseline_metrics": baseline_metrics,
        "reranked_metrics": reranked_metrics,
        "delta_reranked_minus_baseline": delta,
        "per_query": per_query,
        "activation_gate": {
            "eligible": eligible,
            "requirements": "NDCG delta >= 0.01 and MAP delta >= 0 on job-held-out predictions",
        },
    }
