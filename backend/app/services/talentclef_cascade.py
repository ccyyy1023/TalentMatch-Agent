from __future__ import annotations

import random

from app.services.talentclef_benchmark import evaluate_rankings


CASCADE_ALPHA_GRID = tuple(round(value / 10, 1) for value in range(11))


def _normalize(rows: list[tuple[str, float]]) -> dict[str, float]:
    values = {item: float(score) for item, score in rows}
    minimum = min(values.values(), default=0.0)
    maximum = max(values.values(), default=0.0)
    width = maximum - minimum
    return {item: (score - minimum) / width if width > 0 else 0.0 for item, score in values.items()}


def fuse_rankings(
    lexical: dict[str, list[tuple[str, float]]],
    evidence: dict[str, list[tuple[str, float]]],
    lexical_weight: float,
) -> dict[str, list[tuple[str, float]]]:
    if not 0 <= lexical_weight <= 1:
        raise ValueError("lexical_weight must be between zero and one")
    if set(lexical) != set(evidence):
        raise ValueError("lexical and evidence query IDs must match")
    output = {}
    for query_id in lexical:
        left = _normalize(lexical[query_id])
        right = _normalize(evidence[query_id])
        if set(left) != set(right):
            raise ValueError(f"candidate IDs differ for query {query_id}")
        output[query_id] = sorted(
            (
                (item, lexical_weight * left[item] + (1 - lexical_weight) * right[item])
                for item in left
            ),
            key=lambda row: (-row[1], row[0]),
        )
    return output


def tune_and_evaluate_cascade(
    lexical: dict[str, list[tuple[str, float]]],
    evidence: dict[str, list[tuple[str, float]]],
    qrels: dict[str, dict[str, int]],
    *,
    seed: int = 20260902,
    tuning_fraction: float = 0.7,
) -> dict:
    query_ids = sorted(qrels)
    if len(query_ids) < 3:
        raise ValueError("cascade tuning requires at least three queries")
    rng = random.Random(seed)
    rng.shuffle(query_ids)
    tuning_size = min(len(query_ids) - 1, max(2, round(len(query_ids) * tuning_fraction)))
    tuning = tuple(sorted(query_ids[:tuning_size]))
    holdout = tuple(sorted(query_ids[tuning_size:]))

    def metrics(rankings, selected):
        return evaluate_rankings(
            {item: rankings[item] for item in selected},
            {item: qrels[item] for item in selected},
        )[0]

    trials = []
    for weight in CASCADE_ALPHA_GRID:
        rankings = fuse_rankings(lexical, evidence, weight)
        trials.append({"lexical_weight": weight, "metrics": metrics(rankings, tuning)})
    selected = max(
        trials,
        key=lambda row: (row["metrics"]["ndcg"], row["metrics"]["map"], -abs(row["lexical_weight"] - 0.5)),
    )
    rankings = fuse_rankings(lexical, evidence, selected["lexical_weight"])
    lexical_holdout = metrics(lexical, holdout)
    evidence_holdout = metrics(evidence, holdout)
    cascade_holdout = metrics(rankings, holdout)
    return {
        "split": {"seed": seed, "tuning_queries": list(tuning), "holdout_queries": list(holdout)},
        "selected_lexical_weight": selected["lexical_weight"],
        "tuning_trials": trials,
        "holdout": {
            "lexical": lexical_holdout,
            "evidence": evidence_holdout,
            "cascade": cascade_holdout,
            "delta_cascade_minus_lexical": {
                key: round(cascade_holdout[key] - lexical_holdout[key], 6) for key in lexical_holdout
            },
            "delta_cascade_minus_evidence": {
                key: round(cascade_holdout[key] - evidence_holdout[key], 6) for key in evidence_holdout
            },
        },
    }
