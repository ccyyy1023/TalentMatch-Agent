from __future__ import annotations

import math

from app.schemas import AnalysisResponse, EvaluationMetrics


def _dcg(labels: list[int], k: int) -> float:
    return sum((2 ** label - 1) / math.log2(index + 2) for index, label in enumerate(labels[:k]))


def ranking_metrics(order: list[str], labels: dict[str, int]) -> dict[str, float | None]:
    ranked_labels = [labels[item_id] for item_id in order if item_id in labels]
    ideal = sorted(ranked_labels, reverse=True)
    ideal_dcg = _dcg(ideal, 5)
    relevant_total = sum(label > 0 for label in labels.values())
    top3 = ranked_labels[:3]
    top5 = ranked_labels[:5]
    ndcg = _dcg(ranked_labels, 5) / ideal_dcg if ideal_dcg else None
    precision = sum(label > 0 for label in top3) / len(top3) if top3 else None
    recall = sum(label > 0 for label in top5) / relevant_total if relevant_total else None
    mrr = next((1 / (index + 1) for index, label in enumerate(ranked_labels) if label > 0), None)
    return {
        "ndcg_at_5": round(ndcg, 4) if ndcg is not None else None,
        "precision_at_3": round(precision, 4) if precision is not None else None,
        "recall_at_5": round(recall, 4) if recall is not None else None,
        "mrr": round(mrr, 4) if mrr is not None else None,
    }


def evaluate_run(response: AnalysisResponse, labels: dict[str, int]) -> EvaluationMetrics:
    ranked_ids = [item.candidate_id for item in response.ranking]
    ranked_labels = [labels[item_id] for item_id in ranked_ids if item_id in labels]
    ranking = ranking_metrics(ranked_ids, labels)
    criteria = [criterion for result in response.ranking for criterion in result.criteria]
    supported = sum(bool(item.evidence_ids) or item.status.value in {"missing", "review"} for item in criteria)
    evidence_coverage = supported / len(criteria) if criteria else 1.0
    unsupported = sum(item.status.value == "matched" and not item.evidence_ids for item in criteria)
    unsupported_rate = unsupported / len(criteria) if criteria else 0.0
    manual_review = sum(result.recommendation == "manual_review" for result in response.ranking)
    return EvaluationMetrics(
        run_id=response.run_id, candidate_count=len(response.ranking), labeled_count=len(ranked_labels),
        ndcg_at_5=ranking["ndcg_at_5"], precision_at_3=ranking["precision_at_3"],
        recall_at_5=ranking["recall_at_5"], mrr=ranking["mrr"],
        evidence_coverage=round(evidence_coverage, 4), unsupported_claim_rate=round(unsupported_rate, 4),
        manual_review_rate=round(manual_review / len(response.ranking), 4) if response.ranking else 0.0,
    )
