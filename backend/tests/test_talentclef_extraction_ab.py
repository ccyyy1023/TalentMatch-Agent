from pathlib import Path

import pytest

from app.services.talentclef_benchmark import TalentClefDataset
from app.services.talentclef_extraction_ab import (
    build_stratified_sample,
    compare_model_reports,
    evaluate_raw_sample,
)


def _dataset() -> TalentClefDataset:
    queries = {"q1": "python data", "q2": "clinical nurse", "q3": "retail sales"}
    corpus = {
        "c1": "python data engineer", "c2": "sql python", "c3": "clinical nurse",
        "c4": "patient care", "c5": "retail sales", "c6": "inventory sales",
    }
    return TalentClefDataset(
        split="development",
        language="en",
        queries=queries,
        corpus=corpus,
        qrels={"q1": {"c1": 1, "c2": 1}, "q2": {"c3": 1, "c4": 1}, "q3": {"c5": 1, "c6": 1}},
    )


def test_stratified_sample_is_reproducible_and_balanced():
    left = build_stratified_sample(_dataset(), query_limit=2, positives_per_query=1, negatives_per_query=1, seed=7)
    right = build_stratified_sample(_dataset(), query_limit=2, positives_per_query=1, negatives_per_query=1, seed=7)
    assert left == right
    assert all(sorted(labels.values()) == [0, 1] for labels in left.labels.values())
    assert all(len(pool) == 2 for pool in left.pools.values())


def test_raw_sample_metrics_are_computable():
    sample = build_stratified_sample(_dataset(), query_limit=2, positives_per_query=1, negatives_per_query=1)
    metrics = evaluate_raw_sample(_dataset(), sample)
    assert set(metrics) == {"map", "mrr", "ndcg", "precision_at_5", "precision_at_10", "precision_at_100"}
    assert metrics["mrr"] == 1.0


def test_comparison_blocks_claim_when_either_model_falls_back():
    base = {
        "model": "left",
        "fallbacks": {"total": 0},
        "model_calls_expected": 2,
        "sample_ranking_metrics": {"map": 0.5, "mrr": 1.0},
    }
    other = {
        "model": "right",
        "fallbacks": {"total": 1},
        "model_calls_expected": 2,
        "sample_ranking_metrics": {"map": 0.6, "mrr": 1.0},
    }
    comparison = compare_model_reports(base, other)
    assert comparison["delta_right_minus_left"]["map"] == 0.1
    assert comparison["model_only_interpretation_gate"] == "invalid_for_model_quality_claim_due_to_fallbacks"
    assert comparison["operational_pipeline_gate"] == "eligible_with_fallback_rates_disclosed"


def test_rejects_unlabeled_or_oversized_samples():
    dataset = _dataset()
    dataset_without_qrels = TalentClefDataset(
        split="test", language="en", queries=dataset.queries, corpus=dataset.corpus, qrels=None
    )
    with pytest.raises(ValueError, match="public qrels"):
        build_stratified_sample(dataset_without_qrels)
    with pytest.raises(ValueError, match="query_limit"):
        build_stratified_sample(dataset, query_limit=10)
    with pytest.raises(ValueError, match="negative_strategy"):
        build_stratified_sample(dataset, negative_strategy="unknown")


def test_bm25_hard_negative_strategy_selects_lexically_confusing_documents():
    dataset = TalentClefDataset(
        split="development",
        language="en",
        queries={"q1": "python data engineer"},
        corpus={
            "positive": "python data engineer",
            "hard": "python data engineer internship",
            "easy": "clinical nurse",
        },
        qrels={"q1": {"positive": 1, "hard": 0, "easy": 0}},
    )
    sample = build_stratified_sample(
        dataset, query_limit=1, positives_per_query=1, negatives_per_query=1,
        negative_strategy="bm25_hard",
    )
    assert set(sample.pools["q1"]) == {"positive", "hard"}
