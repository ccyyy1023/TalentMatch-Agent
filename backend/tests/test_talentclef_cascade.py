import pytest

from app.services.talentclef_cascade import fuse_rankings, tune_and_evaluate_cascade


def _rankings():
    lexical = {
        f"q{i}": [("a", 2.0), ("b", 1.0), ("c", 0.0)]
        for i in range(4)
    }
    evidence = {
        f"q{i}": [("b", 2.0), ("a", 1.0), ("c", 0.0)]
        for i in range(4)
    }
    qrels = {f"q{i}": {"a": 0, "b": 1, "c": 0} for i in range(4)}
    return lexical, evidence, qrels


def test_fusion_validates_weight_and_candidate_scope():
    lexical, evidence, _ = _rankings()
    with pytest.raises(ValueError, match="lexical_weight"):
        fuse_rankings(lexical, evidence, 1.2)
    broken = {**evidence, "q0": [("other", 1.0)]}
    with pytest.raises(ValueError, match="candidate IDs"):
        fuse_rankings(lexical, broken, 0.5)


def test_cascade_selects_on_tuning_and_reports_separate_holdout():
    lexical, evidence, qrels = _rankings()
    report = tune_and_evaluate_cascade(lexical, evidence, qrels, seed=4)
    assert report["selected_lexical_weight"] < 0.5
    assert set(report["split"]["tuning_queries"]).isdisjoint(report["split"]["holdout_queries"])
    assert report["holdout"]["cascade"]["ndcg"] == 1.0
