from app.services.hybrid_skill_extractor import JobBertDocumentSkillExtractor
from app.services.skillspan_benchmark import SkillSpan
from app.services.talentclef_benchmark import TalentClefDataset
from app.services.talentclef_hybrid_benchmark import (
    fused_rankings,
    run_hybrid_benchmark,
    split_development_queries,
)


class KeywordPredictor:
    def predict_batch(self, token_batches, batch_size=32):
        outputs = []
        for tokens in token_batches:
            lowered = [item.casefold() for item in tokens]
            matches = []
            for keyword in ("gardening", "nursing", "welding"):
                if keyword in lowered:
                    index = lowered.index(keyword)
                    matches.append(SkillSpan(index, index + 1, "knowledge", keyword))
            outputs.append(matches)
        return outputs


def _dataset():
    queries = {
        "q1": "gardening role", "q2": "nursing role", "q3": "welding role",
        "q4": "gardening work", "q5": "nursing work", "q6": "welding work",
    }
    corpus = {
        "c1": "gardening", "c2": "nursing", "c3": "welding", "c4": "unrelated",
    }
    qrels = {
        query_id: {candidate_id: int(skill in corpus[candidate_id]) for candidate_id in corpus}
        for query_id, skill in {
            "q1": "gardening", "q2": "nursing", "q3": "welding",
            "q4": "gardening", "q5": "nursing", "q6": "welding",
        }.items()
    }
    return TalentClefDataset("development", "en", queries, corpus, qrels)


def test_query_split_is_reproducible_and_disjoint():
    left = split_development_queries(list(_dataset().queries), seed=9)
    right = split_development_queries(list(_dataset().queries), seed=9)
    assert left == right
    assert set(left.tuning).isdisjoint(left.holdout)
    assert set(left.tuning) | set(left.holdout) == set(_dataset().queries)


def test_fused_ranking_rejects_invalid_alpha():
    dataset = _dataset()
    views = {key: value for key, value in dataset.queries.items()}
    candidates = {key: value for key, value in dataset.corpus.items()}
    try:
        fused_rankings(dataset, views, candidates, 1.1)
    except ValueError as error:
        assert "alpha" in str(error)
    else:
        raise AssertionError("invalid alpha must fail")


def test_hybrid_report_keeps_tuning_and_holdout_separate():
    report = run_hybrid_benchmark(
        _dataset(), JobBertDocumentSkillExtractor(predictor=KeywordPredictor()), seed=9,
    )
    assert report["scope"]["ranked_pairs"] == 24
    assert report["claim_gate"]["holdout_not_used_for_alpha_selection"] is True
    assert report["claim_gate"]["resume_metric_eligible"] is False
    assert report["methods"]["jobbert_catalog_fusion"]["holdout_metrics"]["ndcg"] == 1.0
