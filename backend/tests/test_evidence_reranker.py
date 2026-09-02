from app.services.evidence_reranker import leave_one_job_out_rerank, rerank_top_k


def test_invalid_quote_score_cannot_change_base_order():
    base = [("a", 0.9), ("b", 0.8), ("c", 0.1)]
    output = rerank_top_k(base, {"a": 0, "b": 100}, {"a"}, top_k=2, base_weight=0.0)
    assert [item[0] for item in output] == ["a", "b", "c"]


def test_valid_grounded_score_can_rerank_only_the_head():
    base = [("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.1)]
    output = rerank_top_k(
        base, {"a": 0, "b": 100, "c": 50}, {"a", "b", "c"}, top_k=3, base_weight=0.0,
    )
    assert [item[0] for item in output] == ["b", "c", "a", "d"]


def test_leave_one_job_out_never_tunes_on_held_out_job():
    base = {
        "j1": [("a", 2), ("b", 1), ("c", 0)],
        "j2": [("a", 2), ("b", 1), ("c", 0)],
        "j3": [("a", 2), ("b", 1), ("c", 0)],
    }
    direct = {job: {"a": 0, "b": 100, "c": 50} for job in base}
    valid = {job: {"a", "b", "c"} for job in base}
    qrels = {job: {"a": 0, "b": 2, "c": 1} for job in base}
    report = leave_one_job_out_rerank(base, direct, valid, qrels)
    assert set(report["selected_by_fold"]) == set(base)
    assert report["reranked_metrics"]["ndcg"] > report["baseline_metrics"]["ndcg"]
    assert report["activation_gate"]["eligible"] is True
