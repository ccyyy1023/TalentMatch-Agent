import threading
import time
from types import SimpleNamespace

from app.schemas import Evidence, ParsedCandidate
from app.services.baselines import keyword_coverage_ranking
from app.services.evaluation import evaluate_run, ranking_metrics
from app.services.workflow import TalentMatchWorkflow


def test_rules_workflow_runs_end_to_end(sample_request):
    response = TalentMatchWorkflow().run(sample_request)
    assert len(response.ranking) == len(sample_request.candidates)
    assert response.traces[0].node == "jd_analyzer_agent"
    assert response.traces[-1].node == "ranking_and_explanation"
    assert response.compliance.automatic_rejection_enabled is False


def test_workflow_reports_monotonic_node_progress(sample_request):
    updates = []
    TalentMatchWorkflow().run(
        sample_request,
        progress_callback=lambda stage, progress, detail: updates.append((stage, progress, detail)),
    )
    stages = [stage for stage, _, _ in updates]
    progress = [value for _, value, _ in updates]
    assert stages[0] == "job_ready"
    assert "candidates_ready" in stages
    assert "matching_ready" in stages
    assert "compliance_ready" in stages
    assert stages[-1] == "ranking_ready"
    assert progress == sorted(progress)
    assert progress[0] == 15 and progress[-1] == 95


def test_ranking_is_descending(sample_request):
    response = TalentMatchWorkflow().run(sample_request)
    scores = [item.score for item in response.ranking]
    assert scores == sorted(scores, reverse=True)


def test_metrics_are_bounded(sample_request):
    response = TalentMatchWorkflow().run(sample_request)
    labels = {item.id: item.relevance_label for item in sample_request.candidates}
    metrics = evaluate_run(response, labels)
    for value in (metrics.ndcg_at_5, metrics.precision_at_3, metrics.recall_at_5, metrics.mrr, metrics.evidence_coverage):
        assert value is not None and 0 <= value <= 1


def test_unsupported_claim_rate_is_zero(sample_request):
    response = TalentMatchWorkflow().run(sample_request)
    labels = {item.id: item.relevance_label for item in sample_request.candidates}
    metrics = evaluate_run(response, labels)
    assert metrics.unsupported_claim_rate == 0


def test_sensitive_attributes_never_enter_score(sample_request):
    response = TalentMatchWorkflow().run(sample_request)
    assert response.compliance.protected_attributes_used_in_score is False


def test_keyword_baseline_is_evaluable(sample_request):
    ranking = keyword_coverage_ranking(sample_request)
    labels = {item.id: item.relevance_label for item in sample_request.candidates}
    metrics = ranking_metrics([item_id for item_id, _ in ranking], labels)
    assert metrics["ndcg_at_5"] is not None


def test_ollama_candidate_agents_run_with_bounded_parallelism_and_keep_order(sample_request, monkeypatch):
    workflow = TalentMatchWorkflow()
    lock = threading.Lock()
    active = 0
    max_active = 0

    class RecordingAnalyzer:
        def analyze(self, candidate_id, name, text, mode):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.03)
            with lock:
                active -= 1
            return ParsedCandidate(id=candidate_id, display_name=name, masked_name=f"候选人-{candidate_id}"), "ollama"

    workflow.candidate_analyzer = RecordingAnalyzer()
    monkeypatch.setattr("app.services.workflow.settings", SimpleNamespace(ollama_workers=2))
    request = sample_request.model_copy(update={"mode": "ollama", "candidates": sample_request.candidates[:4]})
    output = workflow._candidate_node({"request": request, "traces": []})
    assert max_active == 2
    assert [item.id for item in output["candidates"]] == [item.id for item in request.candidates]
    assert "并行度:2" in output["traces"][-1].detail


def test_adaptive_mode_routes_only_weak_candidate_to_llm(sample_request, monkeypatch):
    workflow = TalentMatchWorkflow()
    calls = []

    class AdaptiveAnalyzer:
        def analyze(self, candidate_id, name, text, mode):
            calls.append((candidate_id, mode))
            if mode == "rules" and candidate_id.endswith("001"):
                return ParsedCandidate(
                    id=candidate_id, display_name=name, masked_name="候选人-STRONG",
                    years_experience=3, education="本科",
                    evidence=[Evidence(
                        id="ev-1", kind="skill", value="Python", normalized_skill="python",
                        source_quote="项目使用Python", section="project", strength=1.0,
                    )],
                ), "rules"
            if mode == "rules":
                return ParsedCandidate(
                    id=candidate_id, display_name=name, masked_name="候选人-WEAK",
                    parse_warnings=["未从简历中提取到已知技能"],
                ), "rules"
            return ParsedCandidate(
                id=candidate_id, display_name=name, masked_name="候选人-ENRICHED",
                years_experience=2, education="本科",
            ), "ollama"

    workflow.candidate_analyzer = AdaptiveAnalyzer()
    monkeypatch.setattr("app.services.workflow.settings", SimpleNamespace(ollama_workers=2))
    request = sample_request.model_copy(update={"mode": "adaptive", "candidates": sample_request.candidates[:2]})
    output = workflow._candidate_node({"request": request, "traces": []})
    assert calls.count((request.candidates[0].id, "ollama")) == 0
    assert calls.count((request.candidates[1].id, "ollama")) == 1
    assert "LLM路由:1/2" in output["traces"][-1].detail
