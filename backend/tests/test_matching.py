from app.services.analyzers import CandidateAnalyzer, JDAnalyzer
from app.schemas import Evidence, ParsedCandidate, ParsedJD, Priority, Requirement
from app.services.matcher import MatchingEngine, PRIORITY_GROUP_WEIGHTS
from app.services.ollama_client import OllamaClient
from app.services.reviewer import ConflictReviewer


def build_results(sample_request):
    client = OllamaClient()
    job, _ = JDAnalyzer(client).analyze(sample_request.job_description, "rules")
    analyzer = CandidateAnalyzer(client)
    candidates = [analyzer.analyze(item.id, item.name, item.text, "rules")[0] for item in sample_request.candidates]
    engine = MatchingEngine()
    return client, job, candidates, [engine.match(job, item) for item in candidates]


def test_strong_candidate_scores_above_irrelevant_candidate(sample_request):
    _, _, _, results = build_results(sample_request)
    by_id = {item.candidate_id: item for item in results}
    assert by_id["cand-001"].score > by_id["cand-006"].score


def test_hard_requirement_missing_is_visible(sample_request):
    _, _, _, results = build_results(sample_request)
    irrelevant = next(item for item in results if item.candidate_id == "cand-006")
    assert irrelevant.recommendation == "insufficient_hard_requirement_evidence"
    assert any(finding.code == "HARD_REQUIREMENT_MISSING" for finding in irrelevant.findings)


def test_every_matched_criterion_has_evidence(sample_request):
    _, _, _, results = build_results(sample_request)
    for result in results:
        for criterion in result.criteria:
            if criterion.status.value == "matched":
                assert criterion.evidence_ids


def test_weak_skill_list_triggers_review(sample_request):
    client, job, candidates, results = build_results(sample_request)
    candidate = next(item for item in candidates if item.id == "cand-008")
    result = next(item for item in results if item.candidate_id == "cand-008")
    reviewed = ConflictReviewer(client).review(job, candidate, result, "rules")
    assert reviewed.recommendation == "manual_review" or reviewed.recommendation == "insufficient_hard_requirement_evidence"
    assert any(f.code == "SKILLS_WITHOUT_PROJECT_EVIDENCE" for f in reviewed.findings)


def test_candidate_display_name_is_anonymous(sample_request):
    _, _, _, results = build_results(sample_request)
    assert all(result.display_name.startswith("候选人-") for result in results)


def test_llm_reviewer_routes_only_ambiguous_noncritical_cases(sample_request):
    client, _, candidates, results = build_results(sample_request)
    reviewer = ConflictReviewer(client)
    decisions = [
        reviewer.needs_llm_review(candidate, result, "ollama")
        for candidate, result in zip(candidates, results)
    ]
    assert any(decisions)
    assert not all(decisions)
    for result, decision in zip(results, decisions):
        if any(item.severity == "critical" for item in result.findings):
            assert decision is False


def test_priority_group_weights_are_explicit_and_normalized():
    assert PRIORITY_GROUP_WEIGHTS == {
        Priority.hard: 0.85,
        Priority.preferred: 0.10,
        Priority.context: 0.05,
    }
    assert sum(PRIORITY_GROUP_WEIGHTS.values()) == 1.0


def test_repeated_context_clauses_do_not_dilute_hard_requirement_group():
    candidate = ParsedCandidate(
        id="candidate-group-test",
        display_name="测试候选人",
        masked_name="候选人-TEST",
        skills=["python"],
        evidence=[Evidence(
            id="ev-python", kind="skill", value="Python", normalized_skill="python",
            source_quote="项目经历：使用Python完成服务开发。", section="project", strength=1.0,
        )],
    )
    hard = Requirement(
        id="hard-python", text="Python能力", category="skill", priority=Priority.hard,
        normalized_skill="python", source_quote="必须掌握Python",
    )
    context = Requirement(
        id="context-1", text="负责跨团队协作", category="responsibility", priority=Priority.context,
        source_quote="负责跨团队协作",
    )
    engine = MatchingEngine()
    one_context = engine.match(ParsedJD(title="测试岗位", requirements=[hard, context]), candidate)
    repeated_contexts = [
        context.model_copy(update={"id": f"context-{index}", "text": f"上下文职责{index}"})
        for index in range(1, 11)
    ]
    many_contexts = engine.match(ParsedJD(title="测试岗位", requirements=[hard, *repeated_contexts]), candidate)
    assert one_context.score == many_contexts.score
