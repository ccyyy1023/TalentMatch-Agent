from app.services.reviewer import ALLOWED_LLM_FINDING_CODES, ConflictReviewer
from app.services.reviewer_benchmark import build_reviewer_cases, compare_reviewer_models


class FakeReviewerClient:
    def __init__(self, payload):
        self.payload = payload

    def generate_json(self, *args, **kwargs):
        return self.payload


def test_controlled_reviewer_suite_is_balanced_and_routed():
    cases = build_reviewer_cases()
    assert len(cases) == 24
    assert sum(case.expected_conflict for case in cases) == 12
    reviewer = ConflictReviewer(FakeReviewerClient({"findings": []}))
    assert all(reviewer.needs_llm_review(case.candidate, case.result, "ollama") for case in cases)


def test_reviewer_accepts_only_existing_cited_ids():
    case = build_reviewer_cases()[0]
    valid = {
        "severity": "warning", "code": "EVIDENCE_CONTRADICTS_REQUIREMENT",
        "message": "证据与结论矛盾", "requirement_id": "req-1", "evidence_ids": ["ev-1"],
    }
    invalid = {
        "severity": "critical", "code": "INVENTED_CODE", "message": "虚构",
        "requirement_id": "req-missing", "evidence_ids": ["ev-missing"],
    }
    reviewed = ConflictReviewer(FakeReviewerClient({"findings": [valid, invalid]})).review(
        case.job, case.candidate, case.result.model_copy(deep=True), "ollama"
    )
    accepted = [item for item in reviewed.findings if item.code in ALLOWED_LLM_FINDING_CODES]
    assert len(accepted) == 1
    assert accepted[0].requirement_id == "req-1"
    assert accepted[0].evidence_ids == ["ev-1"]


def test_grounding_gate_rejects_false_conflict_on_strong_supported_evidence():
    case = next(item for item in build_reviewer_cases() if item.case_id == "skill-supported")
    payload = {"findings": [{
        "severity": "warning", "code": "RESULT_OVERSTATES_EVIDENCE",
        "message": "错误地认为证据不足", "requirement_id": "req-1", "evidence_ids": ["ev-1"],
    }]}
    reviewed = ConflictReviewer(FakeReviewerClient(payload)).review(
        case.job, case.candidate, case.result.model_copy(deep=True), "ollama"
    )
    assert not any(item.code in ALLOWED_LLM_FINDING_CODES for item in reviewed.findings)
    assert any(item.code == "UNGROUNDED_LLM_FINDING_REJECTED" for item in reviewed.findings)


def test_grounding_gate_keeps_numeric_experience_conflict():
    case = next(item for item in build_reviewer_cases() if item.case_id == "experience-conflict")
    payload = {"findings": [{
        "severity": "critical", "code": "EXPERIENCE_CONFLICT",
        "message": "2年低于5年", "requirement_id": "req-1", "evidence_ids": ["ev-1"],
    }]}
    reviewed = ConflictReviewer(FakeReviewerClient(payload)).review(
        case.job, case.candidate, case.result.model_copy(deep=True), "ollama"
    )
    assert any(item.code == "EXPERIENCE_CONFLICT" for item in reviewed.findings)


def test_grounding_gate_requires_direct_negation_for_skill_conflict():
    supported = next(item for item in build_reviewer_cases() if item.case_id == "sql-supported")
    conflicted = next(item for item in build_reviewer_cases() if item.case_id == "sql-conflict")
    payload = {"findings": [{
        "severity": "warning", "code": "EVIDENCE_CONTRADICTS_REQUIREMENT",
        "message": "检查技能证据", "requirement_id": "req-1", "evidence_ids": ["ev-1"],
    }]}
    supported_result = ConflictReviewer(FakeReviewerClient(payload)).review(
        supported.job, supported.candidate, supported.result.model_copy(deep=True), "ollama"
    )
    conflicted_result = ConflictReviewer(FakeReviewerClient(payload)).review(
        conflicted.job, conflicted.candidate, conflicted.result.model_copy(deep=True), "ollama"
    )
    assert not any(item.code in ALLOWED_LLM_FINDING_CODES for item in supported_result.findings)
    assert any(item.code == "EVIDENCE_CONTRADICTS_REQUIREMENT" for item in conflicted_result.findings)


def test_heterogeneous_gate_requires_predeclared_f1_gain_without_more_fallbacks():
    labels = [True, False] * 12
    homogeneous_cases = [
        {"expected_conflict": label, "predicted_conflict": False} for label in labels
    ]
    improved_cases = [
        {"expected_conflict": label, "predicted_conflict": label} for label in labels
    ]
    homogeneous = {"metrics": {"f1": 0.0, "accuracy": 0.5}, "fallbacks": 0, "cases": homogeneous_cases}
    improved = {"metrics": {"f1": 1.0, "accuracy": 1.0}, "fallbacks": 0, "cases": improved_cases}
    worse = {"metrics": {"f1": 1.0, "accuracy": 1.0}, "fallbacks": 1, "cases": improved_cases}
    assert compare_reviewer_models(homogeneous, improved)["decision"] == "keep_heterogeneous_reviewer"
    assert compare_reviewer_models(homogeneous, worse)["decision"] == "keep_homogeneous_reviewer"
    assert compare_reviewer_models(homogeneous, improved)["resume_claim_allowed"] is False
