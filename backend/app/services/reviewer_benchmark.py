from __future__ import annotations

import random
from dataclasses import dataclass
from time import perf_counter

from app.schemas import (
    CandidateResult,
    CriterionMatch,
    Evidence,
    MatchStatus,
    ParsedCandidate,
    ParsedJD,
    Priority,
    Requirement,
)
from app.services.ollama_client import OllamaClient
from app.services.reviewer import ALLOWED_LLM_FINDING_CODES, ConflictReviewer


@dataclass(frozen=True)
class ReviewerCase:
    case_id: str
    expected_conflict: bool
    job: ParsedJD
    candidate: ParsedCandidate
    result: CandidateResult


def _case(
    case_id: str,
    requirement_text: str,
    category: str,
    evidence_text: str,
    evidence_kind: str,
    explanation: str,
    expected_conflict: bool,
    *,
    normalized_skill: str | None = None,
    minimum_years: float | None = None,
    candidate_years: float | None = None,
    education: str | None = None,
    strength: float = 0.9,
) -> ReviewerCase:
    requirement = Requirement(
        id="req-1", text=requirement_text, category=category, priority=Priority.hard,
        normalized_skill=normalized_skill, minimum_years=minimum_years, source_quote=requirement_text,
    )
    evidence = Evidence(
        id="ev-1", kind=evidence_kind, value=evidence_text, normalized_skill=normalized_skill,
        years=candidate_years if evidence_kind == "experience" else None,
        source_quote=evidence_text, section="work", strength=strength,
    )
    candidate = ParsedCandidate(
        id=case_id, display_name="受控候选人", masked_name=f"候选人-{case_id.upper()}",
        skills=[normalized_skill] if normalized_skill else [], years_experience=candidate_years,
        education=education, evidence=[evidence], parse_warnings=[], pii_detected=[], security_flags=[],
    )
    criterion = CriterionMatch(
        requirement_id="req-1", requirement_text=requirement_text, priority=Priority.hard,
        status=MatchStatus.review, score=0.55, evidence_ids=["ev-1"], explanation=explanation,
    )
    result = CandidateResult(
        candidate_id=case_id, display_name=candidate.masked_name, score=65.0, confidence=0.7,
        recommendation="manual_review", criteria=[criterion], matched_evidence=[evidence],
    )
    return ReviewerCase(
        case_id=case_id, expected_conflict=expected_conflict,
        job=ParsedJD(title="受控岗位", requirements=[requirement]), candidate=candidate, result=result,
    )


def build_reviewer_cases() -> list[ReviewerCase]:
    return [
        _case("skill-conflict", "必须掌握Python", "skill", "项目仅使用Java，未使用Python", "skill",
              "已有证据证明Python要求满足", True, normalized_skill="python"),
        _case("experience-conflict", "至少5年开发经验", "experience", "具有2年开发经验", "experience",
              "候选人满足5年经验要求", True, minimum_years=5, candidate_years=2),
        _case("education-conflict", "硕士及以上学历", "education", "最高学历为本科", "education",
              "候选人满足硕士学历要求", True, education="本科"),
        _case("weak-overstated", "具备RAG项目经验", "skill", "个人总结：了解RAG概念", "skill",
              "已具备充分的RAG生产项目证据", True, normalized_skill="rag", strength=0.4),
        _case("skill-supported", "必须掌握Python", "skill", "使用Python开发数据处理服务", "skill",
              "Python证据存在，仍建议人工确认深度", False, normalized_skill="python"),
        _case("experience-supported", "至少5年开发经验", "experience", "具有6年开发经验", "experience",
              "候选人经验年限满足要求", False, minimum_years=5, candidate_years=6),
        _case("education-supported", "硕士及以上学历", "education", "最高学历为硕士", "education",
              "候选人学历满足要求", False, education="硕士"),
        _case("project-supported", "具备RAG项目经验", "skill", "负责RAG知识库项目的检索与评测", "project",
              "存在RAG项目证据，结论保持人工复核", False, normalized_skill="rag"),
        _case("sql-conflict", "必须掌握SQL", "skill", "工作中仅使用Excel，未使用SQL", "skill",
              "已有证据证明SQL要求满足", True, normalized_skill="sql"),
        _case("docker-conflict", "必须具备Docker部署经验", "skill", "采用手工部署，未使用Docker", "skill",
              "Docker部署经验充分", True, normalized_skill="docker"),
        _case("experience-3v1", "至少3年后端经验", "experience", "具有1年后端经验", "experience",
              "经验年限满足岗位要求", True, minimum_years=3, candidate_years=1),
        _case("experience-8v4", "至少8年工程经验", "experience", "具有4年工程经验", "experience",
              "经验年限达到8年要求", True, minimum_years=8, candidate_years=4),
        _case("bachelor-conflict", "本科及以上学历", "education", "最高学历为大专", "education",
              "学历满足本科要求", True, education="大专"),
        _case("phd-conflict", "博士学历", "education", "最高学历为硕士", "education",
              "学历满足博士要求", True, education="硕士"),
        _case("docker-weak-overstated", "具备Docker生产部署经验", "skill", "技能列表：Docker", "skill",
              "具有充分的Docker生产部署证据", True, normalized_skill="docker", strength=0.4),
        _case("ml-weak-overstated", "具备机器学习落地经验", "skill", "自我评价：了解机器学习", "skill",
              "已具备机器学习生产落地经验", True, normalized_skill="machine_learning", strength=0.4),
        _case("sql-supported", "必须掌握SQL", "skill", "使用SQL完成报表查询与性能优化", "skill",
              "SQL证据存在，仍建议复核熟练度", False, normalized_skill="sql"),
        _case("docker-supported", "必须具备Docker部署经验", "skill", "使用Docker部署三个生产服务", "project",
              "Docker项目证据支持要求", False, normalized_skill="docker"),
        _case("experience-4v3", "至少3年后端经验", "experience", "具有4年后端经验", "experience",
              "经验年限满足岗位要求", False, minimum_years=3, candidate_years=4),
        _case("experience-10v8", "至少8年工程经验", "experience", "具有10年工程经验", "experience",
              "经验年限满足岗位要求", False, minimum_years=8, candidate_years=10),
        _case("master-over-bachelor", "本科及以上学历", "education", "最高学历为硕士", "education",
              "学历满足本科要求", False, education="硕士"),
        _case("phd-supported", "博士学历", "education", "最高学历为博士", "education",
              "学历满足博士要求", False, education="博士"),
        _case("docker-project-supported", "具备Docker生产部署经验", "skill", "负责Docker生产集群发布与回滚", "project",
              "存在Docker生产项目证据", False, normalized_skill="docker"),
        _case("ml-project-supported", "具备机器学习落地经验", "skill", "负责机器学习模型上线与效果监控", "project",
              "存在机器学习落地项目证据", False, normalized_skill="machine_learning"),
    ]


def run_reviewer_model(model: str, cases: list[ReviewerCase] | None = None) -> dict:
    cases = cases or build_reviewer_cases()
    reviewer = ConflictReviewer(OllamaClient(chat_model=model, cache_enabled=False))
    started = perf_counter()
    rows = []
    for case in cases:
        reviewed = reviewer.review(case.job, case.candidate, case.result.model_copy(deep=True), "ollama")
        llm_findings = [item for item in reviewed.findings if item.code in ALLOWED_LLM_FINDING_CODES]
        fallback = any(item.code == "LLM_REVIEW_FALLBACK" for item in reviewed.findings)
        predicted = bool(llm_findings)
        rows.append({
            "case_id": case.case_id,
            "expected_conflict": case.expected_conflict,
            "predicted_conflict": predicted,
            "fallback": fallback,
            "finding_codes": [item.code for item in llm_findings],
            "citations_valid": all(
                item.requirement_id == "req-1" and set(item.evidence_ids) == {"ev-1"}
                for item in llm_findings
            ),
        })
    true_positive = sum(row["expected_conflict"] and row["predicted_conflict"] for row in rows)
    false_positive = sum(not row["expected_conflict"] and row["predicted_conflict"] for row in rows)
    false_negative = sum(row["expected_conflict"] and not row["predicted_conflict"] for row in rows)
    true_negative = sum(not row["expected_conflict"] and not row["predicted_conflict"] for row in rows)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    positive_count = sum(case.expected_conflict for case in cases)
    return {
        "model": model,
        "scope": {
            "cases": len(rows), "positive": positive_count,
            "negative": len(rows) - positive_count, "cache_enabled": False,
        },
        "elapsed_seconds": round(perf_counter() - started, 3),
        "fallbacks": sum(row["fallback"] for row in rows),
        "confusion": {"tp": true_positive, "fp": false_positive, "fn": false_negative, "tn": true_negative},
        "metrics": {
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "accuracy": round((true_positive + true_negative) / len(rows), 4),
            "citation_validity": round(
                sum(row["citations_valid"] for row in rows) / len(rows), 4
            ),
        },
        "cases": rows,
    }


def compare_reviewer_models(homogeneous: dict, heterogeneous: dict) -> dict:
    f1_delta = heterogeneous["metrics"]["f1"] - homogeneous["metrics"]["f1"]
    accuracy_delta = heterogeneous["metrics"]["accuracy"] - homogeneous["metrics"]["accuracy"]
    correctness_deltas = []
    for left, right in zip(homogeneous["cases"], heterogeneous["cases"]):
        left_correct = left["expected_conflict"] == left["predicted_conflict"]
        right_correct = right["expected_conflict"] == right["predicted_conflict"]
        correctness_deltas.append(int(right_correct) - int(left_correct))
    rng = random.Random(20260901)
    size = len(correctness_deltas)
    estimates = sorted(
        sum(correctness_deltas[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(5000)
    )
    confidence_interval = [estimates[125], estimates[4874]]
    keep_heterogeneous = (
        f1_delta >= 0.05
        and accuracy_delta >= 0
        and confidence_interval[0] > 0
        and heterogeneous["fallbacks"] <= homogeneous["fallbacks"]
    )
    return {
        "f1_delta_heterogeneous_minus_homogeneous": round(f1_delta, 4),
        "accuracy_delta_heterogeneous_minus_homogeneous": round(accuracy_delta, 4),
        "fallback_delta_heterogeneous_minus_homogeneous": heterogeneous["fallbacks"] - homogeneous["fallbacks"],
        "paired_bootstrap_95ci_accuracy_delta": [round(value, 4) for value in confidence_interval],
        "decision": "keep_heterogeneous_reviewer" if keep_heterogeneous else "keep_homogeneous_reviewer",
        "resume_claim_allowed": False,
        "reason": (
            "Controlled-set improvement observed; independent human-labeled validation is still required before a resume claim."
            if keep_heterogeneous else
            "The heterogeneous reviewer did not meet the predeclared improvement gate."
        ),
    }
