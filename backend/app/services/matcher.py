from __future__ import annotations

from collections.abc import Callable

from app.schemas import (
    CandidateResult, CriterionMatch, Evidence, MatchStatus, ParsedCandidate, ParsedJD, Priority, ReviewFinding,
)
from app.services.skill_catalog import display_name, lexical_relatedness


class MatchingEngine:
    def __init__(self, semantic_similarity: Callable[[str, str], float] | None = None):
        self.semantic_similarity = semantic_similarity

    def match(self, job: ParsedJD, candidate: ParsedCandidate) -> CandidateResult:
        criteria: list[CriterionMatch] = []
        all_matched_evidence: dict[str, Evidence] = {}
        weighted_sum = 0.0
        weight_total = 0.0
        findings: list[ReviewFinding] = []

        for requirement in job.requirements:
            weight = {Priority.hard: 1.5, Priority.preferred: 0.75, Priority.context: 0.35}[requirement.priority]
            criterion = self._match_requirement(requirement, candidate)
            criteria.append(criterion)
            weighted_sum += criterion.score * weight
            weight_total += weight
            for evidence_id in criterion.evidence_ids:
                evidence = next((ev for ev in candidate.evidence if ev.id == evidence_id), None)
                if evidence:
                    all_matched_evidence[evidence.id] = evidence
            if requirement.priority == Priority.hard and criterion.status == MatchStatus.missing:
                findings.append(ReviewFinding(
                    severity="critical", code="HARD_REQUIREMENT_MISSING",
                    message=f"硬性条件未找到证据：{requirement.text}", requirement_id=requirement.id,
                ))
            if requirement.priority == Priority.hard and criterion.status in {MatchStatus.partial, MatchStatus.review}:
                findings.append(ReviewFinding(
                    severity="warning", code="WEAK_HARD_REQUIREMENT_EVIDENCE",
                    message=f"硬性条件只有弱证据：{requirement.text}", requirement_id=requirement.id,
                    evidence_ids=criterion.evidence_ids,
                ))

        score = round(100 * weighted_sum / weight_total, 2) if weight_total else 0.0
        hard_missing = any(item.priority == Priority.hard and item.status == MatchStatus.missing for item in criteria)
        weak_count = sum(item.status in {MatchStatus.partial, MatchStatus.review} for item in criteria)
        confidence = max(0.25, min(0.98, 0.94 - 0.08 * weak_count - 0.12 * len(candidate.parse_warnings)))
        if hard_missing:
            recommendation = "insufficient_hard_requirement_evidence"
        elif score >= 72 and confidence >= 0.65:
            recommendation = "recommended"
        else:
            recommendation = "manual_review"
        strengths = [item.explanation for item in criteria if item.status == MatchStatus.matched][:5]
        gaps = [item.explanation for item in criteria if item.status in {MatchStatus.missing, MatchStatus.partial, MatchStatus.review}][:5]
        return CandidateResult(
            candidate_id=candidate.id, display_name=candidate.masked_name, score=score,
            confidence=round(confidence, 3), recommendation=recommendation, criteria=criteria,
            strengths=strengths, gaps=gaps, findings=findings,
            matched_evidence=list(all_matched_evidence.values()),
        )

    def _match_requirement(self, requirement, candidate: ParsedCandidate) -> CriterionMatch:
        if requirement.category == "skill" and requirement.normalized_skill:
            candidates = [ev for ev in candidate.evidence if ev.normalized_skill]
            exact = [ev for ev in candidates if ev.normalized_skill == requirement.normalized_skill]
            if exact:
                best = max(exact, key=lambda item: item.strength)
                status = MatchStatus.matched if best.strength >= 0.8 else MatchStatus.review
                score = best.strength
                explanation = (
                    f"匹配{display_name(requirement.normalized_skill)}，证据来自{best.section}：{best.source_quote[:70]}"
                    if status == MatchStatus.matched else
                    f"提到{display_name(requirement.normalized_skill)}，但证据仅来自{best.section}，需要复核"
                )
                return CriterionMatch(
                    requirement_id=requirement.id, requirement_text=requirement.text, priority=requirement.priority,
                    status=status, score=score, evidence_ids=[best.id], explanation=explanation,
                )
            best_related: tuple[float, Evidence] | None = None
            for evidence in candidates:
                related = lexical_relatedness(requirement.normalized_skill, evidence.normalized_skill or "")
                if self.semantic_similarity and related < 0.7:
                    try:
                        semantic = self.semantic_similarity(display_name(requirement.normalized_skill), display_name(evidence.normalized_skill or ""))
                        related = max(related, max(0.0, (semantic - 0.55) / 0.45))
                    except Exception:
                        pass
                adjusted = related * evidence.strength
                if best_related is None or adjusted > best_related[0]:
                    best_related = (adjusted, evidence)
            if best_related and best_related[0] >= 0.35:
                score, evidence = best_related
                return CriterionMatch(
                    requirement_id=requirement.id, requirement_text=requirement.text, priority=requirement.priority,
                    status=MatchStatus.partial, score=min(score, 0.65), evidence_ids=[evidence.id],
                    explanation=f"存在相邻技能{display_name(evidence.normalized_skill or '')}，不能视为完全满足{display_name(requirement.normalized_skill)}",
                )
            return CriterionMatch(
                requirement_id=requirement.id, requirement_text=requirement.text, priority=requirement.priority,
                status=MatchStatus.missing, score=0.0, explanation=f"未找到{display_name(requirement.normalized_skill)}的可核验证据",
            )

        if requirement.category == "experience" and requirement.minimum_years is not None:
            if candidate.years_experience is None:
                return CriterionMatch(
                    requirement_id=requirement.id, requirement_text=requirement.text, priority=requirement.priority,
                    status=MatchStatus.review, score=0.35, explanation="简历未提供可稳定计算的工作年限，需要人工复核",
                )
            ratio = min(1.0, candidate.years_experience / requirement.minimum_years) if requirement.minimum_years else 1.0
            year_evidence = [ev for ev in candidate.evidence if ev.kind == "experience"]
            best_evidence = max(year_evidence, key=lambda ev: ev.strength, default=None)
            evidence_strength = best_evidence.strength if best_evidence else 0.55
            score = ratio * evidence_strength
            status = MatchStatus.matched if ratio >= 1 and evidence_strength >= 0.8 else MatchStatus.partial if ratio < 1 else MatchStatus.review
            return CriterionMatch(
                requirement_id=requirement.id, requirement_text=requirement.text, priority=requirement.priority,
                status=status, score=score, evidence_ids=[best_evidence.id] if best_evidence else [],
                explanation=f"可识别经验约{candidate.years_experience:g}年，岗位要求{requirement.minimum_years:g}年",
            )

        if requirement.category == "education":
            if candidate.education:
                levels = {"大专": 1, "本科": 2, "硕士": 3, "博士": 4}
                required = next((level for level in levels if level in requirement.text), None)
                met = not required or levels.get(candidate.education, 0) >= levels[required]
                education_evidence = [ev.id for ev in candidate.evidence if ev.kind == "education" and candidate.education in ev.source_quote]
                return CriterionMatch(
                    requirement_id=requirement.id, requirement_text=requirement.text, priority=requirement.priority,
                    status=MatchStatus.matched if met else MatchStatus.missing, score=1.0 if met else 0.0,
                    evidence_ids=education_evidence,
                    explanation=f"简历学历为{candidate.education}" + ("，满足要求" if met else "，低于岗位要求"),
                )
            return CriterionMatch(
                requirement_id=requirement.id, requirement_text=requirement.text, priority=requirement.priority,
                status=MatchStatus.review, score=0.35, explanation="未提取到学历证据，需要人工复核",
            )

        return CriterionMatch(
            requirement_id=requirement.id, requirement_text=requirement.text, priority=requirement.priority,
            status=MatchStatus.review, score=0.5, explanation="该职责属于语义性要求，暂不作为自动淘汰依据",
        )
