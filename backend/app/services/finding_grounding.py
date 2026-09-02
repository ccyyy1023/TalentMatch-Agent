from __future__ import annotations

import re

from app.schemas import Evidence, Requirement


class FindingGroundingGate:
    """Deterministic semantic gate for model-generated review findings."""

    _education_rank = {
        "大专": 0,
        "associate": 0,
        "本科": 1,
        "学士": 1,
        "bachelor": 1,
        "硕士": 2,
        "研究生": 2,
        "master": 2,
        "博士": 3,
        "phd": 3,
        "doctorate": 3,
    }

    def accepts(self, code: str, requirement: Requirement, evidence: list[Evidence]) -> bool:
        if not evidence:
            return False
        if code == "EXPERIENCE_CONFLICT":
            return self._experience_conflict(requirement, evidence)
        if code == "EDUCATION_CONFLICT":
            return self._education_conflict(requirement, evidence)
        if code == "RESULT_OVERSTATES_EVIDENCE":
            return self._result_overstates_evidence(evidence)
        if code == "EVIDENCE_CONTRADICTS_REQUIREMENT":
            return self._skill_contradiction(requirement, evidence)
        return False

    @staticmethod
    def _experience_conflict(requirement: Requirement, evidence: list[Evidence]) -> bool:
        if requirement.category != "experience" or requirement.minimum_years is None:
            return False
        observed = [item.years for item in evidence if item.kind == "experience" and item.years is not None]
        return bool(observed) and max(observed) < requirement.minimum_years

    def _education_conflict(self, requirement: Requirement, evidence: list[Evidence]) -> bool:
        if requirement.category != "education":
            return False
        required_rank = self._rank(requirement.text)
        observed = [self._rank(f"{item.value} {item.source_quote}") for item in evidence if item.kind == "education"]
        observed = [rank for rank in observed if rank is not None]
        return required_rank is not None and bool(observed) and max(observed) < required_rank

    @staticmethod
    def _result_overstates_evidence(evidence: list[Evidence]) -> bool:
        # A model may question a conclusion only when every cited item is weak.
        # Evidence strength already incorporates its source section, so the
        # gate does not reinterpret a strong certification or other source.
        return all(item.strength < 0.7 for item in evidence)

    @staticmethod
    def _skill_contradiction(requirement: Requirement, evidence: list[Evidence]) -> bool:
        if requirement.category != "skill" or not requirement.normalized_skill:
            return False
        skill_tokens = {
            requirement.normalized_skill.lower(),
            requirement.normalized_skill.lower().replace("_", " "),
            requirement.normalized_skill.lower().replace("_", ""),
        }
        for item in evidence:
            text = f"{item.value} {item.source_quote}".lower().replace(" ", "")
            for skill in skill_tokens:
                compact_skill = skill.replace(" ", "")
                if not compact_skill or compact_skill not in text:
                    continue
                negative_patterns = (
                    rf"未(?:曾)?使用.{{0,8}}{re.escape(compact_skill)}",
                    rf"没有.{{0,8}}{re.escape(compact_skill)}",
                    rf"不具备.{{0,8}}{re.escape(compact_skill)}",
                    rf"无.{{0,8}}{re.escape(compact_skill)}(?:经验|能力|技能)",
                    rf"(?:no|without|notuse|didnotuse).{{0,8}}{re.escape(compact_skill)}",
                )
                if any(re.search(pattern, text) for pattern in negative_patterns):
                    return True
        return False

    def _rank(self, text: str) -> int | None:
        lowered = text.lower()
        matches = [rank for token, rank in self._education_rank.items() if token in lowered]
        return max(matches) if matches else None
