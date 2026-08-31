from __future__ import annotations

from app.schemas import CandidateResult, ParsedCandidate, ParsedJD, ReviewFinding
from app.services.ollama_client import OllamaClient


class ConflictReviewer:
    def __init__(self, ollama: OllamaClient):
        self.ollama = ollama

    def review(self, job: ParsedJD, candidate: ParsedCandidate, result: CandidateResult, mode: str) -> CandidateResult:
        findings = list(result.findings)
        skill_evidence = [ev for ev in candidate.evidence if ev.kind == "skill"]
        if skill_evidence and all(ev.section not in {"work", "project"} for ev in skill_evidence):
            findings.append(ReviewFinding(
                severity="warning", code="SKILLS_WITHOUT_PROJECT_EVIDENCE",
                message="提取到的技能均缺少工作或项目经历支撑，禁止直接视为强证据",
                evidence_ids=[ev.id for ev in skill_evidence[:8]],
            ))
        if "protected_attribute" in candidate.pii_detected:
            findings.append(ReviewFinding(
                severity="info", code="PROTECTED_ATTRIBUTE_MASKED",
                message="检测到性别、年龄或婚育等敏感信息，已从评分输入中排除",
            ))
        if candidate.security_flags:
            findings.append(ReviewFinding(
                severity="warning", code="PROMPT_INJECTION_SUSPECTED",
                message="候选人材料包含疑似提示注入指令，已跳过LLM解析并仅保留确定性证据检查",
            ))
        if candidate.parse_warnings:
            for warning in candidate.parse_warnings:
                findings.append(ReviewFinding(severity="warning", code="PARSE_WARNING", message=warning))

        if self.needs_llm_review(candidate, result, mode):
            try:
                payload = self.ollama.generate_json(
                    "你是招聘结果复核员。只检查已有证据与结论是否矛盾，不得新增候选人能力。"
                    "返回JSON字段findings，每项含severity(info/warning/critical)、code、message。",
                    f"岗位要求：{job.model_dump_json()}\n候选人证据：{candidate.model_dump_json()}\n初始结果：{result.model_dump_json()}",
                    cache_namespace="conflict_reviewer", prompt_version="reviewer-v2",
                )
                for item in payload.get("findings", [])[:5]:
                    if item.get("severity") in {"info", "warning", "critical"} and item.get("message"):
                        findings.append(ReviewFinding(
                            severity=item["severity"], code=str(item.get("code") or "LLM_REVIEW"),
                            message=str(item["message"])[:300],
                        ))
            except Exception:
                findings.append(ReviewFinding(
                    severity="info", code="LLM_REVIEW_FALLBACK",
                    message="模型复核不可用，已保留确定性冲突检查结果",
                ))

        result.findings = self._deduplicate(findings)
        if any(item.severity == "critical" for item in result.findings):
            result.recommendation = "insufficient_hard_requirement_evidence"
        elif any(item.severity == "warning" for item in result.findings):
            result.recommendation = "manual_review" if result.recommendation != "insufficient_hard_requirement_evidence" else result.recommendation
            result.confidence = round(max(0.25, result.confidence - 0.08), 3)
        return result

    @staticmethod
    def needs_llm_review(candidate: ParsedCandidate, result: CandidateResult, mode: str) -> bool:
        if mode not in {"ollama", "adaptive"} or candidate.security_flags:
            return False
        if any(item.severity == "critical" for item in result.findings):
            return False
        ambiguous = any(item.status.value in {"partial", "review"} for item in result.criteria)
        return ambiguous and result.confidence < 0.85

    @staticmethod
    def _deduplicate(items: list[ReviewFinding]) -> list[ReviewFinding]:
        seen: set[tuple[str, str]] = set()
        output = []
        for item in items:
            key = (item.code, item.message)
            if key not in seen:
                seen.add(key)
                output.append(item)
        return output
