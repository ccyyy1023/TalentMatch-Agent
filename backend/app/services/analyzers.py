from __future__ import annotations

import re
from time import perf_counter
from typing import TYPE_CHECKING

from app.schemas import Evidence, ParsedCandidate, ParsedJD, Priority, Requirement, TraceEvent
from app.services.ollama_client import OllamaClient, OllamaUnavailable
from app.services.skill_catalog import display_name, extract_skills, normalize_skill
from app.services.security import detect_prompt_injection

if TYPE_CHECKING:
    from app.services.hybrid_skill_extractor import DocumentSkillExtractor

MAX_JOBBERT_JD_REQUIREMENTS = 12


HARD_MARKERS = (
    "必须", "要求", "精通", "熟练掌握", "至少", "不少于", "本科及以上",
    "must", "required", "requirement", "minimum", "at least",
)
PREFERRED_MARKERS = (
    "优先", "加分", "最好", "熟悉", "了解", "preferred", "nice to have", "plus",
)


def infer_priority(text: str) -> Priority:
    lowered = text.casefold()
    if any(marker in lowered for marker in HARD_MARKERS):
        return Priority.hard
    if any(marker in lowered for marker in PREFERRED_MARKERS):
        return Priority.preferred
    return Priority.context


def coerce_optional_number(value) -> float | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else None


def split_units(text: str) -> list[str]:
    units = re.split(r"[\n。；;]+", text)
    return [re.sub(r"^[\s\-•·\d.、）)]+", "", unit).strip() for unit in units if unit.strip()]


class JDAnalyzer:
    def __init__(self, ollama: OllamaClient, skill_extractor: "DocumentSkillExtractor | None" = None):
        self.ollama = ollama
        self.skill_extractor = skill_extractor

    def analyze(self, text: str, mode: str) -> tuple[ParsedJD, TraceEvent]:
        started = perf_counter()
        if mode == "ollama":
            try:
                parsed = self._analyze_with_llm(text)
                return parsed, TraceEvent(
                    node="jd_analyzer_agent", status="completed",
                    detail=f"Ollama抽取{len(parsed.requirements)}项岗位要求" + ("（缓存命中）" if getattr(self.ollama, "last_call_cache_hit", False) else ""),
                    elapsed_ms=(perf_counter() - started) * 1000,
                )
            except Exception as exc:
                parsed = self._analyze_rules(text)
                return parsed, TraceEvent(
                    node="jd_analyzer_agent", status="fallback",
                    detail=f"模型抽取失败，已回退规则：{type(exc).__name__}:{str(exc)[:80]}",
                    elapsed_ms=(perf_counter() - started) * 1000,
                )
        parsed = self._analyze_rules(text)
        return parsed, TraceEvent(
            node="jd_analyzer_agent", status="completed", detail=f"规则抽取{len(parsed.requirements)}项岗位要求", elapsed_ms=(perf_counter() - started) * 1000
        )

    def _analyze_with_llm(self, text: str) -> ParsedJD:
        system = (
            "你是招聘岗位结构化专家。只抽取原文明确表达的内容，禁止补充常识。"
            "输出JSON，字段为title、summary、requirements、ambiguities。requirements每项包含"
            "text、category(skill/experience/education/responsibility/other)、priority(hard/preferred/context)、"
            "normalized_skill、minimum_years、source_quote。source_quote必须是输入中的连续原文。"
            "requirements只保留对筛选有区分度的要求，总数不超过20项，字段内容保持简洁。"
        )
        raw = self.ollama.generate_json(system, text, cache_namespace="jd_analyzer", prompt_version="jd-v3")
        requirements: list[Requirement] = []
        for index, item in enumerate(raw.get("requirements", []), start=1):
            quote = str(item.get("source_quote") or item.get("text") or "").strip()
            if not quote or quote not in text:
                continue
            normalized = item.get("normalized_skill")
            if normalized:
                normalized = normalize_skill(str(normalized)) or normalize_skill(quote)
            requirements.append(Requirement(
                id=f"req-{index}",
                text=str(item.get("text") or quote),
                category=item.get("category") if item.get("category") in {"skill", "experience", "education", "responsibility", "other"} else "other",
                priority=item.get("priority") if item.get("priority") in {"hard", "preferred", "context"} else "context",
                normalized_skill=normalized,
                minimum_years=coerce_optional_number(item.get("minimum_years")),
                source_quote=quote,
            ))
        if not requirements:
            raise OllamaUnavailable("模型没有返回可核验的岗位要求")
        rules_parsed = self._analyze_rules(text)
        # Semantic responsibility/other clauses must not become automatic hard
        # filters. Preserve concrete skill, experience and education items;
        # the previous list filter accidentally discarded all of them.
        normalized_requirements: list[Requirement] = []
        for item in requirements:
            if (
                item.category in {"responsibility", "other"}
                and not extract_skills(item.source_quote)
                and not re.search(r"\d+(?:\.\d+)?\s*年", item.source_quote)
                and not re.search(r"博士|硕士|本科|大专", item.source_quote)
            ):
                item = item.model_copy(update={"priority": Priority.context})
            normalized_requirements.append(item)
        requirements = normalized_requirements
        existing_keys = {(item.category, item.normalized_skill, item.minimum_years, item.source_quote) for item in requirements}
        for item in rules_parsed.requirements:
            key = (item.category, item.normalized_skill, item.minimum_years, item.source_quote)
            if key not in existing_keys:
                requirements.append(item.model_copy(update={"id": f"req-{len(requirements) + 1}"}))
                existing_keys.add(key)
        return ParsedJD(
            title=str(raw.get("title") or self._guess_title(text)),
            summary=str(raw.get("summary") or ""),
            requirements=requirements,
            ambiguities=list(dict.fromkeys([str(x) for x in raw.get("ambiguities", [])] + rules_parsed.ambiguities))[:10],
        )

    def _analyze_rules(self, text: str) -> ParsedJD:
        requirements: list[Requirement] = []
        ambiguities: list[str] = []
        seen: set[tuple[str, str]] = set()
        for unit in split_units(text):
            priority = infer_priority(unit)
            years_match = re.search(r"(\d+(?:\.\d+)?)\s*年(?:[^，。；;\n]{0,10})?(?:以上|及以上|经验)", unit)
            skills = extract_skills(unit)
            for skill, _ in skills:
                key = (skill, priority.value)
                if key in seen:
                    continue
                seen.add(key)
                requirements.append(Requirement(
                    id=f"req-{len(requirements) + 1}", text=f"{display_name(skill)}能力", category="skill",
                    priority=priority, normalized_skill=skill, source_quote=unit,
                ))
            if years_match:
                requirements.append(Requirement(
                    id=f"req-{len(requirements) + 1}", text=f"至少{years_match.group(1)}年相关经验", category="experience",
                    priority=priority if priority != Priority.context else Priority.hard,
                    minimum_years=float(years_match.group(1)), source_quote=unit,
                ))
            if any(word in unit for word in ("本科", "硕士", "博士", "大专")):
                requirements.append(Requirement(
                    id=f"req-{len(requirements) + 1}", text=unit, category="education", priority=priority,
                    source_quote=unit,
                ))
            if len(unit) > 12 and not skills and not years_match and any(word in unit for word in ("负责", "参与", "设计", "建设")):
                requirements.append(Requirement(
                    id=f"req-{len(requirements) + 1}", text=unit, category="responsibility", priority=Priority.context,
                    source_quote=unit,
                ))
            if any(marker in unit for marker in ("优秀", "较强", "良好", "相关经验")) and not skills:
                ambiguities.append(unit)
        if self.skill_extractor is not None:
            try:
                mentions = self.skill_extractor.extract(text)
            except Exception:
                mentions = []
            priority_order = {Priority.hard: 0, Priority.preferred: 1, Priority.context: 2}
            mentions = sorted(
                mentions,
                key=lambda item: (
                    priority_order[infer_priority(item.source_quote)],
                    item.label != "knowledge",
                    len(item.text.split()),
                    item.start,
                ),
            )
            added = 0
            for mention in mentions:
                priority = infer_priority(mention.source_quote)
                key = (mention.normalized_skill, priority.value)
                if key in seen:
                    continue
                seen.add(key)
                requirements.append(Requirement(
                    id=f"req-{len(requirements) + 1}",
                    text=f"{mention.text} competency",
                    category="skill",
                    priority=priority,
                    normalized_skill=mention.normalized_skill,
                    source_quote=mention.source_quote,
                ))
                added += 1
                if added >= MAX_JOBBERT_JD_REQUIREMENTS:
                    break
        if not requirements:
            requirements.append(Requirement(
                id="req-1", text="岗位整体语义匹配", category="other", priority=Priority.context,
                source_quote=text[:200],
            ))
            ambiguities.append("岗位描述缺少可结构化的技能或年限条件")
        return ParsedJD(title=self._guess_title(text), summary=text[:160].strip(), requirements=requirements, ambiguities=ambiguities[:10])

    @staticmethod
    def _guess_title(text: str) -> str:
        first = split_units(text)[0] if split_units(text) else "未命名岗位"
        title = re.sub(r"^(岗位名称|职位名称|招聘岗位)[:：]\s*", "", first)
        return title[:40]


class CandidateAnalyzer:
    SECTION_MARKERS = {
        "技能": "skills", "专业技能": "skills", "技术栈": "skills",
        "项目": "project", "项目经历": "project",
        "工作": "work", "工作经历": "work", "实习经历": "work",
        "教育": "education", "教育经历": "education",
        "个人总结": "summary", "自我评价": "summary",
    }

    def __init__(self, ollama: OllamaClient, skill_extractor: "DocumentSkillExtractor | None" = None):
        self.ollama = ollama
        self.skill_extractor = skill_extractor

    def analyze(
        self, candidate_id: str, name: str, text: str, mode: str,
        target_skills: list[str] | None = None,
    ) -> tuple[ParsedCandidate, str]:
        security_flags = detect_prompt_injection(text)
        if mode == "ollama" and security_flags:
            parsed = self._analyze_rules(candidate_id, name, text, target_skills)
            parsed.security_flags = security_flags
            return parsed, "security_fallback"
        if mode == "ollama":
            try:
                parsed = self._analyze_with_llm(candidate_id, name, text, target_skills)
                return parsed, "ollama_cache" if getattr(self.ollama, "last_call_cache_hit", False) else "ollama"
            except Exception as exc:
                return self._analyze_rules(candidate_id, name, text, target_skills), f"fallback:{type(exc).__name__}:{str(exc)[:80]}"
        return self._analyze_rules(candidate_id, name, text, target_skills), "rules"

    def _analyze_with_llm(
        self, candidate_id: str, name: str, text: str, target_skills: list[str] | None = None,
    ) -> ParsedCandidate:
        system = (
            "你是简历证据抽取专家。只抽取简历原文明确支持的信息，不推测候选人能力。"
            "输出JSON字段：skills、years_experience、education、evidence、parse_warnings。"
            "evidence每项字段kind、value、normalized_skill、years、source_quote、section、strength。"
            "source_quote必须逐字复制输入中的一整行，禁止改写、概括或添加标点；"
            "section只能为work、project、skills、education、summary、unknown；"
            "工作或项目证据strength最高1，技能列表不高于0.65，自我评价不高于0.45。"
            "normalized_skill优先使用python、fastapi、langgraph、langchain、rag、postgresql、docker等小写规范名。"
            "只保留最能支持筛选判断的证据，每类最多2条、总数不超过6条，value保持简洁。"
        )
        raw = self.ollama.generate_json(system, text, cache_namespace="candidate_analyzer", prompt_version="candidate-v3")
        evidence: list[Evidence] = []
        for index, item in enumerate(raw.get("evidence", []), start=1):
            quote = str(item.get("source_quote") or "").strip()
            if not quote or quote not in text:
                continue
            section = str(item.get("section") or "unknown")
            strength_cap = 1.0 if section in {"work", "project"} else 0.65 if section == "skills" else 0.45 if section == "summary" else 0.7
            strength = min(float(item.get("strength") or strength_cap), strength_cap)
            normalized = normalize_skill(str(item.get("normalized_skill") or item.get("value") or ""))
            kind = item.get("kind") if item.get("kind") in {"skill", "experience", "education", "project", "achievement", "other"} else "other"
            evidence.append(Evidence(
                id=f"{candidate_id}-ev-{index}", kind=kind, value=str(item.get("value") or quote),
                normalized_skill=normalized, years=coerce_optional_number(item.get("years")), source_quote=quote, section=section,
                strength=strength,
            ))
        if not evidence:
            raise OllamaUnavailable("模型没有返回可核验证据")
        rules_parsed = self._analyze_rules(candidate_id, name, text, target_skills)
        merged: list[Evidence] = []
        seen_evidence: set[tuple] = set()
        for item in [*evidence, *rules_parsed.evidence]:
            key = (item.kind, item.normalized_skill, item.source_quote)
            if key in seen_evidence:
                continue
            seen_evidence.add(key)
            merged.append(item.model_copy(update={"id": f"{candidate_id}-ev-{len(merged) + 1}"}))
        evidence = merged
        skills = sorted({ev.normalized_skill for ev in evidence if ev.normalized_skill})
        education_raw = raw.get("education")
        if isinstance(education_raw, dict):
            education_text = " ".join(str(value) for value in education_raw.values() if value not in (None, ""))
            degree_match = re.search(r"博士|硕士|本科|大专", education_text)
            education_raw = degree_match.group() if degree_match else education_raw.get("degree") or education_raw.get("level") or education_raw.get("学历") or education_raw.get("value")
        elif isinstance(education_raw, list):
            education_raw = education_raw[0] if education_raw else None
        education_text = str(education_raw).strip() if education_raw not in (None, "") else ""
        education_match = re.search(r"博士|硕士|本科|大专", education_text)
        education = education_match.group() if education_match else education_text or None
        years_raw = raw.get("years_experience")
        if isinstance(years_raw, str):
            years_match = re.search(r"\d+(?:\.\d+)?", years_raw)
            years_raw = float(years_match.group()) if years_match else None
        elif not isinstance(years_raw, (int, float)):
            years_raw = None
        year_candidates = [value for value in (years_raw, rules_parsed.years_experience) if value is not None]
        combined_years = max(year_candidates) if year_candidates else None
        warning_values = []
        for warning in raw.get("parse_warnings", []):
            if isinstance(warning, dict):
                warning = warning.get("message") or warning.get("value") or warning.get("warning")
            if warning:
                warning_values.append(str(warning))
        if rules_parsed.years_experience is not None:
            warning_values = [warning for warning in warning_values if not ("年限" in warning or "经验" in warning)]
        return ParsedCandidate(
            id=candidate_id, display_name=name, masked_name=self._masked_name(candidate_id), skills=skills,
            years_experience=combined_years, education=education or rules_parsed.education, evidence=evidence,
            pii_detected=self._detect_pii(text), security_flags=detect_prompt_injection(text),
            parse_warnings=list(dict.fromkeys(warning_values + rules_parsed.parse_warnings))[:10],
        )

    def _analyze_rules(
        self, candidate_id: str, name: str, text: str, target_skills: list[str] | None = None,
    ) -> ParsedCandidate:
        section = "unknown"
        evidence: list[Evidence] = []
        line_sections: dict[str, str] = {}
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            normalized_header = re.sub(r"[：:\s【】\[\]]", "", line)
            for marker, mapped in self.SECTION_MARKERS.items():
                if normalized_header == marker or normalized_header.startswith(marker):
                    section = mapped
                    break
            line_sections[line] = section
            section_strength = {"work": 1.0, "project": 1.0, "skills": 0.65, "education": 0.8, "summary": 0.45}.get(section, 0.55)
            for skill, alias in extract_skills(line):
                evidence.append(Evidence(
                    id=f"{candidate_id}-ev-{len(evidence) + 1}", kind="skill", value=alias,
                    normalized_skill=skill, source_quote=line[:300], section=section, strength=section_strength,
                ))
            if re.search(r"\b(19|20)\d{2}[./年-]", line) and section in {"work", "project"}:
                evidence.append(Evidence(
                    id=f"{candidate_id}-ev-{len(evidence) + 1}", kind="experience", value=line[:100],
                    source_quote=line[:300], section=section, strength=section_strength,
                ))
            years_in_line = re.search(r"(\d+(?:\.\d+)?)\s*年(?:[^，。；;\n]{0,10})?(?:以上|及以上|经验)", line)
            if years_in_line:
                evidence.append(Evidence(
                    id=f"{candidate_id}-ev-{len(evidence) + 1}", kind="experience",
                    value=f"{years_in_line.group(1)}年", years=float(years_in_line.group(1)),
                    source_quote=line[:300], section=section, strength=section_strength,
                ))
            education_in_line = re.search(r"(博士|硕士|本科|大专)", line)
            if education_in_line:
                evidence.append(Evidence(
                    id=f"{candidate_id}-ev-{len(evidence) + 1}", kind="education",
                    value=education_in_line.group(1), source_quote=line[:300], section=section,
                    strength=0.8 if section == "education" else 0.65,
                ))
            if any(token in line for token in ("提升", "降低", "优化", "%", "准确率", "F1", "QPS")):
                evidence.append(Evidence(
                    id=f"{candidate_id}-ev-{len(evidence) + 1}", kind="achievement", value=line[:100],
                    source_quote=line[:300], section=section, strength=section_strength,
                ))
        if target_skills:
            from app.services.hybrid_skill_extractor import verify_target_skills

            mentions = verify_target_skills(text, target_skills)
            existing = {(item.normalized_skill, item.source_quote) for item in evidence if item.normalized_skill}
            for mention in mentions:
                key = (mention.normalized_skill, mention.source_quote)
                if key in existing:
                    continue
                existing.add(key)
                mention_section = line_sections.get(mention.source_quote, "unknown")
                strength = {
                    "work": 1.0, "project": 1.0, "skills": 0.65,
                    "education": 0.8, "summary": 0.45,
                }.get(mention_section, 0.55)
                evidence.append(Evidence(
                    id=f"{candidate_id}-ev-{len(evidence) + 1}",
                    kind="skill",
                    value=mention.text,
                    normalized_skill=mention.normalized_skill,
                    source_quote=mention.source_quote,
                    section=mention_section,
                    strength=strength,
                ))
        years_values = [float(value) for value in re.findall(r"(\d+(?:\.\d+)?)\s*年(?:[^，。；;\n]{0,10})?(?:以上|及以上|经验)", text)]
        years = max(years_values) if years_values else self._estimate_years_from_dates(text)
        education_match = re.search(r"(博士|硕士|本科|大专)", text)
        skills = sorted({ev.normalized_skill for ev in evidence if ev.normalized_skill})
        warnings = []
        if not skills:
            warnings.append("未从简历中提取到已知技能")
        if not any(ev.section in {"work", "project"} for ev in evidence if ev.kind == "skill"):
            warnings.append("技能缺少项目或工作经历证据")
        return ParsedCandidate(
            id=candidate_id, display_name=name, masked_name=self._masked_name(candidate_id), skills=skills,
            years_experience=years, education=education_match.group(1) if education_match else None,
            evidence=evidence, pii_detected=self._detect_pii(text),
            security_flags=detect_prompt_injection(text), parse_warnings=warnings,
        )

    @staticmethod
    def _estimate_years_from_dates(text: str) -> float | None:
        years = [int(value) for value in re.findall(r"\b(20\d{2})\b", text)]
        if len(years) < 2:
            return None
        return float(max(0, min(20, max(years) - min(years))))

    @staticmethod
    def _detect_pii(text: str) -> list[str]:
        pii = []
        if re.search(r"1[3-9]\d{9}", text):
            pii.append("phone")
        if re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text):
            pii.append("email")
        if any(word in text for word in ("男", "女", "年龄", "出生年月", "婚姻")):
            pii.append("protected_attribute")
        return pii

    @staticmethod
    def _masked_name(candidate_id: str) -> str:
        suffix = re.sub(r"\W", "", candidate_id)[-4:] or "0000"
        return f"候选人-{suffix.upper()}"
