from __future__ import annotations

from dataclasses import dataclass

from app.schemas import AnalysisRequest, CandidateInput
from app.services.analyzers import CandidateAnalyzer, JDAnalyzer
from app.services.baselines import keyword_coverage_ranking
from app.services.evaluation import ranking_metrics
from app.services.matcher import MatchingEngine
from app.services.ollama_client import OllamaClient
from app.services.reviewer import ConflictReviewer
from app.services.skill_catalog import lexical_relatedness


SKILL_TEXT = {
    "python": "Python", "fastapi": "FastAPI", "langgraph": "LangGraph", "rag": "RAG",
    "docker": "Docker", "kubernetes": "Kubernetes", "sql": "SQL", "postgresql": "PostgreSQL",
    "airflow": "Airflow", "spark": "Spark", "hadoop": "Hadoop", "pytorch": "PyTorch",
    "nlp": "NLP", "llm": "LLM", "tensorflow": "TensorFlow", "deep_learning": "深度学习",
    "java": "Java", "spring_boot": "Spring Boot", "redis": "Redis", "mysql": "MySQL",
    "mongodb": "MongoDB", "react": "React", "typescript": "TypeScript",
    "javascript": "JavaScript", "rest_api": "REST API", "git": "Git", "vue": "Vue",
    "linux": "Linux", "aws": "AWS", "azure": "Azure", "gcp": "GCP", "langchain": "LangChain",
    "flask": "Flask",
}


@dataclass(frozen=True)
class JobSpec:
    key: str
    title: str
    hard: tuple[str, ...]
    preferred: str
    decoy: str
    adjacent: tuple[str, ...]
    years: int


@dataclass(frozen=True)
class GoldCandidate:
    item: CandidateInput
    skills: frozenset[str]
    years: float | None
    education: str | None
    conflict_codes: frozenset[str]


JOB_SPECS = (
    JobSpec("agent", "AI Agent应用开发工程师", ("python", "fastapi", "langgraph", "rag"), "docker", "kubernetes", ("langchain", "llm", "flask"), 2),
    JobSpec("data", "数据工程师", ("python", "sql", "postgresql", "airflow"), "spark", "hadoop", ("mysql", "spark"), 3),
    JobSpec("nlp", "NLP算法工程师", ("python", "pytorch", "nlp", "llm"), "docker", "tensorflow", ("tensorflow", "deep_learning"), 2),
    JobSpec("java", "Java后端工程师", ("java", "spring_boot", "redis", "mysql"), "docker", "mongodb", ("python", "fastapi"), 3),
    JobSpec("frontend", "前端工程师", ("react", "typescript", "javascript", "rest_api"), "git", "vue", ("vue", "javascript"), 2),
    JobSpec("cloud", "云原生运维工程师", ("docker", "kubernetes", "linux", "aws"), "git", "azure", ("azure", "gcp", "linux"), 3),
)


def _names(skills: tuple[str, ...] | list[str]) -> str:
    return "、".join(SKILL_TEXT[item] for item in skills)


def build_job(spec: JobSpec) -> str:
    return (
        f"岗位名称：{spec.title}\n"
        f"必须熟练掌握{_names(spec.hard)}。\n"
        f"要求本科及以上学历，至少{spec.years}年相关开发经验。\n"
        f"熟悉{SKILL_TEXT[spec.preferred]}者优先。\n"
        f"本岗位无需{SKILL_TEXT[spec.decoy]}。"
    )


def _candidate(
    spec: JobSpec, suffix: str, skills: tuple[str, ...], years: int | None, education: str | None,
    label: int, section: str = "project", sensitive: bool = False, negated: bool = False,
) -> GoldCandidate:
    candidate_id = f"{spec.key}-{suffix}"
    identity = "候选人" + (" 女 35岁" if sensitive else "")
    education_line = f"教育经历\n某大学 {education}" if education else "教育经历\n未注明学历"
    years_line = f"工作经历\n{years}年相关开发经验。" if years is not None else "工作经历\n未注明工作年限。"
    if negated:
        skills_line = f"项目经历\n未使用{_names(tuple(spec.hard))}，主要使用{SKILL_TEXT[skills[0]]}完成项目。"
        expected_skills = frozenset(skills)
    elif section == "skills":
        skills_line = f"专业技能\n{_names(skills)}"
        expected_skills = frozenset(skills)
    else:
        skills_line = f"项目经历\n在交付项目中使用{_names(skills)}完成开发与测试。"
        expected_skills = frozenset(skills)
    text = "\n".join((identity, education_line, years_line, skills_line))

    expected_codes: set[str] = set()
    if sensitive:
        expected_codes.add("PROTECTED_ATTRIBUTE_MASKED")
    missing_hard = not set(spec.hard) <= set(expected_skills) or education not in {"本科", "硕士", "博士"}
    if missing_hard:
        expected_codes.add("HARD_REQUIREMENT_MISSING")
    if years is None or years < spec.years:
        expected_codes.add("WEAK_HARD_REQUIREMENT_EVIDENCE")
    has_related_hard = any(
        required not in expected_skills
        and any(lexical_relatedness(required, available) > 0 for available in expected_skills)
        for required in spec.hard
    )
    if has_related_hard or (section == "skills" and bool(set(spec.hard) & set(expected_skills))):
        expected_codes.add("WEAK_HARD_REQUIREMENT_EVIDENCE")
    if section == "skills" and expected_skills:
        expected_codes.update({"SKILLS_WITHOUT_PROJECT_EVIDENCE", "PARSE_WARNING"})
    if not expected_skills:
        expected_codes.add("PARSE_WARNING")
    return GoldCandidate(
        item=CandidateInput(id=candidate_id, name=identity, text=text, relevance_label=label),
        skills=expected_skills, years=float(years) if years is not None else None,
        education=education, conflict_codes=frozenset(expected_codes),
    )


def build_candidates(spec: JobSpec) -> list[GoldCandidate]:
    all_skills = (*spec.hard, spec.preferred)
    half = spec.hard[:2]
    unrelated = ("react",) if "react" not in all_skills else ("python",)
    return [
        _candidate(spec, "expert", all_skills, spec.years + 2, "本科", 2),
        _candidate(spec, "sensitive", all_skills, spec.years + 2, "本科", 2, sensitive=True),
        _candidate(spec, "core", spec.hard, spec.years + 1, "硕士", 2),
        _candidate(spec, "partial", half, spec.years + 1, "本科", 1),
        _candidate(spec, "adjacent", spec.adjacent, spec.years + 1, "本科", 1),
        _candidate(spec, "short", spec.hard, 1, "本科", 1),
        _candidate(spec, "degree", all_skills, spec.years + 1, "大专", 0),
        _candidate(spec, "list", all_skills, spec.years + 1, "本科", 0, section="skills"),
        _candidate(spec, "negated", unrelated, spec.years + 1, "本科", 0, negated=True),
        _candidate(spec, "unrelated", unrelated, None, None, 0),
    ]


def _set_prf(predicted: set[tuple], expected: set[tuple]) -> dict[str, float | int]:
    tp = len(predicted & expected)
    fp = len(predicted - expected)
    fn = len(expected - predicted)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def run_controlled_benchmark() -> dict:
    ollama = OllamaClient()
    jd_analyzer = JDAnalyzer(ollama)
    candidate_analyzer = CandidateAnalyzer(ollama)
    matcher = MatchingEngine()
    reviewer = ConflictReviewer(ollama)
    expected_jd_skills: set[tuple] = set()
    predicted_jd_skills: set[tuple] = set()
    expected_candidate_skills: set[tuple] = set()
    predicted_candidate_skills: set[tuple] = set()
    expected_conflicts: set[tuple] = set()
    matcher_conflicts: set[tuple] = set()
    reviewed_conflicts: set[tuple] = set()
    years_correct = education_correct = candidate_count = quote_valid = quote_total = 0
    full_ranking_rows = []
    baseline_ranking_rows = []
    fairness_pairs = fairness_invariant = 0

    for spec in JOB_SPECS:
        job_text = build_job(spec)
        parsed_job, _ = jd_analyzer.analyze(job_text, "rules")
        expected = set((*spec.hard, spec.preferred))
        expected_jd_skills |= {(spec.key, skill) for skill in expected}
        predicted_jd_skills |= {(spec.key, req.normalized_skill) for req in parsed_job.requirements if req.normalized_skill}
        gold_candidates = build_candidates(spec)
        labels = {gold.item.id: gold.item.relevance_label or 0 for gold in gold_candidates}
        results = []
        result_by_suffix = {}
        for gold in gold_candidates:
            parsed, _ = candidate_analyzer.analyze(gold.item.id, gold.item.name, gold.item.text, "rules")
            candidate_count += 1
            expected_candidate_skills |= {(gold.item.id, skill) for skill in gold.skills}
            predicted_candidate_skills |= {(gold.item.id, skill) for skill in parsed.skills}
            years_correct += parsed.years_experience == gold.years
            education_correct += parsed.education == gold.education
            for evidence in parsed.evidence:
                quote_total += 1
                quote_valid += evidence.source_quote in gold.item.text
            raw_result = matcher.match(parsed_job, parsed)
            matcher_conflicts |= {(gold.item.id, finding.code) for finding in raw_result.findings}
            reviewed = reviewer.review(parsed_job, parsed, raw_result, "rules")
            reviewed_conflicts |= {(gold.item.id, finding.code) for finding in reviewed.findings}
            expected_conflicts |= {(gold.item.id, code) for code in gold.conflict_codes}
            results.append(reviewed)
            result_by_suffix[gold.item.id.rsplit("-", 1)[-1]] = reviewed
        fairness_pairs += 1
        expert = result_by_suffix["expert"]
        sensitive = result_by_suffix["sensitive"]
        expert_signature = [(item.requirement_id, item.status, item.score) for item in expert.criteria]
        sensitive_signature = [(item.requirement_id, item.status, item.score) for item in sensitive.criteria]
        fairness_invariant += expert.score == sensitive.score and expert_signature == sensitive_signature
        full_order = [item.candidate_id for item in sorted(results, key=lambda item: (item.score, item.confidence), reverse=True)]
        request = AnalysisRequest(job_description=job_text, candidates=[gold.item for gold in gold_candidates], mode="rules")
        baseline_order = [item_id for item_id, _ in keyword_coverage_ranking(request)]
        full_ranking_rows.append(ranking_metrics(full_order, labels))
        baseline_ranking_rows.append(ranking_metrics(baseline_order, labels))

    def average(rows: list[dict], key: str) -> float:
        values = [row[key] for row in rows if row[key] is not None]
        return round(sum(values) / len(values), 4) if values else 0.0

    metric_keys = ("ndcg_at_5", "precision_at_3", "recall_at_5", "mrr")
    return {
        "benchmark": "Chinese controlled synthetic evidence suite",
        "scope": {"job_queries": len(JOB_SPECS), "candidate_documents": candidate_count, "candidate_job_pairs": candidate_count},
        "jd_skill_extraction": _set_prf(predicted_jd_skills, expected_jd_skills),
        "candidate_skill_extraction": _set_prf(predicted_candidate_skills, expected_candidate_skills),
        "candidate_field_accuracy": {
            "years_experience": round(years_correct / candidate_count, 4),
            "education": round(education_correct / candidate_count, 4),
            "source_quote_substring_validity": round(quote_valid / quote_total, 4) if quote_total else 1.0,
        },
        "conflict_detection_ablation": {
            "matcher_only": _set_prf(matcher_conflicts, expected_conflicts),
            "with_conflict_reviewer": _set_prf(reviewed_conflicts, expected_conflicts),
        },
        "ranking": {
            "keyword_baseline": {key: average(baseline_ranking_rows, key) for key in metric_keys},
            "evidence_workflow": {key: average(full_ranking_rows, key) for key in metric_keys},
        },
        "fairness_pair_check": {
            "pairs": fairness_pairs, "score_and_criteria_invariant_pairs": fairness_invariant,
            "attributes_swapped": ["name", "sex", "age"],
        },
        "limitations": [
            "Cases are template-generated controlled tests, not real resumes or independent human annotations.",
            "Results measure deterministic boundary handling and regression safety, not production hiring validity.",
            "The six job families cover only skills present in the current normalization catalog.",
        ],
    }
