from app.services.analyzers import CandidateAnalyzer, JDAnalyzer, coerce_optional_number
from app.services.ollama_client import OllamaClient


class FakeOllama:
    def generate_json(self, system, user, **kwargs):
        project_line = user.splitlines()[-1]
        return {
            "skills": ["python"],
            "years_experience": "3年",
            "education": {"kind": "education", "value": "某大学 计算机 本科", "source_quote": "教育经历：计算机本科"},
            "evidence": [{
                "kind": "skill", "value": "Python", "normalized_skill": "python",
                "years": None, "source_quote": project_line,
                "section": "project", "strength": 1.0,
            }],
            "parse_warnings": [],
        }


class FakeJDOllama:
    last_call_cache_hit = False

    def generate_json(self, system, user, **kwargs):
        return {
            "title": "Backend Engineer",
            "summary": "API role",
            "requirements": [
                {
                    "text": "Python",
                    "category": "skill",
                    "priority": "hard",
                    "normalized_skill": "python",
                    "minimum_years": None,
                    "source_quote": "Python is required",
                },
                {
                    "text": "Build APIs",
                    "category": "responsibility",
                    "priority": "hard",
                    "normalized_skill": None,
                    "minimum_years": None,
                    "source_quote": "Build internal APIs",
                },
            ],
            "ambiguities": [],
        }

    def status(self):
        return {"available": True}


def test_jd_rules_extracts_hard_skill_and_years(sample_request):
    parsed, trace = JDAnalyzer(OllamaClient()).analyze(sample_request.job_description, "rules")
    assert parsed.title == "AI Agent应用开发工程师"
    assert any(req.normalized_skill == "python" and req.priority.value == "hard" for req in parsed.requirements)
    assert any(req.minimum_years == 2 for req in parsed.requirements)
    assert trace.status == "completed"


def test_llm_jd_preserves_concrete_requirements_and_downgrades_semantic_clause():
    text = "Backend Engineer\nPython is required\nBuild internal APIs"
    parsed, trace = JDAnalyzer(FakeJDOllama()).analyze(text, "ollama")
    assert trace.status == "completed"
    python = next(item for item in parsed.requirements if item.normalized_skill == "python")
    responsibility = next(item for item in parsed.requirements if item.category == "responsibility")
    assert python.priority.value == "hard"
    assert responsibility.priority.value == "context"


def test_optional_number_coercion_accepts_llm_text_formats():
    assert coerce_optional_number("3+ years") == 3.0
    assert coerce_optional_number("about 2.5 years") == 2.5
    assert coerce_optional_number(None) is None
    assert coerce_optional_number("unknown") is None


def test_candidate_project_evidence_is_strong(sample_request):
    item = sample_request.candidates[0]
    parsed, mode = CandidateAnalyzer(OllamaClient()).analyze(item.id, item.name, item.text, "rules")
    assert mode == "rules"
    assert "langgraph" in parsed.skills
    evidence = [ev for ev in parsed.evidence if ev.normalized_skill == "langgraph"]
    assert evidence and max(ev.strength for ev in evidence) == 1.0


def test_skill_list_evidence_is_capped(sample_request):
    item = sample_request.candidates[-1]
    parsed, _ = CandidateAnalyzer(OllamaClient()).analyze(item.id, item.name, item.text, "rules")
    evidence = [ev for ev in parsed.evidence if ev.kind == "skill"]
    assert evidence
    assert max(ev.strength for ev in evidence) <= 0.65


def test_pii_detection_and_masking(sample_request):
    item = sample_request.candidates[2]
    parsed, _ = CandidateAnalyzer(OllamaClient()).analyze(item.id, item.name, item.text, "rules")
    assert "protected_attribute" in parsed.pii_detected
    assert item.name not in parsed.masked_name


def test_phone_and_email_are_detected(sample_request):
    item = sample_request.candidates[0]
    parsed, _ = CandidateAnalyzer(OllamaClient()).analyze(item.id, item.name, item.text, "rules")
    assert {"phone", "email"}.issubset(set(parsed.pii_detected))


def test_llm_candidate_output_normalizes_structured_education():
    text = "教育经历：计算机本科\n项目经历：使用Python开发接口。"
    parsed, mode = CandidateAnalyzer(FakeOllama()).analyze("cand-test", "测试", text, "ollama")
    assert mode in {"ollama", "ollama_cache"}
    assert parsed.education == "本科"
    assert parsed.years_experience == 3


def test_llm_and_rules_evidence_are_merged():
    text = "教育经历：计算机本科\n项目经历：使用Python、FastAPI开发接口。"
    parsed, mode = CandidateAnalyzer(FakeOllama()).analyze("cand-test", "测试", text, "ollama")
    assert mode in {"ollama", "ollama_cache"}
    assert {"python", "fastapi"}.issubset(set(parsed.skills))


def test_stringified_education_object_is_normalized():
    class StringEducationOllama(FakeOllama):
        def generate_json(self, system, user, **kwargs):
            payload = super().generate_json(system, user, **kwargs)
            payload["education"] = "{'kind': 'education', 'value': '某大学 计算机 本科'}"
            return payload

    text = "教育经历：计算机本科\n项目经历：使用Python开发接口。"
    parsed, mode = CandidateAnalyzer(StringEducationOllama()).analyze("cand-test", "测试", text, "ollama")
    assert mode in {"ollama", "ollama_cache"}
    assert parsed.education == "本科"
