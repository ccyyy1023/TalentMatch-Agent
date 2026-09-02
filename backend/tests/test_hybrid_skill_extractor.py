from app.services.analyzers import CandidateAnalyzer, JDAnalyzer
from app.services.hybrid_skill_extractor import (
    JobBertDocumentSkillExtractor,
    detect_document_language,
    normalize_open_skill,
)
from app.services.ollama_client import OllamaClient
from app.services.skill_catalog import display_name
from app.services.skillspan_benchmark import SkillSpan


class FakePredictor:
    def __init__(self):
        self.calls = 0

    def predict_batch(self, token_batches, batch_size=32):
        self.calls += 1
        outputs = []
        for tokens in token_batches:
            lowered = [item.casefold() for item in tokens]
            if "project" in lowered and "management" in lowered:
                start = lowered.index("project")
                outputs.append([SkillSpan(start, start + 2, "skill", "project management")])
            else:
                outputs.append([])
        return outputs


def test_language_router_is_conservative_for_supported_english():
    assert detect_document_language("The role requires project management experience and communication skills.") == "en"
    assert detect_document_language("候选人需要具备项目管理和沟通能力。") == "zh"
    assert detect_document_language("El puesto requiere experiencia con gestión de proyectos y trabajo en equipo.") == "es"


def test_jobbert_mentions_keep_exact_source_quote_and_open_skill_key():
    predictor = FakePredictor()
    extractor = JobBertDocumentSkillExtractor(predictor=predictor)
    text = "Required skills:\nStrong project management experience."
    mentions = extractor.extract(text, language="en")
    assert len(mentions) == 1
    assert mentions[0].text == "project management"
    assert mentions[0].source_quote == "Strong project management experience."
    assert mentions[0].normalized_skill == "open_skill:project_management"
    assert display_name(mentions[0].normalized_skill) == "project management"


def test_jobbert_route_skips_non_english_documents():
    predictor = FakePredictor()
    extractor = JobBertDocumentSkillExtractor(predictor=predictor)
    assert extractor.extract("需要项目管理能力。", language="zh") == []
    assert extractor.extract("Se requiere gestión de proyectos.", language="es") == []
    assert predictor.calls == 0


def test_hybrid_mentions_reach_existing_jd_and_candidate_schemas():
    extractor = JobBertDocumentSkillExtractor(predictor=FakePredictor())
    ollama = OllamaClient(chat_model="unused")
    jd, _ = JDAnalyzer(ollama, extractor).analyze(
        "The role requires strong project management experience.", "rules",
    )
    candidate, origin = CandidateAnalyzer(ollama, extractor).analyze(
        "c1", "Candidate", "WORK EXPERIENCE\nDelivered project management for a migration.", "rules",
        target_skills=["open_skill:project_management"],
    )
    assert origin == "rules"
    assert any(item.normalized_skill == "open_skill:project_management" for item in jd.requirements)
    assert "open_skill:project_management" in candidate.skills
    assert any(
        item.normalized_skill == "open_skill:project_management" and item.source_quote in candidate.evidence[0].source_quote
        for item in candidate.evidence
    )


def test_candidate_profile_stays_job_independent_until_exact_target_verification():
    analyzer = CandidateAnalyzer(OllamaClient(chat_model="unused"))
    text = "WORK EXPERIENCE\nDelivered project management for a migration."
    profile, origin = analyzer.analyze_profile("c1", "Candidate", text, "rules")
    enriched = analyzer.enrich_for_job(profile, text, ["open_skill:project_management"])
    enriched_again = analyzer.enrich_for_job(enriched, text, ["open_skill:project_management"])

    assert origin == "rules"
    assert "open_skill:project_management" not in profile.skills
    assert "open_skill:project_management" in enriched.skills
    assert len(enriched_again.evidence) == len(enriched.evidence)
    assert any(
        item.normalized_skill == "open_skill:project_management"
        and item.source_quote == "Delivered project management for a migration."
        and item.section == "work"
        for item in enriched.evidence
    )


def test_open_skill_normalization_rejects_generic_labels():
    assert normalize_open_skill("skills") is None
    assert normalize_open_skill("Stakeholder Management") == "open_skill:stakeholder_management"
