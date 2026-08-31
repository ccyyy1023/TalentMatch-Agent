import pytest

from app.services.skill_catalog import extract_skills, lexical_relatedness, normalize_skill


@pytest.mark.parametrize("raw,expected", [
    ("Python", "python"), ("检索增强生成", "rag"), ("K8S", "kubernetes"),
    ("SpringBoot", "spring_boot"), ("Postgres", "postgresql"),
    ("自然语言处理", "nlp"), ("Vue.js", "vue"), ("容器化", "docker"),
])
def test_normalize_skill_aliases(raw, expected):
    assert normalize_skill(raw) == expected


def test_extract_skills_respects_word_boundaries():
    skills = {skill for skill, _ in extract_skills("使用Python、FastAPI和PostgreSQL构建API")}
    assert {"python", "fastapi", "postgresql"}.issubset(skills)


def test_related_skill_is_not_exact_match():
    assert 0 < lexical_relatedness("pytorch", "tensorflow") < 1


def test_unrelated_skills_have_zero_relatedness():
    assert lexical_relatedness("python", "react") == 0


def test_negated_skill_is_not_extracted_but_positive_skill_is():
    skills = {skill for skill, _ in extract_skills("未使用Python，也未接触Docker，主要使用Java开发。")}
    assert skills == {"java"}


def test_job_description_no_requirement_phrase_is_respected():
    skills = {skill for skill, _ in extract_skills("无需Kubernetes，必须掌握FastAPI。")}
    assert skills == {"fastapi"}


def test_negation_scope_covers_skill_list_until_clause_boundary():
    skills = {skill for skill, _ in extract_skills("未使用Python、Docker、Redis，主要使用Java。")}
    assert skills == {"java"}
