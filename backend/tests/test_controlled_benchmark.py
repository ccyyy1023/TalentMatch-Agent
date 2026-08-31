from app.services.controlled_benchmark import JOB_SPECS, build_candidates, build_job
from app.services.skill_catalog import extract_skills


def test_controlled_suite_has_expected_scope():
    assert len(JOB_SPECS) == 6
    assert sum(len(build_candidates(spec)) for spec in JOB_SPECS) == 60


def test_decoy_skill_is_not_extracted_from_no_requirement_sentence():
    for spec in JOB_SPECS:
        extracted = {skill for skill, _ in extract_skills(build_job(spec))}
        assert spec.decoy not in extracted
        assert set(spec.hard) <= extracted


def test_sensitive_pair_changes_only_identity_line():
    for spec in JOB_SPECS:
        cases = {case.item.id.rsplit("-", 1)[-1]: case for case in build_candidates(spec)}
        expert = cases["expert"]
        sensitive = cases["sensitive"]
        assert expert.skills == sensitive.skills
        assert expert.years == sensitive.years
        assert expert.education == sensitive.education
