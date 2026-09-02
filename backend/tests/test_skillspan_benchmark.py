import json

from app.services.skillspan_benchmark import (
    JobBertSpanPredictor, SkillSpan, bio_to_spans, catalog_predict, load_skillspan, span_metrics, stratified_sample,
)


def test_bio_to_spans_closes_on_new_begin_and_end():
    tokens = ["design", "APIs", "and", "communicate", "clearly"]
    tags = ["B", "I", "O", "B", "I"]
    assert bio_to_spans(tokens, tags, "skill") == [
        SkillSpan(0, 2, "skill", "design APIs"),
        SkillSpan(3, 5, "skill", "communicate clearly"),
    ]


def test_loader_combines_skill_and_knowledge_layers(tmp_path):
    path = tmp_path / "sample.json"
    path.write_text(json.dumps({
        "idx": 7, "tokens": ["communicate", "with", "Python"],
        "tags_skill": ["B", "O", "O"], "tags_knowledge": ["O", "O", "B"], "source": "tech",
    }) + "\n", encoding="utf-8")
    record = load_skillspan(path)[0]
    assert {(item.text, item.label) for item in record.gold} == {
        ("communicate", "skill"), ("Python", "knowledge"),
    }


def test_catalog_predicts_known_technology_token_boundaries():
    predicted = catalog_predict(["Experience", "with", "Python", "and", "Docker", "."])
    assert {(item.start, item.end, item.text) for item in predicted} == {
        (2, 3, "Python"), (4, 5, "Docker"),
    }
    assert all(item.label == "knowledge" for item in predicted)


def test_span_metrics_distinguish_typed_exact_and_overlap():
    gold = [SkillSpan(1, 3, "skill", "clear communication")]
    predicted = [SkillSpan(2, 3, "knowledge", "communication")]
    metrics = span_metrics([(predicted, gold)])
    assert metrics["typed_exact"]["f1"] == 0.0
    assert metrics["boundary_exact"]["f1"] == 0.0
    assert metrics["boundary_overlap"]["f1"] == 1.0


def test_stratified_sample_is_reproducible():
    records = []
    for index in range(20):
        path_record = type("Record", (), {})()
        path_record.record_id = f"tech:{index}"
        path_record.tokens = ["Python"]
        path_record.source = "tech" if index % 2 else "house"
        path_record.gold = [SkillSpan(0, 1, "knowledge", "Python")] if index % 3 else []
        records.append(path_record)
    left = [item.record_id for item in stratified_sample(records, 8, 42)]
    right = [item.record_id for item in stratified_sample(records, 8, 42)]
    assert left == right and len(left) == 8


def test_jobbert_uses_separate_official_endpoints_for_each_label():
    assert JobBertSpanPredictor.MODEL_IDS == {
        "skill": "jjzha/jobbert_skill_extraction",
        "knowledge": "jjzha/jobbert_knowledge_extraction",
    }
    assert all(len(value) == 40 for value in JobBertSpanPredictor.MODEL_REVISIONS.values())
