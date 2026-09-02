from pathlib import Path

import pytest

from app.services.talentclef_benchmark import (
    BM25Index,
    evaluate_rankings,
    load_talentclef_task_a,
    run_bm25_benchmark,
    write_trec_run,
)


def _write_fixture(root: Path) -> None:
    language_dir = root / "development" / "en"
    (language_dir / "queries").mkdir(parents=True)
    (language_dir / "corpus").mkdir()
    (language_dir / "queries" / "job-data").write_text(
        "Data engineer Python SQL pipelines", encoding="utf-8"
    )
    (language_dir / "queries" / "job-care").write_text(
        "Nurse patient clinical care", encoding="utf-8"
    )
    (language_dir / "corpus" / "cv-data").write_text(
        "Python SQL data pipelines engineer", encoding="utf-8"
    )
    (language_dir / "corpus" / "cv-care").write_text(
        "Registered nurse provides clinical patient care", encoding="utf-8"
    )
    (language_dir / "corpus" / "cv-other").write_text(
        "Retail sales and inventory", encoding="utf-8"
    )
    (language_dir / "qrels.tsv").write_text(
        "job-data\t0\tcv-data\t1\njob-care\t0\tcv-care\t1\n",
        encoding="utf-8",
    )


def test_loads_extensionless_documents_and_binary_qrels(tmp_path: Path):
    _write_fixture(tmp_path)
    dataset = load_talentclef_task_a(tmp_path, split="dev", language="en")
    assert dataset.split == "development"
    assert set(dataset.queries) == {"job-data", "job-care"}
    assert len(dataset.corpus) == 3
    assert dataset.qrels == {"job-data": {"cv-data": 1}, "job-care": {"cv-care": 1}}


def test_bm25_ranks_domain_match_first_and_reports_official_metrics(tmp_path: Path):
    _write_fixture(tmp_path)
    dataset = load_talentclef_task_a(tmp_path)
    report, rankings = run_bm25_benchmark(dataset)
    assert rankings["job-data"][0][0] == "cv-data"
    assert rankings["job-care"][0][0] == "cv-care"
    assert report["official_task_a_metrics"]["map"] == 1.0
    assert report["official_task_a_metrics"]["mrr"] == 1.0
    assert report["scope"]["ranked_pairs"] == 6


def test_test_split_loads_without_qrels_but_cannot_be_locally_evaluated(tmp_path: Path):
    test_dir = tmp_path / "test" / "en"
    (test_dir / "queries").mkdir(parents=True)
    (test_dir / "corpus").mkdir()
    (test_dir / "queries" / "q1").write_text("job", encoding="utf-8")
    (test_dir / "corpus" / "c1").write_text("candidate", encoding="utf-8")
    dataset = load_talentclef_task_a(tmp_path, split="test")
    assert dataset.qrels is None
    with pytest.raises(ValueError, match="without public qrels"):
        run_bm25_benchmark(dataset)


def test_rejects_qrels_ids_missing_from_corpus(tmp_path: Path):
    _write_fixture(tmp_path)
    qrels = tmp_path / "development" / "en" / "qrels.tsv"
    qrels.write_text("job-data\t0\tmissing-cv\t1\njob-care\t0\tcv-care\t1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown_documents"):
        load_talentclef_task_a(tmp_path)


def test_writes_official_trec_run_shape(tmp_path: Path):
    path = tmp_path / "run.txt"
    write_trec_run(path, {"q1": [("c1", 0.9), ("c2", 0.1)]}, tag="test_run")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[0].split() == ["q1", "Q0", "c1", "1", "0.900000000000", "test_run"]


def test_evaluation_requires_all_qrels_queries():
    with pytest.raises(ValueError, match="exactly match"):
        evaluate_rankings({"q1": [("c1", 1.0)]}, {"q2": {"c1": 1}})


def test_bm25_requires_documents():
    with pytest.raises(ValueError, match="at least one"):
        BM25Index({})
