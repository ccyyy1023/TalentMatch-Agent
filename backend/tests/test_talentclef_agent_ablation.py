from app.services.talentclef_agent_ablation import run_direct_pair_agent
from app.services.talentclef_benchmark import TalentClefDataset
from app.services.talentclef_extraction_ab import ExtractionABSample


class FakeClient:
    def generate_json(self, system, prompt, **kwargs):
        if "strong match" in prompt:
            return {
                "score": 90,
                "job_quote": "python role",
                "cv_quote": "strong match python",
                "reason": "explicit match",
            }
        return {"score": 10, "job_quote": "invented", "cv_quote": "invented", "reason": "weak"}


class EchoClient:
    def generate_json(self, system, prompt, **kwargs):
        return {"job": "echo", "cv": "echo"}


def test_direct_pair_agent_reports_ranking_and_quote_grounding():
    dataset = TalentClefDataset(
        split="development",
        language="en",
        queries={"q1": "python role"},
        corpus={"good": "strong match python", "bad": "unrelated profile"},
        qrels={"q1": {"good": 1, "bad": 0}},
    )
    sample = ExtractionABSample(
        query_ids=("q1",),
        pools={"q1": ("good", "bad")},
        labels={"q1": {"good": 1, "bad": 0}},
        candidate_ids=("bad", "good"),
        seed=1,
    )
    report = run_direct_pair_agent(dataset, sample, client=FakeClient())
    assert report["ranking_metrics"]["map"] == 1.0
    assert report["grounding"]["both_quotes_valid_rate"] == 0.5
    assert report["fallbacks"] == 0


def test_direct_pair_agent_counts_schema_violation_as_failure():
    dataset = TalentClefDataset(
        split="development", language="en", queries={"q1": "python role"},
        corpus={"c1": "python"}, qrels={"q1": {"c1": 1}},
    )
    sample = ExtractionABSample(
        query_ids=("q1",), pools={"q1": ("c1",)}, labels={"q1": {"c1": 1}},
        candidate_ids=("c1",), seed=1,
    )
    report = run_direct_pair_agent(dataset, sample, client=EchoClient())
    assert report["fallbacks"] == 1
    assert "missing score" in report["details"]["q1:c1"]["error"]
