from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from app.services.model_cache import ModelResponseCache
from app.services.ollama_client import OllamaClient
from app.services.analyzers import CandidateAnalyzer


def test_cache_key_changes_with_model_prompt_or_input(tmp_path: Path):
    cache = ModelResponseCache(tmp_path / "cache.db")
    options = {"temperature": 0.0, "seed": 42}
    base = cache.build_key("qwen@a", "jd", "v1", "system", "input", options)
    assert base != cache.build_key("qwen@b", "jd", "v1", "system", "input", options)
    assert base != cache.build_key("qwen@a", "jd", "v2", "system", "input", options)
    assert base != cache.build_key("qwen@a", "jd", "v1", "system", "changed", options)


def test_cache_round_trip_and_hit_count(tmp_path: Path):
    cache = ModelResponseCache(tmp_path / "cache.db")
    key = cache.build_key("model", "candidate", "v1", "s", "u", {})
    assert cache.get(key) is None
    cache.put(key, "model", "candidate", {"skills": ["python"]})
    assert cache.get(key) == {"skills": ["python"]}
    assert cache.stats() == {"entries": 1, "hits": 1}


def test_ollama_client_reuses_cached_json_without_second_http_call(tmp_path: Path, monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": '{"ok": true}'}}

    calls = {"post": 0, "payload": None}

    def fake_post(*args, **kwargs):
        calls["post"] += 1
        calls["payload"] = kwargs["json"]
        return FakeResponse()

    monkeypatch.setattr("app.services.ollama_client.httpx.post", fake_post)
    client = OllamaClient(cache=ModelResponseCache(tmp_path / "cache.db"))
    client._chat_model_identity = "qwen@test-digest"
    first = client.generate_json("system", "user", cache_namespace="test", prompt_version="v1")
    second = client.generate_json("system", "user", cache_namespace="test", prompt_version="v1")
    assert first == second == {"ok": True}
    assert calls["post"] == 1
    assert calls["payload"]["model"] == client.chat_model
    assert calls["payload"]["format"] == "json"
    assert calls["payload"]["think"] is False
    assert calls["payload"]["options"]["num_predict"] == 1536
    assert client.last_call_cache_hit is True


def test_cache_hit_state_is_isolated_per_worker_thread(tmp_path: Path):
    client = OllamaClient(cache=ModelResponseCache(tmp_path / "cache.db"))
    client.last_call_cache_hit = True

    def worker():
        before = client.last_call_cache_hit
        client.last_call_cache_hit = True
        return before, client.last_call_cache_hit

    with ThreadPoolExecutor(max_workers=1) as executor:
        before, after = executor.submit(worker).result()
    assert before is False and after is True
    assert client.last_call_cache_hit is True


def test_candidate_profile_llm_cache_is_reused_across_job_targets(tmp_path: Path, monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": '{"skills":["Python"],"years_experience":2,"education":"本科","evidence":[{"kind":"skill","value":"Python","normalized_skill":"python","source_quote":"项目使用Python完成接口开发。","section":"project","strength":1.0}],"parse_warnings":[]}'}}

    calls = {"post": 0}

    def fake_post(*args, **kwargs):
        calls["post"] += 1
        return FakeResponse()

    monkeypatch.setattr("app.services.ollama_client.httpx.post", fake_post)
    client = OllamaClient(cache=ModelResponseCache(tmp_path / "cache.db"))
    client._chat_model_identity = "qwen@test-digest"
    analyzer = CandidateAnalyzer(client)
    text = "项目经历\n项目使用Python完成接口开发。\n熟悉project management。"

    first, first_origin = analyzer.analyze(
        "c1", "Candidate", text, "ollama", target_skills=["python"],
    )
    second, second_origin = analyzer.analyze(
        "c1", "Candidate", text, "ollama", target_skills=["open_skill:project_management"],
    )

    assert calls["post"] == 1
    assert first_origin == "ollama"
    assert second_origin == "ollama_cache"
    assert "python" in first.skills
    assert "open_skill:project_management" in second.skills
