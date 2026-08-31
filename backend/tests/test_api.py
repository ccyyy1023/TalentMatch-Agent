import copy
import time
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.main import app, auth
from app.schemas import ParsedJD, TraceEvent, UserCreate


client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def authenticated_client():
    username = f"test-admin-{uuid4().hex[:8]}"
    password = "test-password-2026"
    auth.create_user(UserCreate(username=username, password=password, role="admin"))
    logged_in = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert logged_in.status_code == 200
    client.headers.update({"Authorization": f"Bearer {logged_in.json()['access_token']}"})
    yield
    client.headers.pop("Authorization", None)


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["database"]["available"] is True
    assert response.json()["task_queue"]["backend"] == "memory"


def test_liveness_and_readiness_endpoints():
    assert client.get("/api/v1/health/live").json()["status"] == "alive"
    ready = client.get("/api/v1/health/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"


def test_sensitive_endpoint_requires_login():
    anonymous = TestClient(app)
    response = anonymous.get("/api/v1/demo")
    assert response.status_code == 401


def test_admin_can_create_recruiter_and_recruiter_cannot_manage_users():
    username = f"test-recruiter-{uuid4().hex[:8]}"
    password = "recruiter-password-2026"
    created = client.post(
        "/api/v1/users", json={"username": username, "password": password, "role": "recruiter"},
    )
    assert created.status_code == 201
    recruiter_login = TestClient(app).post(
        "/api/v1/auth/login", json={"username": username, "password": password},
    )
    token = recruiter_login.json()["access_token"]
    denied = TestClient(app).get("/api/v1/users", headers={"Authorization": f"Bearer {token}"})
    assert denied.status_code == 403


def test_admin_audit_contains_login_and_user_creation():
    response = client.get("/api/v1/audit")
    assert response.status_code == 200
    actions = {(item["action"], item["resource_type"]) for item in response.json()}
    assert ("login", "session") in actions
    assert ("create", "user") in actions


def test_managed_job_lifecycle():
    created = client.post("/api/v1/jobs", json={
        "title": "AI应用工程师",
        "description": "负责智能体工作流、后端接口与招聘场景评测体系建设。",
        "status": "draft",
    })
    assert created.status_code == 201, created.text
    job_id = created.json()["id"]
    updated = client.patch(f"/api/v1/jobs/{job_id}", json={"status": "active"})
    assert updated.status_code == 200
    assert updated.json()["status"] == "active"
    listed = client.get("/api/v1/jobs?status=active")
    assert any(item["id"] == job_id for item in listed.json())


def test_managed_candidate_lifecycle():
    created = client.post("/api/v1/candidates", json={
        "display_name": "测试候选人",
        "external_ref": "ATS-TEST-001",
        "resume_text": "三年Python后端经验，使用FastAPI、PostgreSQL和Docker建设模型服务。",
    })
    assert created.status_code == 201, created.text
    candidate_id = created.json()["id"]
    updated = client.patch(f"/api/v1/candidates/{candidate_id}", json={"status": "reviewing"})
    assert updated.status_code == 200
    assert updated.json()["status"] == "reviewing"
    fetched = client.get(f"/api/v1/candidates/{candidate_id}")
    assert fetched.json()["external_ref"] == "ATS-TEST-001"


def test_screening_batch_business_loop():
    job = client.post("/api/v1/jobs", json={
        "title": "Python智能体工程师",
        "description": "负责使用Python、FastAPI和SQL建设可追踪的多智能体招聘分析服务。",
        "status": "active",
    })
    assert job.status_code == 201, job.text
    candidate = client.post("/api/v1/candidates", json={
        "display_name": "业务闭环测试候选人",
        "external_ref": f"ATS-BATCH-{uuid4().hex[:8]}",
        "resume_text": "拥有三年Python与FastAPI开发经验，使用SQL数据库交付过智能体工作流项目。",
    })
    assert candidate.status_code == 201, candidate.text

    created = client.post("/api/v1/screening-batches", json={
        "job_id": job.json()["id"],
        "candidate_ids": [candidate.json()["id"]],
        "mode": "rules",
    })
    assert created.status_code == 202, created.text
    batch_id = created.json()["id"]
    assert created.json()["status"] == "queued"

    batch = None
    for _ in range(100):
        batch = client.get(f"/api/v1/screening-batches/{batch_id}")
        assert batch.status_code == 200, batch.text
        if batch.json()["status"] in {"awaiting_review", "failed"}:
            break
        time.sleep(0.02)
    assert batch.json()["status"] == "awaiting_review", batch.json()
    assert batch.json()["run_id"].startswith("run-")
    assert batch.json()["items"][0]["stage"] == "analyzed"

    decided = client.post(
        f"/api/v1/screening-batches/{batch_id}/candidates/{candidate.json()['id']}/decision",
        json={"decision": "advance", "note": "核心能力证据完整，进入技术面试"},
    )
    assert decided.status_code == 200, decided.text
    assert decided.json()["status"] == "reviewed"
    assert decided.json()["reviewed_count"] == 1
    assert decided.json()["items"][0]["stage"] == "advanced"

    history = client.get(f"/api/v1/candidates/{candidate.json()['id']}/screening-history")
    assert history.status_code == 200
    assert history.json()[0]["batch_id"] == batch_id
    assert history.json()[0]["decision"] == "advance"


def test_screening_batch_rejects_closed_job():
    job = client.post("/api/v1/jobs", json={
        "title": "已关闭测试岗位",
        "description": "该岗位用于验证关闭状态下不能继续创建新的候选人筛选批次。",
        "status": "closed",
    }).json()
    candidate = client.post("/api/v1/candidates", json={
        "display_name": "关闭岗位测试候选人",
        "resume_text": "具有Python后端工程经验，并参与过招聘数据处理系统的建设工作。",
    }).json()
    response = client.post("/api/v1/screening-batches", json={
        "job_id": job["id"], "candidate_ids": [candidate["id"]], "mode": "rules",
    })
    assert response.status_code == 409


def test_prometheus_metrics_expose_core_service_signals():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "talentmatch_http_requests_total" in response.text
    assert "talentmatch_analysis_tasks_total" in response.text
    assert "talentmatch_analysis_queue_depth" in response.text
    assert "talentmatch_analysis_task_outcomes_total" in response.text
    assert "talentmatch_analysis_queue_wait_seconds" in response.text
    assert "talentmatch_agent_node_duration_seconds" in response.text


def test_demo_endpoint_returns_candidates():
    response = client.get("/api/v1/demo")
    assert response.status_code == 200
    assert len(response.json()["candidates"]) >= 5


def test_analyze_rules_endpoint(sample_payload):
    payload = copy.deepcopy(sample_payload)
    payload["mode"] = "rules"
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ranking"]
    assert body["run_id"].startswith("run-")


def test_background_analysis_task_completes_and_returns_result(sample_payload):
    payload = copy.deepcopy(sample_payload)
    payload["mode"] = "rules"
    created = client.post("/api/v1/analysis/tasks", json=payload)
    assert created.status_code == 202, created.text
    task_id = created.json()["task_id"]
    assert task_id.startswith("task-")

    status = None
    for _ in range(100):
        status = client.get(f"/api/v1/analysis/tasks/{task_id}")
        assert status.status_code == 200
        if status.json()["status"] in {"completed", "failed"}:
            break
        time.sleep(0.02)

    body = status.json()
    assert body["status"] == "completed", body
    assert body["progress"] == 100
    assert body["stage"] == "completed"
    assert body["result"]["ranking"]


def test_unknown_background_analysis_task_returns_404():
    response = client.get("/api/v1/analysis/tasks/task-not-found")
    assert response.status_code == 404


def test_job_parse_and_human_confirmed_criteria(sample_payload):
    parsed = client.post(
        "/api/v1/jobs/parse", json={"text": sample_payload["job_description"], "mode": "rules"},
    )
    assert parsed.status_code == 200
    job = parsed.json()["job"]
    assert job["requirements"]
    payload = copy.deepcopy(sample_payload)
    payload.update({"mode": "rules", "confirmed_job": job, "criteria_confirmed_by_human": True})
    analyzed = client.post("/api/v1/analyze", json=payload)
    assert analyzed.status_code == 200
    assert "已确认" in analyzed.json()["traces"][0]["detail"]


def test_adaptive_job_parse_uses_ollama_analyzer(sample_payload, monkeypatch):
    called = {}

    def fake_analyze(text, mode):
        called["mode"] = mode
        return ParsedJD(title="测试岗位", summary=text), TraceEvent(
            node="jd_analyzer_agent", status="completed", detail="fake", elapsed_ms=0,
        )

    monkeypatch.setattr("app.main.workflow.jd_analyzer.analyze", fake_analyze)
    response = client.post(
        "/api/v1/jobs/parse", json={"text": sample_payload["job_description"], "mode": "adaptive"},
    )
    assert response.status_code == 200
    assert called["mode"] == "ollama"


def test_invalid_mode_is_rejected(sample_payload):
    payload = copy.deepcopy(sample_payload)
    payload["mode"] = "invalid"
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 422


def test_txt_document_parse():
    response = client.post(
        "/api/v1/documents/parse",
        files={"file": ("resume.txt", "Python开发工程师，具有FastAPI项目经验。".encode("utf-8"), "text/plain")},
    )
    assert response.status_code == 200
    assert "FastAPI" in response.json()["text"]


def test_unsupported_document_is_rejected():
    response = client.post(
        "/api/v1/documents/parse",
        files={"file": ("resume.exe", b"invalid", "application/octet-stream")},
    )
    assert response.status_code == 422


def test_human_decision_is_persisted(sample_payload):
    payload = copy.deepcopy(sample_payload)
    payload["mode"] = "rules"
    analysis = client.post("/api/v1/analyze", json=payload).json()
    run_id = analysis["run_id"]
    candidate_id = analysis["ranking"][0]["candidate_id"]
    saved = client.post(
        f"/api/v1/runs/{run_id}/decisions",
        json={"candidate_id": candidate_id, "decision": "advance", "note": "证据完整，进入面试"},
    )
    assert saved.status_code == 200
    assert saved.json()["reviewer"].startswith("test-admin-")
    listed = client.get(f"/api/v1/runs/{run_id}/decisions")
    assert listed.status_code == 200
    assert listed.json()[0]["decision"] == "advance"
    assert listed.json()[0]["reviewer"] == saved.json()["reviewer"]
