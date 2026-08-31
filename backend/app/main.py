from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings
from app.schemas import (
    AnalysisRequest, AnalysisResponse, AnalysisTaskCreated, AnalysisTaskStatus, AuditRecord, EvaluationMetrics,
    CandidateCreate, CandidateRecord, CandidateUpdate, HumanDecisionRecord, HumanDecisionRequest,
    JobCreate, JobParseRequest, JobParseResponse, JobRecord, JobUpdate, LoginRequest, LoginResponse,
    ScreeningBatchCreate, ScreeningBatchRecord, ScreeningDecisionRequest, UserCreate, UserView,
)
from app.services.auth import AuthService
from app.services.catalog import RecruitmentCatalog
from app.services.database import RunStore
from app.services.document_parser import UnsupportedDocument, parse_document
from app.services.evaluation import evaluate_run
from app.services.ollama_client import OllamaClient
from app.services.observability import (
    ANALYSIS_TASKS, TASK_POLLS, configure_structured_logging, metrics_payload, observe_http_request,
    record_task_observation,
)
from app.services.relational import database_health
from app.services.task_manager import AnalysisTaskManager, RedisAnalysisTaskManager
from app.services.workflow import TalentMatchWorkflow


configure_structured_logging()
logger = logging.getLogger("talentmatch.api")

app = FastAPI(
    title="TalentMatch Agent API",
    version="0.1.0",
    description="证据驱动的多智能体人岗匹配与招聘决策支持系统",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_observability(request: Request, call_next):
    return await observe_http_request(request, call_next)

ollama = OllamaClient()
workflow = TalentMatchWorkflow(ollama)
store = RunStore()
if settings.task_backend == "redis":
    task_manager = RedisAnalysisTaskManager()
elif settings.task_backend == "memory":
    task_manager = AnalysisTaskManager(workflow, store)
else:
    raise RuntimeError("TALENTMATCH_TASK_BACKEND必须为memory或redis")
auth = AuthService()
catalog = RecruitmentCatalog()
bearer = HTTPBearer(auto_error=False)
sample_path = settings.project_root / "data" / "sample_dataset.json"


def reconcile_screening_batch(batch: ScreeningBatchRecord) -> ScreeningBatchRecord:
    task = task_manager.get(batch.task_id)
    if task is None:
        return batch
    record_task_observation(task)
    target_status = {
        "queued": "queued", "running": "running", "completed": "awaiting_review", "failed": "failed",
    }[task.status]
    run_id = task.result.run_id if task.result is not None else None
    error = task.error if task.status == "failed" else None
    if batch.status == "reviewed" and target_status == "awaiting_review":
        return batch
    if batch.status == target_status and batch.run_id == run_id and batch.error == error:
        return batch
    updated = catalog.update_screening_batch_from_task(batch.id, target_status, run_id, error)
    return updated or batch


def current_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> UserView:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="请先登录")
    user = auth.authenticate(credentials.credentials)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="登录已失效，请重新登录")
    return user


def admin_user(user: UserView = Depends(current_user)) -> UserView:
    if user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="需要管理员权限")
    return user


@app.get("/api/v1/health")
def health() -> dict:
    database = database_health()
    task_queue = task_manager.health()
    return {
        "status": "ok" if database["available"] and task_queue["available"] else "degraded",
        "service": "talentmatch-agent",
        "ollama": ollama.status(), "model_cache": ollama.cache_status(),
        "database": database, "task_queue": task_queue,
    }


@app.get("/api/v1/health/live")
def liveness() -> dict:
    return {"status": "alive"}


@app.get("/api/v1/health/ready")
def readiness() -> dict:
    database = database_health()
    task_queue = task_manager.health()
    if not database["available"] or not task_queue["available"]:
        raise HTTPException(status_code=503, detail={"database": database, "task_queue": task_queue})
    return {"status": "ready", "database": database, "task_queue": task_queue}


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics() -> Response:
    queue_health = task_manager.health()
    payload, content_type = metrics_payload(queue_health.get("queued"))
    return Response(content=payload, media_type=content_type)


@app.post("/api/v1/auth/login", response_model=LoginResponse)
def login(request: LoginRequest) -> LoginResponse:
    result = auth.login(request.username, request.password)
    if result is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")
    auth.record_audit(result.user, "login", "session")
    return result


@app.get("/api/v1/auth/me", response_model=UserView)
def get_me(user: UserView = Depends(current_user)) -> UserView:
    return user


@app.post("/api/v1/auth/logout")
def logout(
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
    user: UserView = Depends(current_user),
) -> dict:
    auth.record_audit(user, "logout", "session")
    auth.logout(credentials.credentials)
    return {"status": "logged_out"}


@app.post("/api/v1/users", response_model=UserView, status_code=201)
def create_user(request: UserCreate, actor: UserView = Depends(admin_user)) -> UserView:
    try:
        created = auth.create_user(request)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    auth.record_audit(actor, "create", "user", created.id, {"username": created.username, "role": created.role})
    return created


@app.get("/api/v1/users", response_model=list[UserView])
def list_users(_: UserView = Depends(admin_user)) -> list[UserView]:
    return auth.list_users()


@app.get("/api/v1/audit", response_model=list[AuditRecord])
def list_audit(limit: int = 100, _: UserView = Depends(admin_user)) -> list[AuditRecord]:
    return auth.list_audit(limit)


@app.post("/api/v1/jobs", response_model=JobRecord, status_code=201)
def create_job(request: JobCreate, user: UserView = Depends(current_user)) -> JobRecord:
    record = catalog.create_job(request, user.id)
    auth.record_audit(user, "create", "job", record.id, {"status": record.status})
    return record


@app.get("/api/v1/jobs", response_model=list[JobRecord])
def list_jobs(status: str | None = None, limit: int = 100, _: UserView = Depends(current_user)) -> list[JobRecord]:
    if status not in {None, "draft", "active", "closed"}:
        raise HTTPException(status_code=422, detail="无效的岗位状态")
    return catalog.list_jobs(status, min(max(limit, 1), 100))


@app.get("/api/v1/jobs/{job_id}", response_model=JobRecord)
def get_job(job_id: str, _: UserView = Depends(current_user)) -> JobRecord:
    record = catalog.get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="岗位不存在")
    return record


@app.patch("/api/v1/jobs/{job_id}", response_model=JobRecord)
def update_job(job_id: str, request: JobUpdate, user: UserView = Depends(current_user)) -> JobRecord:
    record = catalog.update_job(job_id, request)
    if record is None:
        raise HTTPException(status_code=404, detail="岗位不存在")
    auth.record_audit(user, "update", "job", job_id, request.model_dump(exclude_none=True))
    return record


@app.post("/api/v1/candidates", response_model=CandidateRecord, status_code=201)
def create_candidate(request: CandidateCreate, user: UserView = Depends(current_user)) -> CandidateRecord:
    record = catalog.create_candidate(request, user.id)
    auth.record_audit(user, "create", "candidate", record.id, {"status": record.status})
    return record


@app.get("/api/v1/candidates", response_model=list[CandidateRecord])
def list_candidates(status: str | None = None, limit: int = 100, _: UserView = Depends(current_user)) -> list[CandidateRecord]:
    if status not in {None, "new", "reviewing", "archived"}:
        raise HTTPException(status_code=422, detail="无效的候选人状态")
    return catalog.list_candidates(status, min(max(limit, 1), 100))


@app.get("/api/v1/candidates/{candidate_id}", response_model=CandidateRecord)
def get_candidate(candidate_id: str, _: UserView = Depends(current_user)) -> CandidateRecord:
    record = catalog.get_candidate(candidate_id)
    if record is None:
        raise HTTPException(status_code=404, detail="候选人不存在")
    return record


@app.get("/api/v1/candidates/{candidate_id}/screening-history")
def get_candidate_screening_history(candidate_id: str, _: UserView = Depends(current_user)) -> list[dict]:
    if catalog.get_candidate(candidate_id) is None:
        raise HTTPException(status_code=404, detail="候选人不存在")
    return catalog.candidate_screening_history(candidate_id)


@app.patch("/api/v1/candidates/{candidate_id}", response_model=CandidateRecord)
def update_candidate(
    candidate_id: str, request: CandidateUpdate, user: UserView = Depends(current_user),
) -> CandidateRecord:
    record = catalog.update_candidate(candidate_id, request)
    if record is None:
        raise HTTPException(status_code=404, detail="候选人不存在")
    auth.record_audit(user, "update", "candidate", candidate_id, request.model_dump(exclude_none=True))
    return record


@app.get("/api/v1/demo")
def get_demo(_: UserView = Depends(current_user)) -> dict:
    return json.loads(sample_path.read_text(encoding="utf-8"))


@app.post("/api/v1/documents/parse")
async def parse_uploaded_document(
    file: UploadFile = File(...), user: UserView = Depends(current_user),
) -> dict:
    content = await file.read()
    if len(content) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="文件不能超过8MB")
    try:
        text = parse_document(file.filename or "document", content)
    except UnsupportedDocument as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    auth.record_audit(
        user, "parse", "candidate_document", detail={"filename": file.filename, "characters": len(text)},
    )
    return {"filename": file.filename, "characters": len(text), "text": text}


@app.post("/api/v1/analyze", response_model=AnalysisResponse)
def analyze(request: AnalysisRequest, user: UserView = Depends(current_user)) -> AnalysisResponse:
    try:
        response = workflow.run(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"分析失败：{type(exc).__name__}: {exc}") from exc
    labels = {item.id: item.relevance_label for item in request.candidates if item.relevance_label is not None}
    metrics = evaluate_run(response, labels) if labels else None
    store.save(response, metrics)
    auth.record_audit(user, "analyze", "run", response.run_id, {"candidate_count": len(response.ranking), "mode": request.mode})
    return response


@app.post("/api/v1/analysis/tasks", response_model=AnalysisTaskCreated, status_code=202)
def create_analysis_task(
    request: AnalysisRequest, user: UserView = Depends(current_user),
) -> AnalysisTaskCreated:
    try:
        created = task_manager.submit(request)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="分析任务队列暂不可用") from exc
    ANALYSIS_TASKS.labels("workbench", request.mode).inc()
    auth.record_audit(user, "create", "analysis_task", created.task_id, {"candidate_count": len(request.candidates), "mode": request.mode})
    return created


@app.get("/api/v1/analysis/tasks/{task_id}", response_model=AnalysisTaskStatus)
def get_analysis_task(task_id: str, _: UserView = Depends(current_user)) -> AnalysisTaskStatus:
    status = task_manager.get(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail="分析任务不存在或已过期")
    TASK_POLLS.labels(status.status).inc()
    record_task_observation(status)
    return status


@app.post("/api/v1/screening-batches", response_model=ScreeningBatchRecord, status_code=202)
def create_screening_batch(
    request: ScreeningBatchCreate, user: UserView = Depends(current_user),
) -> ScreeningBatchRecord:
    job = catalog.get_job(request.job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="岗位不存在")
    if job.status == "closed":
        raise HTTPException(status_code=409, detail="已关闭岗位不能创建筛选批次")
    candidate_ids = list(dict.fromkeys(request.candidate_ids))
    if len(candidate_ids) != len(request.candidate_ids):
        raise HTTPException(status_code=422, detail="候选人不能重复")
    candidate_records = []
    for candidate_id in candidate_ids:
        candidate = catalog.get_candidate(candidate_id)
        if candidate is None:
            raise HTTPException(status_code=404, detail=f"候选人不存在：{candidate_id}")
        if candidate.status == "archived":
            raise HTTPException(status_code=409, detail=f"已归档候选人不能加入批次：{candidate_id}")
        candidate_records.append(candidate)
    analysis_request = AnalysisRequest(
        job_description=job.description,
        candidates=[
            {"id": candidate.id, "name": candidate.display_name, "text": candidate.resume_text}
            for candidate in candidate_records
        ],
        mode=request.mode,
        enable_semantic_matching=request.enable_semantic_matching,
    )
    try:
        task = task_manager.submit(analysis_request)
        batch = catalog.create_screening_batch(job, candidate_records, task.task_id, request.mode, user.id)
    except Exception as exc:
        logger.exception("screening_batch_submit_failed")
        raise HTTPException(status_code=503, detail="筛选批次暂时无法创建") from exc
    ANALYSIS_TASKS.labels("screening_batch", request.mode).inc()
    auth.record_audit(
        user, "create", "screening_batch", batch.id,
        {"job_id": job.id, "candidate_count": len(candidate_records), "task_id": task.task_id},
    )
    return batch


@app.get("/api/v1/screening-batches", response_model=list[ScreeningBatchRecord])
def list_screening_batches(limit: int = 50, _: UserView = Depends(current_user)) -> list[ScreeningBatchRecord]:
    return [reconcile_screening_batch(batch) for batch in catalog.list_screening_batches(min(max(limit, 1), 100))]


@app.get("/api/v1/screening-batches/{batch_id}", response_model=ScreeningBatchRecord)
def get_screening_batch(batch_id: str, _: UserView = Depends(current_user)) -> ScreeningBatchRecord:
    batch = catalog.get_screening_batch(batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="筛选批次不存在")
    reconciled = reconcile_screening_batch(batch)
    task = task_manager.get(reconciled.task_id)
    if task is not None:
        TASK_POLLS.labels(task.status).inc()
    return reconciled


@app.post(
    "/api/v1/screening-batches/{batch_id}/candidates/{candidate_id}/decision",
    response_model=ScreeningBatchRecord,
)
def save_screening_batch_decision(
    batch_id: str,
    candidate_id: str,
    request: ScreeningDecisionRequest,
    user: UserView = Depends(current_user),
) -> ScreeningBatchRecord:
    batch = catalog.get_screening_batch(batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="筛选批次不存在")
    batch = reconcile_screening_batch(batch)
    if batch.status not in {"awaiting_review", "reviewed"} or batch.run_id is None:
        raise HTTPException(status_code=409, detail="批次分析尚未完成，不能记录人工决策")
    decision = HumanDecisionRequest(
        candidate_id=candidate_id, decision=request.decision, note=request.note, reviewer=user.username,
    )
    try:
        store.save_decision(batch.run_id, decision)
    except KeyError as exc:
        raise HTTPException(status_code=409, detail="批次运行结果尚未持久化") from exc
    updated = catalog.save_screening_decision(
        batch_id, candidate_id, request.decision, request.note, user.username,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="候选人不在该筛选批次中")
    auth.record_audit(
        user, "decide", "screening_batch_candidate", candidate_id,
        {"batch_id": batch_id, "run_id": batch.run_id, "decision": request.decision},
    )
    return updated


@app.post("/api/v1/jobs/parse", response_model=JobParseResponse)
def parse_job(request: JobParseRequest, user: UserView = Depends(current_user)) -> JobParseResponse:
    analyzer_mode = "ollama" if request.mode == "adaptive" else request.mode
    job, trace = workflow.jd_analyzer.analyze(request.text, analyzer_mode)
    auth.record_audit(user, "parse", "job_description", detail={"mode": request.mode, "requirements": len(job.requirements)})
    return JobParseResponse(job=job, trace=trace)


@app.post("/api/v1/analyze/demo", response_model=AnalysisResponse)
def analyze_demo(mode: str = "rules", user: UserView = Depends(current_user)) -> AnalysisResponse:
    if mode not in {"rules", "adaptive", "ollama"}:
        raise HTTPException(status_code=422, detail="mode必须为rules、adaptive或ollama")
    payload = json.loads(sample_path.read_text(encoding="utf-8"))
    payload["mode"] = mode
    request = AnalysisRequest.model_validate(payload)
    response = workflow.run(request)
    labels = {item.id: item.relevance_label for item in request.candidates if item.relevance_label is not None}
    metrics = evaluate_run(response, labels)
    store.save(response, metrics)
    auth.record_audit(user, "analyze", "run", response.run_id, {"candidate_count": len(response.ranking), "mode": mode, "demo": True})
    return response


@app.get("/api/v1/runs")
def list_runs(limit: int = 20, _: UserView = Depends(current_user)) -> list[dict]:
    return store.list_runs(min(max(limit, 1), 100))


@app.get("/api/v1/runs/{run_id}")
def get_run(run_id: str, user: UserView = Depends(current_user)) -> dict:
    result = store.get(run_id)
    if not result:
        raise HTTPException(status_code=404, detail="运行记录不存在")
    auth.record_audit(user, "read", "run", run_id)
    return result


@app.get("/api/v1/runs/{run_id}/metrics", response_model=EvaluationMetrics)
def get_metrics(run_id: str, _: UserView = Depends(current_user)) -> EvaluationMetrics:
    result = store.get(run_id)
    if not result:
        raise HTTPException(status_code=404, detail="运行记录不存在")
    if not result["metrics"]:
        raise HTTPException(status_code=404, detail="该运行没有标注标签，无法计算排序指标")
    return EvaluationMetrics.model_validate(result["metrics"])


@app.post("/api/v1/runs/{run_id}/decisions", response_model=HumanDecisionRecord)
def save_human_decision(
    run_id: str, request: HumanDecisionRequest, user: UserView = Depends(current_user),
) -> HumanDecisionRecord:
    attributed_request = request.model_copy(update={"reviewer": user.username})
    try:
        record = store.save_decision(run_id, attributed_request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="运行记录不存在") from exc
    auth.record_audit(user, "decide", "candidate", request.candidate_id, {"run_id": run_id, "decision": request.decision})
    return record


@app.get("/api/v1/runs/{run_id}/decisions")
def list_human_decisions(run_id: str, _: UserView = Depends(current_user)) -> list[dict]:
    if not store.get(run_id):
        raise HTTPException(status_code=404, detail="运行记录不存在")
    return store.list_decisions(run_id)
