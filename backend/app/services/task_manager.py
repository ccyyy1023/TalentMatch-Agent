from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import RLock
from uuid import uuid4

from redis import Redis
from rq import Queue, Retry, get_current_job

from app.config import settings
from app.schemas import AnalysisRequest, AnalysisResponse, AnalysisTaskCreated, AnalysisTaskStatus
from app.services.database import RunStore
from app.services.evaluation import evaluate_run
from app.services.workflow import TalentMatchWorkflow


class AnalysisTaskManager:
    """Small in-process task runner for long local-model analyses.

    Task state intentionally lives in memory. Completed analysis results are still
    persisted by RunStore, but queued/running tasks do not survive a backend restart.
    """

    def __init__(
        self,
        workflow: TalentMatchWorkflow,
        store: RunStore,
        max_workers: int = 2,
        max_tasks: int = 100,
    ) -> None:
        self.workflow = workflow
        self.store = store
        self.max_tasks = max_tasks
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="analysis-task")
        self._lock = RLock()
        self._tasks: dict[str, AnalysisTaskStatus] = {}

    def submit(self, request: AnalysisRequest) -> AnalysisTaskCreated:
        task_id = f"task-{uuid4().hex[:12]}"
        now = self._now()
        status = AnalysisTaskStatus(
            task_id=task_id,
            status="queued",
            progress=0,
            stage="queued",
            detail="任务已进入本地分析队列",
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._evict_finished_locked()
            self._tasks[task_id] = status
        self._executor.submit(self._run, task_id, request)
        return AnalysisTaskCreated(task_id=task_id, status="queued")

    def get(self, task_id: str) -> AnalysisTaskStatus | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return task.model_copy(deep=True) if task else None

    def health(self) -> dict:
        return {"backend": "memory", "available": True, "queued": 0}

    def _run(self, task_id: str, request: AnalysisRequest) -> None:
        self._update(
            task_id, status="running", progress=5, stage="starting", detail="正在初始化分析工作流",
            started_at=self._now(),
        )

        def on_progress(stage: str, progress: int, detail: str) -> None:
            self._update(task_id, status="running", progress=progress, stage=stage, detail=detail)

        try:
            response = self.workflow.run(request, progress_callback=on_progress)
            labels = {
                item.id: item.relevance_label
                for item in request.candidates
                if item.relevance_label is not None
            }
            metrics = evaluate_run(response, labels) if labels else None
            self.store.save(response, metrics)
            self._update(
                task_id,
                status="completed",
                progress=100,
                stage="completed",
                detail="分析完成，结果与评测记录已保存",
                result=response,
            )
        except Exception as exc:  # API exposes a bounded local diagnostic, not a traceback.
            message = f"{type(exc).__name__}: {str(exc)[:300]}"
            self._update(
                task_id,
                status="failed",
                progress=100,
                stage="failed",
                detail="分析任务执行失败",
                error=message,
            )

    def _update(self, task_id: str, **changes) -> None:
        with self._lock:
            current = self._tasks.get(task_id)
            if current is None:
                return
            payload = current.model_dump()
            payload.update(changes)
            payload["updated_at"] = self._now()
            self._tasks[task_id] = AnalysisTaskStatus.model_validate(payload)

    def _evict_finished_locked(self) -> None:
        overflow = len(self._tasks) - self.max_tasks + 1
        if overflow <= 0:
            return
        finished = sorted(
            (task for task in self._tasks.values() if task.status in {"completed", "failed"}),
            key=lambda task: task.updated_at,
        )
        for task in finished[:overflow]:
            self._tasks.pop(task.task_id, None)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()


class RedisAnalysisTaskManager:
    """Redis-backed task state plus an RQ queue consumed by a separate worker."""

    def __init__(self, redis_url: str | None = None, queue_name: str | None = None) -> None:
        self.redis_url = redis_url or settings.redis_url
        self.queue_name = queue_name or settings.task_queue
        self.redis = Redis.from_url(self.redis_url)
        self.queue = Queue(self.queue_name, connection=self.redis)

    def submit(self, request: AnalysisRequest) -> AnalysisTaskCreated:
        task_id = f"task-{uuid4().hex[:12]}"
        now = self._now()
        self._update(
            task_id,
            status="queued",
            progress=0,
            stage="queued",
            detail="任务已进入持久化分析队列",
            created_at=now,
            updated_at=now,
            result="",
            error="",
        )
        try:
            self.queue.enqueue(
                execute_redis_analysis_task,
                task_id,
                request.model_dump(mode="json"),
                job_id=task_id,
                job_timeout=1800,
                result_ttl=settings.task_result_ttl,
                failure_ttl=settings.task_result_ttl,
                retry=Retry(max=2, interval=[5, 15]),
            )
        except Exception:
            self.redis.delete(self._key(task_id))
            raise
        return AnalysisTaskCreated(task_id=task_id, status="queued")

    def get(self, task_id: str) -> AnalysisTaskStatus | None:
        raw = self.redis.hgetall(self._key(task_id))
        if not raw:
            return None
        payload = {self._decode(key): self._decode(value) for key, value in raw.items()}
        result = AnalysisResponse.model_validate_json(payload["result"]) if payload.get("result") else None
        return AnalysisTaskStatus(
            task_id=task_id,
            status=payload["status"],
            progress=int(payload["progress"]),
            stage=payload["stage"],
            detail=payload.get("detail", ""),
            created_at=payload["created_at"],
            updated_at=payload["updated_at"],
            started_at=payload.get("started_at") or None,
            result=result,
            error=payload.get("error") or None,
        )

    def health(self) -> dict:
        try:
            available = bool(self.redis.ping())
            queued = self.queue.count
        except Exception:
            available = False
            queued = None
        return {"backend": "redis", "available": available, "queue": self.queue_name, "queued": queued}

    def _update(self, task_id: str, **changes) -> None:
        mapping = {key: self._serialize(value) for key, value in changes.items()}
        self.redis.hset(self._key(task_id), mapping=mapping)
        self.redis.expire(self._key(task_id), settings.task_result_ttl)

    @staticmethod
    def _serialize(value) -> str:
        if isinstance(value, AnalysisResponse):
            return value.model_dump_json()
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return "" if value is None else str(value)

    @staticmethod
    def _decode(value: bytes | str) -> str:
        return value.decode("utf-8") if isinstance(value, bytes) else value

    @staticmethod
    def _key(task_id: str) -> str:
        return f"talentmatch:analysis:{task_id}"

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()


def execute_redis_analysis_task(task_id: str, request_payload: dict) -> str:
    """RQ entrypoint. It is module-level so a separate worker can import it."""
    redis_client = Redis.from_url(settings.redis_url)
    manager = RedisAnalysisTaskManager(settings.redis_url, settings.task_queue)
    manager._update(
        task_id,
        status="running",
        progress=5,
        stage="starting",
        detail="独立Worker正在初始化分析工作流",
        started_at=manager._now(),
        updated_at=manager._now(),
        error="",
    )

    def on_progress(stage: str, progress: int, detail: str) -> None:
        manager._update(
            task_id, status="running", progress=progress, stage=stage,
            detail=detail, updated_at=manager._now(),
        )

    try:
        request = AnalysisRequest.model_validate(request_payload)
        workflow = TalentMatchWorkflow()
        response = workflow.run(request, progress_callback=on_progress)
        labels = {
            item.id: item.relevance_label
            for item in request.candidates
            if item.relevance_label is not None
        }
        metrics = evaluate_run(response, labels) if labels else None
        RunStore().save(response, metrics)
        manager._update(
            task_id,
            status="completed",
            progress=100,
            stage="completed",
            detail="独立Worker分析完成，结果与评测记录已保存",
            result=response,
            error="",
            updated_at=manager._now(),
        )
        return response.run_id
    except Exception as exc:
        job = get_current_job(connection=redis_client)
        retries_left = getattr(job, "retries_left", 0) if job is not None else 0
        manager._update(
            task_id,
            status="queued" if retries_left else "failed",
            progress=0 if retries_left else 100,
            stage="retrying" if retries_left else "failed",
            detail="任务失败，等待Worker自动重试" if retries_left else "分析任务执行失败",
            error=f"{type(exc).__name__}: {str(exc)[:300]}",
            updated_at=manager._now(),
        )
        raise
