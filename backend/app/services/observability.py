from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from threading import Lock
from uuid import uuid4

from fastapi import Request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest


HTTP_REQUESTS = Counter(
    "talentmatch_http_requests_total", "HTTP requests handled by the API", ["method", "path", "status"],
)
HTTP_LATENCY = Histogram(
    "talentmatch_http_request_duration_seconds", "HTTP request duration", ["method", "path"],
)
ANALYSIS_TASKS = Counter(
    "talentmatch_analysis_tasks_total", "Analysis tasks submitted", ["source", "mode"],
)
TASK_POLLS = Counter(
    "talentmatch_analysis_task_polls_total", "Observed analysis task states", ["status"],
)
QUEUE_DEPTH = Gauge("talentmatch_analysis_queue_depth", "Current analysis queue depth")
TASK_OUTCOMES = Counter(
    "talentmatch_analysis_task_outcomes_total", "Terminal analysis task outcomes", ["status", "mode"],
)
QUEUE_WAIT = Histogram(
    "talentmatch_analysis_queue_wait_seconds", "Time between task submission and worker start", ["mode"],
)
ANALYSIS_DURATION = Histogram(
    "talentmatch_analysis_duration_seconds", "End-to-end analysis workflow duration", ["mode"],
)
AGENT_NODE_DURATION = Histogram(
    "talentmatch_agent_node_duration_seconds", "Per-node workflow duration", ["node", "status"],
)
_observed: set[tuple[str, str]] = set()
_observed_lock = Lock()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("request_id", "method", "path", "status_code", "elapsed_ms", "task_id", "batch_id"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_structured_logging() -> None:
    root = logging.getLogger()
    if any(getattr(handler, "_talentmatch_json", False) for handler in root.handlers):
        return
    handler = logging.StreamHandler()
    handler._talentmatch_json = True  # type: ignore[attr-defined]
    handler.setFormatter(JsonFormatter())
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


async def observe_http_request(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", "")[:64] or uuid4().hex
    started = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - started
        route = request.scope.get("route")
        path = getattr(route, "path", request.url.path)
        HTTP_REQUESTS.labels(request.method, path, str(status_code)).inc()
        HTTP_LATENCY.labels(request.method, path).observe(elapsed)
        logging.getLogger("talentmatch.http").info(
            "request_completed",
            extra={
                "request_id": request_id, "method": request.method, "path": path,
                "status_code": status_code, "elapsed_ms": round(elapsed * 1000, 2),
            },
        )
        if "response" in locals():
            response.headers["X-Request-ID"] = request_id


def metrics_payload(queue_depth: int | None) -> tuple[bytes, str]:
    if queue_depth is not None:
        QUEUE_DEPTH.set(queue_depth)
    return generate_latest(), CONTENT_TYPE_LATEST


def record_task_observation(task) -> None:
    mode = task.result.mode if task.result is not None else "unknown"
    if task.started_at and task.status in {"completed", "failed"}:
        key = (task.task_id, "started")
        with _observed_lock:
            first_started_observation = key not in _observed
            if first_started_observation:
                _observed.add(key)
        if first_started_observation:
            created = datetime.fromisoformat(task.created_at)
            started = datetime.fromisoformat(task.started_at)
            QUEUE_WAIT.labels(mode).observe(max(0.0, (started - created).total_seconds()))
    if task.status not in {"completed", "failed"}:
        return
    key = (task.task_id, task.status)
    with _observed_lock:
        if key in _observed:
            return
        _observed.add(key)
    TASK_OUTCOMES.labels(task.status, mode).inc()
    if task.result is not None:
        ANALYSIS_DURATION.labels(mode).observe(task.result.elapsed_ms / 1000)
        for trace in task.result.traces:
            AGENT_NODE_DURATION.labels(trace.node, trace.status).observe(trace.elapsed_ms / 1000)
