from __future__ import annotations

import os
import socket
import logging

from redis import Redis
from rq import Queue, SimpleWorker, Worker

from app.config import settings
from app.services.observability import configure_structured_logging


def main() -> None:
    configure_structured_logging()
    connection = Redis.from_url(settings.redis_url)
    queue = Queue(settings.task_queue, connection=connection)
    # RQ's SpawnWorker still calls os.wait4() while supervising the child in
    # rq 2.11, which is unavailable on Windows. SimpleWorker executes jobs in
    # the worker process and is the supported local-development fallback.
    # Container/Linux deployments retain Worker process isolation.
    worker_type = SimpleWorker if os.name == "nt" else Worker
    # Containerized workers all run as PID 1, so PID-only names collide when
    # the service is scaled. Docker assigns a unique hostname per replica.
    worker_name = f"talentmatch-{socket.gethostname()}-{os.getpid()}"
    logging.getLogger("talentmatch.worker").info(
        "worker_started", extra={"task_id": worker_name, "path": settings.task_queue},
    )
    worker = worker_type([queue], connection=connection, name=worker_name)
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    main()
