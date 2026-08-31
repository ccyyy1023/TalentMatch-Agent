from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlalchemy import func, insert, select, update
from sqlalchemy.engine import Engine

from app.schemas import (
    CandidateCreate, CandidateRecord, CandidateUpdate, JobCreate, JobRecord, JobUpdate,
    ScreeningBatchItemRecord, ScreeningBatchRecord,
)
from app.services.relational import candidates, get_engine, jobs, screening_batch_items, screening_batches


def _now() -> str:
    return datetime.now().astimezone().isoformat()


class RecruitmentCatalog:
    def __init__(self, engine: Engine | None = None) -> None:
        self.engine = engine or get_engine()

    def create_job(self, payload: JobCreate, actor_id: str) -> JobRecord:
        now = _now()
        values = {"id": f"job-{uuid4().hex[:12]}", **payload.model_dump(), "created_by": actor_id,
                  "created_at": now, "updated_at": now}
        with self.engine.begin() as connection:
            connection.execute(insert(jobs).values(**values))
        return JobRecord.model_validate(values)

    def list_jobs(self, status: str | None = None, limit: int = 100) -> list[JobRecord]:
        statement = select(jobs)
        if status:
            statement = statement.where(jobs.c.status == status)
        statement = statement.order_by(jobs.c.updated_at.desc()).limit(limit)
        with self.engine.connect() as connection:
            return [JobRecord.model_validate(dict(row)) for row in connection.execute(statement).mappings()]

    def get_job(self, job_id: str) -> JobRecord | None:
        with self.engine.connect() as connection:
            row = connection.execute(select(jobs).where(jobs.c.id == job_id)).mappings().first()
        return JobRecord.model_validate(dict(row)) if row else None

    def update_job(self, job_id: str, payload: JobUpdate) -> JobRecord | None:
        values = payload.model_dump(exclude_none=True)
        if not values:
            return self.get_job(job_id)
        values["updated_at"] = _now()
        with self.engine.begin() as connection:
            result = connection.execute(update(jobs).where(jobs.c.id == job_id).values(**values))
        return self.get_job(job_id) if result.rowcount else None

    def create_candidate(self, payload: CandidateCreate, actor_id: str) -> CandidateRecord:
        now = _now()
        values = {"id": f"candidate-{uuid4().hex[:12]}", **payload.model_dump(), "created_by": actor_id,
                  "created_at": now, "updated_at": now}
        with self.engine.begin() as connection:
            connection.execute(insert(candidates).values(**values))
        return CandidateRecord.model_validate(values)

    def list_candidates(self, status: str | None = None, limit: int = 100) -> list[CandidateRecord]:
        statement = select(candidates)
        if status:
            statement = statement.where(candidates.c.status == status)
        statement = statement.order_by(candidates.c.updated_at.desc()).limit(limit)
        with self.engine.connect() as connection:
            return [CandidateRecord.model_validate(dict(row)) for row in connection.execute(statement).mappings()]

    def get_candidate(self, candidate_id: str) -> CandidateRecord | None:
        with self.engine.connect() as connection:
            row = connection.execute(select(candidates).where(candidates.c.id == candidate_id)).mappings().first()
        return CandidateRecord.model_validate(dict(row)) if row else None

    def update_candidate(self, candidate_id: str, payload: CandidateUpdate) -> CandidateRecord | None:
        values = payload.model_dump(exclude_none=True)
        if not values:
            return self.get_candidate(candidate_id)
        values["updated_at"] = _now()
        with self.engine.begin() as connection:
            result = connection.execute(update(candidates).where(candidates.c.id == candidate_id).values(**values))
        return self.get_candidate(candidate_id) if result.rowcount else None

    def create_screening_batch(
        self,
        job: JobRecord,
        candidate_records: list[CandidateRecord],
        task_id: str,
        mode: str,
        actor_id: str,
    ) -> ScreeningBatchRecord:
        now = _now()
        batch_id = f"batch-{uuid4().hex[:12]}"
        batch_values = {
            "id": batch_id, "job_id": job.id, "task_id": task_id, "run_id": None,
            "status": "queued", "mode": mode, "error": None, "created_by": actor_id,
            "created_at": now, "updated_at": now,
        }
        item_values = [
            {"batch_id": batch_id, "candidate_id": candidate.id, "stage": "pending", "note": ""}
            for candidate in candidate_records
        ]
        with self.engine.begin() as connection:
            connection.execute(insert(screening_batches).values(**batch_values))
            connection.execute(insert(screening_batch_items), item_values)
            connection.execute(
                update(candidates)
                .where(candidates.c.id.in_([candidate.id for candidate in candidate_records]))
                .values(status="reviewing", updated_at=now)
            )
        record = self.get_screening_batch(batch_id)
        assert record is not None
        return record

    def list_screening_batches(self, limit: int = 50) -> list[ScreeningBatchRecord]:
        statement = select(screening_batches.c.id).order_by(screening_batches.c.updated_at.desc()).limit(limit)
        with self.engine.connect() as connection:
            ids = [row[0] for row in connection.execute(statement).all()]
        return [record for batch_id in ids if (record := self.get_screening_batch(batch_id)) is not None]

    def get_screening_batch(self, batch_id: str) -> ScreeningBatchRecord | None:
        statement = (
            select(screening_batches, jobs.c.title.label("job_title"))
            .join(jobs, jobs.c.id == screening_batches.c.job_id)
            .where(screening_batches.c.id == batch_id)
        )
        with self.engine.connect() as connection:
            batch = connection.execute(statement).mappings().first()
            if not batch:
                return None
            item_rows = connection.execute(
                select(
                    screening_batch_items.c.candidate_id, candidates.c.display_name,
                    screening_batch_items.c.stage, screening_batch_items.c.decision,
                    screening_batch_items.c.note, screening_batch_items.c.reviewer,
                    screening_batch_items.c.decided_at,
                )
                .join(candidates, candidates.c.id == screening_batch_items.c.candidate_id)
                .where(screening_batch_items.c.batch_id == batch_id)
                .order_by(screening_batch_items.c.id)
            ).mappings().all()
        items = [ScreeningBatchItemRecord.model_validate(dict(row)) for row in item_rows]
        return ScreeningBatchRecord(
            id=batch["id"], job_id=batch["job_id"], job_title=batch["job_title"],
            task_id=batch["task_id"], run_id=batch["run_id"], status=batch["status"],
            mode=batch["mode"], candidate_count=len(items),
            reviewed_count=sum(item.decision is not None for item in items), error=batch["error"],
            created_by=batch["created_by"], created_at=batch["created_at"],
            updated_at=batch["updated_at"], items=items,
        )

    def update_screening_batch_from_task(
        self, batch_id: str, status: str, run_id: str | None = None, error: str | None = None,
    ) -> ScreeningBatchRecord | None:
        now = _now()
        values = {"status": status, "updated_at": now, "error": error}
        if run_id is not None:
            values["run_id"] = run_id
        with self.engine.begin() as connection:
            result = connection.execute(
                update(screening_batches).where(screening_batches.c.id == batch_id).values(**values)
            )
            if status == "awaiting_review":
                connection.execute(
                    update(screening_batch_items)
                    .where(screening_batch_items.c.batch_id == batch_id, screening_batch_items.c.stage == "pending")
                    .values(stage="analyzed")
                )
        return self.get_screening_batch(batch_id) if result.rowcount else None

    def save_screening_decision(
        self, batch_id: str, candidate_id: str, decision: str, note: str, reviewer: str,
    ) -> ScreeningBatchRecord | None:
        stage_by_decision = {"advance": "advanced", "hold": "held", "not_advance": "not_advanced"}
        now = _now()
        with self.engine.begin() as connection:
            result = connection.execute(
                update(screening_batch_items)
                .where(
                    screening_batch_items.c.batch_id == batch_id,
                    screening_batch_items.c.candidate_id == candidate_id,
                )
                .values(stage=stage_by_decision[decision], decision=decision, note=note,
                        reviewer=reviewer, decided_at=now)
            )
            if not result.rowcount:
                return None
            total = connection.scalar(
                select(func.count()).select_from(screening_batch_items)
                .where(screening_batch_items.c.batch_id == batch_id)
            )
            reviewed = connection.scalar(
                select(func.count()).select_from(screening_batch_items)
                .where(screening_batch_items.c.batch_id == batch_id, screening_batch_items.c.decision.is_not(None))
            )
            connection.execute(
                update(screening_batches).where(screening_batches.c.id == batch_id)
                .values(status="reviewed" if total == reviewed else "awaiting_review", updated_at=now)
            )
        return self.get_screening_batch(batch_id)

    def candidate_screening_history(self, candidate_id: str) -> list[dict]:
        statement = (
            select(
                screening_batches.c.id.label("batch_id"), screening_batches.c.job_id,
                jobs.c.title.label("job_title"), screening_batches.c.run_id,
                screening_batch_items.c.stage, screening_batch_items.c.decision,
                screening_batch_items.c.note, screening_batch_items.c.reviewer,
                screening_batch_items.c.decided_at, screening_batches.c.created_at,
            )
            .join(screening_batch_items, screening_batch_items.c.batch_id == screening_batches.c.id)
            .join(jobs, jobs.c.id == screening_batches.c.job_id)
            .where(screening_batch_items.c.candidate_id == candidate_id)
            .order_by(screening_batches.c.created_at.desc())
        )
        with self.engine.connect() as connection:
            return [dict(row) for row in connection.execute(statement).mappings().all()]
