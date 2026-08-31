from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import insert, select, update

from app.schemas import AnalysisResponse, EvaluationMetrics, HumanDecisionRecord, HumanDecisionRequest
from app.services.relational import get_engine, human_decisions, runs


class RunStore:
    def __init__(self, path: Path | None = None):
        self.engine = get_engine(path)

    def save(self, response: AnalysisResponse, metrics: EvaluationMetrics | None = None) -> None:
        payload = {
            "run_id": response.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": response.mode,
            "job_title": response.job.title,
            "candidate_count": len(response.ranking),
            "response_json": response.model_dump_json(),
            "metrics_json": metrics.model_dump_json() if metrics else None,
        }
        with self.engine.begin() as connection:
            exists = connection.execute(select(runs.c.run_id).where(runs.c.run_id == response.run_id)).first()
            if exists:
                connection.execute(update(runs).where(runs.c.run_id == response.run_id).values(**payload))
            else:
                connection.execute(insert(runs).values(**payload))

    def list_runs(self, limit: int = 20) -> list[dict]:
        statement = (
            select(
                runs.c.run_id, runs.c.created_at, runs.c.mode, runs.c.job_title,
                runs.c.candidate_count, runs.c.metrics_json,
            )
            .order_by(runs.c.created_at.desc())
            .limit(limit)
        )
        with self.engine.connect() as connection:
            rows = connection.execute(statement).mappings().all()
        return [
            {
                "run_id": row["run_id"], "created_at": row["created_at"], "mode": row["mode"],
                "job_title": row["job_title"], "candidate_count": row["candidate_count"],
                "metrics": json.loads(row["metrics_json"]) if row["metrics_json"] else None,
            }
            for row in rows
        ]

    def get(self, run_id: str) -> dict | None:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(runs.c.response_json, runs.c.metrics_json).where(runs.c.run_id == run_id)
            ).mappings().first()
        if not row:
            return None
        return {
            "response": json.loads(row["response_json"]),
            "metrics": json.loads(row["metrics_json"]) if row["metrics_json"] else None,
        }

    def save_decision(self, run_id: str, request: HumanDecisionRequest) -> HumanDecisionRecord:
        created_at = datetime.now(timezone.utc).isoformat()
        values = {
            "run_id": run_id,
            "candidate_id": request.candidate_id,
            "decision": request.decision,
            "note": request.note,
            "reviewer": request.reviewer,
            "created_at": created_at,
        }
        with self.engine.begin() as connection:
            if not connection.execute(select(runs.c.run_id).where(runs.c.run_id == run_id)).first():
                raise KeyError(run_id)
            existing = connection.execute(
                select(human_decisions.c.id).where(
                    human_decisions.c.run_id == run_id,
                    human_decisions.c.candidate_id == request.candidate_id,
                )
            ).first()
            if existing:
                connection.execute(
                    update(human_decisions)
                    .where(human_decisions.c.id == existing[0])
                    .values(**values)
                )
            else:
                connection.execute(insert(human_decisions).values(**values))
        return HumanDecisionRecord(run_id=run_id, created_at=created_at, **request.model_dump())

    def list_decisions(self, run_id: str) -> list[dict]:
        statement = (
            select(
                human_decisions.c.run_id, human_decisions.c.candidate_id, human_decisions.c.decision,
                human_decisions.c.note, human_decisions.c.reviewer, human_decisions.c.created_at,
            )
            .where(human_decisions.c.run_id == run_id)
            .order_by(human_decisions.c.created_at.desc())
        )
        with self.engine.connect() as connection:
            return [dict(row) for row in connection.execute(statement).mappings().all()]
