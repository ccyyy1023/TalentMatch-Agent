from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path = Path(__file__).resolve().parents[2]
    llm_provider: str = os.getenv("TALENTMATCH_LLM_PROVIDER", "ollama")
    ollama_base_url: str = os.getenv("TALENTMATCH_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    chat_model: str = os.getenv("TALENTMATCH_CHAT_MODEL", "qwen3:4b")
    jd_model: str = os.getenv("TALENTMATCH_JD_MODEL", os.getenv("TALENTMATCH_CHAT_MODEL", "qwen3:4b"))
    candidate_model: str = os.getenv(
        "TALENTMATCH_CANDIDATE_MODEL", os.getenv("TALENTMATCH_CHAT_MODEL", "qwen3:4b"),
    )
    reviewer_model: str = os.getenv(
        "TALENTMATCH_REVIEWER_MODEL", os.getenv("TALENTMATCH_CHAT_MODEL", "qwen2.5:7b"),
    )
    embed_model: str = os.getenv("TALENTMATCH_EMBED_MODEL", "embeddinggemma")
    skill_extractor: str = os.getenv("TALENTMATCH_SKILL_EXTRACTOR", "catalog")
    jobbert_cache_dir_raw: str = os.getenv(
        "TALENTMATCH_JOBBERT_CACHE_DIR", "data/external/skillspan_models",
    )
    ollama_workers_raw: str = os.getenv("TALENTMATCH_OLLAMA_WORKERS", "2")
    auth_session_hours_raw: str = os.getenv("TALENTMATCH_AUTH_SESSION_HOURS", "8")
    task_backend: str = os.getenv("TALENTMATCH_TASK_BACKEND", "memory")
    redis_url: str = os.getenv("TALENTMATCH_REDIS_URL", "redis://127.0.0.1:6379/0")
    task_queue: str = os.getenv("TALENTMATCH_TASK_QUEUE", "talentmatch")
    task_result_ttl_raw: str = os.getenv("TALENTMATCH_TASK_RESULT_TTL", "86400")
    database_path_raw: str = os.getenv("TALENTMATCH_DATABASE_PATH", "data/talentmatch.db")
    database_url_raw: str = os.getenv("TALENTMATCH_DATABASE_URL", "")
    allow_origins_raw: str = os.getenv("TALENTMATCH_ALLOW_ORIGINS", "http://localhost:5173")

    @property
    def database_path(self) -> Path:
        path = Path(self.database_path_raw)
        return path if path.is_absolute() else self.project_root / path

    @property
    def jobbert_cache_dir(self) -> Path:
        path = Path(self.jobbert_cache_dir_raw)
        return path if path.is_absolute() else self.project_root / path

    @property
    def database_url(self) -> str:
        if self.database_url_raw:
            return self.database_url_raw
        return f"sqlite:///{self.database_path.as_posix()}"

    @property
    def allow_origins(self) -> list[str]:
        return [item.strip() for item in self.allow_origins_raw.split(",") if item.strip()]

    @property
    def ollama_workers(self) -> int:
        try:
            return min(4, max(1, int(self.ollama_workers_raw)))
        except ValueError:
            return 2

    @property
    def auth_session_hours(self) -> int:
        try:
            return min(24, max(1, int(self.auth_session_hours_raw)))
        except ValueError:
            return 8

    @property
    def task_result_ttl(self) -> int:
        try:
            return min(604800, max(3600, int(self.task_result_ttl_raw)))
        except ValueError:
            return 86400


settings = Settings()
