from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Priority(str, Enum):
    hard = "hard"
    preferred = "preferred"
    context = "context"


class MatchStatus(str, Enum):
    matched = "matched"
    partial = "partial"
    missing = "missing"
    review = "review"


class Requirement(BaseModel):
    id: str
    text: str
    category: Literal["skill", "experience", "education", "responsibility", "other"] = "other"
    priority: Priority = Priority.context
    normalized_skill: str | None = None
    minimum_years: float | None = None
    source_quote: str


class ParsedJD(BaseModel):
    title: str = "未命名岗位"
    summary: str = ""
    requirements: list[Requirement] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)


class Evidence(BaseModel):
    id: str
    kind: Literal["skill", "experience", "education", "project", "achievement", "other"]
    value: str
    normalized_skill: str | None = None
    years: float | None = None
    source_quote: str
    section: str = "unknown"
    strength: float = Field(default=0.5, ge=0, le=1)


class ParsedCandidate(BaseModel):
    id: str
    display_name: str
    masked_name: str
    skills: list[str] = Field(default_factory=list)
    years_experience: float | None = None
    education: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    pii_detected: list[str] = Field(default_factory=list)
    security_flags: list[str] = Field(default_factory=list)
    parse_warnings: list[str] = Field(default_factory=list)


class CriterionMatch(BaseModel):
    requirement_id: str
    requirement_text: str
    priority: Priority
    status: MatchStatus
    score: float = Field(ge=0, le=1)
    evidence_ids: list[str] = Field(default_factory=list)
    explanation: str


class ReviewFinding(BaseModel):
    severity: Literal["info", "warning", "critical"]
    code: str
    message: str
    requirement_id: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class CandidateResult(BaseModel):
    candidate_id: str
    display_name: str
    score: float = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=1)
    recommendation: Literal["recommended", "manual_review", "insufficient_hard_requirement_evidence"]
    criteria: list[CriterionMatch]
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    findings: list[ReviewFinding] = Field(default_factory=list)
    matched_evidence: list[Evidence] = Field(default_factory=list)


class CandidateInput(BaseModel):
    id: str
    name: str = "候选人"
    text: str = Field(min_length=20)
    relevance_label: int | None = Field(default=None, ge=0, le=2)


class AnalysisRequest(BaseModel):
    job_description: str = Field(min_length=20)
    candidates: list[CandidateInput] = Field(min_length=1, max_length=50)
    mode: Literal["rules", "adaptive", "ollama"] = "rules"
    enable_semantic_matching: bool = True
    confirmed_job: ParsedJD | None = None
    criteria_confirmed_by_human: bool = False


class JobParseRequest(BaseModel):
    text: str = Field(min_length=20)
    mode: Literal["rules", "adaptive", "ollama"] = "rules"


class JobParseResponse(BaseModel):
    job: ParsedJD
    trace: TraceEvent


class TraceEvent(BaseModel):
    node: str
    status: Literal["completed", "fallback", "skipped"]
    detail: str
    elapsed_ms: float


class ComplianceAudit(BaseModel):
    pii_masking_enabled: bool = True
    protected_attributes_used_in_score: bool = False
    automatic_rejection_enabled: bool = False
    manual_review_count: int = 0
    warnings: list[str] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    run_id: str
    mode: str
    job: ParsedJD
    ranking: list[CandidateResult]
    compliance: ComplianceAudit
    traces: list[TraceEvent]
    elapsed_ms: float
    model_info: dict[str, Any] = Field(default_factory=dict)


class EvaluationMetrics(BaseModel):
    run_id: str
    candidate_count: int
    labeled_count: int
    ndcg_at_5: float | None = None
    precision_at_3: float | None = None
    recall_at_5: float | None = None
    mrr: float | None = None
    evidence_coverage: float
    unsupported_claim_rate: float
    manual_review_rate: float


class HumanDecisionRequest(BaseModel):
    candidate_id: str
    decision: Literal["advance", "hold", "not_advance"]
    note: str = Field(default="", max_length=500)
    reviewer: str = Field(default="demo_recruiter", max_length=80)


class HumanDecisionRecord(HumanDecisionRequest):
    run_id: str
    created_at: str


class LoginRequest(BaseModel):
    username: str = Field(min_length=3, max_length=80)
    password: str = Field(min_length=10, max_length=200)


class UserCreate(LoginRequest):
    role: Literal["admin", "recruiter"] = "recruiter"


class UserView(BaseModel):
    id: str
    username: str
    role: Literal["admin", "recruiter"]
    active: bool
    created_at: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_at: str
    user: UserView


class AuditRecord(BaseModel):
    id: int
    actor_user_id: str
    actor_username: str
    action: str
    resource_type: str
    resource_id: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class JobCreate(BaseModel):
    title: str = Field(min_length=2, max_length=300)
    description: str = Field(min_length=20)
    status: Literal["draft", "active", "closed"] = "draft"


class JobUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=2, max_length=300)
    description: str | None = Field(default=None, min_length=20)
    status: Literal["draft", "active", "closed"] | None = None


class JobRecord(JobCreate):
    id: str
    created_by: str
    created_at: str
    updated_at: str


class CandidateCreate(BaseModel):
    display_name: str = Field(min_length=1, max_length=160)
    resume_text: str = Field(min_length=20)
    external_ref: str | None = Field(default=None, max_length=128)
    status: Literal["new", "reviewing", "archived"] = "new"


class CandidateUpdate(BaseModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=160)
    resume_text: str | None = Field(default=None, min_length=20)
    external_ref: str | None = Field(default=None, max_length=128)
    status: Literal["new", "reviewing", "archived"] | None = None


class CandidateRecord(CandidateCreate):
    id: str
    created_by: str
    created_at: str
    updated_at: str


class ScreeningBatchCreate(BaseModel):
    job_id: str = Field(min_length=1, max_length=64)
    candidate_ids: list[str] = Field(min_length=1, max_length=50)
    mode: Literal["rules", "adaptive", "ollama"] = "rules"
    enable_semantic_matching: bool = True


class ScreeningDecisionRequest(BaseModel):
    decision: Literal["advance", "hold", "not_advance"]
    note: str = Field(default="", max_length=500)


class ScreeningBatchItemRecord(BaseModel):
    candidate_id: str
    display_name: str
    stage: Literal["pending", "analyzed", "advanced", "held", "not_advanced"]
    decision: Literal["advance", "hold", "not_advance"] | None = None
    note: str = ""
    reviewer: str | None = None
    decided_at: str | None = None


class ScreeningBatchRecord(BaseModel):
    id: str
    job_id: str
    job_title: str
    task_id: str
    run_id: str | None = None
    status: Literal["queued", "running", "awaiting_review", "reviewed", "failed"]
    mode: Literal["rules", "adaptive", "ollama"]
    candidate_count: int
    reviewed_count: int = 0
    error: str | None = None
    created_by: str
    created_at: str
    updated_at: str
    items: list[ScreeningBatchItemRecord] = Field(default_factory=list)


class AnalysisTaskCreated(BaseModel):
    task_id: str
    status: Literal["queued", "running", "completed", "failed"]


class AnalysisTaskStatus(AnalysisTaskCreated):
    progress: int = Field(ge=0, le=100)
    stage: str
    detail: str = ""
    created_at: str
    updated_at: str
    started_at: str | None = None
    result: AnalysisResponse | None = None
    error: str | None = None
