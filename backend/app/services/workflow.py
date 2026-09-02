from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Callable, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.schemas import (
    AnalysisRequest, AnalysisResponse, CandidateResult, ComplianceAudit, ParsedCandidate, ParsedJD, TraceEvent,
)
from app.services.analyzers import CandidateAnalyzer, JDAnalyzer
from app.services.hybrid_skill_extractor import DocumentSkillExtractor, JobBertDocumentSkillExtractor
from app.services.matcher import MatchingEngine
from app.services.ollama_client import OllamaClient
from app.services.reviewer import ConflictReviewer


class WorkflowState(TypedDict, total=False):
    request: AnalysisRequest
    job: ParsedJD
    candidates: list[ParsedCandidate]
    results: list[CandidateResult]
    compliance: ComplianceAudit
    traces: list[TraceEvent]
    progress_callback: Callable[[str, int, str], None]


class TalentMatchWorkflow:
    def __init__(
        self,
        ollama: OllamaClient | None = None,
        *,
        jd_ollama: OllamaClient | None = None,
        candidate_ollama: OllamaClient | None = None,
        reviewer_ollama: OllamaClient | None = None,
        skill_extractor: DocumentSkillExtractor | None = None,
    ):
        # Passing the legacy ``ollama`` argument intentionally shares one
        # client across every role. Without it, each Agent can select a model
        # independently while retaining the same deterministic workflow.
        self.jd_ollama = jd_ollama or ollama or OllamaClient(chat_model=settings.jd_model)
        self.candidate_ollama = candidate_ollama or ollama or OllamaClient(chat_model=settings.candidate_model)
        self.reviewer_ollama = reviewer_ollama or ollama or OllamaClient(chat_model=settings.reviewer_model)
        self.skill_extractor = skill_extractor
        if self.skill_extractor is None and settings.skill_extractor == "jobbert":
            self.skill_extractor = JobBertDocumentSkillExtractor(settings.jobbert_cache_dir)
        self.ollama = self.candidate_ollama  # backwards-compatible embedding/client handle
        self.jd_analyzer = JDAnalyzer(self.jd_ollama, self.skill_extractor)
        self.candidate_analyzer = CandidateAnalyzer(self.candidate_ollama)
        self.reviewer = ConflictReviewer(self.reviewer_ollama)
        self.graph = self._build_graph()

    @property
    def agent_models(self) -> dict[str, str]:
        return {
            "jd_analyzer": self.jd_ollama.chat_model,
            "candidate_analyzer": self.candidate_ollama.chat_model,
            "conflict_reviewer": self.reviewer_ollama.chat_model,
        }

    def _unique_clients(self) -> list[OllamaClient]:
        clients = (self.jd_ollama, self.candidate_ollama, self.reviewer_ollama)
        return list({id(client): client for client in clients}.values())

    def status(self) -> dict:
        # Agent roles currently share one Ollama endpoint. Query its model list
        # once so splitting role clients does not triple health-check latency.
        endpoint_status = self.candidate_ollama.status()
        models = endpoint_status.get("models", [])

        def role(model: str) -> dict:
            return {
                "available": endpoint_status.get("available", False),
                "model": model,
                "chat_model_ready": any(name.split(":")[0] == model.split(":")[0] for name in models),
            }

        role_status = {
            "jd_analyzer": role(self.jd_ollama.chat_model),
            "candidate_analyzer": role(self.candidate_ollama.chat_model),
            "conflict_reviewer": role(self.reviewer_ollama.chat_model),
        }
        return {
            "available": endpoint_status.get("available", False),
            "models": models,
            "chat_model_ready": all(item.get("chat_model_ready", False) for item in role_status.values()),
            "embed_model_ready": endpoint_status.get("embed_model_ready", False),
            "agent_models": self.agent_models,
            "roles": role_status,
        }

    def cache_status(self) -> dict:
        unique_stats = [client.cache_status() for client in self._unique_clients()]
        role_stats = {
            "jd_analyzer": self.jd_ollama.cache_status(),
            "candidate_analyzer": self.candidate_ollama.cache_status(),
            "conflict_reviewer": self.reviewer_ollama.cache_status(),
        }
        return {
            "enabled": all(item.get("enabled", False) for item in unique_stats),
            "session_hits": sum(item.get("session_hits", 0) for item in unique_stats),
            "session_misses": sum(item.get("session_misses", 0) for item in unique_stats),
            # Every role uses the same persistent cache database by default;
            # use max rather than summing the same global counters three times.
            "entries": max((item.get("entries", 0) for item in unique_stats), default=0),
            "hits": max((item.get("hits", 0) for item in unique_stats), default=0),
            "roles": role_stats,
        }

    def _build_graph(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("jd_analyzer_agent", self._jd_node)
        graph.add_node("candidate_evidence_agents", self._candidate_node)
        graph.add_node("deterministic_matching_engine", self._match_node)
        graph.add_node("conflict_reviewer_agent", self._review_node)
        graph.add_node("compliance_agent", self._compliance_node)
        graph.add_node("ranking_and_explanation", self._finalize_node)
        graph.add_edge(START, "jd_analyzer_agent")
        graph.add_edge("jd_analyzer_agent", "candidate_evidence_agents")
        graph.add_edge("candidate_evidence_agents", "deterministic_matching_engine")
        graph.add_conditional_edges(
            "deterministic_matching_engine", self._needs_review,
            {"review": "conflict_reviewer_agent", "skip": "compliance_agent"},
        )
        graph.add_edge("conflict_reviewer_agent", "compliance_agent")
        graph.add_edge("compliance_agent", "ranking_and_explanation")
        graph.add_edge("ranking_and_explanation", END)
        return graph.compile()

    def run(
        self,
        request: AnalysisRequest,
        progress_callback: Callable[[str, int, str], None] | None = None,
    ) -> AnalysisResponse:
        started = perf_counter()
        clients = self._unique_clients()
        cache_hits_before = sum(client.cache_hits for client in clients)
        cache_misses_before = sum(client.cache_misses for client in clients)
        initial_state: WorkflowState = {"request": request, "traces": []}
        if progress_callback is not None:
            initial_state["progress_callback"] = progress_callback
        final = self.graph.invoke(initial_state)
        model_status = self.status()
        cache_status = self.cache_status()
        cache_status["run_hits"] = sum(client.cache_hits for client in clients) - cache_hits_before
        cache_status["run_misses"] = sum(client.cache_misses for client in clients) - cache_misses_before
        return AnalysisResponse(
            run_id=f"run-{uuid4().hex[:12]}", mode=request.mode, job=final["job"], ranking=final["results"],
            compliance=final["compliance"], traces=final["traces"],
            elapsed_ms=round((perf_counter() - started) * 1000, 2),
            model_info={
                "provider": "ollama" if request.mode in {"ollama", "adaptive"} else "deterministic",
                "chat_model": self.candidate_ollama.chat_model,
                "agent_models": self.agent_models,
                "embed_model": self.candidate_ollama.embed_model,
                "skill_extractor": "jobbert" if self.skill_extractor is not None else "catalog_and_llm",
                "criteria_confirmed_by_human": request.criteria_confirmed_by_human,
                "cache": cache_status,
                **model_status,
            },
        )

    def _jd_node(self, state: WorkflowState) -> dict:
        if state["request"].confirmed_job is not None and state["request"].criteria_confirmed_by_human:
            trace = TraceEvent(
                node="jd_analyzer_agent", status="completed",
                detail=f"使用招聘人员已确认的{len(state['request'].confirmed_job.requirements)}项岗位条件，跳过重复抽取",
                elapsed_ms=0.0,
            )
            self._notify(state, "job_ready", 15, trace.detail)
            return {"job": state["request"].confirmed_job, "traces": [*state.get("traces", []), trace]}
        analyzer_mode = "ollama" if state["request"].mode == "adaptive" else state["request"].mode
        job, trace = self.jd_analyzer.analyze(state["request"].job_description, analyzer_mode)
        self._notify(state, "job_ready", 15, trace.detail)
        return {"job": job, "traces": [*state.get("traces", []), trace]}

    def _candidate_node(self, state: WorkflowState) -> dict:
        started = perf_counter()
        request_mode = state["request"].mode
        workers = settings.ollama_workers if request_mode in {"ollama", "adaptive"} else 1
        target_skills = [
            item.normalized_skill
            for item in state.get("job", ParsedJD()).requirements
            if item.normalized_skill
        ]

        def analyze_item(item):
            return self.candidate_analyzer.analyze(
                item.id, item.name, item.text, request_mode, target_skills=target_skills,
            )

        routed_to_llm = len(state["request"].candidates) if request_mode == "ollama" else 0
        if request_mode == "adaptive":
            base = [
                self.candidate_analyzer.analyze(
                    item.id, item.name, item.text, "rules", target_skills=target_skills,
                )
                for item in state["request"].candidates
            ]
            selected = [index for index, (candidate, _) in enumerate(base) if self._needs_llm_enrichment(candidate)]
            routed_to_llm = len(selected)

            def enrich(index):
                item = state["request"].candidates[index]
                return index, self.candidate_analyzer.analyze(
                    item.id, item.name, item.text, "ollama", target_skills=target_skills,
                )

            enriched = {}
            if selected:
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="candidate-agent") as executor:
                    enriched = dict(executor.map(enrich, selected))
            analyzed = [enriched.get(index, (candidate, "adaptive_rules")) for index, (candidate, _) in enumerate(base)]
        elif workers > 1 and len(state["request"].candidates) > 1:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="candidate-agent") as executor:
                analyzed = list(executor.map(analyze_item, state["request"].candidates))
        else:
            analyzed = [analyze_item(item) for item in state["request"].candidates]
        candidates = [item[0] for item in analyzed]
        modes: dict[str, int] = {}
        for _, used_mode in analyzed:
            modes[used_mode] = modes.get(used_mode, 0) + 1
        detail = "，".join(f"{key}:{value}" for key, value in modes.items())
        used_fallback = any(key.startswith("fallback") or key == "security_fallback" for key in modes)
        trace = TraceEvent(
            node="candidate_evidence_agents", status="fallback" if used_fallback else "completed",
            detail=f"解析{len(candidates)}份候选人材料（{detail}，LLM路由:{routed_to_llm}/{len(candidates)}，并行度:{workers}）",
            elapsed_ms=(perf_counter() - started) * 1000,
        )
        self._notify(state, "candidates_ready", 45, trace.detail)
        return {"candidates": candidates, "traces": [*state["traces"], trace]}

    def _match_node(self, state: WorkflowState) -> dict:
        started = perf_counter()
        semantic = None
        if state["request"].enable_semantic_matching and state["request"].mode in {"ollama", "adaptive"}:
            semantic = self.ollama.similarity
        engine = MatchingEngine(semantic_similarity=semantic)
        results = [engine.match(state["job"], candidate) for candidate in state["candidates"]]
        trace = TraceEvent(
            node="deterministic_matching_engine", status="completed",
            detail="按硬性、加分项和上下文分组权重结合证据强度完成评分；敏感属性未进入评分",
            elapsed_ms=(perf_counter() - started) * 1000,
        )
        self._notify(state, "matching_ready", 60, trace.detail)
        return {"results": results, "traces": [*state["traces"], trace]}

    @staticmethod
    def _needs_review(state: WorkflowState) -> str:
        return "review" if any(result.recommendation != "recommended" or result.findings for result in state["results"]) else "skip"

    def _review_node(self, state: WorkflowState) -> dict:
        started = perf_counter()
        candidate_map = {item.id: item for item in state["candidates"]}
        workers = settings.ollama_workers if state["request"].mode in {"ollama", "adaptive"} else 1
        llm_review_count = sum(
            self.reviewer.needs_llm_review(candidate_map[result.candidate_id], result, state["request"].mode)
            for result in state["results"]
        )

        def review_item(result):
            return self.reviewer.review(state["job"], candidate_map[result.candidate_id], result, state["request"].mode)

        if workers > 1 and len(state["results"]) > 1:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="reviewer-agent") as executor:
                reviewed = list(executor.map(review_item, state["results"]))
        else:
            reviewed = [review_item(result) for result in state["results"]]
        trace = TraceEvent(
            node="conflict_reviewer_agent", status="completed",
            detail=f"确定性复核{len(reviewed)}名候选人，其中{llm_review_count}名模糊案例进入LLM复核（并行度:{workers}）",
            elapsed_ms=(perf_counter() - started) * 1000,
        )
        self._notify(state, "review_ready", 80, trace.detail)
        return {"results": reviewed, "traces": [*state["traces"], trace]}

    @staticmethod
    def _needs_llm_enrichment(candidate: ParsedCandidate) -> bool:
        if candidate.security_flags:
            return False
        strong_skill_evidence = any(
            item.kind == "skill" and item.section in {"work", "project"} and item.strength >= 0.8
            for item in candidate.evidence
        )
        return bool(candidate.parse_warnings) or not strong_skill_evidence or candidate.years_experience is None or candidate.education is None

    def _compliance_node(self, state: WorkflowState) -> dict:
        started = perf_counter()
        warnings = []
        if any("protected_attribute" in candidate.pii_detected for candidate in state["candidates"]):
            warnings.append("部分简历包含敏感属性，界面只展示匿名候选人编号")
        if any(candidate.security_flags for candidate in state["candidates"]):
            warnings.append("部分简历包含疑似提示注入，相关材料已强制使用确定性解析")
        manual_count = sum(result.recommendation == "manual_review" for result in state["results"])
        audit = ComplianceAudit(manual_review_count=manual_count, warnings=warnings)
        trace = TraceEvent(
            node="compliance_agent", status="completed",
            detail="确认关闭自动拒绝，屏蔽姓名及敏感属性，保留人工复核出口",
            elapsed_ms=(perf_counter() - started) * 1000,
        )
        self._notify(state, "compliance_ready", 90, trace.detail)
        return {"compliance": audit, "traces": [*state["traces"], trace]}

    def _finalize_node(self, state: WorkflowState) -> dict:
        started = perf_counter()
        ranking = sorted(state["results"], key=lambda item: (item.score, item.confidence), reverse=True)
        trace = TraceEvent(
            node="ranking_and_explanation", status="completed",
            detail="按透明匹配分与置信度排序，并保留每项结论的证据ID",
            elapsed_ms=(perf_counter() - started) * 1000,
        )
        self._notify(state, "ranking_ready", 95, trace.detail)
        return {"results": ranking, "traces": [*state["traces"], trace]}

    @staticmethod
    def _notify(state: WorkflowState, stage: str, progress: int, detail: str) -> None:
        callback = state.get("progress_callback")
        if callback is not None:
            callback(stage, progress, detail)
