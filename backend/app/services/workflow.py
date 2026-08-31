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
    def __init__(self, ollama: OllamaClient | None = None):
        self.ollama = ollama or OllamaClient()
        self.jd_analyzer = JDAnalyzer(self.ollama)
        self.candidate_analyzer = CandidateAnalyzer(self.ollama)
        self.reviewer = ConflictReviewer(self.ollama)
        self.graph = self._build_graph()

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
        cache_hits_before = self.ollama.cache_hits
        cache_misses_before = self.ollama.cache_misses
        initial_state: WorkflowState = {"request": request, "traces": []}
        if progress_callback is not None:
            initial_state["progress_callback"] = progress_callback
        final = self.graph.invoke(initial_state)
        model_status = self.ollama.status()
        cache_status = self.ollama.cache_status()
        cache_status["run_hits"] = self.ollama.cache_hits - cache_hits_before
        cache_status["run_misses"] = self.ollama.cache_misses - cache_misses_before
        return AnalysisResponse(
            run_id=f"run-{uuid4().hex[:12]}", mode=request.mode, job=final["job"], ranking=final["results"],
            compliance=final["compliance"], traces=final["traces"],
            elapsed_ms=round((perf_counter() - started) * 1000, 2),
            model_info={
                "provider": "ollama" if request.mode in {"ollama", "adaptive"} else "deterministic",
                "chat_model": self.ollama.chat_model,
                "embed_model": self.ollama.embed_model,
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

        def analyze_item(item):
            return self.candidate_analyzer.analyze(item.id, item.name, item.text, request_mode)

        routed_to_llm = len(state["request"].candidates) if request_mode == "ollama" else 0
        if request_mode == "adaptive":
            base = [
                self.candidate_analyzer.analyze(item.id, item.name, item.text, "rules")
                for item in state["request"].candidates
            ]
            selected = [index for index, (candidate, _) in enumerate(base) if self._needs_llm_enrichment(candidate)]
            routed_to_llm = len(selected)

            def enrich(index):
                item = state["request"].candidates[index]
                return index, self.candidate_analyzer.analyze(item.id, item.name, item.text, "ollama")

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
            detail="使用透明权重、硬性条件和证据强度完成评分；敏感属性未进入评分",
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
