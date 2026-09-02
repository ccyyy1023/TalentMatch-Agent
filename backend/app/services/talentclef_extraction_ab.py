from __future__ import annotations

import random
from dataclasses import dataclass
from time import perf_counter

from app.services.analyzers import CandidateAnalyzer, JDAnalyzer
from app.services.hybrid_skill_extractor import DocumentSkillExtractor
from app.services.matcher import MatchingEngine
from app.services.ollama_client import OllamaClient
from app.services.talentclef_benchmark import BM25Index, TalentClefDataset, evaluate_rankings


@dataclass(frozen=True)
class ExtractionABSample:
    query_ids: tuple[str, ...]
    pools: dict[str, tuple[str, ...]]
    labels: dict[str, dict[str, int]]
    candidate_ids: tuple[str, ...]
    seed: int


def build_stratified_sample(
    dataset: TalentClefDataset,
    query_limit: int = 3,
    positives_per_query: int = 4,
    negatives_per_query: int = 4,
    seed: int = 20260901,
    negative_strategy: str = "random",
) -> ExtractionABSample:
    if dataset.qrels is None:
        raise ValueError("Extraction A/B requires public qrels")
    if min(query_limit, positives_per_query, negatives_per_query) < 1:
        raise ValueError("Sample sizes must be positive")
    if negative_strategy not in {"random", "bm25_hard"}:
        raise ValueError("negative_strategy must be random or bm25_hard")
    rng = random.Random(seed)
    hard_negative_index = BM25Index(dataset.corpus) if negative_strategy == "bm25_hard" else None
    all_queries = sorted(dataset.queries)
    if query_limit > len(all_queries):
        raise ValueError("query_limit exceeds available queries")
    selected_queries = tuple(sorted(rng.sample(all_queries, query_limit)))
    pools: dict[str, tuple[str, ...]] = {}
    labels: dict[str, dict[str, int]] = {}
    all_candidates: set[str] = set()
    for query_id in selected_queries:
        positives = sorted(
            document_id
            for document_id, relevance in dataset.qrels[query_id].items()
            if relevance > 0
        )
        negatives = sorted(set(dataset.corpus) - set(positives))
        if positives_per_query > len(positives) or negatives_per_query > len(negatives):
            raise ValueError(f"Not enough labeled candidates for query {query_id}")
        selected_positive = rng.sample(positives, positives_per_query)
        if hard_negative_index is not None:
            negative_set = set(negatives)
            ranked_negatives = [
                document_id
                for document_id, _ in hard_negative_index.rank(dataset.queries[query_id])
                if document_id in negative_set
            ]
            selected_negative = ranked_negatives[:negatives_per_query]
        else:
            selected_negative = rng.sample(negatives, negatives_per_query)
        pool = selected_positive + selected_negative
        rng.shuffle(pool)
        pools[query_id] = tuple(pool)
        labels[query_id] = {
            document_id: int(document_id in selected_positive)
            for document_id in pool
        }
        all_candidates.update(pool)
    return ExtractionABSample(
        query_ids=selected_queries,
        pools=pools,
        labels=labels,
        candidate_ids=tuple(sorted(all_candidates)),
        seed=seed,
    )


def _rank_sample(
    query_texts: dict[str, str],
    candidate_texts: dict[str, str],
    sample: ExtractionABSample,
) -> dict[str, list[tuple[str, float]]]:
    index = BM25Index(candidate_texts)
    return {
        query_id: sorted(
            ((candidate_id, index.score(query_texts[query_id], candidate_id)) for candidate_id in sample.pools[query_id]),
            key=lambda item: (-item[1], item[0]),
        )
        for query_id in sample.query_ids
    }


def evaluate_raw_sample(dataset: TalentClefDataset, sample: ExtractionABSample) -> dict[str, float]:
    rankings = raw_sample_rankings(dataset, sample)
    metrics, _ = evaluate_rankings(rankings, sample.labels)
    return metrics


def raw_sample_rankings(
    dataset: TalentClefDataset, sample: ExtractionABSample,
) -> dict[str, list[tuple[str, float]]]:
    return _rank_sample(dataset.queries, dataset.corpus, sample)


def run_model_extraction(
    dataset: TalentClefDataset,
    sample: ExtractionABSample,
    model: str,
    base_url: str | None = None,
    *,
    skill_extractor: DocumentSkillExtractor | None = None,
    cache_enabled: bool = False,
    variant: str | None = None,
) -> dict:
    client = OllamaClient(base_url=base_url, chat_model=model, cache_enabled=cache_enabled)
    jd_analyzer = JDAnalyzer(client, skill_extractor)
    candidate_analyzer = CandidateAnalyzer(client, skill_extractor)
    started = perf_counter()

    parsed_jobs = {}
    job_details = {}
    for query_id in sample.query_ids:
        parsed, trace = jd_analyzer.analyze(dataset.queries[query_id], "ollama")
        parsed_jobs[query_id] = parsed
        job_details[query_id] = {
            "status": trace.status,
            "detail": trace.detail,
            "requirements": len(parsed.requirements),
            "normalized_skills": sum(bool(item.normalized_skill) for item in parsed.requirements),
            "valid_source_quotes": sum(item.source_quote in dataset.queries[query_id] for item in parsed.requirements),
            "elapsed_ms": round(trace.elapsed_ms, 3),
        }

    parsed_candidates = {}
    candidate_details = {}
    target_skills = sorted({
        item.normalized_skill
        for parsed_job in parsed_jobs.values()
        for item in parsed_job.requirements
        if item.normalized_skill
    })
    for candidate_id in sample.candidate_ids:
        parsed, origin = candidate_analyzer.analyze(
            candidate_id,
            f"Candidate {candidate_id}",
            dataset.corpus[candidate_id],
            "ollama",
            target_skills=target_skills,
        )
        parsed_candidates[candidate_id] = parsed
        candidate_details[candidate_id] = {
            "origin": origin,
            "evidence": len(parsed.evidence),
            "normalized_skills": len(parsed.skills),
            "valid_source_quotes": sum(
                item.source_quote in dataset.corpus[candidate_id]
                for item in parsed.evidence
            ),
            "warnings": len(parsed.parse_warnings),
        }

    extracted_queries = {
        query_id: "\n".join(
            [parsed_jobs[query_id].title]
            + [f"{item.text}\n{item.source_quote}" for item in parsed_jobs[query_id].requirements]
        )
        for query_id in sample.query_ids
    }
    extracted_candidates = {
        candidate_id: "\n".join(
            parsed_candidates[candidate_id].skills
            + [f"{item.value}\n{item.source_quote}" for item in parsed_candidates[candidate_id].evidence]
        )
        for candidate_id in sample.candidate_ids
    }
    rankings = _rank_sample(extracted_queries, extracted_candidates, sample)
    metrics, per_query = evaluate_rankings(rankings, sample.labels)
    engine = MatchingEngine()
    matcher_rankings = {
        query_id: sorted(
            (
                (
                    candidate_id,
                    engine.match(parsed_jobs[query_id], parsed_candidates[candidate_id]).score,
                )
                for candidate_id in sample.pools[query_id]
            ),
            key=lambda item: (-item[1], item[0]),
        )
        for query_id in sample.query_ids
    }
    matcher_metrics, matcher_per_query = evaluate_rankings(matcher_rankings, sample.labels)

    job_fallbacks = sum(row["status"] == "fallback" for row in job_details.values())
    candidate_fallbacks = sum(str(row["origin"]).startswith("fallback:") for row in candidate_details.values())
    job_requirements = sum(row["requirements"] for row in job_details.values())
    candidate_evidence = sum(row["evidence"] for row in candidate_details.values())
    return {
        "model": model,
        "variant": variant or model,
        "cache_enabled": cache_enabled,
        "elapsed_seconds": round(perf_counter() - started, 3),
        "model_calls_expected": len(sample.query_ids) + len(sample.candidate_ids),
        "fallbacks": {
            "jd": job_fallbacks,
            "candidate": candidate_fallbacks,
            "total": job_fallbacks + candidate_fallbacks,
        },
        "extraction": {
            "job_requirements": job_requirements,
            "job_normalized_skills": sum(row["normalized_skills"] for row in job_details.values()),
            "job_valid_quote_rate": round(
                sum(row["valid_source_quotes"] for row in job_details.values()) / job_requirements,
                6,
            ) if job_requirements else 0.0,
            "candidate_evidence": candidate_evidence,
            "candidate_normalized_skills": sum(row["normalized_skills"] for row in candidate_details.values()),
            "candidate_valid_quote_rate": round(
                sum(row["valid_source_quotes"] for row in candidate_details.values()) / candidate_evidence,
                6,
            ) if candidate_evidence else 0.0,
            "candidate_warnings": sum(row["warnings"] for row in candidate_details.values()),
        },
        "sample_ranking_metrics": metrics,
        "sample_rankings": {
            query_id: [[candidate_id, round(score, 8)] for candidate_id, score in rows]
            for query_id, rows in rankings.items()
        },
        "deterministic_matching_metrics": matcher_metrics,
        "deterministic_matching_rankings": {
            query_id: [[candidate_id, round(score, 8)] for candidate_id, score in rows]
            for query_id, rows in matcher_rankings.items()
        },
        "deterministic_matching_per_query": {
            query_id: {name: round(value, 6) for name, value in row.items()}
            for query_id, row in matcher_per_query.items()
        },
        "per_query_ranking_metrics": {
            query_id: {name: round(value, 6) for name, value in row.items()}
            for query_id, row in per_query.items()
        },
        "job_details": job_details,
        "candidate_details": candidate_details,
    }


def compare_model_reports(left: dict, right: dict) -> dict:
    metric_names = left["sample_ranking_metrics"]
    return {
        "left_model": left["model"],
        "right_model": right["model"],
        "delta_right_minus_left": {
            name: round(right["sample_ranking_metrics"][name] - left["sample_ranking_metrics"][name], 6)
            for name in metric_names
        },
        "fallback_delta_right_minus_left": right["fallbacks"]["total"] - left["fallbacks"]["total"],
        "model_only_interpretation_gate": (
            "eligible_for_quality_comparison"
            if left["fallbacks"]["total"] == 0 and right["fallbacks"]["total"] == 0
            else "invalid_for_model_quality_claim_due_to_fallbacks"
        ),
        "operational_pipeline_gate": (
            "eligible_with_fallback_rates_disclosed"
            if left["fallbacks"]["total"] < left["model_calls_expected"]
            and right["fallbacks"]["total"] < right["model_calls_expected"]
            else "invalid_because_a_model_never_completed_extraction"
        ),
    }
