from __future__ import annotations

import random
from dataclasses import dataclass

from app.services.hybrid_skill_extractor import JobBertDocumentSkillExtractor
from app.services.skill_catalog import display_name, extract_skills
from app.services.talentclef_benchmark import BM25Index, TalentClefDataset, evaluate_rankings


ALPHA_GRID = tuple(round(value / 10, 1) for value in range(3, 11))


@dataclass(frozen=True)
class QuerySplit:
    tuning: tuple[str, ...]
    holdout: tuple[str, ...]
    seed: int


def split_development_queries(
    query_ids: list[str] | tuple[str, ...], seed: int = 20260902, tuning_fraction: float = 0.7,
) -> QuerySplit:
    unique = sorted(set(query_ids))
    if len(unique) < 3:
        raise ValueError("at least three queries are required for tuning and holdout evaluation")
    rng = random.Random(seed)
    rng.shuffle(unique)
    tuning_size = min(len(unique) - 1, max(2, round(len(unique) * tuning_fraction)))
    return QuerySplit(
        tuning=tuple(sorted(unique[:tuning_size])),
        holdout=tuple(sorted(unique[tuning_size:])),
        seed=seed,
    )


def _catalog_text(text: str) -> str:
    values = [display_name(skill) for skill, _ in extract_skills(text)]
    return " ".join(dict.fromkeys(values)) or "__no_skill__"


def build_skill_views(
    dataset: TalentClefDataset,
    extractor: JobBertDocumentSkillExtractor,
) -> tuple[dict[str, str], dict[str, str], dict]:
    query_ids = sorted(dataset.queries)
    candidate_ids = sorted(dataset.corpus)
    all_ids = [("query", item) for item in query_ids] + [("candidate", item) for item in candidate_ids]
    texts = [dataset.queries[item] for _, item in all_ids[:len(query_ids)]] + [
        dataset.corpus[item] for _, item in all_ids[len(query_ids):]
    ]
    predicted = extractor.extract_many(texts, languages=[dataset.language] * len(texts))
    views: dict[tuple[str, str], str] = {}
    mention_counts = {"query": 0, "candidate": 0}
    open_counts = {"query": 0, "candidate": 0}
    for (kind, item_id), text, mentions in zip(all_ids, texts, predicted):
        catalog = _catalog_text(text)
        values = [mention.text for mention in mentions]
        views[(kind, item_id)] = " ".join(dict.fromkeys([catalog, *values])) or "__no_skill__"
        mention_counts[kind] += len(mentions)
        open_counts[kind] += sum(item.normalized_skill.startswith("open_skill:") for item in mentions)
    return (
        {item: views[("query", item)] for item in query_ids},
        {item: views[("candidate", item)] for item in candidate_ids},
        {
            "jobbert_mentions": mention_counts,
            "open_vocabulary_mentions": open_counts,
            "documents": {"queries": len(query_ids), "candidates": len(candidate_ids)},
        },
    )


def _normalize(values: dict[str, float]) -> dict[str, float]:
    maximum = max(values.values(), default=0.0)
    return {key: value / maximum if maximum > 0 else 0.0 for key, value in values.items()}


def fused_rankings(
    dataset: TalentClefDataset,
    query_skill_texts: dict[str, str],
    candidate_skill_texts: dict[str, str],
    alpha: float,
) -> dict[str, list[tuple[str, float]]]:
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between zero and one")
    raw_index = BM25Index(dataset.corpus)
    skill_index = BM25Index(candidate_skill_texts)
    rankings = {}
    for query_id, query_text in dataset.queries.items():
        raw = _normalize({item: raw_index.score(query_text, item) for item in dataset.corpus})
        skill_query = query_skill_texts.get(query_id, "__no_skill__")
        skills = _normalize({item: skill_index.score(skill_query, item) for item in dataset.corpus})
        rankings[query_id] = sorted(
            (
                (item, alpha * raw[item] + (1 - alpha) * skills[item])
                for item in dataset.corpus
            ),
            key=lambda row: (-row[1], row[0]),
        )
    return rankings


def _subset_metrics(
    rankings: dict[str, list[tuple[str, float]]],
    qrels: dict[str, dict[str, int]],
    query_ids: tuple[str, ...],
) -> dict[str, float]:
    selected_rankings = {item: rankings[item] for item in query_ids}
    selected_qrels = {item: qrels[item] for item in query_ids}
    return evaluate_rankings(selected_rankings, selected_qrels)[0]


def tune_fusion_alpha(
    dataset: TalentClefDataset,
    query_skill_texts: dict[str, str],
    candidate_skill_texts: dict[str, str],
    tuning_queries: tuple[str, ...],
    grid: tuple[float, ...] = ALPHA_GRID,
) -> tuple[float, list[dict]]:
    if dataset.qrels is None:
        raise ValueError("fusion tuning requires public development qrels")
    trials = []
    for alpha in grid:
        rankings = fused_rankings(dataset, query_skill_texts, candidate_skill_texts, alpha)
        metrics = _subset_metrics(rankings, dataset.qrels, tuning_queries)
        trials.append({"alpha": alpha, "metrics": metrics})
    best = max(trials, key=lambda row: (row["metrics"]["ndcg"], row["metrics"]["map"], row["alpha"]))
    return float(best["alpha"]), trials


def run_hybrid_benchmark(
    dataset: TalentClefDataset,
    extractor: JobBertDocumentSkillExtractor,
    *,
    seed: int = 20260902,
) -> dict:
    if dataset.split != "development" or dataset.qrels is None:
        raise ValueError("hybrid effectiveness evaluation requires the labeled development split")
    if dataset.language != "en":
        raise ValueError("fixed JobBERT endpoints are English-only")
    split = split_development_queries(list(dataset.queries), seed=seed)
    jobbert_queries, jobbert_candidates, diagnostics = build_skill_views(dataset, extractor)
    catalog_queries = {key: _catalog_text(value) for key, value in dataset.queries.items()}
    catalog_candidates = {key: _catalog_text(value) for key, value in dataset.corpus.items()}

    methods = {}
    for name, queries, candidates in (
        ("catalog_fusion", catalog_queries, catalog_candidates),
        ("jobbert_catalog_fusion", jobbert_queries, jobbert_candidates),
    ):
        alpha, trials = tune_fusion_alpha(dataset, queries, candidates, split.tuning)
        rankings = fused_rankings(dataset, queries, candidates, alpha)
        methods[name] = {
            "selected_alpha_raw_text": alpha,
            "tuning_trials": trials,
            "tuning_metrics": _subset_metrics(rankings, dataset.qrels, split.tuning),
            "holdout_metrics": _subset_metrics(rankings, dataset.qrels, split.holdout),
            "full_development_metrics_descriptive_only": evaluate_rankings(rankings, dataset.qrels)[0],
        }

    raw_rankings = fused_rankings(dataset, catalog_queries, catalog_candidates, 1.0)
    raw = {
        "selected_alpha_raw_text": 1.0,
        "tuning_metrics": _subset_metrics(raw_rankings, dataset.qrels, split.tuning),
        "holdout_metrics": _subset_metrics(raw_rankings, dataset.qrels, split.holdout),
        "full_development_metrics_descriptive_only": evaluate_rankings(raw_rankings, dataset.qrels)[0],
    }
    return {
        "benchmark": "TalentCLEF development holdout hybrid extraction and ranking",
        "dataset_version": "0.3.0",
        "language": dataset.language,
        "scope": {
            "queries": len(dataset.queries),
            "candidates": len(dataset.corpus),
            "ranked_pairs": len(dataset.queries) * len(dataset.corpus),
        },
        "split": {"seed": split.seed, "tuning_queries": list(split.tuning), "holdout_queries": list(split.holdout)},
        "raw_bm25": raw,
        "methods": methods,
        "extraction_diagnostics": diagnostics,
        "claim_gate": {
            "uses_official_test_labels": False,
            "holdout_not_used_for_alpha_selection": True,
            "resume_metric_eligible": False,
            "reason": "TalentCLEF development has only ten queries; use as architecture evidence, not a headline effect claim.",
        },
    }
