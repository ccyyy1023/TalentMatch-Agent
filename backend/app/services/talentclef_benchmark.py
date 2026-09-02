from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_LANGUAGES = {"en", "es"}
SPLIT_ALIASES = {"dev": "development", "development": "development", "test": "test"}
TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


@dataclass(frozen=True)
class TalentClefDataset:
    split: str
    language: str
    queries: dict[str, str]
    corpus: dict[str, str]
    qrels: dict[str, dict[str, int]] | None


def _read_text_directory(path: Path) -> dict[str, str]:
    if not path.is_dir():
        raise FileNotFoundError(f"TalentCLEF directory not found: {path}")
    documents = {
        item.name: item.read_text(encoding="utf-8-sig").strip()
        for item in sorted(path.iterdir(), key=lambda value: value.name)
        if item.is_file()
    }
    if not documents:
        raise ValueError(f"TalentCLEF directory is empty: {path}")
    if any(not text for text in documents.values()):
        raise ValueError(f"TalentCLEF contains an empty document in: {path}")
    return documents


def _read_qrels(path: Path) -> dict[str, dict[str, int]]:
    judgments: dict[str, dict[str, int]] = defaultdict(dict)
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not raw_line.strip():
            continue
        fields = raw_line.split()
        if len(fields) != 4:
            raise ValueError(f"Invalid qrels row at {path}:{line_number}")
        query_id, iteration, document_id, relevance_text = fields
        if iteration != "0":
            raise ValueError(f"Unexpected qrels iteration at {path}:{line_number}: {iteration}")
        try:
            relevance = int(relevance_text)
        except ValueError as error:
            raise ValueError(f"Invalid qrels relevance at {path}:{line_number}") from error
        if relevance not in {0, 1}:
            raise ValueError(f"Task A relevance must be binary at {path}:{line_number}")
        if document_id in judgments[query_id]:
            raise ValueError(f"Duplicate qrels pair at {path}:{line_number}")
        judgments[query_id][document_id] = relevance
    if not judgments:
        raise ValueError(f"TalentCLEF qrels is empty: {path}")
    return dict(judgments)


def load_talentclef_task_a(root: Path, split: str = "development", language: str = "en") -> TalentClefDataset:
    normalized_split = SPLIT_ALIASES.get(split.lower())
    if normalized_split is None:
        raise ValueError(f"Unsupported TalentCLEF split: {split}")
    normalized_language = language.lower()
    if normalized_language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported TalentCLEF language: {language}")

    split_dir = root / normalized_split / normalized_language
    queries = _read_text_directory(split_dir / "queries")
    corpus = _read_text_directory(split_dir / "corpus")
    qrels_path = split_dir / "qrels.tsv"
    qrels = _read_qrels(qrels_path) if qrels_path.is_file() else None

    if normalized_split == "development" and qrels is None:
        raise FileNotFoundError(f"Development qrels not found: {qrels_path}")
    if qrels is not None:
        unknown_queries = set(qrels) - set(queries)
        unknown_documents = {
            document_id
            for rows in qrels.values()
            for document_id in rows
            if document_id not in corpus
        }
        missing_judgments = set(queries) - set(qrels)
        if unknown_queries or unknown_documents or missing_judgments:
            raise ValueError(
                "TalentCLEF IDs are inconsistent: "
                f"unknown_queries={sorted(unknown_queries)}, "
                f"unknown_documents={sorted(unknown_documents)}, "
                f"queries_without_qrels={sorted(missing_judgments)}"
            )
    return TalentClefDataset(
        split=normalized_split,
        language=normalized_language,
        queries=queries,
        corpus=corpus,
        qrels=qrels,
    )


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_PATTERN.findall(text)]


class BM25Index:
    """Small dependency-free BM25 baseline for auditable full-text ranking."""

    def __init__(self, documents: dict[str, str], k1: float = 1.5, b: float = 0.75):
        if not documents:
            raise ValueError("BM25 requires at least one document")
        self.k1 = k1
        self.b = b
        self.term_frequencies = {document_id: Counter(tokenize(text)) for document_id, text in documents.items()}
        self.document_lengths = {
            document_id: sum(frequencies.values())
            for document_id, frequencies in self.term_frequencies.items()
        }
        self.average_length = sum(self.document_lengths.values()) / len(self.document_lengths)
        document_frequency: Counter[str] = Counter()
        for frequencies in self.term_frequencies.values():
            document_frequency.update(frequencies.keys())
        document_count = len(documents)
        self.idf = {
            term: math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def score(self, query: str, document_id: str) -> float:
        frequencies = self.term_frequencies[document_id]
        document_length = self.document_lengths[document_id]
        normalization = self.k1 * (1 - self.b + self.b * document_length / self.average_length)
        score = 0.0
        for term, query_frequency in Counter(tokenize(query)).items():
            term_frequency = frequencies.get(term, 0)
            if not term_frequency:
                continue
            score += self.idf.get(term, 0.0) * (
                term_frequency * (self.k1 + 1) / (term_frequency + normalization)
            ) * (1 + math.log(query_frequency))
        return score

    def rank(self, query: str) -> list[tuple[str, float]]:
        return sorted(
            ((document_id, self.score(query, document_id)) for document_id in self.term_frequencies),
            key=lambda item: (-item[1], item[0]),
        )


def _average_precision(order: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for rank, document_id in enumerate(order, start=1):
        if document_id in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant)


def _reciprocal_rank(order: list[str], relevant: set[str]) -> float:
    return next((1 / rank for rank, document_id in enumerate(order, start=1) if document_id in relevant), 0.0)


def _ndcg(order: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    dcg = sum(1 / math.log2(rank + 1) for rank, document_id in enumerate(order, start=1) if document_id in relevant)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, len(relevant) + 1))
    return dcg / ideal


def _precision_at(order: list[str], relevant: set[str], cutoff: int) -> float:
    return sum(document_id in relevant for document_id in order[:cutoff]) / cutoff


def evaluate_rankings(
    rankings: dict[str, list[tuple[str, float]]],
    qrels: dict[str, dict[str, int]],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    if set(rankings) != set(qrels):
        raise ValueError("Ranking query IDs must exactly match qrels query IDs")
    per_query: dict[str, dict[str, float]] = {}
    for query_id, ranked_rows in rankings.items():
        order = [document_id for document_id, _ in ranked_rows]
        if len(order) != len(set(order)):
            raise ValueError(f"Ranking contains duplicate document IDs for query {query_id}")
        relevant = {document_id for document_id, relevance in qrels[query_id].items() if relevance > 0}
        per_query[query_id] = {
            "map": _average_precision(order, relevant),
            "mrr": _reciprocal_rank(order, relevant),
            "ndcg": _ndcg(order, relevant),
            "precision_at_5": _precision_at(order, relevant, 5),
            "precision_at_10": _precision_at(order, relevant, 10),
            "precision_at_100": _precision_at(order, relevant, 100),
        }
    metrics = {
        name: round(sum(row[name] for row in per_query.values()) / len(per_query), 6)
        for name in next(iter(per_query.values()))
    }
    return metrics, per_query


def run_bm25_benchmark(dataset: TalentClefDataset) -> tuple[dict, dict[str, list[tuple[str, float]]]]:
    if dataset.qrels is None:
        raise ValueError("Cannot evaluate a TalentCLEF split without public qrels")
    index = BM25Index(dataset.corpus)
    rankings = {query_id: index.rank(text) for query_id, text in dataset.queries.items()}
    metrics, per_query = evaluate_rankings(rankings, dataset.qrels)
    positive_labels = sum(relevance > 0 for rows in dataset.qrels.values() for relevance in rows.values())
    report = {
        "benchmark": "TalentCLEF 2026 Task A full-text ranking",
        "dataset_version": "0.3.0",
        "split": dataset.split,
        "language": dataset.language,
        "label_type": "human-expert binary contextual job-person relevance",
        "scope": {
            "queries": len(dataset.queries),
            "candidates": len(dataset.corpus),
            "ranked_pairs": len(dataset.queries) * len(dataset.corpus),
            "positive_qrels": positive_labels,
        },
        "method": {
            "name": "BM25 full-text lexical baseline",
            "parameters": {"k1": index.k1, "b": index.b},
            "llm_calls": 0,
            "trained_on_development_labels": False,
        },
        "official_task_a_metrics": metrics,
        "per_query": {
            query_id: {name: round(value, 6) for name, value in row.items()}
            for query_id, row in per_query.items()
        },
        "limitations": [
            "The development set contains only 10 queries per language, so aggregate metrics have high variance.",
            "The corpus text is synthetic and privacy-preserving, not raw production ATS data.",
            "This lexical baseline does not use TalentMatch Agents or an LLM and is only an ingestion/evaluation check.",
            "The public test split has no qrels and is not assigned local effectiveness metrics.",
        ],
    }
    return report, rankings


def write_trec_run(path: Path, rankings: dict[str, list[tuple[str, float]]], tag: str = "talentmatch_bm25") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # The official 2026 evaluator reads with header=None, so the run must not
    # include a column-name row even though older submission docs showed one.
    lines: list[str] = []
    for query_id, ranked_rows in rankings.items():
        lines.extend(
            f"{query_id} Q0 {document_id} {rank} {score:.12f} {tag}"
            for rank, (document_id, score) in enumerate(ranked_rows, start=1)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
