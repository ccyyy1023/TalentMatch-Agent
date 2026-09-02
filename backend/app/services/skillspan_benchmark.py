from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from app.services.ollama_client import OllamaClient
from app.services.skill_catalog import SKILL_ALIASES


@dataclass(frozen=True, order=True)
class SkillSpan:
    start: int
    end: int
    label: str
    text: str


@dataclass(frozen=True)
class SkillSpanRecord:
    record_id: str
    tokens: list[str]
    source: str
    gold: list[SkillSpan]


class JobBertSpanPredictor:
    """Run the two official SkillSpan token-classification endpoints."""

    MODEL_IDS = {
        "skill": "jjzha/jobbert_skill_extraction",
        "knowledge": "jjzha/jobbert_knowledge_extraction",
    }
    MODEL_REVISIONS = {
        "skill": "a6211b675e287f3a07ffd4834b8182a2b519589b",
        "knowledge": "9dea9759b7e290413cb4d92e8cae338affab9be4",
    }

    def __init__(self, cache_dir: Path | None = None):
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install requirements-skillspan.txt before running JobBERT evaluation") from exc
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.endpoints = {}
        for label, model_id in self.MODEL_IDS.items():
            revision = self.MODEL_REVISIONS[label]
            tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, cache_dir=cache_dir)
            model = AutoModelForTokenClassification.from_pretrained(
                model_id, revision=revision, cache_dir=cache_dir, use_safetensors=True,
            ).to(self.device)
            model.eval()
            self.endpoints[label] = (tokenizer, model)

    def predict(self, tokens: list[str]) -> list[SkillSpan]:
        return self.predict_batch([tokens], batch_size=1)[0]

    def predict_batch(self, token_batches: list[list[str]], batch_size: int = 32) -> list[list[SkillSpan]]:
        results: list[list[SkillSpan]] = [[] for _ in token_batches]
        for label, (tokenizer, model) in self.endpoints.items():
            for offset in range(0, len(token_batches), batch_size):
                chunk = token_batches[offset:offset + batch_size]
                encoded = tokenizer(
                    chunk, is_split_into_words=True, return_tensors="pt", padding=True,
                    truncation=True, max_length=512,
                )
                inputs = {key: value.to(self.device) for key, value in encoded.items()}
                with self.torch.inference_mode():
                    predicted = model(**inputs).logits.argmax(dim=-1).tolist()
                for batch_index, (tokens, predicted_ids) in enumerate(zip(chunk, predicted)):
                    word_ids = encoded.word_ids(batch_index=batch_index)
                    tags = ["O"] * len(tokens)
                    seen_words: set[int] = set()
                    for token_index, word_index in enumerate(word_ids):
                        if word_index is None or word_index in seen_words or word_index >= len(tokens):
                            continue
                        seen_words.add(word_index)
                        tags[word_index] = str(model.config.id2label[predicted_ids[token_index]])
                    results[offset + batch_index].extend(bio_to_spans(tokens, tags, label))
        return [sorted(set(spans)) for spans in results]


def bio_to_spans(tokens: list[str], tags: list[str], label: str) -> list[SkillSpan]:
    if len(tokens) != len(tags):
        raise ValueError("tokens and BIO tags must have identical lengths")
    spans: list[SkillSpan] = []
    start: int | None = None
    for index, tag in enumerate([*tags, "O"]):
        if tag == "B" or (tag == "I" and start is None):
            if start is not None:
                spans.append(SkillSpan(start, index, label, " ".join(tokens[start:index])))
            start = index
        elif tag == "O" and start is not None:
            spans.append(SkillSpan(start, index, label, " ".join(tokens[start:index])))
            start = None
        elif tag not in {"I", "O"}:
            raise ValueError(f"unsupported BIO tag: {tag}")
    return spans


def load_skillspan(path: Path) -> list[SkillSpanRecord]:
    records: list[SkillSpanRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        tokens = [str(value) for value in raw["tokens"]]
        gold = [
            *bio_to_spans(tokens, raw["tags_skill"], "skill"),
            *bio_to_spans(tokens, raw["tags_knowledge"], "knowledge"),
        ]
        records.append(SkillSpanRecord(
            record_id=f"{raw.get('source', 'unknown')}:{raw.get('idx', 0)}:{line_number}",
            tokens=tokens, source=str(raw.get("source") or "unknown"), gold=sorted(gold),
        ))
    return records


def stratified_sample(records: list[SkillSpanRecord], size: int, seed: int = 20260901) -> list[SkillSpanRecord]:
    if size <= 0 or size >= len(records):
        return list(records)
    buckets: dict[tuple[str, str], list[SkillSpanRecord]] = {}
    for record in records:
        labels = {span.label for span in record.gold}
        kind = "both" if len(labels) == 2 else next(iter(labels), "negative")
        buckets.setdefault((record.source, kind), []).append(record)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    ordered: list[SkillSpanRecord] = []
    keys = sorted(buckets)
    while len(ordered) < size and any(buckets.values()):
        for key in keys:
            if buckets[key] and len(ordered) < size:
                ordered.append(buckets[key].pop())
    return sorted(ordered, key=lambda item: item.record_id)


def catalog_predict(tokens: list[str]) -> list[SkillSpan]:
    text = " ".join(tokens)
    lowered = text.lower()
    token_ranges: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        token_ranges.append((cursor, cursor + len(token)))
        cursor += len(token) + 1
    spans: set[SkillSpan] = set()
    for aliases in SKILL_ALIASES.values():
        for alias in sorted(aliases, key=len, reverse=True):
            if not alias.isascii():
                continue
            pattern = rf"(?<![a-z0-9_]){re.escape(alias.lower())}(?![a-z0-9_])"
            for match in re.finditer(pattern, lowered):
                covered = [index for index, (start, end) in enumerate(token_ranges) if start < match.end() and end > match.start()]
                if not covered:
                    continue
                start, end = covered[0], covered[-1] + 1
                if token_ranges[start][0] == match.start() and token_ranges[end - 1][1] == match.end():
                    spans.add(SkillSpan(start, end, "knowledge", " ".join(tokens[start:end])))
    return sorted(spans)


def ollama_predict(tokens: list[str], client: OllamaClient) -> tuple[list[SkillSpan], int]:
    indexed = " | ".join(f"{index}={token}" for index, token in enumerate(tokens))
    payload = client.generate_json(
        "You are a strict NER annotator for English job postings. Output exactly one JSON object with only a "
        "spans array. Every span object contains integer start, integer end, and string label. Do not repeat the "
        "input tokens and do not add any other key. start is inclusive and end is exclusive. Use label=skill for an ability, action, or "
        "soft competency; use label=knowledge for a technology, subject, domain, method, or qualification. "
        "Never tag pronouns, determiners, punctuation, company information, or generic words by themselves. "
        "Example input: 0=The | 1=role | 2=requires | 3=Python | 4=and | 5=clear | 6=communication. "
        "Example output: {\"spans\":[{\"start\":3,\"end\":4,\"label\":\"knowledge\"},"
        "{\"start\":5,\"end\":7,\"label\":\"skill\"}]}. Return {\"spans\":[]} only when no competency is explicitly present.",
        f"TOKENS: {indexed}",
        timeout=60, cache_namespace="skillspan_extractor", prompt_version="skillspan-v3",
    )
    accepted: set[SkillSpan] = set()
    rejected = 0
    for item in payload.get("spans", [])[:20]:
        try:
            start, end = int(item["start"]), int(item["end"])
            label = str(item["label"]).lower()
        except (KeyError, TypeError, ValueError):
            rejected += 1
            continue
        if label not in {"skill", "knowledge"} or not 0 <= start < end <= len(tokens):
            rejected += 1
            continue
        accepted.add(SkillSpan(start, end, label, " ".join(tokens[start:end])))
    return sorted(accepted), rejected


def _prf(tp: int, predicted: int, gold: int) -> dict[str, float | int]:
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "predicted": predicted, "gold": gold,
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
    }


def span_metrics(rows: list[tuple[list[SkillSpan], list[SkillSpan]]]) -> dict:
    exact_typed_tp = exact_untyped_tp = overlap_tp = 0
    predicted_total = gold_total = 0
    per_label = {label: {"tp": 0, "predicted": 0, "gold": 0} for label in ("skill", "knowledge")}
    for predicted, gold in rows:
        predicted_total += len(predicted)
        gold_total += len(gold)
        predicted_typed = {(item.start, item.end, item.label) for item in predicted}
        gold_typed = {(item.start, item.end, item.label) for item in gold}
        exact_typed_tp += len(predicted_typed & gold_typed)
        exact_untyped_tp += len(
            {(item.start, item.end) for item in predicted} & {(item.start, item.end) for item in gold}
        )
        for label in per_label:
            predicted_label = {item for item in predicted_typed if item[2] == label}
            gold_label = {item for item in gold_typed if item[2] == label}
            per_label[label]["tp"] += len(predicted_label & gold_label)
            per_label[label]["predicted"] += len(predicted_label)
            per_label[label]["gold"] += len(gold_label)
        unmatched = set(range(len(gold)))
        for item in predicted:
            candidates = [
                index for index in unmatched
                if item.start < gold[index].end and gold[index].start < item.end
            ]
            if candidates:
                best = max(candidates, key=lambda index: min(item.end, gold[index].end) - max(item.start, gold[index].start))
                unmatched.remove(best)
                overlap_tp += 1
    return {
        "typed_exact": _prf(exact_typed_tp, predicted_total, gold_total),
        "boundary_exact": _prf(exact_untyped_tp, predicted_total, gold_total),
        "boundary_overlap": _prf(overlap_tp, predicted_total, gold_total),
        "per_label_typed_exact": {
            label: _prf(values["tp"], values["predicted"], values["gold"])
            for label, values in per_label.items()
        },
    }


def evaluate_skillspan(
    records: list[SkillSpanRecord], mode: str, *, model: str = "qwen3:4b",
    model_cache_dir: Path | None = None,
) -> dict:
    started = perf_counter()
    client = OllamaClient(chat_model=model, cache_enabled=False) if mode == "ollama" else None
    jobbert = JobBertSpanPredictor(model_cache_dir) if mode == "jobbert" else None
    predictor_ready = perf_counter()
    jobbert_predictions = jobbert.predict_batch([record.tokens for record in records]) if jobbert else None
    predictions_ready = perf_counter()
    rows = []
    diagnostics = []
    failures = rejected = 0
    for record_index, record in enumerate(records):
        if mode == "catalog":
            predicted = catalog_predict(record.tokens)
        elif mode == "ollama":
            try:
                predicted, invalid = ollama_predict(record.tokens, client)
                rejected += invalid
            except Exception:
                predicted = []
                failures += 1
        elif mode == "jobbert":
            predicted = jobbert_predictions[record_index]
        else:
            raise ValueError("mode must be catalog, ollama or jobbert")
        rows.append((predicted, record.gold))
        if len(diagnostics) < 30 and (predicted or record.gold):
            diagnostics.append({
                "record_id": record.record_id,
                "tokens": record.tokens,
                "gold": [item.__dict__ for item in record.gold],
                "predicted": [item.__dict__ for item in predicted],
            })
    scoring_finished = perf_counter()
    return {
        "dataset": "SkillSpan", "mode": mode,
        "model": model if mode == "ollama" else list(JobBertSpanPredictor.MODEL_IDS.values()) if mode == "jobbert" else None,
        "scope": {
            "sentences": len(records),
            "sources": {source: sum(item.source == source for item in records) for source in sorted({item.source for item in records})},
            "positive_sentences": sum(bool(item.gold) for item in records),
            "gold_spans": sum(len(item.gold) for item in records),
        },
        "metrics": span_metrics(rows), "failures": failures, "rejected_invalid_spans": rejected,
        "timing_seconds": {
            "model_setup": round(predictor_ready - started, 3),
            "prediction": round(predictions_ready - predictor_ready, 3),
            "scoring": round(scoring_finished - predictions_ready, 3),
            "total": round(scoring_finished - started, 3),
        },
        "elapsed_seconds": round(scoring_finished - started, 3), "diagnostics": diagnostics,
    }
