from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.services.skill_catalog import normalize_skill
from app.services.skillspan_benchmark import JobBertSpanPredictor


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[.+#/-][A-Za-z0-9+#]+)*|[^\W_]+|[^\w\s]", re.UNICODE)
SPAN_EDGE_PATTERN = re.compile(r"^[\s,.;:!?()\[\]{}'\"/\\|+-]+|[\s,.;:!?()\[\]{}'\"/\\|+-]+$")
ENGLISH_HINTS = {
    "and", "with", "experience", "skills", "knowledge", "required", "requirements",
    "work", "role", "candidate", "develop", "engineering", "management",
}
SPANISH_HINTS = {
    "de", "la", "el", "y", "con", "experiencia", "habilidades", "conocimientos",
    "requisitos", "trabajo", "puesto", "candidato", "desarrollo", "gestión",
}
GENERIC_SPANS = {
    "ability", "abilities", "candidate", "experience", "knowledge", "requirement",
    "requirements", "role", "skill", "skills", "work", "working",
}


class SpanPredictor(Protocol):
    def predict_batch(self, token_batches: list[list[str]], batch_size: int = 32): ...


class DocumentSkillExtractor(Protocol):
    def extract(self, text: str, language: str | None = None) -> list["SkillMention"]: ...


@dataclass(frozen=True, order=True)
class SkillMention:
    start: int
    end: int
    label: str
    text: str
    normalized_skill: str
    source_quote: str


@dataclass(frozen=True)
class _Token:
    text: str
    start: int
    end: int


def detect_document_language(text: str) -> str:
    """Return a conservative language route used only for optional extraction."""
    cjk_count = len(re.findall(r"[\u3400-\u9fff]", text))
    latin_words = re.findall(r"[A-Za-zÀ-ÿ]+", text.casefold())
    if cjk_count >= max(4, len(latin_words) // 4):
        return "zh"
    words = set(latin_words)
    spanish_score = len(words & SPANISH_HINTS)
    english_score = len(words & ENGLISH_HINTS)
    if spanish_score >= 3 and spanish_score > english_score:
        return "es"
    return "en"


def normalize_open_skill(value: str) -> str | None:
    cleaned = SPAN_EDGE_PATTERN.sub("", value).strip()
    if not cleaned or len(cleaned) > 80 or cleaned.casefold() in GENERIC_SPANS:
        return None
    known = normalize_skill(cleaned)
    if known:
        return known
    tokens = re.findall(r"[a-z0-9]+(?:[.+#/-][a-z0-9+#]+)*", cleaned.casefold())
    if not tokens:
        return None
    canonical = "_".join(tokens)
    return f"open_skill:{canonical}" if canonical not in GENERIC_SPANS else None


def _tokenize(text: str) -> list[_Token]:
    return [_Token(match.group(), match.start(), match.end()) for match in TOKEN_PATTERN.finditer(text)]


def _line_quote(text: str, start: int, end: int) -> str:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end < 0:
        line_end = len(text)
    quote = text[line_start:line_end].strip()
    return quote[:500] if quote else text[start:end]


def verify_target_skills(text: str, normalized_skills: list[str]) -> list[SkillMention]:
    """Ground job-specific open skills in candidate text without another model call."""
    mentions: list[SkillMention] = []
    seen: set[str] = set()
    for normalized in normalized_skills:
        if not normalized.startswith("open_skill:") or normalized in seen:
            continue
        seen.add(normalized)
        phrase = normalized.removeprefix("open_skill:").replace("_", " ")
        if not phrase:
            continue
        parts = [re.escape(part) for part in phrase.split()]
        pattern = r"(?<![a-z0-9])" + r"[\s_-]+".join(parts) + r"(?![a-z0-9])"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match is None:
            continue
        mentions.append(SkillMention(
            start=match.start(),
            end=match.end(),
            label="knowledge",
            text=text[match.start():match.end()],
            normalized_skill=normalized,
            source_quote=_line_quote(text, match.start(), match.end()),
        ))
    return mentions


class JobBertDocumentSkillExtractor:
    """Optional English open-vocabulary extractor backed by fixed JobBERT endpoints.

    Models are loaded lazily. The online workflow can therefore keep this feature
    disabled without importing Torch, while evaluation can inject a fake or real
    predictor through the same interface.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        *,
        predictor: SpanPredictor | None = None,
        chunk_size: int = 384,
        overlap: int = 32,
    ):
        if chunk_size < 32 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("invalid JobBERT chunk configuration")
        self.cache_dir = cache_dir
        self._predictor = predictor
        self.chunk_size = chunk_size
        self.overlap = overlap

    @property
    def loaded(self) -> bool:
        return self._predictor is not None

    def _get_predictor(self) -> SpanPredictor:
        if self._predictor is None:
            self._predictor = JobBertSpanPredictor(self.cache_dir)
        return self._predictor

    def extract(self, text: str, language: str | None = None) -> list[SkillMention]:
        return self.extract_many([text], languages=[language] if language else None)[0]

    def extract_many(
        self,
        texts: list[str],
        *,
        languages: list[str | None] | None = None,
        batch_size: int = 32,
    ) -> list[list[SkillMention]]:
        if languages is not None and len(languages) != len(texts):
            raise ValueError("languages and texts must have identical lengths")
        outputs: list[list[SkillMention]] = [[] for _ in texts]
        chunks: list[list[str]] = []
        chunk_meta: list[tuple[int, int, list[_Token]]] = []
        step = self.chunk_size - self.overlap
        for document_index, text in enumerate(texts):
            language = (languages[document_index] if languages else None) or detect_document_language(text)
            if language != "en":
                continue
            tokens = _tokenize(text)
            for offset in range(0, len(tokens), step):
                token_chunk = tokens[offset:offset + self.chunk_size]
                if not token_chunk:
                    continue
                chunks.append([item.text for item in token_chunk])
                chunk_meta.append((document_index, offset, token_chunk))
                if offset + self.chunk_size >= len(tokens):
                    break
        if not chunks:
            return outputs
        predictions = self._get_predictor().predict_batch(chunks, batch_size=batch_size)
        seen: list[set[tuple[int, int, str]]] = [set() for _ in texts]
        for predicted, (document_index, _, token_chunk) in zip(predictions, chunk_meta):
            text = texts[document_index]
            for span in predicted:
                if not 0 <= span.start < span.end <= len(token_chunk):
                    continue
                start = token_chunk[span.start].start
                end = token_chunk[span.end - 1].end
                raw = text[start:end]
                normalized = normalize_open_skill(raw)
                key = (start, end, span.label)
                if normalized is None or key in seen[document_index]:
                    continue
                seen[document_index].add(key)
                outputs[document_index].append(SkillMention(
                    start=start,
                    end=end,
                    label=span.label,
                    text=raw,
                    normalized_skill=normalized,
                    source_quote=_line_quote(text, start, end),
                ))
        return [sorted(items) for items in outputs]
