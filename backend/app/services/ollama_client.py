from __future__ import annotations

import json
import math
import re
import threading
from functools import lru_cache
from typing import Any

import httpx

from app.config import settings
from app.services.model_cache import ModelResponseCache


class OllamaUnavailable(RuntimeError):
    pass


class OllamaClient:
    CACHE_PROMPT_VERSION = "ollama-json-v1"

    def __init__(
        self, base_url: str | None = None, chat_model: str | None = None,
        embed_model: str | None = None, cache: ModelResponseCache | None = None,
        cache_enabled: bool = True,
    ):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.chat_model = chat_model or settings.chat_model
        self.embed_model = embed_model or settings.embed_model
        self.cache = cache or ModelResponseCache()
        self.cache_enabled = cache_enabled
        self._call_context = threading.local()
        self._stats_lock = threading.Lock()
        self.cache_hits = 0
        self.cache_misses = 0
        self._chat_model_identity: str | None = None

    @property
    def last_call_cache_hit(self) -> bool:
        return bool(getattr(self._call_context, "cache_hit", False))

    @last_call_cache_hit.setter
    def last_call_cache_hit(self, value: bool) -> None:
        self._call_context.cache_hit = value

    def status(self) -> dict[str, Any]:
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=3)
            response.raise_for_status()
            models = [item["name"] for item in response.json().get("models", [])]
            return {
                "available": True,
                "models": models,
                "chat_model_ready": any(name.split(":")[0] == self.chat_model.split(":")[0] for name in models),
                "embed_model_ready": any(name.split(":")[0] == self.embed_model.split(":")[0] for name in models),
            }
        except Exception as exc:
            return {"available": False, "models": [], "error": str(exc)}

    def _model_identity(self) -> str:
        if self._chat_model_identity:
            return self._chat_model_identity
        identity = self.chat_model
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=3)
            response.raise_for_status()
            for item in response.json().get("models", []):
                if item.get("name") == self.chat_model or item.get("name", "").split(":")[0] == self.chat_model.split(":")[0]:
                    identity = f"{self.chat_model}@{item.get('digest', 'unknown')}"
                    break
        except Exception:
            pass
        self._chat_model_identity = identity
        return identity

    def generate_json(
        self, system: str, user: str, timeout: float = 120,
        cache_namespace: str = "generic", prompt_version: str | None = None,
    ) -> dict[str, Any]:
        # Structured extraction must be bounded. Without num_predict, a local
        # model can spend the full request timeout expanding repetitive JSON on
        # long resumes, making both production latency and A/B results unstable.
        options = {"temperature": 0.0, "seed": 42, "num_predict": 1536}
        cache_options = {**options, "format": "json", "think": False}
        model_identity = self._model_identity()
        cache_key = self.cache.build_key(
            model_identity, cache_namespace, prompt_version or self.CACHE_PROMPT_VERSION,
            system, user, cache_options,
        )
        if self.cache_enabled:
            cached = self.cache.get(cache_key)
            if cached is not None:
                self.last_call_cache_hit = True
                with self._stats_lock:
                    self.cache_hits += 1
                return cached
        self.last_call_cache_hit = False
        with self._stats_lock:
            self.cache_misses += 1
        payload = {
            "model": self.chat_model,
            "stream": False,
            "format": "json",
            "think": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": options,
        }
        try:
            response = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=timeout)
            response.raise_for_status()
            content = response.json()["message"]["content"]
            parsed = self._parse_json(content)
            if self.cache_enabled:
                self.cache.put(cache_key, model_identity, cache_namespace, parsed)
            return parsed
        except Exception as exc:
            raise OllamaUnavailable(str(exc)) from exc

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

    @lru_cache(maxsize=1024)
    def embed_one(self, text: str) -> tuple[float, ...]:
        try:
            response = httpx.post(
                f"{self.base_url}/api/embed",
                json={"model": self.embed_model, "input": text},
                timeout=60,
            )
            response.raise_for_status()
            vector = response.json()["embeddings"][0]
            return tuple(float(value) for value in vector)
        except Exception as exc:
            raise OllamaUnavailable(str(exc)) from exc

    def similarity(self, left: str, right: str) -> float:
        a = self.embed_one(left)
        b = self.embed_one(right)
        numerator = sum(x * y for x, y in zip(a, b))
        denominator = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
        return numerator / denominator if denominator else 0.0

    def cache_status(self) -> dict[str, Any]:
        with self._stats_lock:
            session_hits = self.cache_hits
            session_misses = self.cache_misses
        try:
            persistent = self.cache.stats()
        except Exception as exc:
            persistent = {"entries": 0, "hits": 0, "available": False, "error": str(exc)[:200]}
        return {
            "enabled": self.cache_enabled,
            "session_hits": session_hits,
            "session_misses": session_misses,
            **persistent,
        }
