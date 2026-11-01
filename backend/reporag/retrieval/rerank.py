"""Cross-encoder reranking: score (query, chunk) pairs jointly for precision."""

from __future__ import annotations

import threading
from typing import Protocol


class Reranker(Protocol):
    def score(self, query: str, documents: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device=device, max_length=512)

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        scores = self.model.predict(
            [(query, d) for d in documents], batch_size=32, show_progress_bar=False
        )
        return [float(s) for s in scores]


_lock = threading.Lock()
_rerankers: dict[tuple[str, str], CrossEncoderReranker] = {}


def get_reranker(model_name: str, device: str) -> CrossEncoderReranker:
    with _lock:
        key = (model_name, device)
        if key not in _rerankers:
            _rerankers[key] = CrossEncoderReranker(model_name, device)
        return _rerankers[key]
