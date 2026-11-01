"""Dense retrieval: sentence-transformer embeddings stored in a FAISS index."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Protocol

import numpy as np

# bge models are trained with this instruction prefix for queries (not documents).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    dim: int

    def encode_documents(self, texts: list[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, device: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
        self._query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype="float32")

    def encode_query(self, text: str) -> np.ndarray:
        vector = self.model.encode(
            [self._query_prefix + text], normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vector, dtype="float32")


_lock = threading.Lock()
_embedders: dict[tuple[str, str], SentenceTransformerEmbedder] = {}


def get_embedder(model_name: str, device: str) -> SentenceTransformerEmbedder:
    """Load each model once per process; loading takes seconds, encoding is fast."""
    with _lock:
        key = (model_name, device)
        if key not in _embedders:
            _embedders[key] = SentenceTransformerEmbedder(model_name, device)
        return _embedders[key]


class DenseIndex:
    """Exact inner-product search over L2-normalized vectors (= cosine similarity).

    A flat index is exact and fast enough for the few thousand chunks a single
    repo produces; approximate indexes (IVF/HNSW) only pay off at much larger scale.
    """

    def __init__(self, index):
        self.index = index

    @classmethod
    def build(cls, vectors: np.ndarray) -> DenseIndex:
        import faiss

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        k = min(k, self.index.ntotal)
        if k == 0:
            return []
        scores, ids = self.index.search(query_vector, k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=True) if i >= 0]

    def save(self, path: Path) -> None:
        import faiss

        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path) -> DenseIndex:
        import faiss

        return cls(faiss.read_index(str(path)))
