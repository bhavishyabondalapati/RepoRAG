"""Lightweight stand-ins for ML models and the LLM, so tests run fast and offline."""

import hashlib

import numpy as np

from reporag.retrieval.bm25 import tokenize

DIM = 64


class FakeEmbedder:
    """Bag-of-hashed-tokens vectors: texts sharing words get high cosine similarity."""

    dim = DIM

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(DIM, dtype="float32")
        for tok in tokenize(text):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def encode_documents(self, texts):
        return np.vstack([self._vec(t) for t in texts])

    def encode_query(self, text):
        return self._vec(text)[None, :]


class KeywordReranker:
    """Scores by word overlap; anything with zero overlap gets a very low score."""

    def score(self, query, documents):
        q = set(tokenize(query))
        return [float(len(q & set(tokenize(d)))) * 2 - 8 for d in documents]


class FakeLLM:
    name = "fake"

    def __init__(self, reply: str = "It verifies the signature [1]."):
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        self.prompts.append(prompt)
        return self.reply
