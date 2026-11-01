"""Retrieval pipeline with four modes, so the eval can compare them head to head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from reporag.index_store import RepoIndex
from reporag.ingest.chunking import Chunk
from reporag.retrieval.dense import Embedder
from reporag.retrieval.fusion import reciprocal_rank_fusion
from reporag.retrieval.rerank import Reranker

Mode = Literal["bm25", "dense", "hybrid", "hybrid_rerank"]
MODES: tuple[Mode, ...] = ("bm25", "dense", "hybrid", "hybrid_rerank")


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float  # mode-specific: BM25 score, cosine, RRF score, or rerank logit
    rerank_score: float | None = None


class Retriever:
    def __init__(
        self,
        index: RepoIndex,
        embedder: Embedder,
        reranker: Reranker | None = None,
        candidates_per_retriever: int = 30,
        rerank_top_n: int = 30,
    ):
        self.index = index
        self.embedder = embedder
        self.reranker = reranker
        self.candidates = candidates_per_retriever
        self.rerank_top_n = rerank_top_n

    def _bm25(self, query: str, k: int) -> list[tuple[int, float]]:
        return self.index.bm25.search(query, k)

    def _dense(self, query: str, k: int) -> list[tuple[int, float]]:
        return self.index.dense.search(self.embedder.encode_query(query), k)

    def retrieve(self, query: str, mode: Mode = "hybrid_rerank", top_k: int = 6) -> list[
        RetrievedChunk
    ]:
        chunks = self.index.chunks
        if mode == "bm25":
            hits = self._bm25(query, top_k)
        elif mode == "dense":
            hits = self._dense(query, top_k)
        elif mode in ("hybrid", "hybrid_rerank"):
            bm25_ids = [i for i, _ in self._bm25(query, self.candidates)]
            dense_ids = [i for i, _ in self._dense(query, self.candidates)]
            hits = reciprocal_rank_fusion([bm25_ids, dense_ids])
            if mode == "hybrid_rerank":
                return self._rerank(query, [i for i, _ in hits[: self.rerank_top_n]], top_k)
            hits = hits[:top_k]
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")
        return [RetrievedChunk(chunks[i], s) for i, s in hits]

    def _rerank(self, query: str, ids: list[int], top_k: int) -> list[RetrievedChunk]:
        if self.reranker is None:
            raise ValueError("hybrid_rerank mode needs a reranker")
        chunks = [self.index.chunks[i] for i in ids]
        scores = self.reranker.score(query, [c.search_text() for c in chunks])
        ranked = sorted(zip(chunks, scores, strict=True), key=lambda cs: -cs[1])[:top_k]
        return [RetrievedChunk(c, s, rerank_score=s) for c, s in ranked]
