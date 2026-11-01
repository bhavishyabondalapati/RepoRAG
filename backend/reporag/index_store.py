"""Build, save, and load the per-repo search index (chunks + BM25 + FAISS)."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from reporag.ingest.chunking import Chunk
from reporag.retrieval.bm25 import BM25Index
from reporag.retrieval.dense import DenseIndex, Embedder

EMBED_BATCH = 64
INDEX_VERSION = 1


@dataclass
class RepoIndex:
    repo: str
    commit: str
    chunks: list[Chunk]
    bm25: BM25Index
    dense: DenseIndex
    files_indexed: int = 0

    @classmethod
    def build(
        cls,
        repo: str,
        commit: str,
        chunks: list[Chunk],
        embedder: Embedder,
        files_indexed: int = 0,
        on_progress: Callable[[float], None] | None = None,
    ) -> RepoIndex:
        if not chunks:
            raise ValueError("No indexable content found in this repository.")
        texts = [c.search_text() for c in chunks]
        batches = []
        for start in range(0, len(texts), EMBED_BATCH):
            batches.append(embedder.encode_documents(texts[start : start + EMBED_BATCH]))
            if on_progress:
                on_progress(min(1.0, (start + EMBED_BATCH) / len(texts)))
        vectors = np.vstack(batches).astype("float32")
        return cls(repo, commit, chunks, BM25Index(texts), DenseIndex.build(vectors), files_indexed)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "chunks.jsonl").open("w") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk.to_dict()) + "\n")
        self.dense.save(directory / "dense.faiss")
        meta = {
            "version": INDEX_VERSION,
            "repo": self.repo,
            "commit": self.commit,
            "num_chunks": len(self.chunks),
            "files_indexed": self.files_indexed,
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: Path) -> RepoIndex:
        meta = json.loads((directory / "meta.json").read_text())
        if meta.get("version") != INDEX_VERSION:
            raise ValueError("Index was built by an incompatible version; re-index the repo.")
        with (directory / "chunks.jsonl").open() as f:
            chunks = [Chunk(**json.loads(line)) for line in f]
        # BM25 is cheap to rebuild, so we don't persist it.
        bm25 = BM25Index([c.search_text() for c in chunks])
        dense = DenseIndex.load(directory / "dense.faiss")
        return cls(meta["repo"], meta["commit"], chunks, bm25, dense, meta.get("files_indexed", 0))

    @staticmethod
    def exists(directory: Path) -> bool:
        return (directory / "meta.json").exists() and (directory / "dense.faiss").exists()
