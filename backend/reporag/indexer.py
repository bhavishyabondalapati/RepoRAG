"""End-to-end indexing: validate -> clone -> select files -> chunk -> embed -> save."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from reporag.config import Settings
from reporag.index_store import RepoIndex
from reporag.ingest.chunking import Chunk, chunk_file
from reporag.ingest.github import RepoRef, check_repo_size, clone_repo, iter_indexable_files
from reporag.retrieval.dense import Embedder

# on_stage(stage, progress 0..1, message)
StageCallback = Callable[[str, float, str], None]


def _noop(stage: str, progress: float, message: str) -> None:
    pass


def chunk_repository(root: Path, settings: Settings) -> tuple[list[Chunk], int]:
    files = iter_indexable_files(root, settings.max_files, settings.max_file_kb)
    chunks: list[Chunk] = []
    for rel, language in files:
        try:
            source = (root / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable despite the extension
        chunks.extend(chunk_file(rel.as_posix(), language, source))
    return chunks, len(files)


def index_repository(
    ref: RepoRef,
    settings: Settings,
    embedder: Embedder,
    commit: str | None = None,
    on_stage: StageCallback = _noop,
) -> RepoIndex:
    on_stage("cloning", 0.05, f"Checking and cloning {ref.slug}")
    check_repo_size(ref, settings.max_repo_mb, settings.github_token)
    root, sha = clone_repo(ref, settings.repos_dir, commit=commit)

    on_stage("chunking", 0.2, "Splitting files into chunks")
    chunks, num_files = chunk_repository(root, settings)
    on_stage("embedding", 0.3, f"Embedding {len(chunks)} chunks from {num_files} files")

    def embed_progress(fraction: float) -> None:
        on_stage("embedding", 0.3 + 0.65 * fraction, f"Embedding {len(chunks)} chunks")

    index = RepoIndex.build(ref.slug, sha, chunks, embedder, num_files, embed_progress)
    index.save(settings.indexes_dir / ref.key)
    return index
