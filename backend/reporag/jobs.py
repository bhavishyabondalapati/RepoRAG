"""Background indexing jobs and an in-memory cache of loaded indexes."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from reporag.config import Settings
from reporag.index_store import RepoIndex
from reporag.indexer import index_repository
from reporag.ingest.github import RepoRef
from reporag.retrieval.dense import Embedder

log = logging.getLogger(__name__)


@dataclass
class JobState:
    repo_id: str
    repo: str
    status: str = "queued"  # queued | cloning | chunking | embedding | ready | failed
    progress: float = 0.0
    message: str = "Waiting to start"
    error: str | None = None
    commit: str | None = None
    files_indexed: int = 0
    chunks: int = 0

    def mark_ready(self, index: RepoIndex) -> None:
        self.status, self.progress, self.message = "ready", 1.0, "Ready"
        self.commit, self.files_indexed, self.chunks = (
            index.commit,
            index.files_indexed,
            len(index.chunks),
        )


IndexFn = Callable[..., RepoIndex]


class JobManager:
    """Runs one indexing job at a time (embedding is memory- and GPU-heavy)."""

    def __init__(
        self,
        settings: Settings,
        embedder_factory: Callable[[], Embedder],
        index_fn: IndexFn = index_repository,
    ):
        self.settings = settings
        self._embedder_factory = embedder_factory
        self._index_fn = index_fn
        self._jobs: dict[str, JobState] = {}
        self._indexes: dict[str, RepoIndex] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="indexer")

    def submit(self, ref: RepoRef, force: bool = False) -> JobState:
        with self._lock:
            existing = self._jobs.get(ref.key)
            if existing and existing.status not in ("ready", "failed"):
                return existing  # already running
            if not force and self._load_cached(ref.key):
                return self._jobs[ref.key]
            job = JobState(repo_id=ref.key, repo=ref.slug)
            self._jobs[ref.key] = job
            self._indexes.pop(ref.key, None)
        self._executor.submit(self._run, ref, job)
        return job

    def _run(self, ref: RepoRef, job: JobState) -> None:
        def on_stage(stage: str, progress: float, message: str) -> None:
            job.status, job.progress, job.message = stage, round(progress, 3), message

        try:
            index = self._index_fn(ref, self.settings, self._embedder_factory(), on_stage=on_stage)
        except Exception as exc:  # surface any failure to the UI instead of crashing
            log.exception("Indexing %s failed", ref.slug)
            job.status, job.error, job.message = "failed", str(exc), "Indexing failed"
            return
        with self._lock:
            self._indexes[ref.key] = index
            job.mark_ready(index)

    def _load_cached(self, repo_id: str) -> bool:
        """Load an index built in a previous run from disk. Caller holds the lock."""
        if repo_id in self._indexes:
            return True
        directory = self.settings.indexes_dir / repo_id
        if not RepoIndex.exists(directory):
            return False
        try:
            index = RepoIndex.load(directory)
        except Exception:
            log.warning("Ignoring unreadable cached index for %s", repo_id)
            return False
        self._indexes[repo_id] = index
        job = JobState(repo_id=repo_id, repo=index.repo)
        job.mark_ready(index)
        self._jobs[repo_id] = job
        return True

    def get(self, repo_id: str) -> JobState | None:
        with self._lock:
            if repo_id not in self._jobs:
                self._load_cached(repo_id)
            return self._jobs.get(repo_id)

    def get_index(self, repo_id: str) -> RepoIndex | None:
        with self._lock:
            return self._indexes.get(repo_id)
