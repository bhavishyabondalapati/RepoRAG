"""FastAPI application: index a GitHub repo, poll progress, ask cited questions."""

from __future__ import annotations

import logging
import time
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from reporag.config import Settings, get_settings, resolve_device
from reporag.ingest.github import InvalidRepoURL, parse_github_url
from reporag.jobs import JobManager, JobState
from reporag.llm.base import LLMError, LLMProvider, get_llm
from reporag.qa import answer_question
from reporag.retrieval.dense import get_embedder
from reporag.retrieval.pipeline import Retriever
from reporag.retrieval.rerank import Reranker, get_reranker
from reporag.schemas import AskRequest, AskResponse, Citation, Health, IndexRequest, RepoStatus

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# --------------------------------------------------------------------------- #
# Dependencies (overridden in tests with fakes)
# --------------------------------------------------------------------------- #


@lru_cache
def get_job_manager() -> JobManager:
    settings = get_settings()
    device = resolve_device(settings.device)
    return JobManager(settings, lambda: get_embedder(settings.embedding_model, device))


def get_query_embedder():
    settings = get_settings()
    return get_embedder(settings.embedding_model, resolve_device(settings.device))


def get_query_reranker() -> Reranker:
    settings = get_settings()
    return get_reranker(settings.reranker_model, resolve_device(settings.device))


def get_llm_provider() -> LLMProvider:
    settings = get_settings()
    try:
        return get_llm("gemini", api_key=settings.gemini_api_key, model=settings.gemini_model)
    except LLMError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #

app = FastAPI(title="RepoRAG", version="0.1.0", description=__doc__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in get_settings().cors_origins.split(",")],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def _status(job: JobState) -> RepoStatus:
    return RepoStatus(**job.__dict__)


@app.get("/api/health", response_model=Health)
def health(settings: Settings = Depends(get_settings)) -> Health:
    return Health(status="ok", llm_configured=bool(settings.gemini_api_key))


@app.post("/api/repos", response_model=RepoStatus, status_code=status.HTTP_202_ACCEPTED)
def index_repo(req: IndexRequest, manager: JobManager = Depends(get_job_manager)) -> RepoStatus:
    try:
        ref = parse_github_url(req.url)
    except InvalidRepoURL as exc:
        raise HTTPException(422, str(exc)) from exc
    return _status(manager.submit(ref, force=req.force))


@app.get("/api/repos/{repo_id}", response_model=RepoStatus)
def repo_status(repo_id: str, manager: JobManager = Depends(get_job_manager)) -> RepoStatus:
    job = manager.get(repo_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown repository; index it first.")
    return _status(job)


@app.post("/api/repos/{repo_id}/ask", response_model=AskResponse)
def ask(
    repo_id: str,
    req: AskRequest,
    manager: JobManager = Depends(get_job_manager),
    embedder=Depends(get_query_embedder),
    reranker: Reranker = Depends(get_query_reranker),
    llm: LLMProvider = Depends(get_llm_provider),
    settings: Settings = Depends(get_settings),
) -> AskResponse:
    job = manager.get(repo_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown repository; index it first.")
    index = manager.get_index(repo_id)
    if job.status != "ready" or index is None:
        raise HTTPException(status.HTTP_409_CONFLICT, f"Repository is not ready ({job.status}).")

    started = time.perf_counter()
    retriever = Retriever(
        index,
        embedder,
        reranker if req.mode == "hybrid_rerank" else None,
        candidates_per_retriever=settings.candidates_per_retriever,
        rerank_top_n=settings.rerank_top_n,
    )
    try:
        result = answer_question(
            req.question,
            retriever,
            llm,
            mode=req.mode,
            top_k=settings.final_top_k,
            min_rerank_score=settings.min_rerank_score,
        )
    except LLMError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc

    citations = [
        Citation(
            ref=ref,
            path=r.chunk.path,
            start_line=r.chunk.start_line,
            end_line=r.chunk.end_line,
            language=r.chunk.language,
            symbol=r.chunk.symbol,
            snippet=r.chunk.text,
            score=round(r.score, 4),
            url=(
                f"https://github.com/{index.repo}/blob/{index.commit}/{r.chunk.path}"
                f"#L{r.chunk.start_line}-L{r.chunk.end_line}"
            ),
        )
        for ref, r in result.citations
    ]
    return AskResponse(
        answer=result.text,
        insufficient=result.insufficient,
        citations=citations,
        mode=req.mode,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
