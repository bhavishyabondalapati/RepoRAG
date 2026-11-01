"""Pydantic models for API requests and responses."""

from __future__ import annotations

from pydantic import BaseModel, Field

from reporag.retrieval.pipeline import Mode


class IndexRequest(BaseModel):
    url: str = Field(..., max_length=300, examples=["https://github.com/pallets/itsdangerous"])
    force: bool = Field(False, description="Re-index even if a cached index exists")


class RepoStatus(BaseModel):
    repo_id: str
    repo: str
    status: str
    progress: float
    message: str
    error: str | None = None
    commit: str | None = None
    files_indexed: int = 0
    chunks: int = 0


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    mode: Mode = "hybrid_rerank"


class Citation(BaseModel):
    ref: int = Field(..., description="The [n] marker used in the answer text")
    path: str
    start_line: int
    end_line: int
    language: str
    symbol: str | None = None
    snippet: str
    score: float
    url: str


class AskResponse(BaseModel):
    answer: str
    insufficient: bool
    citations: list[Citation]
    mode: Mode
    latency_ms: int


class Health(BaseModel):
    status: str
    llm_configured: bool
