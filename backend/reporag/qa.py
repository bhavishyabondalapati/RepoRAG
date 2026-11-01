"""Grounded question answering: retrieve, prompt with numbered excerpts, parse citations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from reporag.llm.base import LLMProvider
from reporag.retrieval.pipeline import Mode, RetrievedChunk, Retriever

INSUFFICIENT = "not enough information in this repo"
INSUFFICIENT_MESSAGE = "Not enough information in this repo to answer that."

SYSTEM_PROMPT = f"""You answer questions about one code repository using ONLY the numbered \
source excerpts provided by the user.

Rules:
1. Use only facts stated in the excerpts. Do not rely on outside knowledge of the project.
2. Cite every factual sentence with the excerpt number(s) in square brackets, like [2] or [1][3].
3. If the excerpts do not contain enough information to answer, reply with exactly:
{INSUFFICIENT}
4. Be concise. Use Markdown and put identifiers in backticks."""

_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass
class Answer:
    text: str
    insufficient: bool
    citations: list[tuple[int, RetrievedChunk]] = field(default_factory=list)  # (ref, chunk)
    retrieved: list[RetrievedChunk] = field(default_factory=list)


def build_prompt(question: str, retrieved: list[RetrievedChunk]) -> str:
    blocks = []
    for n, r in enumerate(retrieved, start=1):
        c = r.chunk
        label = f"[{n}] {c.path} lines {c.start_line}-{c.end_line}"
        if c.symbol:
            label += f" ({c.symbol})"
        blocks.append(f"{label}\n```{c.language}\n{c.text}\n```")
    excerpts = "\n\n".join(blocks)
    return f"Source excerpts:\n\n{excerpts}\n\nQuestion: {question}"


def parse_citations(text: str, num_sources: int) -> list[int]:
    """Citation numbers in order of first appearance, ignoring out-of-range ones."""
    seen: list[int] = []
    for group in _CITATION_RE.findall(text):
        for part in group.split(","):
            n = int(part)
            if 1 <= n <= num_sources and n not in seen:
                seen.append(n)
    return seen


def is_insufficient(text: str) -> bool:
    normalized = re.sub(r"[^a-z ]", "", text.lower()).strip()
    return normalized.startswith(INSUFFICIENT) or normalized == ""


def answer_question(
    question: str,
    retriever: Retriever,
    llm: LLMProvider,
    mode: Mode = "hybrid_rerank",
    top_k: int = 6,
    min_rerank_score: float | None = None,
) -> Answer:
    retrieved = retriever.retrieve(question, mode=mode, top_k=top_k)

    # Guard 1: if the reranker says nothing is relevant, don't even ask the LLM.
    top = retrieved[0] if retrieved else None
    if top is None or (
        min_rerank_score is not None
        and top.rerank_score is not None
        and top.rerank_score < min_rerank_score
    ):
        return Answer(INSUFFICIENT_MESSAGE, insufficient=True, retrieved=retrieved)

    raw = llm.generate(build_prompt(question, retrieved), system=SYSTEM_PROMPT).strip()

    # Guard 2: the model itself says the context is insufficient.
    if is_insufficient(raw):
        return Answer(INSUFFICIENT_MESSAGE, insufficient=True, retrieved=retrieved)

    # Guard 3: an answer with no valid citations is ungrounded, so we refuse it.
    refs = parse_citations(raw, len(retrieved))
    if not refs:
        return Answer(INSUFFICIENT_MESSAGE, insufficient=True, retrieved=retrieved)

    return Answer(
        raw,
        insufficient=False,
        citations=[(n, retrieved[n - 1]) for n in refs],
        retrieved=retrieved,
    )
