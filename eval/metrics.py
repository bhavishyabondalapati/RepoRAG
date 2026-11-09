"""Retrieval metrics and an LLM-as-judge faithfulness score."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    path: str
    start: int
    end: int


def overlaps(path: str, start: int, end: int, gold: Span) -> bool:
    """A retrieved chunk counts as relevant if it is in the gold file and its lines overlap."""
    return path == gold.path and start <= gold.end and end >= gold.start


def recall_at_k(retrieved: list[tuple[str, int, int]], gold: list[Span], k: int) -> float:
    """Fraction of gold evidence spans covered by at least one of the top-k chunks."""
    if not gold:
        raise ValueError("recall is undefined for questions without gold evidence")
    top = retrieved[:k]
    hit = sum(1 for g in gold if any(overlaps(p, s, e, g) for p, s, e in top))
    return hit / len(gold)


def reciprocal_rank(retrieved: list[tuple[str, int, int]], gold: list[Span], k: int = 10) -> float:
    """1 / rank of the first relevant chunk within the top k (0 if none)."""
    for rank, (p, s, e) in enumerate(retrieved[:k], start=1):
        if any(overlaps(p, s, e, g) for g in gold):
            return 1.0 / rank
    return 0.0


def citation_precision(cited: list[tuple[str, int, int]], gold: list[Span]) -> float | None:
    """Share of cited chunks that overlap a gold evidence span (None if nothing was cited)."""
    if not cited:
        return None
    return sum(1 for p, s, e in cited if any(overlaps(p, s, e, g) for g in gold)) / len(cited)


def citation_recall(cited: list[tuple[str, int, int]], gold: list[Span]) -> float:
    """Fraction of gold evidence spans covered by at least one cited chunk."""
    return recall_at_k(cited, gold, k=len(cited))


JUDGE_SYSTEM = """You are a strict evaluator of answer faithfulness for a retrieval system.
You check whether every factual claim in an ANSWER is directly supported by the CONTEXT.
Ignore citation markers like [1]. Do not use outside knowledge: a claim that is true in
general but not stated in the CONTEXT is unsupported."""

JUDGE_TEMPLATE = """CONTEXT:
{context}

ANSWER:
{answer}

Split the ANSWER into its individual factual claims. For each claim decide whether the
CONTEXT supports it. Respond with only a JSON object, no prose:
{{"claims": [{{"claim": "...", "supported": true}}], "supported": <int>, "total": <int>}}"""


def build_judge_prompt(context: str, answer: str) -> str:
    return JUDGE_TEMPLATE.format(context=context, answer=answer)


def parse_judge_score(raw: str) -> float | None:
    """Return supported/total from the judge's JSON, or None if unparseable."""
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    claims = data.get("claims")
    if isinstance(claims, list) and claims:
        supported = sum(1 for c in claims if isinstance(c, dict) and c.get("supported") is True)
        return supported / len(claims)
    total = data.get("total")
    if isinstance(total, int) and total > 0 and isinstance(data.get("supported"), int):
        return min(1.0, data["supported"] / total)
    return None
