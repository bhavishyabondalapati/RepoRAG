"""Keyword retrieval with BM25, using a code-friendly tokenizer."""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
# Splits camelCase / PascalCase / digits: "parseHTTPResponse2" -> parse, HTTP, Response, 2
_SUBWORD_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "does",
    "do",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "can",
    "i",
    "you",
    "repo",
    "code",
}


def tokenize(text: str) -> list[str]:
    """Lowercased tokens; identifiers also contribute their sub-words.

    `get_user_name` and `getUserName` both yield get / user / name, so a natural
    language query like "get user name" can match either style.
    """
    tokens: list[str] = []
    for word in _WORD_RE.findall(text):
        lower = word.lower()
        parts = [p.lower() for piece in word.split("_") for p in _SUBWORD_RE.findall(piece)]
        if lower not in _STOPWORDS:
            tokens.append(lower)
        if len(parts) > 1:
            tokens.extend(p for p in parts if p not in _STOPWORDS)
    return tokens


class BM25Index:
    def __init__(self, documents: list[str]):
        self._bm25 = BM25Okapi([tokenize(d) or ["<empty>"] for d in documents])

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return (doc_index, score) for the top-k documents with a positive score."""
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]
