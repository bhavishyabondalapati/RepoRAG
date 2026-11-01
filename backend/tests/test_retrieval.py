import pytest

from reporag.index_store import RepoIndex
from reporag.ingest.chunking import Chunk
from reporag.retrieval.bm25 import BM25Index, tokenize
from reporag.retrieval.fusion import reciprocal_rank_fusion
from reporag.retrieval.pipeline import MODES, Retriever
from tests.fakes import FakeEmbedder, KeywordReranker

# ---------------------------------------------------------------- fusion ---- #


def test_rrf_rewards_items_ranked_by_both_lists():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a", "d"]])
    order = [item for item, _ in fused]
    assert order[0] == "a"  # rank 1 + rank 2 beats rank 3 + rank 1
    assert order[1] == "c"
    assert set(order) == {"a", "b", "c", "d"}


def test_rrf_score_formula():
    fused = dict(reciprocal_rank_fusion([["x"], ["y", "x"]], k=60))
    assert fused["x"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["y"] == pytest.approx(1 / 61)


def test_rrf_handles_empty_and_single_lists():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
    assert [i for i, _ in reciprocal_rank_fusion([[3, 1, 2]])] == [3, 1, 2]


def test_rrf_ties_are_deterministic():
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]])
    assert [i for i, _ in fused] == ["a", "b"]


# ------------------------------------------------------------------ bm25 ---- #


def test_tokenize_splits_identifiers():
    tokens = tokenize("def getUserName(user_id): parseHTTPResponse")
    for expected in ["getusername", "get", "user", "name", "user_id", "id", "http", "response"]:
        assert expected in tokens
    assert "the" not in tokenize("the user")


def test_bm25_finds_keyword_match():
    index = BM25Index(["def sign(value): ...", "def unsign(token): ...", "readme intro"])
    hits = index.search("how does unsign work", k=3)
    assert hits[0][0] == 1
    assert index.search("the of and", k=3) == []


# -------------------------------------------------------------- pipeline ---- #


def _chunk(path, text, symbol=None):
    return Chunk(path, "python", 1, text.count("\n") + 1, text, "function", symbol)


@pytest.fixture
def retriever():
    chunks = [
        _chunk("signer.py", "def sign(value): return hmac signature of value", "sign"),
        _chunk("signer.py", "def unsign(token): verify signature and raise BadSignature", "unsign"),
        _chunk("timed.py", "def loads(token, max_age): check timestamp expiry", "loads"),
        _chunk("README.md", "itsdangerous helps you send data to untrusted environments"),
    ]
    embedder = FakeEmbedder()
    index = RepoIndex.build("o/r", "abc", chunks, embedder)
    return Retriever(index, embedder, KeywordReranker(), candidates_per_retriever=4)


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_returns_ranked_chunks(retriever, mode):
    results = retriever.retrieve("verify signature BadSignature", mode=mode, top_k=2)
    assert 1 <= len(results) <= 2
    assert results[0].chunk.symbol == "unsign"


def test_rerank_reorders_and_sets_scores(retriever):
    results = retriever.retrieve("timestamp expiry", mode="hybrid_rerank", top_k=4)
    assert results[0].chunk.symbol == "loads"
    assert all(r.rerank_score is not None for r in results)
    scores = [r.rerank_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_unknown_mode_raises(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("x", mode="magic")


def test_index_round_trip(tmp_path, retriever):
    retriever.index.save(tmp_path)
    assert RepoIndex.exists(tmp_path)
    loaded = RepoIndex.load(tmp_path)
    assert loaded.repo == "o/r" and loaded.commit == "abc"
    assert [c.id for c in loaded.chunks] == [c.id for c in retriever.index.chunks]
    assert loaded.dense.index.ntotal == 4


def test_build_rejects_empty_repo():
    with pytest.raises(ValueError):
        RepoIndex.build("o/r", "abc", [], FakeEmbedder())
