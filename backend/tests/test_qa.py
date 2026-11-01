from reporag.index_store import RepoIndex
from reporag.ingest.chunking import Chunk
from reporag.qa import (
    INSUFFICIENT_MESSAGE,
    answer_question,
    build_prompt,
    is_insufficient,
    parse_citations,
)
from reporag.retrieval.pipeline import Retriever
from tests.fakes import FakeEmbedder, FakeLLM, KeywordReranker


def _retriever():
    chunks = [
        Chunk(
            "signer.py",
            "python",
            10,
            20,
            "def unsign(token): verify signature",
            "function",
            "unsign",
        ),
        Chunk("README.md", "markdown", 1, 5, "Install with pip", "section", "Install"),
    ]
    embedder = FakeEmbedder()
    return Retriever(RepoIndex.build("o/r", "abc", chunks, embedder), embedder, KeywordReranker())


def test_parse_citations_order_dedup_and_range():
    text = "First [2]. Then [1][2]. Combined [1, 3]. Bogus [9] and [0]."
    assert parse_citations(text, num_sources=3) == [2, 1, 3]


def test_is_insufficient_variants():
    assert is_insufficient("not enough information in this repo")
    assert is_insufficient("Not enough information in this repo.")
    assert is_insufficient("")
    assert not is_insufficient("The function verifies [1].")


def test_prompt_contains_numbered_sources_with_locations():
    prompt = build_prompt("How?", _retriever().retrieve("verify signature", top_k=2))
    assert "[1] signer.py lines 10-20 (unsign)" in prompt
    assert "Question: How?" in prompt


def test_answer_with_citations():
    llm = FakeLLM("`unsign` verifies the signature [1].")
    result = answer_question("how is the signature verified", _retriever(), llm, top_k=2)
    assert not result.insufficient
    assert [(ref, c.chunk.path) for ref, c in result.citations] == [(1, "signer.py")]


def test_llm_says_insufficient():
    llm = FakeLLM("not enough information in this repo")
    result = answer_question("verify signature", _retriever(), llm, top_k=2)
    assert result.insufficient and result.text == INSUFFICIENT_MESSAGE
    assert result.citations == []


def test_uncited_answer_is_rejected():
    llm = FakeLLM("It probably uses HMAC.")  # no [n] markers -> ungrounded
    result = answer_question("verify signature", _retriever(), llm, top_k=2)
    assert result.insufficient


def test_low_rerank_score_skips_llm():
    llm = FakeLLM()
    result = answer_question(
        "kubernetes helm chart", _retriever(), llm, top_k=2, min_rerank_score=-5.0
    )
    assert result.insufficient
    assert llm.prompts == []  # never called
