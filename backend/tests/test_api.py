import time

import pytest
from fastapi.testclient import TestClient

from reporag import main
from reporag.config import Settings
from reporag.index_store import RepoIndex
from reporag.ingest.chunking import Chunk
from reporag.jobs import JobManager
from tests.fakes import FakeEmbedder, FakeLLM, KeywordReranker

CHUNKS = [
    Chunk(
        "src/signer.py", "python", 3, 9, "def unsign(token): verify signature", "function", "unsign"
    ),
    Chunk("README.md", "markdown", 1, 4, "# Install\npip install thing", "section", "Install"),
]


def fake_index_fn(ref, settings, embedder, on_stage=lambda *a: None, commit=None):
    if ref.name == "broken":
        raise RuntimeError("clone failed")
    on_stage("embedding", 0.5, "Embedding")
    index = RepoIndex.build(ref.slug, "deadbeef", CHUNKS, embedder, files_indexed=2)
    index.save(settings.indexes_dir / ref.key)
    return index


@pytest.fixture
def client(tmp_path):
    settings = Settings(REPORAG_DATA_DIR=tmp_path, gemini_api_key=None)
    manager = JobManager(settings, FakeEmbedder, index_fn=fake_index_fn)
    llm = FakeLLM("`unsign` checks the signature [1].")
    overrides = {
        main.get_settings: lambda: settings,
        main.get_job_manager: lambda: manager,
        main.get_query_embedder: FakeEmbedder,
        main.get_query_reranker: KeywordReranker,
        main.get_llm_provider: lambda: llm,
    }
    main.app.dependency_overrides.update(overrides)
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def wait_until_done(client, repo_id, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/repos/{repo_id}").json()
        if body["status"] in ("ready", "failed"):
            return body
        time.sleep(0.02)
    raise AssertionError("indexing did not finish")


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok", "llm_configured": False}


def test_rejects_non_github_url(client):
    resp = client.post("/api/repos", json={"url": "https://gitlab.com/a/b"})
    assert resp.status_code == 422
    assert "GitHub" in resp.json()["detail"]


def test_index_then_ask_returns_cited_answer(client):
    resp = client.post("/api/repos", json={"url": "https://github.com/pallets/itsdangerous"})
    assert resp.status_code == 202
    repo_id = resp.json()["repo_id"]
    assert repo_id == "pallets__itsdangerous"

    status = wait_until_done(client, repo_id)
    assert status["status"] == "ready" and status["chunks"] == 2 and status["commit"] == "deadbeef"

    resp = client.post(f"/api/repos/{repo_id}/ask", json={"question": "how is signature verified"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["insufficient"] is False
    citation = body["citations"][0]
    assert citation["ref"] == 1 and citation["path"] == "src/signer.py"
    assert (citation["start_line"], citation["end_line"]) == (3, 9)
    assert citation["url"].endswith("/pallets/itsdangerous/blob/deadbeef/src/signer.py#L3-L9")


def test_cached_index_is_reused(client):
    client.post("/api/repos", json={"url": "https://github.com/o/cached"})
    wait_until_done(client, "o__cached")
    # A fresh manager (like after a server restart) should find the index on disk.
    settings = main.app.dependency_overrides[main.get_settings]()
    fresh = JobManager(settings, FakeEmbedder, index_fn=fake_index_fn)
    main.app.dependency_overrides[main.get_job_manager] = lambda: fresh
    assert client.get("/api/repos/o__cached").json()["status"] == "ready"


def test_failed_indexing_is_reported(client):
    client.post("/api/repos", json={"url": "https://github.com/o/broken"})
    status = wait_until_done(client, "o__broken")
    assert status["status"] == "failed" and "clone failed" in status["error"]


def test_ask_unknown_repo_404(client):
    resp = client.post("/api/repos/nobody__nothing/ask", json={"question": "anything here?"})
    assert resp.status_code == 404


def test_ask_validates_question_and_mode(client):
    client.post("/api/repos", json={"url": "https://github.com/o/r"})
    wait_until_done(client, "o__r")
    assert client.post("/api/repos/o__r/ask", json={"question": ""}).status_code == 422
    bad_mode = {"question": "valid question", "mode": "magic"}
    assert client.post("/api/repos/o__r/ask", json=bad_mode).status_code == 422


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid", "hybrid_rerank"])
def test_ask_supports_every_mode(client, mode):
    client.post("/api/repos", json={"url": "https://github.com/o/r"})
    wait_until_done(client, "o__r")
    resp = client.post("/api/repos/o__r/ask", json={"question": "verify signature", "mode": mode})
    assert resp.status_code == 200 and resp.json()["mode"] == mode
