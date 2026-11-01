import pytest

from reporag.ingest.github import InvalidRepoURL, iter_indexable_files, parse_github_url


@pytest.mark.parametrize(
    "url,slug",
    [
        ("https://github.com/pallets/itsdangerous", "pallets/itsdangerous"),
        ("https://github.com/sindresorhus/ky/", "sindresorhus/ky"),
        ("https://github.com/owner/repo.git", "owner/repo"),
        ("  https://www.github.com/a-b/c.d_e  ", "a-b/c.d_e"),
        ("https://github.com/user/user.github.io", "user/user.github.io"),
    ],
)
def test_valid_urls(url, slug):
    assert parse_github_url(url).slug == slug


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/a/b",  # not https
        "https://gitlab.com/a/b",
        "https://github.com.evil.com/a/b",
        "https://user:pass@github.com/a/b",
        "https://github.com/a/b/tree/main",
        "https://github.com/a/b?x=1",
        "https://github.com/a",
        "https://github.com/-bad/repo",
        "https://github.com/a/..",
        "git@github.com:a/b.git",
        "https://github.com/a/b; rm -rf /",
        "",
    ],
)
def test_invalid_urls(url):
    with pytest.raises(InvalidRepoURL):
        parse_github_url(url)


def test_iter_indexable_files_filters(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print(1)")
    (tmp_path / "README.md").write_text("# Hi")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / "node_modules" / "x").mkdir(parents=True)
    (tmp_path / "node_modules" / "x" / "index.js").write_text("x")
    (tmp_path / "package-lock.json").write_text("{}")
    (tmp_path / "bundle.min.js").write_text("x")
    (tmp_path / "huge.py").write_text("x" * 5000)

    files = dict(iter_indexable_files(tmp_path, max_files=100, max_file_kb=1))
    assert {str(p) for p in files} == {"src/app.py", "README.md"}
    assert files[next(p for p in files if p.name == "app.py")] == "python"


def test_iter_indexable_files_respects_max_files(tmp_path):
    for i in range(5):
        (tmp_path / f"f{i}.py").write_text("x = 1")
    assert len(iter_indexable_files(tmp_path, max_files=3, max_file_kb=10)) == 3
