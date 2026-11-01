"""GitHub URL validation, shallow cloning, and selecting which files to index."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import httpx

from reporag.config import (
    CODE_EXTENSIONS,
    CONFIG_EXTENSIONS,
    DOC_EXTENSIONS,
    SKIP_DIRS,
    SKIP_FILE_SUFFIXES,
    SKIP_FILENAMES,
    SPECIAL_FILENAMES,
)

# GitHub rules: owner is alphanumeric + single hyphens (max 39); repo allows . _ -
_OWNER = r"[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}"
_REPO = r"[A-Za-z0-9._-]{1,100}"
_URL_RE = re.compile(
    rf"^https://(?:www\.)?github\.com/(?P<owner>{_OWNER})/(?P<repo>{_REPO}?)(?:\.git)?/?$"
)


class InvalidRepoURL(ValueError):
    pass


class RepoTooLarge(ValueError):
    pass


@dataclass(frozen=True)
class RepoRef:
    owner: str
    name: str

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def key(self) -> str:
        """Filesystem-safe identifier, used for clone and index directories."""
        return f"{self.owner}__{self.name}".lower()

    @property
    def clone_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.name}.git"


def parse_github_url(url: str) -> RepoRef:
    """Accept only https://github.com/<owner>/<repo> URLs.

    Rejects other hosts, credentials, ports, query strings, and deep links, so
    the URL can be safely handed to `git clone`.
    """
    url = (url or "").strip()
    match = _URL_RE.match(url)
    if not match:
        raise InvalidRepoURL(
            "Please provide a public GitHub repository URL like https://github.com/owner/repo"
        )
    repo = match["repo"]
    if repo in {".", ".."}:
        raise InvalidRepoURL("Invalid repository name.")
    return RepoRef(owner=match["owner"], name=repo)


def check_repo_size(ref: RepoRef, max_mb: int, token: str | None = None) -> None:
    """Use the GitHub API to reject huge or private repos before cloning.

    If the API is unreachable or rate-limited we skip the check; the post-clone
    file limits still protect us.
    """
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = httpx.get(f"https://api.github.com/repos/{ref.slug}", headers=headers, timeout=10)
    except httpx.HTTPError:
        return
    if resp.status_code == 404:
        raise InvalidRepoURL(f"Repository {ref.slug} was not found or is private.")
    if resp.status_code != 200:
        return
    size_mb = resp.json().get("size", 0) / 1024  # API reports KB
    if size_mb > max_mb:
        raise RepoTooLarge(f"{ref.slug} is {size_mb:.0f} MB; the limit is {max_mb} MB.")


def clone_repo(ref: RepoRef, dest_root: Path, commit: str | None = None) -> tuple[Path, str]:
    """Shallow-clone a repo (optionally at a specific commit). Returns (path, commit_sha)."""
    dest = dest_root / ref.key
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Never prompt for credentials: private repos should fail fast, not hang.
    env = {"GIT_TERMINAL_PROMPT": "0", "PATH": os.environ.get("PATH", "/usr/bin:/bin")}

    def git(*args: str, cwd: Path | None = None) -> str:
        result = subprocess.run(
            ["git", *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            raise RuntimeError(f"git {args[0]} failed: {result.stderr.strip()[:300]}")
        return result.stdout.strip()

    if commit:
        # Fetch exactly one commit so evals are reproducible.
        dest.mkdir()
        git("init", "-q", cwd=dest)
        git("remote", "add", "origin", ref.clone_url, cwd=dest)
        git("fetch", "-q", "--depth", "1", "origin", commit, cwd=dest)
        git("checkout", "-q", "FETCH_HEAD", cwd=dest)
    else:
        git("clone", "-q", "--depth", "1", "--single-branch", ref.clone_url, str(dest))
    sha = git("rev-parse", "HEAD", cwd=dest)
    return dest, sha


def detect_language(path: Path) -> str | None:
    if path.name in SPECIAL_FILENAMES:
        return SPECIAL_FILENAMES[path.name]
    ext = path.suffix.lower()
    return CODE_EXTENSIONS.get(ext) or DOC_EXTENSIONS.get(ext) or CONFIG_EXTENSIONS.get(ext)


def iter_indexable_files(root: Path, max_files: int, max_file_kb: int) -> list[tuple[Path, str]]:
    """Walk the repo and return (relative_path, language) for files worth indexing."""
    selected: list[tuple[Path, str]] = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts[:-1]):
            continue
        if not path.is_file() or path.is_symlink():
            continue
        if path.name in SKIP_FILENAMES or path.name.endswith(SKIP_FILE_SUFFIXES):
            continue
        language = detect_language(path)
        if language is None:
            continue
        if path.stat().st_size > max_file_kb * 1024:
            continue
        selected.append((rel, language))
        if len(selected) >= max_files:
            break
    return selected
