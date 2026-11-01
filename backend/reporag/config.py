"""Application settings, loaded from environment variables (and .env if present)."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# File types we index. Anything else is skipped to keep indexes small and relevant.
CODE_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".swift": "swift",
    ".scala": "scala",
    ".sh": "shell",
    ".sql": "sql",
}
DOC_EXTENSIONS = {
    ".md": "markdown",
    ".mdx": "markdown",
    ".rst": "rst",
    ".txt": "text",
}
CONFIG_EXTENSIONS = {
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".cfg": "ini",
    ".ini": "ini",
}
# Extensionless files that are worth indexing.
SPECIAL_FILENAMES = {"Dockerfile": "dockerfile", "Makefile": "make", "README": "text"}

# Directories that never contain useful source (vendored deps, build output, VCS data).
SKIP_DIRS = {
    ".git", "node_modules", "vendor", "dist", "build", "out", ".next", "__pycache__",
    ".venv", "venv", "env", ".tox", ".mypy_cache", ".pytest_cache", "coverage",
    "target", ".idea", ".vscode", "site-packages", "bower_components",
}
# Lockfiles and minified bundles are large and add noise to retrieval.
SKIP_FILE_SUFFIXES = (".min.js", ".min.css", ".map", ".lock")
SKIP_FILENAMES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "Cargo.lock"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"), env_file_encoding="utf-8", extra="ignore"
    )

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.6-flash"
    github_token: str | None = None

    data_dir: Path = Field(default=Path("./data"), alias="REPORAG_DATA_DIR")
    max_repo_mb: int = Field(default=50, alias="REPORAG_MAX_REPO_MB")
    max_files: int = Field(default=2000, alias="REPORAG_MAX_FILES")
    max_file_kb: int = Field(default=200, alias="REPORAG_MAX_FILE_KB")
    device: str = Field(default="auto", alias="REPORAG_DEVICE")

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Retrieval knobs
    candidates_per_retriever: int = 30
    rerank_top_n: int = 30
    final_top_k: int = 6
    # Cross-encoder logit below which we treat retrieval as "nothing relevant found".
    min_rerank_score: float = -6.0

    cors_origins: str = "http://localhost:3000"

    @property
    def repos_dir(self) -> Path:
        return self.data_dir / "repos"

    @property
    def indexes_dir(self) -> Path:
        return self.data_dir / "indexes"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def resolve_device(preference: str) -> str:
    """Pick the torch device: Apple Silicon (mps) > CUDA > CPU, unless forced."""
    if preference != "auto":
        return preference
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"
