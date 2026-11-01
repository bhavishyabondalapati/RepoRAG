"""Compare retrieval modes on the hand-written dataset.

Usage (from the repo root):
    python eval/run_eval.py              # retrieval metrics + LLM faithfulness
    python eval/run_eval.py --no-llm     # retrieval metrics only (no API key needed)

Reports, per mode (bm25 / dense / hybrid / hybrid_rerank):
    recall@5      fraction of gold evidence spans found in the top 5 chunks
    MRR@10        mean reciprocal rank of the first relevant chunk
    faithfulness  share of answer claims supported by the retrieved context (LLM judge)
    answer rate   answerable questions that got an answer instead of a refusal
    abstention    unanswerable questions correctly refused
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
os.environ.setdefault("REPORAG_DATA_DIR", str(ROOT / "eval" / ".cache"))

from metrics import (  # noqa: E402
    JUDGE_SYSTEM,
    Span,
    build_judge_prompt,
    parse_judge_score,
    recall_at_k,
    reciprocal_rank,
)
from reporag.config import get_settings, resolve_device  # noqa: E402
from reporag.index_store import RepoIndex  # noqa: E402
from reporag.indexer import index_repository  # noqa: E402
from reporag.ingest.github import parse_github_url  # noqa: E402
from reporag.llm.base import get_llm  # noqa: E402
from reporag.qa import answer_question  # noqa: E402
from reporag.retrieval.dense import get_embedder  # noqa: E402
from reporag.retrieval.pipeline import MODES, Retriever  # noqa: E402
from reporag.retrieval.rerank import get_reranker  # noqa: E402

DATASETS = ROOT / "eval" / "datasets"
MODE_LABELS = {
    "bm25": "BM25 only",
    "dense": "Dense only",
    "hybrid": "Hybrid (RRF)",
    "hybrid_rerank": "Hybrid + rerank",
}


class CachedLLM:
    """Wraps an LLM with an on-disk cache so re-running the eval doesn't re-bill calls."""

    def __init__(self, llm, path: Path):
        self.llm, self.path, self.name = llm, path, llm.name
        self.cache = json.loads(path.read_text()) if path.exists() else {}
        self.model = getattr(llm, "model", "unknown")

    def generate(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        key = hashlib.sha256(f"{self.model}\n{system}\n{prompt}".encode()).hexdigest()
        if key not in self.cache:
            self.cache[key] = self.llm.generate(prompt, system=system, temperature=temperature)
            self.path.write_text(json.dumps(self.cache))
        return self.cache[key]


def load_index(name: str, spec: dict, settings, embedder) -> RepoIndex:
    ref = parse_github_url(spec["url"])
    directory = settings.indexes_dir / ref.key
    if RepoIndex.exists(directory):
        index = RepoIndex.load(directory)
        if index.commit == spec["commit"]:
            return index
    print(f"  indexing {ref.slug}@{spec['commit'][:7]} ...")
    return index_repository(ref, settings, embedder, commit=spec["commit"])


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def evaluate(args) -> dict:
    settings = get_settings()
    device = resolve_device(settings.device)
    embedder = get_embedder(settings.embedding_model, device)
    reranker = get_reranker(settings.reranker_model, device)
    llm = None
    if not args.no_llm:
        base = get_llm("gemini", api_key=settings.gemini_api_key, model=settings.gemini_model)
        llm = CachedLLM(base, Path(settings.data_dir) / "llm_cache.json")

    specs = json.loads((DATASETS / "repos.json").read_text())
    per_question: list[dict] = []
    for name, spec in specs.items():
        print(f"[{name}]")
        index = load_index(name, spec, settings, embedder)
        retriever = Retriever(
            index, embedder, reranker, settings.candidates_per_retriever, settings.rerank_top_n
        )
        questions = [json.loads(line) for line in (DATASETS / spec["questions"]).open()]
        for q in questions:
            gold = [Span(g["path"], g["start"], g["end"]) for g in q["gold"]]
            for mode in MODES:
                row = {"repo": name, "id": q["id"], "mode": mode, "answerable": q["answerable"]}
                started = time.perf_counter()
                results = retriever.retrieve(q["question"], mode=mode, top_k=10)
                row["retrieval_ms"] = (time.perf_counter() - started) * 1000
                ranked = [(r.chunk.path, r.chunk.start_line, r.chunk.end_line) for r in results]
                row["top5"] = [f"{p}:{s}-{e}" for p, s, e in ranked[:5]]
                if gold:
                    row["recall@5"] = recall_at_k(ranked, gold, k=5)
                    row["rr@10"] = reciprocal_rank(ranked, gold, k=10)
                if llm is not None:
                    row.update(judge_answer(q, retriever, llm, mode, settings))
                per_question.append(row)
            print(f"  {q['id']} done")
    return summarize(per_question, settings, llm, device)


def judge_answer(q: dict, retriever: Retriever, llm, mode: str, settings) -> dict:
    answer = answer_question(
        q["question"],
        retriever,
        llm,
        mode=mode,
        top_k=settings.final_top_k,
        min_rerank_score=settings.min_rerank_score,
    )
    out = {"answer": answer.text, "refused": answer.insufficient}
    if answer.insufficient:
        return out
    context = "\n\n".join(
        f"[{n}] {r.chunk.path}:{r.chunk.start_line}-{r.chunk.end_line}\n{r.chunk.text}"
        for n, r in enumerate(answer.retrieved, start=1)
    )
    raw = llm.generate(build_judge_prompt(context, answer.text), system=JUDGE_SYSTEM)
    out["faithfulness"] = parse_judge_score(raw)
    return out


def summarize(rows: list[dict], settings, llm, device: str) -> dict:
    summary = {}
    for mode in MODES:
        mode_rows = [r for r in rows if r["mode"] == mode]
        answerable = [r for r in mode_rows if r["answerable"]]
        unanswerable = [r for r in mode_rows if not r["answerable"]]
        entry = {
            "recall@5": mean([r["recall@5"] for r in answerable]),
            "mrr@10": mean([r["rr@10"] for r in answerable]),
            "retrieval_ms": mean([r["retrieval_ms"] for r in mode_rows]),
        }
        if llm is not None:
            faith = [r["faithfulness"] for r in answerable if r.get("faithfulness") is not None]
            entry["faithfulness"] = mean(faith)
            entry["answer_rate"] = mean([0.0 if r["refused"] else 1.0 for r in answerable])
            entry["abstention"] = mean([1.0 if r["refused"] else 0.0 for r in unanswerable])
        summary[mode] = entry
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": {
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model,
            "llm": getattr(llm, "model", None),
            "device": device,
            "final_top_k": settings.final_top_k,
        },
        "num_questions": len({(r["repo"], r["id"]) for r in rows}),
        "num_answerable": len({(r["repo"], r["id"]) for r in rows if r["answerable"]}),
        "summary": summary,
        "per_question": rows,
    }


def fmt(value: float | None, pct: bool = False) -> str:
    if value is None:
        return "TODO"
    return f"{value * 100:.1f}%" if pct else f"{value:.3f}"


def to_markdown(result: dict) -> str:
    cfg = result["config"]
    lines = [
        f"Run: {result['generated_at']} · {result['num_questions']} questions "
        f"({result['num_answerable']} answerable) · device `{cfg['device']}` · "
        f"LLM `{cfg['llm'] or 'not run'}`",
        "",
        "| Retrieval mode | Recall@5 | MRR@10 | Faithfulness | Answer rate | Abstention "
        "| Retrieval latency |",
        "|---|---|---|---|---|---|---|",
    ]
    for mode, s in result["summary"].items():
        lines.append(
            f"| {MODE_LABELS[mode]} | {fmt(s['recall@5'])} | {fmt(s['mrr@10'])} | "
            f"{fmt(s.get('faithfulness'))} | {fmt(s.get('answer_rate'), pct=True)} | "
            f"{fmt(s.get('abstention'), pct=True)} | {s['retrieval_ms']:.0f} ms |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--no-llm", action="store_true", help="skip generation + faithfulness")
    parser.add_argument("--out", type=Path, default=ROOT / "eval" / "results")
    args = parser.parse_args()

    result = evaluate(args)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(result, indent=2))
    table = to_markdown(result)
    (args.out / "results.md").write_text(table)
    print("\n" + table)


if __name__ == "__main__":
    main()
