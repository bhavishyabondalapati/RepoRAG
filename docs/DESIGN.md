# RepoRAG design notes

This document explains *why* RepoRAG is built the way it is. Each section covers the problem, what I chose, what I gave up, and a short answer you could say out loud in an interview.

---

## 1. The overall shape: retrieve, then generate

**Problem.** An LLM on its own doesn't know a random repository, and when it doesn't know, it often invents plausible-looking code. That is the worst possible failure for a "questions about my codebase" tool.

**Choice.** Retrieval-Augmented Generation. First find the pieces of the repo that are relevant to the question. Then give *only those* to the LLM and ask it to answer from them, with citations.

**Why it works.** The model's job shrinks from "remember everything" to "read these six excerpts and summarize." Every claim can be traced back to a file and line range, so a user can verify it in one click.

> *Interview answer:* "I used RAG so answers are grounded in the actual repository. The model only sees retrieved excerpts, must cite them, and is allowed to say it doesn't know. That turns a hallucination-prone chatbot into something you can audit."

---

## 2. Chunking: split code the way programmers think about it

**Problem.** Retrieval returns *chunks*, not files. With fixed 60-line windows, a function gets cut in half, so the retriever returns half of the answer and the citation points at a weird range.

**Choice.** Parse Python, JavaScript and TypeScript with **tree-sitter** (a real parser, the same kind editors use) and cut at function and class boundaries.
- If a class is bigger than 120 lines, it becomes a small "header" chunk plus one chunk per method, labeled `ClassName.method`.
- Comments directly above a function (docstrings, JSDoc) travel with it.
- Module-level code between definitions (imports, constants) is grouped into "module" chunks.
- Markdown is split by headings, keeping the heading path (`Install > macOS`) as a label.
- Anything else uses 60-line windows with 10 lines of overlap, so a sentence on a window edge appears in both windows.

Every chunk stores **path, language, symbol name, start line, end line**. A test checks that each chunk's text exactly equals those lines of the original file. Without that guarantee, citations would quietly drift.

**Tradeoff.** You need a grammar per language. Languages without one get windows, which work but are less precise.

> *Interview answer:* "I chunk by syntax, not by size. tree-sitter gives me function and class boundaries, so each chunk is a complete unit and citations match real code. Big classes are split into methods, and everything keeps exact line numbers."

---

## 3. Two retrievers: keyword (BM25) and meaning (embeddings)

**Problem.** People ask about code in two ways:
- with exact names: "how does `unsign` work?"
- with descriptions: "how do signatures expire?"

No single search method is good at both.

**Choice.** Run both and combine them:
- **BM25** is classic keyword scoring: rare words that appear in a chunk count a lot. I wrote a tokenizer that also splits identifiers (`getUserName` → `get`, `user`, `name`; `max_age` → `max`, `age`), so plain English can match code names.
- **Dense retrieval** turns each chunk and the question into a 384-number vector with `bge-small-en-v1.5`. Chunks whose vectors point in a similar direction (cosine similarity) count as similar in meaning. FAISS does the nearest-neighbour search.

**Why a "flat" FAISS index?** A flat index compares the question with every chunk, which is exact. A repo has at most a few thousand chunks, so that takes milliseconds. Approximate indexes (HNSW, IVF) only pay off at millions of vectors, and they add tuning knobs and small accuracy losses.

**What the eval showed.** BM25 was the strongest *single* retriever. Code questions are full of identifiers, which embeddings blur. Dense search alone was the weakest. That is exactly why both are kept.

> *Interview answer:* "Sparse and dense retrieval fail in different places. BM25 nails exact identifiers; embeddings handle paraphrases. My eval showed BM25 alone beats dense alone on code, and the hybrid finds evidence neither finds alone on the smaller repo."

---

## 4. Combining the two lists: Reciprocal Rank Fusion

**Problem.** BM25 scores can be anything from 0 to 30. Cosine similarities sit between -1 and 1. Adding them is meaningless, and normalizing them is fiddly and fragile.

**Choice.** **Reciprocal Rank Fusion (RRF).** Ignore the scores and use only positions. Each chunk gets `1 / (60 + rank)` from each list it appears in, summed. A chunk that ranks well in *both* lists wins. The constant 60 (from the original paper) stops the #1 spot from dominating everything.

**Tradeoff.** RRF throws away how *confident* each retriever was. In return there's nothing to tune, and it is famously hard to beat.

> *Interview answer:* "RRF merges ranked lists using only ranks, `1/(k + rank)`, so I never compare BM25 scores with cosine similarities. It needs no tuning, which matters when you don't have much labelled data."

---

## 5. Reranking with a cross-encoder, and why I report that it didn't help

**What it is.** The embedding model is a *bi-encoder*: question and chunk are encoded **separately**, which is fast because chunk vectors are computed once at index time. A *cross-encoder* reads the question and a chunk **together**, so it can notice that this specific function answers this specific question. That is more accurate but far slower, so it only re-scores the top 30 fused candidates.

**Choice.** `cross-encoder/ms-marco-MiniLM-L-6-v2`, a small, popular reranker.

**What the eval showed.** On this dataset, reranking *lowered* MRR (0.446 → 0.385). The model was trained on MS MARCO, which is Bing search queries and web passages, not code. That is **domain shift**: a component that helps on web text can hurt on code. I kept it in the pipeline because:
1. its score is also used as a relevance threshold for refusing ("nothing relevant found")
2. the four-way comparison is the point of the eval

The UI lets you switch modes to see the difference yourself.

> *Interview answer:* "I added a cross-encoder reranker and measured it. It made ranking worse on code because it was trained on web search data. I'd rather show that honestly than assume 'rerank' always helps. The next step is a code-trained reranker."

---

## 6. Refusing instead of guessing

**Problem.** If retrieval finds nothing useful, the LLM will happily answer from general knowledge, which may be wrong for *this* repo.

**Choice.** Three independent guards. Any one of them returns *"Not enough information in this repo"*:
1. **Before calling the LLM:** if the reranker's best score is below a threshold, nothing relevant was found, so we don't even ask. This saves money too.
2. **The model says so:** the system prompt tells it to reply with exactly `not enough information in this repo` when the excerpts are insufficient.
3. **No citations means no answer:** if the reply contains no valid `[n]` markers, it isn't grounded, so we refuse it.

**Tradeoff.** Sometimes it refuses a question it could have answered. The eval tracks this as **answer rate**, next to **abstention** (correct refusals on deliberately unanswerable questions). Refusing everything would give perfect faithfulness and a terrible answer rate, so you have to look at both.

> *Interview answer:* "The system can say 'I don't know' in three ways: a retrieval threshold, an explicit instruction to the model, and rejecting any answer without citations. I measure both how often it answers and how often it correctly refuses."

---

## 7. The LLM layer: provider-agnostic, with retries

**Choice.** The app depends on one tiny interface: `generate(prompt, system, temperature) -> str`. Gemini is the first implementation. Adding another provider means writing one class.

**Retries.** API calls fail for temporary reasons: rate limits (429), overloaded servers (503), timeouts. These are retried with **exponential backoff** (wait 1s, 2s, 4s, … capped at 30s) plus **jitter** (±25% randomness, so many clients don't retry at the same instant). Permanent errors, like a bad request or a wrong API key, are **not** retried, because retrying can't fix them.

**Temperature 0.** We want the same, most likely answer every time, not creativity.

> *Interview answer:* "The LLM sits behind a one-method interface so the provider is swappable. Calls retry only on transient errors, with exponential backoff and jitter to avoid thundering herds."

---

## 8. Indexing in the background with progress

**Problem.** Cloning and embedding a repo takes seconds to a minute. An HTTP request shouldn't hang that long.

**Choice.** `POST /api/repos` returns immediately with a job id (HTTP 202 "Accepted"). A background worker clones, chunks and embeds, updating `status` and `progress` as it goes, and the UI polls `GET /api/repos/{id}`. Finished indexes are saved to disk (`chunks.jsonl`, `dense.faiss`, `meta.json`) and reloaded after a restart. BM25 is rebuilt on load because that's faster than storing it.

**Concurrency.** There is only **one** indexing worker, and one lock around model inference. Embedding is memory- and GPU-heavy, and Apple's MPS backend isn't safe to drive from several threads at once. Requests queue instead of crashing.

**Tradeoff.** Job state lives in memory in one process. Fine for a local tool. A multi-user deployment would use a job queue (Celery/RQ) and a shared store.

---

## 9. Safety and limits

- **URL validation.** Only `https://github.com/<owner>/<repo>` (optionally `.git`) is accepted, using GitHub's own naming rules. That rejects other hosts, `http://`, `user:pass@` credentials, deep links, query strings, and characters like `;`, so nothing dangerous can reach the `git` command. The test suite includes these malicious inputs.
- **Never prompt for credentials.** `GIT_TERMINAL_PROMPT=0` makes private repos fail fast instead of hanging.
- **Size limits.** The GitHub API size is checked before cloning (default max 50 MB). Clones are shallow (`--depth 1`). Only known source/doc/config file types are indexed, with at most 2,000 files of up to 200 KB each. `node_modules`, build output, lockfiles and minified bundles are skipped.
- **Secrets.** Only `.env.example` (placeholders) is committed. `.env`, indexes, clones and `node_modules` are git-ignored, and Docker images exclude `.env`.

---

## 10. Evaluation: measure, don't assume

**Dataset.** 32 hand-written questions over two repos pinned to exact commits (so results are reproducible):
- `pallets/itsdangerous`: small, clean Python
- `sindresorhus/ky`: TypeScript with very large test files

For 28 questions I labelled the **gold evidence** (file + line range) by reading the code, and a script verified every span. Four questions are deliberately unanswerable.

**Metrics.**
- **Recall@5:** of the evidence needed, how much showed up in the top 5? A chunk counts if it's in the same file and its lines overlap the gold range.
- **MRR@10:** how high did the first relevant chunk rank? Rank 1 scores 1, rank 2 scores 0.5, and so on.
- **Faithfulness (LLM judge):** a second LLM call splits the answer into claims and checks each against the retrieved context. Score = supported / total. LLM judges are convenient but imperfect, so the prompt forbids outside knowledge and the raw JSON is parsed strictly.
- **Answer rate / abstention:** see section 6.

**Rules I followed.** Numbers in the README come only from real runs of `eval/run_eval.py`. Anything that couldn't be run (faithfulness needs an API key) is marked TODO. I did **not** tune the system on this dataset after seeing the results. With only 32 questions that would just overfit, so the weaknesses it exposed are listed as limitations and future work.

> *Interview answer:* "I built a small labelled eval with gold file-and-line evidence and compared four retrieval setups. Hybrid search won on the clean repo; the web-trained reranker hurt; and big test files were the main failure mode. I reported that instead of tuning to the test set."

---

## 11. Frontend

A two-pane "workbench":
- **Left:** the conversation.
- **Right:** a source viewer showing the cited lines with their real line numbers and a link to the exact range on GitHub, pinned to the indexed commit.

`[n]` markers in the answer become clickable chips. A mode switch (hybrid + rerank / hybrid / dense / BM25) shows the eval's comparison live. Indexing progress shows each stage: clone → chunk → embed → ready.

The look is deliberately plain: paper-and-ink colors, IBM Plex Sans/Mono, thin rules instead of shadowed cards, one accent color. It supports light and dark mode and works at phone width.
