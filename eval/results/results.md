Run: 2026-11-01T00:53:57+00:00 · 32 questions (28 answerable) · device `mps` · LLM `not run`

| Retrieval mode | Recall@5 | MRR@10 | Faithfulness | Answer rate | Abstention | Retrieval latency |
|---|---|---|---|---|---|---|
| BM25 only | 0.589 | 0.486 | TODO | TODO | TODO | 1 ms |
| Dense only | 0.518 | 0.335 | TODO | TODO | TODO | 20 ms |
| Hybrid (RRF) | 0.589 | 0.446 | TODO | TODO | TODO | 8 ms |
| Hybrid + rerank | 0.589 | 0.385 | TODO | TODO | TODO | 117 ms |

Per repository (Recall@5 / MRR@10):

| Retrieval mode | itsdangerous | ky |
|---|---|---|
| BM25 only | 0.821 / 0.722 | 0.357 / 0.250 |
| Dense only | 0.821 / 0.542 | 0.214 / 0.129 |
| Hybrid (RRF) | 0.964 / 0.723 | 0.214 / 0.169 |
| Hybrid + rerank | 0.893 / 0.621 | 0.286 / 0.149 |
