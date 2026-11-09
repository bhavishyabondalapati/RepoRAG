Run: 2026-11-08T23:57:28+00:00 · 32 questions (28 answerable) · device `mps` · LLM `not run`

| Retrieval mode | Recall@5 | MRR@10 | Retrieval latency |
|---|---|---|---|
| BM25 only | 0.589 | 0.486 | 1 ms |
| Dense only | 0.518 | 0.335 | 14 ms |
| Hybrid (RRF) | 0.589 | 0.446 | 8 ms |
| Hybrid + rerank | 0.589 | 0.385 | 110 ms |

Faithfulness, answer rate, abstention and citation precision/recall are not measured in this run (no LLM calls). Retrieval metrics are the measured results.

Per repository (Recall@5 / MRR@10):

| Retrieval mode | itsdangerous | ky |
|---|---|---|
| BM25 only | 0.821 / 0.722 | 0.357 / 0.250 |
| Dense only | 0.821 / 0.542 | 0.214 / 0.129 |
| Hybrid (RRF) | 0.964 / 0.723 | 0.214 / 0.169 |
| Hybrid + rerank | 0.893 / 0.621 | 0.286 / 0.149 |
