Run: 2026-11-01T14:28:13+00:00 · 20 questions (18 answerable) · device `mps` · LLM `not run`

| Retrieval mode | Recall@5 | MRR@10 | Faithfulness | Answer rate | Abstention | Retrieval latency |
|---|---|---|---|---|---|---|
| BM25 only | 0.722 | 0.687 | TODO | TODO | TODO | 1 ms |
| Dense only | 0.667 | 0.631 | TODO | TODO | TODO | 24 ms |
| Hybrid (RRF) | 0.722 | 0.667 | TODO | TODO | TODO | 8 ms |
| Hybrid + rerank | 0.667 | 0.639 | TODO | TODO | TODO | 109 ms |

Per repository (Recall@5 / MRR@10):

| Retrieval mode | itsdangerous | ky |
|---|---|---|
| BM25 only | 1.000 / 0.911 | 0.444 / 0.463 |
| Dense only | 1.000 / 0.917 | 0.333 / 0.344 |
| Hybrid (RRF) | 1.000 / 0.944 | 0.444 / 0.389 |
| Hybrid + rerank | 1.000 / 0.944 | 0.333 / 0.333 |
