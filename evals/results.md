# Evaluation results

<!-- ablation:start -->
### Retrieval ablation

35 hand-labelled questions over the 2025 + 2026 FIA Sporting, Technical and Financial Regulations (`evals/retrieval_gold.json`), filtered by year. Embeddings: `BAAI/bge-small-en-v1.5`; reranker: `ms-marco-MiniLM-L-6-v2` (ONNX). CPU latency, models warm.

| Retrieval | hit@1 | hit@5 | MRR@10 | p50 latency |
|---|---:|---:|---:|---:|
| Dense only (bge-small) | 0.74 | 0.83 | 0.79 | 13 ms |
| BM25 only | 0.54 | 0.83 | 0.68 | 2 ms |
| Hybrid (BM25 + dense, RRF) | 0.71 | 0.89 | 0.79 | 17 ms |
| Hybrid + cross-encoder rerank | 0.86 | 0.94 | 0.90 | 1628 ms |

<!-- ablation:end -->
