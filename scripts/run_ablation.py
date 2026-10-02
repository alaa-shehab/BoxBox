"""Retrieval ablation: dense vs BM25 vs hybrid (RRF) vs hybrid + cross-encoder rerank.

    python scripts/run_ablation.py     # writes the table into evals/results.md

Metrics are article-level: chunks from the same article count once.
    hit@1, hit@5   share of queries with a gold article in the top 1 / top 5
    MRR@10         mean reciprocal rank of the first gold article in the top 10
    p50 ms         median query latency (models warm)
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

from core.config import REPO_ROOT
from core.rag.factory import get_reg_retriever
from core.rag.retriever import Mode, RegRetriever

GOLD = REPO_ROOT / "evals" / "retrieval_gold.json"
RESULTS = REPO_ROOT / "evals" / "results.md"
MODES: tuple[Mode, ...] = ("dense", "bm25", "hybrid", "hybrid_rerank")
START, END = "<!-- ablation:start -->", "<!-- ablation:end -->"


def is_gold(article: str, gold: list[str]) -> bool:
    return any(article == g or (g.endswith(".*") and article.startswith(g[:-1])) for g in gold)


def ranked_articles(retriever: RegRetriever, q: str, year: int, mode: Mode) -> list[str]:
    hits = retriever.search(q, k=20, year=year, mode=mode, exact=False)
    return list(dict.fromkeys(h.chunk.article for h in hits))[:10]


def evaluate(retriever: RegRetriever, queries: list[dict], mode: Mode) -> dict[str, float]:
    hit1 = hit5 = rr = 0.0
    latencies = []
    for item in queries:
        start = time.perf_counter()
        arts = ranked_articles(retriever, item["q"], item["year"], mode)
        latencies.append((time.perf_counter() - start) * 1000)
        ranks = [i for i, a in enumerate(arts, 1) if is_gold(a, item["gold"])]
        first = ranks[0] if ranks else None
        hit1 += first == 1
        hit5 += first is not None and first <= 5
        rr += 1.0 / first if first else 0.0
    n = len(queries)
    return {
        "hit@1": hit1 / n,
        "hit@5": hit5 / n,
        "MRR@10": rr / n,
        "p50_ms": statistics.median(latencies),
    }


def render(results: dict[str, dict[str, float]], n: int, model: str) -> str:
    rows = [
        "| Retrieval | hit@1 | hit@5 | MRR@10 | p50 latency |",
        "|---|---:|---:|---:|---:|",
    ]
    names = {
        "dense": "Dense only (bge-small)",
        "bm25": "BM25 only",
        "hybrid": "Hybrid (BM25 + dense, RRF)",
        "hybrid_rerank": "Hybrid + cross-encoder rerank",
    }
    for mode, m in results.items():
        rows.append(
            f"| {names[mode]} | {m['hit@1']:.2f} | {m['hit@5']:.2f} | "
            f"{m['MRR@10']:.2f} | {m['p50_ms']:.0f} ms |"
        )
    return (
        f"{START}\n### Retrieval ablation\n\n"
        f"{n} hand-labelled questions over the 2025 + 2026 FIA Sporting, Technical and "
        f"Financial Regulations (`evals/retrieval_gold.json`), filtered by year. Embeddings: "
        f"`{model}`; reranker: `ms-marco-MiniLM-L-6-v2` (ONNX). CPU latency, models warm.\n\n"
        + "\n".join(rows)
        + f"\n\n{END}"
    )


def write_section(section: str, path: Path = RESULTS) -> None:
    text = path.read_text() if path.exists() else "# Evaluation results\n"
    if START in text:
        before, rest = text.split(START, 1)
        after = rest.split(END, 1)[1] if END in rest else ""
        text = before + section + after
    else:
        text = text.rstrip() + "\n\n" + section + "\n"
    path.write_text(text)


def main() -> int:
    queries = json.loads(GOLD.read_text())["queries"]
    retriever = get_reg_retriever()
    retriever.search("warm up", k=1, mode="hybrid_rerank", exact=False)  # load models
    results = {mode: evaluate(retriever, queries, mode) for mode in MODES}
    for mode, m in results.items():
        print(f"{mode:14} " + "  ".join(f"{k}={v:.2f}" for k, v in m.items()))
    write_section(render(results, len(queries), retriever.index.model))
    print(f"wrote {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
