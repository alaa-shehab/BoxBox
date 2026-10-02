"""Regulations retrieval: exact article lookup, else BM25 + dense with RRF, then rerank.

    query --(mentions "Article 30.5" / "B6.3.7")--> exact lookup (no semantic search)
          \\-> BM25 top-N  \\
                           RRF (k=60) -> top-M -> cross-encoder rerank -> top-k
          \\-> dense top-N /

`mode` exposes each stage for the ablation: "dense", "bm25", "hybrid", "hybrid_rerank".
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal

from core.log import get_logger
from core.rag.chunking import find_article_refs
from core.rag.index import IndexedChunk, RegIndex
from core.rag.models import Embedder, Reranker

log = get_logger(__name__)

Mode = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]
RRF_K = 60


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: IndexedChunk
    score: float
    method: str  # "exact" | "dense" | "bm25" | "hybrid" | "rerank"

    @property
    def citation(self) -> str:
        c = self.chunk
        return f"{c.doc_title}, Art. {c.article}, p. {c.page}"


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class RegRetriever:
    def __init__(
        self,
        index: RegIndex,
        embedder: Embedder,
        reranker: Reranker | None = None,
        candidates: int = 30,
        rerank_pool: int = 20,
    ) -> None:
        self.index = index
        self.embedder = embedder
        self.reranker = reranker
        self.candidates = candidates
        self.rerank_pool = rerank_pool

    @staticmethod
    def _matches(c: IndexedChunk, year: int | None, doc_type: str | None) -> bool:
        return (year is None or c.year == year) and (doc_type is None or c.doc_type == doc_type)

    def lookup(
        self, article: str, year: int | None = None, doc_type: str | None = None
    ) -> list[RetrievedChunk]:
        """Exact article lookup: the unit itself, or the unit containing a sub-clause."""
        ref = article.upper()
        hits = [
            c
            for c in self.index.chunks
            if self._matches(c, year, doc_type)
            and (c.article == ref or ref in c.subarticles or ref.startswith(c.article + "."))
        ]
        # Prefer the precise unit over a parent that merely contains the clause.
        # and, within a long article, the part that actually contains the clause.
        hits.sort(key=lambda c: (c.article != ref and ref not in c.text, c.year, c.doc_id, c.part))
        return [RetrievedChunk(c, 1.0, "exact") for c in hits]

    def search(
        self,
        query: str,
        k: int = 5,
        year: int | None = None,
        doc_type: str | None = None,
        mode: Mode = "hybrid_rerank",
        exact: bool = True,
    ) -> list[RetrievedChunk]:
        if exact:
            refs = find_article_refs(query)
            found: list[RetrievedChunk] = []
            for ref in refs:
                found += self.lookup(ref, year, doc_type)
            if found:
                return found[:k]

        allowed = {i for i, c in enumerate(self.index.chunks) if self._matches(c, year, doc_type)}
        if not allowed:
            return []
        n = max(self.candidates, k)
        dense: list[tuple[str, float]] = []
        sparse: list[tuple[str, float]] = []
        if mode != "bm25":
            where = self._where(year, doc_type)
            dense = self.index.dense_search(self.embedder.embed_query(query), n, where)
        if mode != "dense":
            sparse = self.index.bm25_search(query, n, allowed)

        if mode == "dense":
            return [RetrievedChunk(self.index.chunk(i), s, "dense") for i, s in dense[:k]]
        if mode == "bm25":
            return [RetrievedChunk(self.index.chunk(i), s, "bm25") for i, s in sparse[:k]]

        fused = rrf([[i for i, _ in dense], [i for i, _ in sparse]])
        if mode == "hybrid" or self.reranker is None:
            return [RetrievedChunk(self.index.chunk(i), s, "hybrid") for i, s in fused[:k]]

        pool = [self.index.chunk(i) for i, _ in fused[: self.rerank_pool]]
        scores = self.reranker.score(query, [f"{c.title}. {c.text}" for c in pool])
        ranked = sorted(zip(pool, scores, strict=True), key=lambda cs: -cs[1])
        return [RetrievedChunk(c, s, "rerank") for c, s in ranked[:k]]

    @staticmethod
    def _where(year: int | None, doc_type: str | None) -> dict[str, Any] | None:
        clauses: list[dict[str, Any]] = []
        if year is not None:
            clauses.append({"year": year})
        if doc_type is not None:
            clauses.append({"doc_type": doc_type})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}
