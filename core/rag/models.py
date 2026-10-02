"""Embedding and reranking models behind small protocols, so tests can use fakes.

Production uses fastembed (ONNX, CPU, no torch). Models load lazily on first use, because
importing core/ must stay cheap for the API and scripts.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from core.log import get_logger

log = get_logger(__name__)


class Embedder(Protocol):
    name: str

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class Reranker(Protocol):
    name: str

    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


class FastEmbedder:
    """BGE-small via fastembed. Query and passage embeddings use the model's own prefixes.

    `threads` caps ONNX Runtime's thread pool (None = all cores, for offline ingestion).
    """

    def __init__(self, model_name: str, threads: int | None = 2) -> None:
        self.name = model_name
        self.threads = threads
        self._model = None
        self._lock = threading.Lock()

    def _load(self):  # type: ignore[no-untyped-def]
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding

                log.info("embedder.load", extra={"fields": {"model": self.name}})
                self._model = TextEmbedding(
                    self.name, threads=self.threads, enable_cpu_mem_arena=False
                )
        return self._model

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        vectors = list(self._load().passage_embed(list(texts), batch_size=32))
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(next(iter(self._load().query_embed(text))), dtype=np.float32)


class FastEmbedReranker:
    """Cross-encoder ms-marco-MiniLM-L-6-v2 (ONNX build) via fastembed.

    Memory (measured, RSS, 15 queries): batch 8, 2 threads and no CPU memory arena keep
    embedder + reranker flat at ~+290 MB. With the arena on, RSS grows to ~+490 MB as input
    shapes vary, and fastembed's default batch of 64 is worse still. The cost is ~20% more
    latency (~1.2 s per reranked query on CPU).
    """

    def __init__(self, model_name: str, threads: int = 2, batch_size: int = 8) -> None:
        self.name = model_name
        self.threads = threads
        self.batch_size = batch_size
        self._model = None
        self._lock = threading.Lock()

    def _load(self):  # type: ignore[no-untyped-def]
        with self._lock:
            if self._model is None:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                log.info("reranker.load", extra={"fields": {"model": self.name}})
                self._model = TextCrossEncoder(
                    self.name, threads=self.threads, enable_cpu_mem_arena=False
                )
        return self._model

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        scores = self._load().rerank(query, list(documents), batch_size=self.batch_size)
        return [float(s) for s in scores]


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder for tests (no model download)."""

    def __init__(self, dims: int = 64) -> None:
        self.name = f"hash-{dims}"
        self.dims = dims

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dims, dtype=np.float32)
        for token in text.lower().split():
            h = int(hashlib.md5(token.strip(".,;:()").encode()).hexdigest(), 16)
            v[h % self.dims] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dims))

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


class OverlapReranker:
    """Test reranker: scores by query-token overlap."""

    name = "overlap"

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        q = set(query.lower().split())
        return [float(len(q & set(d.lower().split()))) for d in documents]
