"""Build the production regulations retriever once per process (lazy, thread-safe)."""

from __future__ import annotations

import threading
from pathlib import Path

from core.config import REPO_ROOT, get_settings
from core.rag.index import RegIndex
from core.rag.models import FastEmbedder, FastEmbedReranker
from core.rag.retriever import RegRetriever

INDEX_DIR = REPO_ROOT / "data" / "index" / "regulations"
_lock = threading.Lock()
_retriever: RegRetriever | None = None


def get_reg_retriever(index_dir: Path = INDEX_DIR, rerank: bool | None = None) -> RegRetriever:
    global _retriever
    with _lock:
        if _retriever is None:
            s = get_settings()
            rerank = s.rerank_enabled if rerank is None else rerank
            index = RegIndex.load(index_dir)
            # Queries must be embedded with the model that embedded the passages.
            embedder = FastEmbedder(index.model, threads=s.onnx_threads)
            reranker = (
                FastEmbedReranker(s.reranker_model, threads=s.onnx_threads) if rerank else None
            )
            _retriever = RegRetriever(index, embedder, reranker)
        return _retriever
