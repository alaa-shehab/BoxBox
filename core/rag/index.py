"""The regulations index: committed chunks + embeddings, loaded into Chroma and BM25.

On disk (`data/index/regulations/`), all committed:
    chunks.jsonl     one chunk per line (text + metadata)
    embeddings.npy   float16 passage embeddings, row-aligned with chunks.jsonl
    manifest.json    embedding model, dims, sources, build time

At startup the embeddings go into an in-memory Chroma collection (no re-embedding, and no
dependency on Chroma's on-disk format), and a BM25 index is built over the same chunks.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from core.log import get_logger
from core.rag.chunking import Chunk

log = get_logger(__name__)

_TOKEN = re.compile(r"[a-z]?\d+(?:\.\d+)*[a-z]?|[a-z]+")
_STOP = frozenset(
    (
        "a an and are as at be by for from has have if in into is it its may must not of on "
        "or shall such that the their there these this to under was were which will with"
    ).split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass(frozen=True)
class IndexedChunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    doc_type: str
    year: int
    article: str
    title: str
    page: int
    text: str
    url: str
    subarticles: tuple[str, ...] = ()
    part: int = 0

    def metadata(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "doc_type": self.doc_type,
            "year": self.year,
            "article": self.article,
            "page": self.page,
        }


@dataclass
class RegIndex:
    chunks: list[IndexedChunk]
    embeddings: np.ndarray
    model: str
    built_at: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.chunks) != len(self.embeddings):
            raise ValueError("chunks and embeddings are not aligned")
        self._by_id = {c.chunk_id: i for i, c in enumerate(self.chunks)}
        self._collection = None
        self._bm25 = None

    # --------------------------------------------------------------- persist
    @classmethod
    def from_chunks(
        cls,
        chunks: Sequence[Chunk],
        embeddings: np.ndarray,
        model: str,
        doc_meta: dict[str, dict[str, str]],
    ) -> RegIndex:
        indexed = [
            IndexedChunk(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                doc_title=doc_meta[c.doc_id]["title"],
                doc_type=c.doc_type,
                year=c.year,
                article=c.article,
                title=c.title,
                page=c.page,
                text=c.text,
                url=doc_meta[c.doc_id]["url"],
                subarticles=tuple(c.subarticles),
                part=c.part,
            )
            for c in chunks
        ]
        return cls(
            indexed,
            embeddings,
            model,
            datetime.now(UTC).isoformat(timespec="seconds"),
            [{"doc_id": k, **v} for k, v in doc_meta.items()],
        )

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        np.save(directory / "embeddings.npy", self.embeddings.astype(np.float16))
        manifest = {
            "model": self.model,
            "dims": int(self.embeddings.shape[1]),
            "count": len(self.chunks),
            "built_at": self.built_at,
            "sources": self.sources,
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")

    @classmethod
    def load(cls, directory: Path) -> RegIndex:
        manifest = json.loads((directory / "manifest.json").read_text())
        chunks = []
        with (directory / "chunks.jsonl").open(encoding="utf-8") as f:
            for line in f:
                raw = json.loads(line)
                raw["subarticles"] = tuple(raw.get("subarticles", ()))
                chunks.append(IndexedChunk(**raw))
        embeddings = np.load(directory / "embeddings.npy").astype(np.float32)
        return cls(
            chunks,
            embeddings,
            manifest["model"],
            manifest.get("built_at", ""),
            manifest.get("sources", []),
        )

    # --------------------------------------------------------------- search
    def chunk(self, chunk_id: str) -> IndexedChunk:
        return self.chunks[self._by_id[chunk_id]]

    @property
    def collection(self):  # type: ignore[no-untyped-def]
        if self._collection is None:
            import chromadb

            # In-memory clients share state within a process: give each index its own name.
            client = chromadb.EphemeralClient()
            name = f"regulations-{uuid.uuid4().hex[:12]}"
            col = client.create_collection(
                name, metadata={"hnsw:space": "cosine"}, embedding_function=None
            )
            for start in range(0, len(self.chunks), 1000):
                batch = self.chunks[start : start + 1000]
                col.add(
                    ids=[c.chunk_id for c in batch],
                    embeddings=self.embeddings[start : start + len(batch)].tolist(),
                    metadatas=[c.metadata() for c in batch],
                )
            self._collection = col
            log.info("regindex.chroma_ready", extra={"fields": {"count": len(self.chunks)}})
        return self._collection

    @property
    def bm25(self):  # type: ignore[no-untyped-def]
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi

            corpus = [tokenize(f"{c.article} {c.title} {c.text}") for c in self.chunks]
            self._bm25 = BM25Okapi(corpus)
        return self._bm25

    def dense_search(
        self, query_vec: np.ndarray, k: int, where: dict[str, Any] | None
    ) -> list[tuple[str, float]]:
        res = self.collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=min(k, len(self.chunks)),
            where=where or None,
            include=["distances"],
        )
        ids, dists = res["ids"][0], res["distances"][0]
        return [(i, 1.0 - float(d)) for i, d in zip(ids, dists, strict=True)]

    def bm25_search(self, query: str, k: int, allowed: set[int] | None) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)
        out = []
        for i in order:
            if allowed is not None and int(i) not in allowed:
                continue
            if scores[i] <= 0:
                break
            out.append((self.chunks[int(i)].chunk_id, float(scores[i])))
            if len(out) >= k:
                break
        return out
