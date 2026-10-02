from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.rag.index import IndexedChunk, RegIndex, tokenize
from core.rag.models import HashEmbedder, OverlapReranker
from core.rag.retriever import RegRetriever, rrf


def _chunk(
    doc: str,
    year: int,
    article: str,
    title: str,
    text: str,
    subs: tuple[str, ...] = (),
    part: int = 0,
) -> IndexedChunk:
    return IndexedChunk(
        chunk_id=f"{doc}:{article}:{part}",
        doc_id=doc,
        doc_title=f"{year} {doc}",
        doc_type=doc.split("_")[1],
        year=year,
        article=article,
        title=title,
        page=10,
        text=text,
        url=f"https://example.org/{doc}.pdf",
        subarticles=subs,
        part=part,
    )


CHUNKS = [
    _chunk("2025_sporting", 2025, "55.1", "SAFETY CAR", "the safety car neutralises the race"),
    _chunk("2025_sporting", 2025, "56.1", "VIRTUAL SAFETY CAR", "virtual safety car delta time"),
    _chunk(
        "2025_sporting",
        2025,
        "30.5",
        "Use of Tyres",
        "each driver must use two different dry compounds in the race",
    ),
    _chunk("2025_sporting", 2025, "30.5", "Use of Tyres", "tyre return procedure", part=1),
    _chunk("2025_technical", 2025, "4.1", "Minimum mass", "the car mass must not be below 800kg"),
    _chunk(
        "2026_sporting",
        2026,
        "B6.3",
        "Use & Return of Tyres",
        "B6.3.7 wet weather tyres must be used behind the safety car",
        subs=("B6.3.1", "B6.3.7"),
    ),
    _chunk("2026_technical", 2026, "C4.1", "Minimum mass", "minimum mass is 724kg"),
]


@pytest.fixture
def retriever() -> RegRetriever:
    emb = HashEmbedder()
    index = RegIndex(CHUNKS, emb.embed_passages([c.text for c in CHUNKS]), emb.name)
    return RegRetriever(index, emb, OverlapReranker(), candidates=5, rerank_pool=5)


def test_rrf_math() -> None:
    fused = dict(rrf([["a", "b"], ["b", "c"]], k=60))
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["a"] == pytest.approx(1 / 61)
    assert rrf([["a", "b"], ["b", "c"]])[0][0] == "b"


def test_exact_lookup_bypasses_search(retriever: RegRetriever) -> None:
    hits = retriever.search("What does Article 30.5 say?")
    assert [h.method for h in hits] == ["exact", "exact"]
    assert [h.chunk.part for h in hits] == [0, 1]
    sub = retriever.search("explain B6.3.7")
    assert sub[0].chunk.article == "B6.3" and sub[0].method == "exact"
    # An article that doesn't exist under the filters falls back to normal search.
    assert all(h.method != "exact" for h in retriever.search("Article 30.5", year=2026))


def test_lookup_of_subclause_prefers_unit_text(retriever: RegRetriever) -> None:
    assert retriever.lookup("55.1.2")[0].chunk.article == "55.1"


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid", "hybrid_rerank"])
def test_modes_respect_filters(retriever: RegRetriever, mode: str) -> None:
    hits = retriever.search("minimum mass", k=3, year=2026, mode=mode)  # type: ignore[arg-type]
    assert hits and all(h.chunk.year == 2026 for h in hits)
    assert hits[0].chunk.article == "C4.1"
    tech = retriever.search("minimum mass", k=5, year=2025, doc_type="technical", mode=mode)  # type: ignore[arg-type]
    assert [h.chunk.article for h in tech] == ["4.1"]


def test_hybrid_rerank_orders_by_reranker(retriever: RegRetriever) -> None:
    hits = retriever.search("virtual safety car delta time", k=2, year=2025)
    assert hits[0].chunk.article == "56.1" and hits[0].method == "rerank"
    assert hits[0].citation == "2025 2025_sporting, Art. 56.1, p. 10"


def test_no_matches_for_impossible_filter(retriever: RegRetriever) -> None:
    assert retriever.search("safety car", year=1999) == []


def test_no_reranker_falls_back_to_hybrid() -> None:
    emb = HashEmbedder()
    index = RegIndex(CHUNKS, emb.embed_passages([c.text for c in CHUNKS]), emb.name)
    hits = RegRetriever(index, emb, None).search("safety car", k=2)
    assert hits and all(h.method == "hybrid" for h in hits)


def test_index_roundtrip(tmp_path: Path) -> None:
    emb = HashEmbedder()
    index = RegIndex(CHUNKS, emb.embed_passages([c.text for c in CHUNKS]), emb.name)
    index.save(tmp_path / "idx")
    loaded = RegIndex.load(tmp_path / "idx")
    assert loaded.chunks == CHUNKS and loaded.model == "hash-64"
    assert np.allclose(loaded.embeddings, index.embeddings, atol=1e-3)  # stored as float16


def test_misaligned_index_rejected() -> None:
    with pytest.raises(ValueError):
        RegIndex(CHUNKS, np.zeros((1, 4)), "x")


def test_tokenize_keeps_article_ids() -> None:
    assert tokenize("Article 55.14 of the Safety Car rules") == [
        "article",
        "55.14",
        "safety",
        "car",
        "rules",
    ]
