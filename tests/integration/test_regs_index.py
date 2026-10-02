"""The committed regulations index: shape, exact lookups, and (opt-in) real retrieval."""

from __future__ import annotations

import json
import os

import pytest

from core.rag.factory import INDEX_DIR
from core.rag.index import RegIndex
from core.rag.models import HashEmbedder
from core.rag.retriever import RegRetriever
from core.rag.sources import REG_SOURCES


@pytest.fixture(scope="module")
def index() -> RegIndex:
    return RegIndex.load(INDEX_DIR)


def test_index_covers_every_source(index: RegIndex) -> None:
    docs = {c.doc_id for c in index.chunks}
    assert docs == {s.doc_id for s in REG_SOURCES}
    assert len(index.chunks) > 1500
    assert index.embeddings.shape == (len(index.chunks), 384)
    assert index.model == "BAAI/bge-small-en-v1.5"
    assert all(c.url.startswith("https://www.fia.com/") for c in index.chunks)
    assert all(c.page >= 1 and c.text for c in index.chunks)


@pytest.mark.parametrize(
    ("query", "doc_id", "article", "must_contain"),
    [
        ("Article 30.5", "2025_sporting", "30.5", "two (2) different"),
        ("Art. 55.12", "2025_sporting", "55.12", "safety car"),
        ("B6.3.7", "2026_sporting", "B6.3", "B6.3.7"),
        ("Article C4.1", "2026_technical", "C4.1", "Minimum Mass"),
        ("Article 4.1", "2025_technical", "4.1", "800kg"),
    ],
)
def test_exact_lookup_on_real_index(
    index: RegIndex, query: str, doc_id: str, article: str, must_contain: str
) -> None:
    # Exact lookup never embeds the query, so a fake embedder is enough.
    year = int(doc_id[:4])
    doc_type = doc_id.split("_")[1]
    hits = RegRetriever(index, HashEmbedder(384)).search(query, year=year, doc_type=doc_type)
    assert hits and hits[0].method == "exact"
    assert hits[0].chunk.doc_id == doc_id and hits[0].chunk.article == article
    assert any(must_contain.lower() in h.chunk.text.lower() for h in hits)


def test_ablation_table_is_recorded() -> None:
    results = (INDEX_DIR.parents[2] / "evals" / "results.md").read_text()
    assert "Hybrid + cross-encoder rerank" in results


@pytest.mark.models
@pytest.mark.skipif(not os.environ.get("RUN_MODEL_TESTS"), reason="set RUN_MODEL_TESTS=1")
def test_real_retrieval_finds_gold_articles() -> None:
    from core.rag.factory import get_reg_retriever

    gold = json.loads((INDEX_DIR.parents[2] / "evals" / "retrieval_gold.json").read_text())
    retriever = get_reg_retriever()
    found = 0
    for item in gold["queries"][:10]:
        hits = retriever.search(item["q"], k=5, year=item["year"], exact=False)
        arts = {h.chunk.article for h in hits}
        found += any(
            a == g or (g.endswith(".*") and a.startswith(g[:-1]))
            for a in arts
            for g in item["gold"]
        )
    assert found >= 8
