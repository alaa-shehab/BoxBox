"""Download the FIA regulations, chunk them by article, embed them and save the index.

    python scripts/ingest_regs.py            # -> data/index/regulations/

PDFs are cached in data/raw/regulations/ (not committed). The derived index is committed,
so the app never embeds documents at runtime. Downloads respect FIA's robots.txt
(Crawl-delay: 10).
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

from core.config import REPO_ROOT, get_settings
from core.log import configure_logging, get_logger
from core.rag.chunking import Chunk, chunk_document
from core.rag.index import RegIndex
from core.rag.models import FastEmbedder
from core.rag.sources import REG_SOURCES, RegSource

log = get_logger("ingest_regs")
RAW_DIR = REPO_ROOT / "data" / "raw" / "regulations"
INDEX_DIR = REPO_ROOT / "data" / "index" / "regulations"
USER_AGENT = "PitWall-fan-project/0.1 (+https://github.com/alaa-shehab/BoxBox)"
CRAWL_DELAY_S = 10


def download(sources: list[RegSource], raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    fetched = 0
    for src in sources:
        dst = raw_dir / src.filename
        if dst.exists():
            continue
        if fetched:
            time.sleep(CRAWL_DELAY_S)
        req = urllib.request.Request(src.url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=120) as resp:
            dst.write_bytes(resp.read())
        fetched += 1
        print(f"downloaded {dst.name}")


def extract_pages(pdf: Path) -> list[str]:
    from pypdf import PdfReader

    return [page.extract_text() or "" for page in PdfReader(str(pdf)).pages]


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=INDEX_DIR)
    args = parser.parse_args(argv)
    configure_logging(settings.log_level, settings.log_json)

    sources = list(REG_SOURCES)
    download(sources, RAW_DIR)
    chunks: list[Chunk] = []
    for src in sources:
        doc_chunks = chunk_document(src, extract_pages(RAW_DIR / src.filename))
        units = len({c.article for c in doc_chunks})
        print(f"{src.doc_id}: {units} articles, {len(doc_chunks)} chunks")
        chunks += doc_chunks

    doc_meta = {s.doc_id: {"title": s.title, "url": s.url} for s in sources}
    embedder = FastEmbedder(settings.embedding_model, threads=None)  # offline: use all cores
    start = time.perf_counter()
    embeddings = embedder.embed_passages(
        [c.embed_text(doc_meta[c.doc_id]["title"]) for c in chunks]
    )
    print(f"embedded {len(chunks)} chunks in {time.perf_counter() - start:.0f}s")
    RegIndex.from_chunks(chunks, embeddings, embedder.name, doc_meta).save(args.out)
    print(f"saved index -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
