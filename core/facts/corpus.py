"""A small cited corpus of driver and team history from Wikipedia (CC BY-SA 4.0).

Built offline by `scripts/build_facts.py` and committed:
    data/facts/wiki/passages.jsonl   passages with subject, section, permalink (oldid)
    data/facts/wiki/embeddings.npy   float16 bge-small embeddings, row-aligned
    data/facts/wiki/manifest.json

Retrieval is per subject (a driver code or team key): BM25 + dense cosine fused with RRF.
That's plenty for at most ~30 passages per subject, so this index doesn't need Chroma.
Each passage records the latest year it mentions (`max_year`), so a replay of an old race
can avoid "future" passages (spoiler safety).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from core.rag.index import tokenize
from core.rag.models import Embedder
from core.rag.retriever import rrf

LICENSE = "CC BY-SA 4.0"
_YEAR = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")
SKIP_SECTIONS = re.compile(
    r"references|external links|see also|notes|footnotes|bibliography|further reading|"
    r"record|results|career summary|statistics|helmet|sponsor|filmography|discography|"
    r"awards|honours|honors|in popular culture|video games?",
    re.IGNORECASE,
)
MIN_CHARS, TARGET_CHARS, MAX_PASSAGES = 120, 700, 30


@dataclass(frozen=True)
class WikiPassage:
    passage_id: str
    subject_type: str  # "driver" | "team"
    subject: str  # driver code or team lineage key
    subject_name: str
    title: str
    section: str
    text: str
    url: str  # permalink to the exact revision used
    max_year: int  # latest year mentioned (0 if none)

    @property
    def citation_label(self) -> str:
        where = f", section '{self.section}'" if self.section else ""
        return f"Wikipedia: {self.title}{where} ({LICENSE})"


def max_year(text: str) -> int:
    years = [int(y) for y in _YEAR.findall(text)]
    return max(years) if years else 0


def split_article(extract: str) -> list[tuple[str, str]]:
    """(section, paragraph) pairs from a plain-text extract with '== Heading ==' markers."""
    section = ""
    skip = False
    out: list[tuple[str, str]] = []
    for block in re.split(r"\n\s*\n|\n(?==)", extract):
        block = block.strip()
        heading = re.match(r"^(=+)\s*(.+?)\s*\1$", block.splitlines()[0]) if block else None
        if heading:
            level, name = len(heading.group(1)), heading.group(2)
            if level == 2:
                section, skip = name, bool(SKIP_SECTIONS.search(name))
            elif SKIP_SECTIONS.search(name):
                skip = True
            block = "\n".join(block.splitlines()[1:]).strip()
        if skip or not block:
            continue
        out.append((section, re.sub(r"\s+", " ", block)))
    return out


def passages_from_article(
    subject_type: str, subject: str, subject_name: str, title: str, url: str, extract: str
) -> list[WikiPassage]:
    merged: list[tuple[str, str]] = []
    for section, para in split_article(extract):
        if merged and merged[-1][0] == section and len(merged[-1][1]) + len(para) < TARGET_CHARS:
            merged[-1] = (section, merged[-1][1] + " " + para)
        else:
            merged.append((section, para))
    out = []
    for section, text in merged:
        if len(text) < MIN_CHARS:
            continue
        out.append(
            WikiPassage(
                passage_id=f"wiki:{subject}:{len(out)}",
                subject_type=subject_type,
                subject=subject,
                subject_name=subject_name,
                title=title,
                section=section,
                text=text[:1500],
                url=url,
                max_year=max_year(text),
            )
        )
        if len(out) >= MAX_PASSAGES:
            break
    return out


class WikiCorpus:
    def __init__(self, passages: Sequence[WikiPassage], embeddings: np.ndarray, model: str) -> None:
        if len(passages) != len(embeddings):
            raise ValueError("passages and embeddings are not aligned")
        self.passages = list(passages)
        self.embeddings = embeddings.astype(np.float32)
        self.model = model
        self._by_subject: dict[str, list[int]] = {}
        for i, p in enumerate(self.passages):
            self._by_subject.setdefault(p.subject, []).append(i)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "passages.jsonl").open("w", encoding="utf-8") as f:
            for p in self.passages:
                f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
        np.save(directory / "embeddings.npy", self.embeddings.astype(np.float16))
        subjects = sorted(self._by_subject)
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "model": self.model,
                    "license": LICENSE,
                    "count": len(self.passages),
                    "subjects": subjects,
                },
                indent=1,
            )
            + "\n"
        )

    @classmethod
    def load(cls, directory: Path) -> WikiCorpus:
        manifest = json.loads((directory / "manifest.json").read_text())
        with (directory / "passages.jsonl").open(encoding="utf-8") as f:
            passages = [WikiPassage(**json.loads(line)) for line in f]
        return cls(passages, np.load(directory / "embeddings.npy"), manifest["model"])

    def subjects(self) -> set[str]:
        return set(self._by_subject)

    def for_subject(self, subject: str, before_year: int | None = None) -> list[WikiPassage]:
        out = [self.passages[i] for i in self._by_subject.get(subject, [])]
        if before_year is not None:
            out = [p for p in out if p.max_year < before_year]
        return out

    def search(
        self,
        subject: str,
        query: str,
        embedder: Embedder | None = None,
        k: int = 3,
        before_year: int | None = None,
    ) -> list[WikiPassage]:
        idx = [
            i
            for i in self._by_subject.get(subject, [])
            if before_year is None or self.passages[i].max_year < before_year
        ]
        if not idx:
            return []
        q = set(tokenize(query))
        lexical = sorted(idx, key=lambda i: -len(q & set(tokenize(self.passages[i].text))))
        rankings = [[self.passages[i].passage_id for i in lexical]]
        if embedder is not None:
            qv = embedder.embed_query(query)
            sims = self.embeddings[idx] @ qv
            dense = [idx[j] for j in np.argsort(-sims)]
            rankings.append([self.passages[i].passage_id for i in dense])
        by_id = {self.passages[i].passage_id: self.passages[i] for i in idx}
        return [by_id[pid] for pid, _ in rrf(rankings)[:k]]
