"""Article-aware chunking of FIA regulation PDFs.

The FIA documents number their clauses in two styles:
    2025:  "34) PIT ENTRY ROAD" (article) -> "34.1 ..." (clause)
    2026:  "B5.10 Formation Lap(s)"       -> "B5.10.1 ..." (clause), section letter prefix

One chunk is one *unit*: an id with exactly two numeric parts ("34.1", "10.2", "B5.10",
"D1.2"), together with every deeper clause beneath it ("B5.10.1", "B5.10.2"). That keeps a
rule and its sub-clauses together, and gives exact-article lookup a stable target. Long
units are split into parts on clause boundaries.

The parser guards against tables of contents, page headers and numbered list items in the
body: a new unit id is accepted only if it moves forward in document order.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field

from core.rag.sources import RegSource

MAX_CHARS = 1800

_ID = re.compile(
    r"^\s*(?P<id>(?P<letter>[A-F])?(?P<nums>\d{1,3}(?:\.\d{1,3}){0,4}))(?P<punct>[).:]?)"
    r"\s+(?P<rest>\S.*)$"
)
_HEADER_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"F[ée]d[ée]ration Internationale de l",
        r"^\s*Issue\s+\d+\s*$",
        r"^\s*\d{1,2}\s+(January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+20\d\d\s*$",
        r"Formula 1 (Sporting|Technical|Financial) Regulations\s+\d+(/\d+)?",
        r"Formula 1 Financial Regulations\s+\d+",
        r"Formula 1 Regulations - Section [A-F]",
        r"^\s*SECTION [A-F]:",
        r"^\s*[A-F]?\d{0,3}\s*$",  # lone page numbers / section letters ("0", "B", "B41")
    )
]
_TOC_LINE = re.compile(r"^\s*[A-F]?\d{1,3}(?:\.\d{1,3})*[).]?\s+.+?\s\d{1,3}\s*$")
_HYPHEN_GAP = re.compile(r"(\w) -(\w)")
_SPACES = re.compile(r"[ \t]+")
ARTICLE_REF = re.compile(
    r"(?:\bart(?:icle)?s?\.?\s*)(?P<id>[A-F]?\d{1,3}(?:\.\d{1,3}){1,4})"
    r"|\b(?P<sid>[A-F]\d{1,3}(?:\.\d{1,3}){1,4})\b",
    re.IGNORECASE,
)


@dataclass
class Chunk:
    doc_id: str
    doc_type: str
    year: int
    article: str  # unit id, e.g. "34.1" or "B5.10"
    title: str  # unit heading if present, else the parent article's title
    parent_title: str
    page: int  # 1-based page where the unit starts
    text: str
    subarticles: list[str] = field(default_factory=list)
    part: int = 0

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}:{self.article}:{self.part}"

    def embed_text(self, doc_title: str) -> str:
        head = f"{doc_title}, Article {self.article}"
        if self.title:
            head += f" ({self.title})"
        return f"{head}\n{self.text}"


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _HYPHEN_GAP.sub(r"\1-\2", text)
    return _SPACES.sub(" ", text).strip()


def _is_header(line: str) -> bool:
    return any(p.search(line) for p in _HEADER_PATTERNS)


def _is_toc_page(lines: list[str]) -> bool:
    content = [ln for ln in lines if ln.strip()]
    if len(content) < 8:
        return False
    toc = sum(1 for ln in content if _TOC_LINE.match(ln))
    return toc / len(content) >= 0.4


def _looks_like_heading(text: str) -> bool:
    """Short title-case text without sentence punctuation: "Formation Lap(s)", "SAFETY CAR"."""
    text = text.strip()
    return (
        0 < len(text) <= 80
        and text[0].isupper()
        and not text.endswith((".", ",", ";", ":", ")"))
        and not re.search(r"\b(must|will|shall|may)\b", text)
    )


def _key(nums: str) -> tuple[int, ...]:
    return tuple(int(n) for n in nums.split("."))


@dataclass
class _Unit:
    article: str
    key: tuple[int, int]
    page: int
    parent_title: str
    title: str = ""
    lines: list[str] = field(default_factory=list)
    subarticles: list[str] = field(default_factory=list)


_ARTICLE_HEADING = re.compile(
    r"^\s*ARTICLE\s+(?P<letter>[A-F])?(?P<num>\d{1,3})\s*[:.\-]\s+"
    r"(?P<title>[A-Z][A-Za-z ,&()/'\-]+?)(?:\s+\d{1,3})?\s*$"
)


def _plausible(key: tuple[int, int], cur: tuple[int, int], seen_with_body: bool) -> bool:
    """A new unit must move forward near the current chapter. Going back to an x.1 section
    is allowed only if that section hasn't been seen with real text yet, which is what
    happens when the body starts after a table of contents."""
    if key == cur:
        return False
    if key > cur:
        return key[0] <= cur[0] + 2 or (key[1] == 1 and key[0] <= cur[0] + 5)
    return key[1] == 1 and not seen_with_body


def parse_units(pages: Iterable[str], letter: str = "") -> list[_Unit]:
    """Split pages into units. `letter` is the document's section prefix ("B" for the 2026
    Sporting Regulations, "" for the 2025 documents). Ids with any other prefix are
    cross-references, not units. Duplicate ids (from a TOC) keep the fullest unit."""
    units: list[_Unit] = []
    latest: dict[str, _Unit] = {}
    current: _Unit | None = None
    parent_title = ""
    for page_no, raw in enumerate(pages, start=1):
        lines = raw.splitlines()
        if _is_toc_page(lines):
            continue
        for line in lines:
            if not line.strip() or _is_header(line):
                continue
            heading = _ARTICLE_HEADING.match(line)
            if heading and heading.group("title") and (heading.group("letter") or "") == letter:
                parent_title = _clean(heading.group("title"))
                continue
            m = _ID.match(line)
            if m and (m.group("letter") or "") == letter:
                nums, rest = _key(m.group("nums")), m.group("rest").strip()
                if len(nums) == 1:
                    if rest[:1].isupper() and (
                        m.group("punct") in (")", ".") or letter or rest.isupper()
                    ):
                        parent_title = _clean(re.sub(r"\s+\d{1,3}$", "", rest))
                        continue
                else:
                    unit_key = (nums[0], nums[1])
                    sub = f"{letter}{m.group('nums')}"
                    if current and unit_key == current.key and len(nums) >= 3:
                        current.subarticles.append(sub)
                        current.lines.append(line)
                        continue
                    cur_key = current.key if current else (0, 0)
                    article = f"{letter}{nums[0]}.{nums[1]}"
                    prior = latest.get(article)
                    seen = prior is not None and sum(len(x) for x in prior.lines) >= 200
                    if current is None or _plausible(unit_key, cur_key, seen):
                        current = _Unit(article, unit_key, page_no, parent_title)
                        latest[article] = current
                        if len(nums) >= 3:
                            current.subarticles.append(sub)
                            current.lines.append(line)
                        elif _looks_like_heading(rest):
                            current.title = _clean(re.sub(r"\s+\d{1,3}$", "", rest))
                        else:
                            current.lines.append(line)
                        units.append(current)
                        continue
            if current is not None:
                current.lines.append(line)

    best: dict[str, _Unit] = {}
    for unit in units:
        size = sum(len(ln) for ln in unit.lines)
        kept = best.get(unit.article)
        if kept is None or size > sum(len(ln) for ln in kept.lines):
            best[unit.article] = unit
    return [u for u in units if best.get(u.article) is u]


def _split(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    # Prefer splitting before sub-clause ids, then at sentence ends.
    pieces = re.split(r"(?=\s[A-F]?\d{1,3}\.\d{1,3}\.\d{1,3}\s)|(?<=\.)\s+", text)
    parts, buf = [], ""
    for piece in pieces:
        if buf and len(buf) + len(piece) + 1 > max_chars:
            parts.append(buf.strip())
            buf = ""
        buf += (" " if buf else "") + piece.strip()
        while len(buf) > max_chars:  # a single enormous sentence
            parts.append(buf[:max_chars])
            buf = buf[max_chars:]
    if buf.strip():
        parts.append(buf.strip())
    return parts


def chunk_document(source: RegSource, pages: Iterable[str]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for unit in parse_units(pages, letter=source.section_letter):
        body = _clean(" ".join(unit.lines))
        if not body:
            continue
        for part, text in enumerate(_split(body, MAX_CHARS)):
            chunks.append(
                Chunk(
                    doc_id=source.doc_id,
                    doc_type=source.doc_type,
                    year=source.year,
                    article=unit.article,
                    title=unit.title or unit.parent_title,
                    parent_title=unit.parent_title,
                    page=unit.page,
                    text=text,
                    subarticles=list(dict.fromkeys(unit.subarticles)),
                    part=part,
                )
            )
    return chunks


def find_article_refs(query: str) -> list[str]:
    """Article ids explicitly referenced in a query ("Article 30.5", "Art. B6.3.7", "B5.10")."""
    out = []
    for m in ARTICLE_REF.finditer(query):
        ref = (m.group("id") or m.group("sid")).upper()
        out.append(ref)
    return list(dict.fromkeys(out))
