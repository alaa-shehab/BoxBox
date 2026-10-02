from __future__ import annotations

from core.rag.chunking import MAX_CHARS, _split, chunk_document, find_article_refs
from core.rag.sources import RegSource

S2025 = RegSource("sporting", 2025, "5", "https://example.org/2025.pdf")
S2026 = RegSource("sporting", 2026, "9", "https://example.org/2026.pdf")

TOC = "\n".join(
    [
        "2025 Formula 1 Sporting Regulations 2/119 30 April 2025",
        "©2025 Fédération Internationale de l'Automobile Issue 5",
    ]
    + [f"{a}) TITLE {a} {10 + a}" for a in range(1, 12)]
)
P1 = """2025 Formula 1 Sporting Regulations 64/119 30 April 2025
©2025 Fédération Internationale de l'Automobile Issue 5
54) INCIDENTS DURING THE RACE
54.3 The stewards may impose any one of the penalties below on any driver involved in an
Incident: a) A five (5) second time penalty.
55) SAFETY CAR
55.1 The FIA safety car will be driven by an FIA appointed safety car driver (see Article 55.
14 below).
55.2 The safety car will be brought into operation to neutralise a race.
"""
P2 = """64
55.3 All cars must line up behind the safety car no more than ten car lengths apart.
1. this numbered list item must stay inside 55.3
56) VOID
"""


def test_2025_style_units_titles_and_pages() -> None:
    chunks = chunk_document(S2025, [TOC, P1, P2])
    by_article = {c.article: c for c in chunks}
    assert list(by_article) == ["54.3", "55.1", "55.2", "55.3"]
    assert by_article["55.1"].title == "SAFETY CAR"  # wrapped cross-ref is not a heading
    assert "14 below" in by_article["55.1"].text
    assert by_article["54.3"].page == 2 and by_article["55.3"].page == 3
    assert "numbered list item" in by_article["55.3"].text
    assert "Fédération" not in by_article["54.3"].text  # page header stripped
    assert by_article["55.1"].chunk_id == "2025_sporting:55.1:0"


P2026 = """SECTION B: SPORTING REGULATIONS
B41
2026 Formula 1 Regulations - Section B [Sporting]
©2026 Fédération Internationale de l'Automobile
01 October 2026
Issue 09
B5.10 Formation Lap(s) Behind Safety Car
B5.10.1 If track conditions are unsuitable formation laps may take place behind the Safety
Car, as described in Article C4.6 of the Technical Regulations.
C4.6 This cross-reference line must not start a unit.
B5.10.2 When the green lights are illuminated the Safety Car will leave the grid.
B5.11 False Start
B5.11.1 All cars must be stationary in their grid position.
"""


def test_2026_style_lettered_units_and_subarticles() -> None:
    chunks = chunk_document(S2026, [P2026])
    by_article = {c.article: c for c in chunks}
    assert list(by_article) == ["B5.10", "B5.11"]
    b510 = by_article["B5.10"]
    assert b510.title == "Formation Lap(s) Behind Safety Car"
    assert b510.subarticles == ["B5.10.1", "B5.10.2"]
    assert "cross-reference line" in b510.text


def test_long_units_are_split_into_parts() -> None:
    long = "\n".join(f"55.1 sentence number {i} about the safety car." for i in range(1))
    long += " " + " ".join(f"Sentence {i} is long enough to matter here." for i in range(120))
    chunks = chunk_document(S2025, ["55) SAFETY CAR\n" + long])
    assert len(chunks) > 1
    assert {c.article for c in chunks} == {"55.1"}
    assert [c.part for c in chunks] == list(range(len(chunks)))
    assert all(len(c.text) <= MAX_CHARS for c in chunks)


def test_split_handles_one_giant_sentence() -> None:
    parts = _split("x" * (MAX_CHARS * 2 + 10), MAX_CHARS)
    assert [len(p) for p in parts] == [MAX_CHARS, MAX_CHARS, 10]


def test_find_article_refs() -> None:
    assert find_article_refs("What does Article 30.5 say?") == ["30.5"]
    assert find_article_refs("art. 55.14 and B6.3.7") == ["55.14", "B6.3.7"]
    assert find_article_refs("Explain B5.10") == ["B5.10"]
    assert find_article_refs("a 1.5 second gap after 12.3 laps") == []


def test_embed_text_includes_doc_and_article() -> None:
    (chunk, *_) = chunk_document(S2025, [P1])
    assert chunk.embed_text(S2025.title).startswith(
        "2025 F1 Sporting Regulations (Issue 5), Article 54.3 (INCIDENTS DURING THE RACE)"
    )
