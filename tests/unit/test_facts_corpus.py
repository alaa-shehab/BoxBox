from __future__ import annotations

from pathlib import Path

import numpy as np

from core.facts.corpus import WikiCorpus, max_year, passages_from_article, split_article
from core.rag.models import HashEmbedder

ARTICLE = """Max Emilian Verstappen is a Dutch racing driver. He competes in Formula One for
Red Bull Racing, and won the championship in 2021, 2022, 2023 and 2024.

== Early life ==
Verstappen was born in Hasselt, Belgium in 1997. His father Jos raced in Formula One in 1994.

Short.

=== Karting ===
He started karting at the age of four and won many titles before 2013.

== Formula One career ==
He debuted with Toro Rosso at the 2015 Australian Grand Prix, aged 17, becoming the
youngest driver to start a Grand Prix.

== Racing record ==
| table rows that should never be indexed |

== References =="""


def test_split_skips_unwanted_sections() -> None:
    sections = [s for s, _ in split_article(ARTICLE)]
    assert "Racing record" not in sections and "References" not in sections
    assert set(sections) == {"", "Early life", "Formula One career"}


def test_passages_merge_and_drop_short_paragraphs() -> None:
    ps = passages_from_article(
        "driver",
        "VER",
        "Max Verstappen",
        "Max Verstappen",
        "https://en.wikipedia.org/w/index.php?title=Max_Verstappen&oldid=1",
        ARTICLE,
    )
    assert [p.section for p in ps] == ["", "Early life", "Formula One career"]
    early = ps[1]
    assert "born in Hasselt" in early.text and "karting" in early.text  # merged
    assert "Short." in early.text  # merged into its section, not a passage on its own
    assert [p.max_year for p in ps] == [2024, 2013, 2015]
    assert ps[0].passage_id == "wiki:VER:0"
    assert ps[0].citation_label == "Wikipedia: Max Verstappen (CC BY-SA 4.0)"


def test_max_year() -> None:
    assert max_year("from 1997 to 2021, then 2024") == 2024
    assert max_year("car number 33 and 1600cc") == 0


def _corpus() -> WikiCorpus:
    ps = passages_from_article("driver", "VER", "Max Verstappen", "Max Verstappen", "u", ARTICLE)
    ps += passages_from_article(
        "team",
        "williams",
        "Williams",
        "Williams Racing",
        "u2",
        "Williams is a British team founded by Frank Williams in 1977 "
        "and Patrick Head, based in Grove, Oxfordshire, and one of the most "
        "successful constructors in the history of the sport.",
    )
    emb = HashEmbedder()
    return WikiCorpus(ps, emb.embed_passages([p.text for p in ps]), emb.name)


def test_search_is_per_subject_and_spoiler_safe() -> None:
    corpus = _corpus()
    assert corpus.subjects() == {"VER", "williams"}
    hits = corpus.search("VER", "born in Hasselt Belgium", HashEmbedder(), k=1)
    assert hits[0].section == "Early life"
    assert all(p.max_year < 2016 for p in corpus.for_subject("VER", before_year=2016))
    assert len(corpus.for_subject("VER", before_year=2016)) == 2
    assert corpus.search("VER", "championship", None, before_year=2010) == []
    assert corpus.search("nobody", "x") == []


def test_corpus_roundtrip(tmp_path: Path) -> None:
    corpus = _corpus()
    corpus.save(tmp_path / "wiki")
    loaded = WikiCorpus.load(tmp_path / "wiki")
    assert loaded.passages == corpus.passages
    assert np.allclose(loaded.embeddings, corpus.embeddings, atol=1e-3)
