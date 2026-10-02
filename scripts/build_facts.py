"""Build stat facts (Jolpica) and the Wikipedia corpus for the showcase races.

    python scripts/build_facts.py              # -> data/facts/races/*.json, data/facts/wiki/

Every fact is computed as of BEFORE its race. API responses are cached under
data/cache/jolpica/ (not committed), so rebuilding is cheap and polite.
"""

from __future__ import annotations

import argparse
import json
import sys

from core.config import REPO_ROOT, get_settings
from core.facts.corpus import WikiCorpus, WikiPassage, passages_from_article
from core.facts.jolpica import JolpicaClient
from core.facts.models import Fact
from core.facts.stats import (
    RaceContext,
    career_win_facts,
    circuit_facts,
    standings_facts,
    teammate_facts,
)
from core.facts.wikipedia import WikipediaClient, title_from_url
from core.feed.builder import race_id
from core.feed.catalog import SHOWCASE_RACES
from core.log import configure_logging
from core.profiles.teams import display_name, team_key
from core.rag.models import FastEmbedder

FACTS_DIR = REPO_ROOT / "data" / "facts"


def build_race(jc: JolpicaClient, season: int, rnd: int) -> tuple[list[Fact], dict, dict]:
    race = jc.races(f"{season}/{rnd}/results")[0]
    circuit = race["Circuit"]
    entrants = {
        r["Driver"]["code"]: (
            r["Driver"]["driverId"],
            f"{r['Driver']['givenName']} {r['Driver']['familyName']}",
            r["Constructor"]["name"],
        )
        for r in race["Results"]
    }
    ctx = RaceContext(
        race_id(season, rnd), season, rnd, circuit["circuitId"], circuit["circuitName"], entrants
    )
    circuit_path = f"circuits/{circuit['circuitId']}/results"
    facts = circuit_facts(ctx, jc.races(circuit_path), jc.url(circuit_path))
    facts += teammate_facts(ctx, jc.races(f"{season}/results"), jc.url(f"{season}/results"))
    if rnd > 1:
        facts += standings_facts(
            ctx, jc.standings(season, rnd - 1), jc.url(f"{season}/{rnd - 1}/driverStandings")
        )
    wins = {d: jc.races(f"drivers/{d}/results/1") for d, _, _ in entrants.values()}
    urls = {d: jc.url(f"drivers/{d}/results/1") for d, _, _ in entrants.values()}
    facts += career_win_facts(ctx, wins, urls)
    drivers = {
        r["Driver"]["code"]: (r["Driver"]["url"], entrants[r["Driver"]["code"]][1])
        for r in race["Results"]
    }
    teams = {team_key(r["Constructor"]["name"]): r["Constructor"]["url"] for r in race["Results"]}
    return facts, drivers, teams


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-wiki", action="store_true")
    parser.add_argument(
        "--only-wiki",
        action="store_true",
        help="rebuild the Wikipedia corpus only (subjects from Jolpica cache/API)",
    )
    args = parser.parse_args(argv)
    configure_logging(settings.log_level, settings.log_json)

    jc = JolpicaClient(cache_dir=settings.data_dir / "cache" / "jolpica")
    all_drivers: dict[str, tuple[str, str]] = {}
    all_teams: dict[str, str] = {}
    (FACTS_DIR / "races").mkdir(parents=True, exist_ok=True)
    for season, rnd in SHOWCASE_RACES:
        facts, drivers, teams = build_race(jc, season, rnd)
        all_drivers.update(drivers)
        all_teams.update(teams)
        if args.only_wiki:
            continue
        out = FACTS_DIR / "races" / f"{race_id(season, rnd)}.json"
        out.write_text(
            json.dumps([f.model_dump(mode="json") for f in facts], indent=1, ensure_ascii=False)
            + "\n"
        )
        kinds: dict[str, int] = {}
        for f in facts:
            kinds[f.kind] = kinds.get(f.kind, 0) + 1
        print(f"{out.name}: {len(facts)} facts {kinds}")
    jc.close()
    if args.skip_wiki:
        return 0

    wiki = WikipediaClient()
    passages: list[WikiPassage] = []
    subjects = [("driver", code, name, url) for code, (url, name) in sorted(all_drivers.items())]
    subjects += [("team", key, display_name(key), url) for key, url in sorted(all_teams.items())]
    failed = []
    for subject_type, subject, name, url in subjects:
        try:
            article = wiki.article(title_from_url(url))
        except Exception as exc:  # blocked / rate limited: keep going, report at the end
            failed.append(subject)
            print(f"  ! {subject}: {type(exc).__name__}: {str(exc)[:120]}")
            continue
        if article is None:
            print(f"  ! no article for {subject} ({url})")
            continue
        found = passages_from_article(
            subject_type, subject, name, article.title, article.permalink, article.extract
        )
        passages += found
        print(f"  {subject:14} {article.title}: {len(found)} passages")
    wiki.close()
    if not passages:
        print("no Wikipedia passages fetched; corpus not written")
        return 1

    embedder = FastEmbedder(settings.embedding_model, threads=None)
    vectors = embedder.embed_passages([f"{p.title}. {p.section}. {p.text}" for p in passages])
    WikiCorpus(passages, vectors, embedder.name).save(FACTS_DIR / "wiki")
    print(f"wiki corpus: {len(passages)} passages for {len(subjects) - len(failed)} subjects")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
