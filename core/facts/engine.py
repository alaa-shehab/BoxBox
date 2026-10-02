"""Facts engine: picks a sourced, unseen fact about the fan's favourites and phrases it.

Sources, in order of preference:
  1. Computed stat facts for this race (Jolpica), most relevant first: circuit-specific,
     then teammate, championship and career facts.
  2. The Wikipedia corpus, restricted to passages that mention no year from the replayed
     season onward (spoiler safety).

The LLM only rephrases. A stat fact's own sentence (or a wiki passage) is the entire
context, and the numeric guard checks the rephrasing against it. Without an LLM, the
deterministic sentence or an extracted passage sentence is shown. Facts without a source
URL can't exist (the Fact model rejects them), and every shown fact is recorded so it
never repeats within a session.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.runnables import Runnable

from core.agents.grounded import generate
from core.config import REPO_ROOT
from core.facts.corpus import WikiCorpus, WikiPassage
from core.facts.models import Fact
from core.log import get_logger
from core.models import Citation, Level, Profile, RaceMeta
from core.profiles.repository import Repository
from core.profiles.teams import display_name, team_key
from core.rag.models import Embedder

log = get_logger(__name__)

FACTS_DIR = REPO_ROOT / "data" / "facts"

STAT_TASK = """\
Task: tell the fan the FACT below in 1-2 lively sentences suited to their level.
- Keep its meaning and every number exactly as given (rounding is fine). Add nothing else.
- Plain text only."""

WIKI_TASK = """\
Task: from the PASSAGE below, tell the fan ONE interesting fact about {subject} in 1-2
sentences suited to their level.
- Use only what the passage says. Don't add facts, dates or numbers from memory.
- Plain text only."""


@dataclass(frozen=True)
class FactCard:
    fact_id: str
    subject: str  # display name of the driver/team/circuit
    text: str
    source: Citation
    origin: str  # "stats" | "wikipedia"
    phrased_by: str  # "llm" | "template"


def load_race_facts(race_id: str, directory: Path = FACTS_DIR / "races") -> list[Fact]:
    path = directory / f"{race_id}.json"
    if not path.exists():
        return []
    facts = []
    for raw in json.loads(path.read_text()):
        try:
            facts.append(Fact.model_validate(raw))
        except ValueError:
            log.warning("facts.dropped_unsourced", extra={"fields": {"id": raw.get("id")}})
    return facts


def load_corpus(directory: Path = FACTS_DIR / "wiki") -> WikiCorpus | None:
    if not (directory / "manifest.json").exists():
        return None
    return WikiCorpus.load(directory)


def _first_sentences(text: str, max_chars: int = 260) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out = ""
    for s in sentences:
        if len(out) + len(s) > max_chars and out:
            break
        out = f"{out} {s}".strip()
    return out


class FactsEngine:
    def __init__(
        self,
        meta: RaceMeta,
        facts: list[Fact],
        corpus: WikiCorpus | None,
        repo: Repository,
        llm: Runnable | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.meta = meta
        self.facts = [f for f in facts if f.race_id == meta.race_id]
        self.corpus = corpus
        self.repo = repo
        self.llm = llm
        self.embedder = embedder

    # ------------------------------------------------------------ subjects
    def _favourite_subjects(self, profile: Profile) -> list[str]:
        subjects = [d for d in profile.favourite_drivers if d in self.meta.drivers]
        subjects.append(team_key(profile.favourite_team))
        return subjects

    def resolve_subject(self, text: str) -> str | None:
        """Map free text ("tell me about Lewis", "Racing Bulls", "HAM") to a subject key."""
        lowered = text.casefold()
        for code, info in self.meta.drivers.items():
            names = {code.casefold(), info.name.casefold(), info.name.split()[-1].casefold()}
            if any(re.search(rf"\b{re.escape(n)}\b", lowered) for n in names):
                return code
        for team in {d.team for d in self.meta.drivers.values()}:
            key = team_key(team)
            if any(n.casefold() in lowered for n in {team, display_name(key)}):
                return key
        return None

    def _name(self, subject: str) -> str:
        if subject in self.meta.drivers:
            return self.meta.drivers[subject].name
        return display_name(subject)

    # ------------------------------------------------------------ picking
    def _stat_candidates(self, subjects: list[str], seen: set[str]) -> list[Fact]:
        order = {s: i for i, s in enumerate(subjects)}
        pool = [f for f in self.facts if f.subject in order and f.id not in seen]
        return sorted(pool, key=lambda f: (-f.relevance, order[f.subject], f.id))

    def _wiki_candidates(
        self, subjects: list[str], seen: set[str], query: str | None
    ) -> list[WikiPassage]:
        if self.corpus is None:
            return []
        out: list[WikiPassage] = []
        for subject in subjects:
            if query:
                found = self.corpus.search(
                    subject, query, self.embedder, k=5, before_year=self.meta.season
                )
            else:
                found = self.corpus.for_subject(subject, before_year=self.meta.season)
            out += [p for p in found if p.passage_id not in seen]
        return out

    def next_fact(
        self,
        profile: Profile,
        session_id: str,
        level: Level | None = None,
        subject_text: str | None = None,
    ) -> FactCard | None:
        """Quiet-period fact about favourites, or on request about `subject_text`."""
        level = level or profile.level
        if subject_text:
            subject = self.resolve_subject(subject_text)
            subjects = [subject] if subject else []
        else:
            subjects = self._favourite_subjects(profile)
        if not subjects:
            return None
        seen = self.repo.seen_facts(session_id)

        for fact in self._stat_candidates(subjects, seen):
            if self.repo.mark_fact_seen(session_id, fact.id):
                return self._phrase_stat(fact, level)
        query = subject_text if subject_text and len(subject_text.split()) > 2 else None
        for passage in self._wiki_candidates(subjects, seen, query):
            if self.repo.mark_fact_seen(session_id, passage.passage_id):
                return self._phrase_wiki(passage, level)
        return None

    # ------------------------------------------------------------ phrasing
    def _phrase_stat(self, fact: Fact, level: Level) -> FactCard:
        out = generate(self.llm, level, STAT_TASK, f"FACT: {fact.text}", fact.text, label=fact.id)
        return FactCard(
            fact.id, self._name(fact.subject), out.text, fact.source, "stats", out.source
        )

    def _phrase_wiki(self, passage: WikiPassage, level: Level) -> FactCard:
        task = WIKI_TASK.format(subject=passage.subject_name)
        context = f"PASSAGE ({passage.title}, {passage.section or 'introduction'}):\n{passage.text}"
        out = generate(
            self.llm, level, task, context, _first_sentences(passage.text), label=passage.passage_id
        )
        source = Citation(source_type="wikipedia", label=passage.citation_label, url=passage.url)
        return FactCard(
            passage.passage_id, passage.subject_name, out.text, source, "wikipedia", out.source
        )
