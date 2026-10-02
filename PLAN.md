# Pit Wall: v1 Plan

> Status: **Step 0, awaiting approval.** No code is written until this plan is approved.

Pit Wall is an AI race companion for F1 fans. It replays any past race lap by lap and explains what happens at the fan's knowledge level. Everything is built around the fan's favourite team and drivers, and a prediction game gives live feedback from a Monte Carlo simulator.

---

## 0. Step 0 findings (verified 2026-10-02)

### 0.1 Library versions and API notes
I installed the latest PyPI releases in a scratch venv and checked their signatures. These notes come from that check, not from memory.

| Package | Version | What matters for this build |
|---|---|---|
| langgraph | 1.2.12 | `StateGraph(state_schema, context_schema=..., input_schema=..., output_schema=...)`, `START`/`END`. `langgraph.prebuilt` still ships `ToolNode`, `tools_condition`, `InjectedState`, and `ToolRuntime`. `create_react_agent` is legacy. |
| langchain / -core / -openai | 1.4.3 / 1.6.6 / 1.6.7 | `langchain.agents.create_agent(model, tools, system_prompt=, middleware=, response_format=, context_schema=, ...)` is the new prebuilt agent. **We use a custom `StateGraph` instead**, with `ChatOpenAI.bind_tools` and `ToolNode`. |
| chromadb | 1.5.9 | Rust core. Collection names must be 3–512 characters. `collection.query(query_embeddings=, where=, where_document=, n_results=)`. **The new `Search`/`Rrf` hybrid API raises `NotImplementedError: Search is not implemented for Local Chroma`**: it works on Chroma Cloud only. Hybrid fusion (BM25 + dense + RRF) therefore lives in our own code. |
| deepeval | 4.2.8 | `LLMTestCaseParams` is now `SingleTurnParams` (the old name still works as an alias). `GEval(name, evaluation_params=, criteria=, evaluation_steps=, rubric=, threshold=, model=)`. Custom metrics subclass `BaseMetric` and implement `measure`, `a_measure`, `is_successful`, and `__name__`. `evaluate(..., async_config=, cache_config=, error_config=)`. **Requires Python <4.0.** |
| langfuse | 4.16.0 | OpenTelemetry-based SDK: `get_client()`, `@observe`, `propagate_attributes`, and `langfuse.langchain.CallbackHandler(public_key=, trace_context=)`. Credentials come from env. If the keys are absent we never construct a handler, so tracing is a no-op. |
| fastf1 | 3.8.3 | `get_session(year, gp, identifier, backend=)`, `Session.load(laps=, telemetry=, weather=, messages=)`. Exposes `laps`, `results`, `race_control_messages`, `weather_data`, `track_status`, and `total_laps`. `fastf1.Cache.enable_cache(dir)`. Data starts in 2018. |
| streamlit | 1.64.0 | `st.fragment(func=None, *, run_every: int\|float\|timedelta\|str\|None, parallel: bool=False, key=)`. **`run_every` is confirmed** in the installed signature. `parallel` is new; I'll confirm its semantics in Phase 10. |
| numpy | 2.5.3 latest | **numpy 2.5 requires Python ≥3.12.** On 3.11 we get 2.4.x. See open question Q3. |
| others | fastapi 0.142, pydantic-settings 2.15, sqlalchemy 2.1, psycopg 3.3, openai 3.24 | — |

`onnxruntime` 1.30 and `tokenizers` 0.23 already come in as chromadb dependencies. This matters for the reranker (§0.4).

### 0.2 OpenF1 real-time access terms: **LiveFeed moves to v2**
- **Historical data (2023 onward) is free** and needs no auth.
- **Real-time data is paid.** A sponsor account costs about €9.90/month and adds REST, MQTT, and WebSocket live access plus higher rate limits. Auth is OAuth2: `POST https://api.openf1.org/token`, then a Bearer token valid for 3,600s.
- Data from **30 minutes before to 30 minutes after a session counts as paid live data.** A free client gets nothing during a race.
- **Decision: `LiveFeed` is v2.** v1 still defines the `RaceFeed` interface so that `LiveFeed` can plug in later without changes elsewhere.
- Caveat: openf1.org is blocked from this container (see 0.3). I took these terms from OpenF1's docs pages as surfaced in search results ([openf1.org/docs](https://openf1.org/docs/), [openf1.org/auth.html](https://openf1.org/auth.html)). Please re-check them before v2.

### 0.3 ⚠️ Blocker: this cloud container cannot reach any data source
The environment's network policy returns 403 for every host the project needs:
`api.openf1.org`, `api.jolpi.ca`, `livetiming.formula1.com` (used by FastF1), `www.fia.com`, `en.wikipedia.org`, `api.openai.com`, `huggingface.co`, and `cloud.langfuse.com`. Only PyPI is reachable.

From this container that means I **cannot** build replays, fit degradation, run the backtest, ingest regulations, build facts, or run LLM evals. There are two fixes, and I recommend doing **both**:
1. **Allow those hosts** in the cloud environment's network settings (Edit environment → Network access). Add the Neon host as well once the database exists.
2. **Make every data build a script that also runs in GitHub Actions** (a `workflow_dispatch` "build-data" job). Actions runners have open egress. The job's outputs (replays, degradation fits, index embeddings, facts) are committed, which the deployment design needs anyway.

Until the hosts are open, I'll build against **small hand-made fixtures** for tests. Every phase's code and tests can be finished that way, but the real data artifacts wait.

### 0.4 Corrections to the spec (stating them directly as asked)
1. **Streamlit Community Cloud memory.** The documented limit is **690MB minimum to 2.7GB maximum** (the docs page dated Feb 2024), not "~1GB". I'll design for **≤600MB RSS**.
2. **The reranker can't use `sentence-transformers`.** It pulls in torch: about 800MB on disk and 400MB+ RSS before any model loads, which breaks the memory budget. Proposal: run `cross-encoder/ms-marco-MiniLM-L-6-v2` as **ONNX through `onnxruntime` and `tokenizers`**. Both are already installed as chromadb dependencies, so no new heavy packages; the model is about 90MB and needs no torch. *This is an architecture choice, so I need your OK.*
3. **The hybrid retrieval packages aren't in the stack list.** BM25 needs `rank_bm25`. Chroma's built-in hybrid search is Cloud-only (see 0.1).
4. **Supabase vs Neon.** Supabase's free tier **pauses a project after 7 days of inactivity and needs a manual restore from the dashboard**. A portfolio link would go dead between job applications. Neon's free tier **scales to zero after 5 minutes and wakes automatically on the next query** (roughly 0.5–1s cold start). → **Neon.**
5. **Jolpica is a volunteer-run free API** (Apache-2.0, has terms of use, and burst and hourly rate limits whose exact numbers I couldn't confirm from here). Stat facts are **precomputed at build time** and committed, so the demo never calls Jolpica at runtime. On-demand races call it with a cache and backoff.
6. **The repo isn't empty.** It contains `pit-window.html`, a "Williams Racing AI Companion" page with Williams branding. That conflicts with the "no official logos or trademarked assets" rule and with "build from scratch". I won't touch it. See Q1.

---

## 1. Architecture

```
            ┌──────────── RaceFeed (ABC) ────────────┐
FastF1 ───► │ ReplayFeed (v1)    LiveFeed (v2, OpenF1)│ ──► RaceState tick (one per lap)
committed   └────────────────────────────────────────┘          │
replays                                                         ▼
                 ┌──────────────┬───────────────┬──────────────┬───────────────┐
                 │ EventDetector│ QuietPeriod   │ Simulator    │ Chat agent    │
                 │ (det.)       │ → FactsEngine │ (numpy MC)   │ (LangGraph)   │
                 └──────┬───────┴──────┬────────┴──────┬───────┴──────┬────────┘
                        ▼              ▼               ▼              ▼
                   Explainer (LLM, level-aware; numbers and facts injected, never generated)
                        │
           RaceSession orchestrator (core/session.py) ──► Repository (SQLite | Postgres)
                        │
     BACKEND_MODE=inprocess ──► Streamlit      BACKEND_MODE=api ──► FastAPI (SSE) ──► Streamlit
```

**Invariant:** nothing outside `core/feed/` imports FastF1 or OpenF1. Every feature consumes `RaceState` ticks.

**LLM numeric/fact guard.** LLM prompts receive the numbers and facts as structured context, and the model is told to quote, not compute. A post-generation **numeric guard** extracts every number from the output and checks it against the provided payload, within rounding. On a mismatch the node regenerates once, then falls back to a templated sentence. The same check doubles as the "numeric consistency" eval.

---

## 2. Data model (`core/models.py`, pydantic v2)

```python
Compound = Literal["SOFT","MEDIUM","HARD","INTERMEDIATE","WET","UNKNOWN"]
TrackFlag = Literal["GREEN","YELLOW","SC","VSC","RED","CHEQUERED"]
Level = Literal["rookie","fan","expert"]

class DriverState:   code, number, team, position, gap_to_leader_s, interval_s,
                     last_lap_s, best_lap_s, compound, tyre_age, stint, pit_count,
                     in_pit: bool, status: Literal["running","dnf","finished"]
class RaceControlMsg: lap, utc, category, flag, message, driver: str|None
class Weather:       air_temp, track_temp, rainfall: bool, humidity
class RaceState:     race_id, season, round, event_name, circuit_key, lap, total_laps,
                     flag: TrackFlag, drivers: list[DriverState], weather,
                     new_messages: list[RaceControlMsg], fastest_lap: (driver, time)|None
class RaceMeta:      race_id, season, round, event_name, circuit_key, total_laps,
                     grid: list[str], drivers: dict[code -> {name, team, team_colour}], is_wet

class Event:         id, lap, type: EventType, drivers: list[str], teams: list[str],
                     payload: dict (numbers only, from data), base_importance: float,
                     importance: float (after favourites boost), needs_regs: bool
EventType = overtake | pit_stop | undercut | overcut | safety_car | vsc | red_flag |
            penalty | race_control | fastest_lap | tyre_cliff | dnf | weather_change
class Explanation:   event_id, level, text, citations: list[Citation]
class Citation:      source_type: "fia_reg"|"wikipedia"|"jolpica"|"fastf1",
                     ref (e.g. "Sporting Regs 2025 Art. 55.3" or a URL), page|None

class Fact:          id, subject (driver/team), text_template, values: dict,
                     source: Citation (required, never optional), relevance_tags
class Profile:       profile_id (uuid4), nickname (unique), favourite_team,
                     favourite_drivers (≤2), level, quiz_score, created_at
class Prediction:    profile_id, race_id, p1, p2, p3, pole_to_win: bool|None,
                     first_dnf: str|None, fastest_lap: str|None, locked_at_lap: int|None
class SimResult:     n_sims, seed, lap, p_top3: dict[code,float],
                     p_exact_podium: float, p_win, finish_dist (20×20 matrix)
class ScenarioResult: name, params, sim: SimResult, delta_p_exact_podium
class Score:         profile_id, race_id, points, breakdown: dict
```

**Persistence** (`core/profiles/repository.py`): there is one `Repository` protocol with a single `SqlRepository` implementation on SQLAlchemy 2.x Core. The `DATABASE_URL` env var picks the backend: `sqlite:///...` locally or `postgresql+psycopg://...` on Neon. Tables: `profiles`, `predictions`, `scores`, `session_facts_seen`. Schema creation is idempotent (`metadata.create_all`). There's no Alembic in v1, which is fine for 4 tables; it goes in FUTURE.md. Tests use in-memory SQLite.

**Why SQLAlchemy Core over two hand-written repos:** the same SQL runs on both backends, so one test suite proves both. **Why not storing state on Cloud's local disk:** that disk is ephemeral and wiped on every reboot.

---

## 3. Key design decisions

| Decision | Choice | Why |
|---|---|---|
| Replay data | Pre-built **compact parquet** per race in `data/replays/{season}_{round}/` (`laps.parquet`, `messages.parquet`, `weather.parquet`, `track_status.parquet`, `meta.json`). Target ≤400KB per race. | No FastF1 download at demo time, and loading takes milliseconds. Non-showcase races go through FastF1 on demand with a temp-dir cache. |
| Showcase races | 5 races, proposed: 2024 Brazil (wet chaos), 2024 Monza (strategy), 2025 Monaco, 2025 British (weather), 2023 Las Vegas (SC + recovery). Final picks once data is reachable. | They cover rain, SC/VSC, undercuts, penalties, and DNFs, so every event type fires. |
| Regulations index | Build-time pipeline: chunk article by article, embed with OpenAI, commit **`chunks.jsonl` + `embeddings.npy` (float16)**. At startup, load them into a Chroma `EphemeralClient` and build BM25 in memory. | Startup needs no embedding API calls and doesn't depend on Chroma's on-disk format, which changes between versions. About 3k chunks × 1,536 dims × 2 bytes ≈ 9MB. Load time is under 3s. |
| Retrieval | Exact article regex (`Art(icle)?\.? \d+(\.\d+)*`) bypasses everything else. Otherwise BM25 top-30 and dense top-30, fused with RRF (k=60), then the ONNX cross-encoder reranks to top-5. Filters on `year` and `doc_type`. | This is the standard robust setup, and the ablation (dense vs hybrid vs hybrid+rerank) goes in `evals/results.md`. |
| Simulator | numpy, **lap-stepped and vectorized over (sims × drivers)**. Each lap: lap time = base pace + deg(compound, age) + fuel effect + noise. Plus a pit policy (stop when deg crosses a threshold or the mandatory compound is still unused), SC events (Bernoulli per lap from the track rate; SC compresses gaps and makes pitting cheap), DNF hazard, and an overtake model (a pass needs a pace delta above the track's difficulty threshold, otherwise the car stays behind with a dirty-air penalty). | 2,000 × 20 × ≤70 laps ≈ 2.8M cells, about 50ms of numpy work per step-pass. The target is **<300ms for 2k sims**, with a benchmark test asserting <1s. Seeded through `np.random.Generator`. |
| Degradation | `scripts/fit_degradation.py`: green-flag laps only, excluding in/out laps and SC laps. Fuel correction is 0.03 s/kg × burn rate (documented). Fit a linear `lap_time ~ tyre_age` per compound, with robust outlier trimming. Cached to `data/degradation/{circuit}_{year}.json`, with a manual override file. | As specified. Every constant is listed in `ASSUMPTIONS.md`. |
| Track parameters | `data/tracks.yaml`: pit loss (median green-flag pit-lane delta from FastF1), SC probability (share of 2018–2025 races with an SC/VSC, from `track_status`), overtaking difficulty (passes per race, normalised), and a DNF hazard (season-level rate per car-lap). | All derived from data by `scripts/build_track_params.py` and committed. |
| Event detector | Pure functions over `(prev_state, state)`, deterministic, no LLM. Importance = a base value per type + context (lead change, podium positions) × **1.5–2.0 boost for favourites**. | Tested on fixture states. |
| Facts | Stat facts are computed at build time from Jolpica plus FastF1 into `data/facts/{season}_{round}.json`, each with a `Citation`. The Wikipedia corpus (driver and team pages through the API, CC BY-SA, attributed) is chunked into a second Chroma collection. The LLM only rephrases `text_template + values`, and the numeric guard checks the output. Facts without a source are dropped at load time. A per-session "seen" set means no repeats. | Grounding is structural: the LLM never holds an unsourced fact. |
| Chat graph | Custom LangGraph: `load_context → agent ⇄ tools(ToolNode) → numeric_guard → respond`. Tools: `get_race_state`, `search_regulations`, `get_historical_stats`, `run_scenario`, `get_fact`. Each tool has a Pydantic input schema and returns structured errors instead of raising. | Every number goes through tools, so it can be audited. |
| UI refresh | `@st.fragment(run_every=tick_interval)` advances the replay one tick per run; only the fragment reruns. The sim runs **once per lap** in the session orchestrator, with results cached by `(race_id, lap, scenario)`. Scenarios run on demand. LLM explanations are generated in a small thread pool, with "explaining…" placeholders until they land. | The UI never blocks for more than one sim run (<300ms). |
| Explainer cost control | Only events with importance ≥ a threshold get an LLM explanation; low-importance events use templates. Explanations are cached by `(event_id, level)`. The model is set by `OPENAI_CHAT_MODEL`. | Keeps a 5x replay affordable. |

---

## 4. Phases (one commit each; tests green before moving on)

| # | Phase | Main files | Done when |
|---|---|---|---|
| 1 | Skeleton, config, Makefile, CI, DB layer | `pyproject.toml` (ruff, pytest, mypy-lite), `core/config.py` (pydantic-settings that also reads `st.secrets` when present), `core/logging.py` (structlog-style JSON via stdlib `logging`), `core/models.py`, `core/profiles/repository.py`, `Makefile` (`install ingest build-data test eval run`), `.github/workflows/ci.yml`, `.env.example`, `FUTURE.md` | Repository tests pass on SQLite; CI is green |
| 2 | RaceFeed + ReplayFeed + showcase data | `core/feed/{base,replay,live_stub,loader}.py`, `scripts/build_replays.py`, `data/replays/*`, `tests/fixtures/mini_race/` | Tick sequence, speed and stepping, and seek work on the fixture; real replays are committed (**needs network**) |
| 3 | Event detector | `core/events/{detector,importance,types}.py` | Every event type is detected on hand-crafted fixture sequences |
| 4 | Profiles, quiz, levels | `core/profiles/{service,quiz}.py`, `core/prompts/{rookie,fan,expert}.md` | Quiz scoring → level; CRUD through the repo |
| 5 | Explainer + regulations RAG | `core/rag/{chunking,index,bm25,rerank,retriever}.py`, `core/agents/explainer.py`, `scripts/ingest_regs.py`, `data/raw/regulations/` | Retrieval tests (exact article path, RRF, filters) with fake embeddings; explainer tests with a mocked LLM; ablation script (**needs network**) |
| 6 | Facts engine + quiet period | `core/facts/{stats,corpus,engine,quiet}.py`, `scripts/build_facts.py` | Sourceless facts are dropped, no repeats, the quiet-period trigger fires after N laps |
| 7 | Simulator + degradation + backtest | `core/sim/{model,params,scenarios}.py`, `core/sim/ASSUMPTIONS.md`, `scripts/{fit_degradation,build_track_params,backtest_sim}.py`, `evals/sim_backtest.md` | Seeded determinism, hand-checked cases, benchmark <1s; Brier and calibration on 10+ races (**needs network**) |
| 8 | Prediction game | `core/game/{predictions,health,scoring,recap}.py` | Lock at lap 1, health per lap, scenario ranking, scoring unit tests |
| 9 | Chat agent | `core/agents/chat_graph.py`, `core/agents/tools.py`, `core/agents/guard.py` | Graph tests with a fake chat model; tool error paths |
| 10 | Streamlit UI | `streamlit_app.py`, `ui/{timing_tower,health_panel,event_feed,chat,onboarding,controls}.py`, `core/session.py`, `core/client.py` (inprocess/api) | Smoke test with `streamlit.testing.v1.AppTest`; non-blocking replay |
| 11 | Evals, CI gate, Langfuse | `evals/{golden.json,thresholds.yaml,results.md}`, `evals/metrics/{level,numeric,grounding}.py`, `scripts/run_evals.py`, `core/tracing.py`, `.github/workflows/evals.yml` | 50+ golden cases; the gate fails below thresholds; tracing is a no-op without keys |
| 12 | Docker Compose | `Dockerfile` (multi-stage, slim), `docker-compose.yml` (api + ui + postgres), `api/main.py` (`/health`, `/chat` SSE, `/race/*`) | `docker compose up` gives a working stack |
| 13 | Deploy + README | Streamlit Cloud config, Neon DB, `README.md` | Live URL replays a race end to end; cold start and RSS measured and reported |

`api/` is drafted in Phase 9–10 (needed for `BACKEND_MODE=api`) and finished in 12.

### Eval plan (Phase 11)
- `golden.json` has 50+ cases across: event explanations (≥15 = 5 events × 3 levels), chat Q&A (≥12), fact grounding (≥8), prediction narration (≥8), and unanswerable or out-of-scope questions (≥7).
- Metrics: DeepEval `FaithfulnessMetric` and `AnswerRelevancyMetric`. A `GEval` "level appropriateness" metric per level. **Rookie:** Flesch reading ease ≥ 60 (`textstat`, see Q2). **Expert:** technical-term density ≥ a threshold, from a curated F1 glossary. **Fact grounding rate** = 100%. **Numeric consistency** = every number in a narration is in the sim payload.
- CI: lint and unit tests on every push. On PRs to main, a ~12-case subset is gated by `thresholds.yaml`. The `OPENAI_API_KEY` repo secret is needed. Fork PRs skip the eval job.

### Memory and cold-start budget (Streamlit Cloud)
streamlit ~120MB, plus pandas/numpy/pyarrow ~120MB, plus chromadb ~60MB, plus onnxruntime and the reranker ~150MB, plus indexes ~30MB, plus langchain ~60MB. **That totals about 540MB.** FastF1 is imported lazily (only for on-demand races). The reranker loads lazily on the first regulations query, with a measured fallback to hybrid-only if RSS runs high. Phase 13 measures this with `psutil` and logs it on `/health`.

---

## 5. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| **Container egress blocked** (§0.3) | Real data, embeddings, and evals can't run here | Allowlist the hosts and/or use the Actions build-data workflow; fixtures in the meantime |
| FIA PDF formats vary (2026 Technical is ~180 pages, with tables) | Bad chunks | Article regex chunker with page tracking; manual spot checks; download PDFs once into `data/raw/regulations/` (check FIA terms; commit only if allowed, otherwise fetch at build) |
| FastF1 timing gaps (missing laps, SC laps without times) | Detector false positives | Forward-fill state, a `data_quality` flag per tick, conservative detector thresholds |
| Simulator calibration is poor | Weak portfolio story | The backtest is honest; report Brier against baselines (grid order, current order); tune only on held-out races |
| LLM invents numbers | Violates a core rule | Structured inputs plus the numeric guard and the numeric-consistency eval |
| Neon cold start / free-tier limits | First-request latency of about 1s | Connection `pool_pre_ping`; DB calls are off the critical render path |
| OpenAI cost during 20x replays | Spend | Importance threshold, caching, a cheap model by default, a per-session LLM call budget |
| Streamlit fragment + threading quirks | UI glitches | Keep all mutable state in `st.session_state`; worker threads return futures only; `AppTest` coverage |
| Jolpica volunteer API downtime | On-demand facts fail | Showcase facts are precomputed; graceful "no facts available" |

---

## 6. Open questions (please answer before Phase 1)

1. **`pit-window.html`** (Williams-branded) is in the repo. Should I delete it, move it out, or leave it alone?
2. **New dependencies** outside the listed stack. OK to add these?
   - `rank_bm25` (BM25; small, pure Python)
   - **ONNX reranker** via `onnxruntime` + `tokenizers` (already transitive), *instead of* `sentence-transformers`/torch. **This is an architecture change, since torch doesn't fit in memory.**
   - `sqlalchemy` + `psycopg[binary]` (repository layer)
   - `textstat` (Flesch reading ease for the Rookie metric)
   - `pyarrow` (parquet; already a Streamlit dependency)
   - `httpx` + `tenacity` (timeouts and retries for Jolpica, Wikipedia, and OpenF1; both already transitive)
   - `ruff` (lint), `pytest-cov`, `psutil` (memory measurement) as dev and ops tools
3. **Python version:** target **3.12** (latest numpy, still supported by DeepEval and Langfuse, available on Streamlit Cloud), or keep 3.11 with `numpy<2.5`? This container has 3.11. My recommendation is `requires-python >=3.11` plus a CI matrix on 3.11 and 3.12, deploying on 3.12.
4. **OpenAI models:** which chat model and embedding model? I'll make both configurable. Defaults: a small, cheap chat model for the explainer and chat, and `text-embedding-3-small` for embeddings. Tell me if you want different defaults.
5. **Network:** can you allowlist these hosts in this environment: `api.openf1.org`, `api.jolpi.ca`, `livetiming.formula1.com`, `www.fia.com`, `en.wikipedia.org`, `api.openai.com`, `huggingface.co` (plus its CDN), and `cloud.langfuse.com`? Alternatively, are you OK with the GitHub Actions build-data workflow (it needs the `OPENAI_API_KEY` secret)? I recommend both.
6. **Showcase races:** are you OK with the 5 proposed in §3, or do you have favourites?
7. **FIA regulation PDFs:** commit them to `data/raw/regulations/` or fetch them at build time? I recommend fetching them and committing only the derived chunks plus embeddings, which avoids redistributing the PDFs.
8. **Neon:** you'll need to create the free project and add `DATABASE_URL` to the Streamlit secrets at deploy time (Phase 13). Is that OK?

---

## 7. Out of scope for v1 (→ `FUTURE.md`)
LiveFeed (OpenF1 paid), leagues and leaderboards, stewards' precedent search, the 2025→2026 regulation diff, personalised debriefs, Alembic migrations, and a semantic cache.
