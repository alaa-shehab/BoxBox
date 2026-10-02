# Pit Wall: v1 Plan

> Status: **Step 0 done.** Revised after feedback (free and open-source only).

Pit Wall is an AI race companion for F1 fans. It replays any past race lap by lap and explains what happens at the fan's knowledge level. Everything is built around the fan's favourite team and drivers, and a prediction game gives live feedback from a Monte Carlo simulator.

---

## 0. Step 0 findings and decisions (updated 2026-10-02 after your feedback)

### 0.0 Your decisions
- `pit-window.html` stays in the repo, and nothing references it.
- **Free and open-source only.** No paid APIs and no OpenAI. Every service below has a **no-credit-card free tier**, or runs locally.
- I'm allowed to choose libraries and the host on reliability.

### 0.1 The free stack (what changed from the spec)

| Concern | Spec said | **Now** | Why |
|---|---|---|---|
| Chat LLM | OpenAI | **Open-weight models on Groq's free tier** (no card; about 30 req/min, 14.4k req/day, per-model token-per-minute caps). **Cerebras free tier** (1M tokens/day, 5 req/min) is the automatic fallback. **Ollama** is the local and Docker option. | Prefer Apache-2.0 models: `openai/gpt-oss-120b` for the tool-calling chat agent, and a small, fast model for the high-volume explainer. All model IDs live in config; I'll confirm them against each provider's `/models` endpoint in Phase 5. |
| LLM client | — | `langchain-groq` 1.1.3 (`ChatGroq`). Cerebras and Ollama go through `init_chat_model` / `langchain-ollama`, wrapped in one `core/llm.py` factory with a **provider fallback chain and a client-side rate limiter**. | One code path for all providers, and the rate limits are handled explicitly rather than surfacing as 429 crashes. |
| Embeddings | OpenAI | **`fastembed` 0.8.1** with `BAAI/bge-small-en-v1.5` (384 dims, 67MB, ONNX, runs on CPU). | Free, local, no torch, deterministic. |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` via sentence-transformers | **The same model, ONNX build, through fastembed's `TextCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")`** | Same model and no torch (torch alone would blow the memory budget). One library covers both embedding and reranking. |
| Eval judge (DeepEval) | OpenAI | `deepeval.models.LocalModel` (any OpenAI-compatible endpoint), pointed at Groq or Cerebras. **The judge uses a different model from the generator** to avoid self-preference bias. | DeepEval 4.x has no Groq class, but `LocalModel` covers it. |
| Hosting | Streamlit Community Cloud | **Still Streamlit Community Cloud.** See 0.2. | It's the only free, no-card host with enough RAM. |
| UI | Streamlit | **Streamlit 1.64** (`st.fragment(run_every=...)` confirmed) | Follows from the hosting choice. |
| Database | Supabase or Neon | **Neon free** (no card; scales to zero and wakes automatically) | Supabase pauses idle projects and needs a manual restore. |
| Tracing | Langfuse | **Langfuse Cloud Hobby (free)**, or self-hosted Langfuse in docker-compose. Optional via env. | — |

### 0.2 Is there anything better than Streamlit that is free? (I checked; no)
"Better" UI frameworks such as NiceGUI 3.17 (FastAPI-based, real server push) or a React frontend need a host that runs arbitrary Python or Docker. Here are the free options as of now:
- **Hugging Face Docker Spaces:** since July 2026, **creating** a Docker or Gradio CPU Space needs PRO ($9/month). Ruled out.
- **Render free:** 512MB RAM, sleeps after 15 minutes, and takes about 1 minute to wake. That's too little RAM for the embedding and reranker models plus the app, and the wake time is a poor first impression for a demo.
- **Koyeb:** now requires a card with a $29 hold. **Cloud Run and Oracle Always Free:** require a card. **Fly.io:** no free tier.
- **Streamlit Community Cloud:** free, no card, documented RAM of 690MB minimum to 2.7GB maximum, and it wakes on click.

→ **Streamlit on Community Cloud is the best free option.** The live-feed weakness of Streamlit's rerun model is handled with `st.fragment(run_every=...)`: only the race panel reruns, and the work runs in background threads. FastAPI stays as the API for the Docker and local stack (`BACKEND_MODE=api`). NiceGUI and a React frontend go into FUTURE.md, for if paid hosting is ever on the table.

### 0.3 Library versions and API notes (verified by installing in a scratch venv)

| Package | Version | What matters for this build |
|---|---|---|
| langgraph | 1.2.12 | `StateGraph(state_schema, context_schema=..., input_schema=..., output_schema=...)`, `START`/`END`. `langgraph.prebuilt` still ships `ToolNode`, `tools_condition`, `InjectedState`, and `ToolRuntime`. `create_react_agent` is legacy. |
| langchain / -core | 1.4.3 / 1.6.6 | `langchain.agents.create_agent(...)` is the new prebuilt agent. **We use a custom `StateGraph` instead**, with `bind_tools` and `ToolNode`. `init_chat_model(model, model_provider=...)`. |
| chromadb | 1.5.9 | Rust core. Collection names must be 3–512 characters. `collection.query(query_embeddings=, where=, where_document=, n_results=)`. **The new `Search`/`Rrf` hybrid API raises `NotImplementedError: Search is not implemented for Local Chroma`**: it works on Chroma Cloud only. Hybrid fusion (BM25 + dense + RRF) therefore lives in our own code. |
| fastembed | 0.8.1 | `TextEmbedding(model_name)`, `TextCrossEncoder(model_name).rerank(query, documents) -> Iterable[float]`. Built on onnxruntime 1.30 and tokenizers. |
| deepeval | 4.2.8 | `LLMTestCaseParams` is now `SingleTurnParams` (the old name still works as an alias). `GEval(name, evaluation_params=, criteria=, evaluation_steps=, rubric=, threshold=, model=)`. Custom metrics subclass `BaseMetric` (`measure`, `a_measure`, `is_successful`, `__name__`). Custom judges use `DeepEvalBaseLLM` or `LocalModel`. **Requires Python <4.0.** |
| langfuse | 4.16.0 | OpenTelemetry-based SDK: `get_client()`, `@observe`, `propagate_attributes`, and `langfuse.langchain.CallbackHandler(public_key=, trace_context=)`. Credentials come from env. If the keys are absent we never construct a handler, so tracing is a no-op. |
| fastf1 | 3.8.3 | `get_session(year, gp, identifier, backend=)`, `Session.load(laps=, telemetry=, weather=, messages=)`. Exposes `laps`, `results`, `race_control_messages`, `weather_data`, `track_status`, and `total_laps`. `fastf1.Cache.enable_cache(dir)`. Data starts in 2018. |
| streamlit | 1.64.0 | `st.fragment(func=None, *, run_every: int\|float\|timedelta\|str\|None, parallel: bool=False, key=)`. **`run_every` is confirmed.** |
| langchain-groq | 1.1.3 | `ChatGroq`. |
| numpy | 2.5.3 latest | numpy 2.5 requires Python ≥3.12; on 3.11 pip picks 2.4.x automatically. **Decision:** `requires-python >=3.11`, CI runs on 3.11 and 3.12, and the deployment uses 3.12. |
| others | fastapi 0.142, pydantic-settings 2.15, sqlalchemy 2.1, psycopg 3.3 | — |

### 0.4 OpenF1 real-time access terms: **LiveFeed moves to v2**
- **Historical data (2023 onward) is free** and needs no auth.
- **Real-time data is paid.** A sponsor account costs about €9.90/month and adds REST, MQTT, and WebSocket live access plus higher rate limits. Auth is OAuth2: `POST https://api.openf1.org/token`, then a Bearer token valid for 3,600s.
- Data from **30 minutes before to 30 minutes after a session counts as paid live data.** A free client gets nothing during a race.
- **Decision: `LiveFeed` is v2**, and it stays v2 under the free-only rule. v1 still defines the `RaceFeed` interface so `LiveFeed` can plug in later.

### 0.5 ⚠️ Blocker: this cloud container cannot reach the data sources
The environment's network policy returns 403 for every host the project needs:
`api.openf1.org`, `api.jolpi.ca`, `livetiming.formula1.com` (used by FastF1), `www.fia.com`, `en.wikipedia.org`, `huggingface.co` (needed for the fastembed model downloads), `api.groq.com`, `api.cerebras.ai`, and `cloud.langfuse.com`. Only PyPI is reachable.

There are two fixes, and I recommend doing **both**:
1. **Allow those hosts** in the cloud environment's network settings (Edit environment → Network access).
2. **Make every data build a script that also runs in a GitHub Actions `build-data` workflow.** Actions is free for public repos, and its runners have open egress. The job commits the outputs (replays, degradation fits, index chunks and embeddings, facts).

Until the hosts are open, I'll build against small hand-made fixtures, and tests use fake embedding and LLM classes.

### 0.6 Other spec corrections
1. **Streamlit Cloud memory** is documented as 690MB to 2.7GB, not "~1GB". I'll design for **≤600MB RSS**.
2. **BM25 needs `rank_bm25`**, because Chroma's built-in hybrid search is Cloud-only.
3. **Jolpica is a volunteer-run free API** (Apache-2.0, has terms of use and rate limits). Stat facts are precomputed at build time and committed, so the demo never calls Jolpica at runtime.
4. **Free LLM rate limits are the main runtime constraint.** A 20x replay can produce events faster than about 30 requests/minute allows. The explainer therefore only calls the LLM for high-importance events, uses templates for the rest, caches by `(event, level)`, and degrades to a template when a request is rate-limited, instead of failing.

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
| Regulations index | Build-time pipeline: chunk article by article, embed with fastembed `bge-small-en-v1.5`, commit **`chunks.jsonl` + `embeddings.npy` (float16, 384 dims)**. At startup, load them into a Chroma `EphemeralClient` and build BM25 in memory. | Startup needs no re-embedding and doesn't depend on Chroma's on-disk format, which changes between versions. About 3k chunks × 384 dims × 2 bytes ≈ 2.3MB. Load time is under 3s. Query embedding runs locally. |
| Retrieval | Exact article regex (`Art(icle)?\.? \d+(\.\d+)*`) bypasses everything else. Otherwise BM25 top-30 and dense top-30, fused with RRF (k=60), then the fastembed ONNX cross-encoder reranks to top-5. Filters on `year` and `doc_type`. | This is the standard robust setup, and the ablation (dense vs hybrid vs hybrid+rerank) goes in `evals/results.md`. |
| Simulator | numpy, **lap-stepped and vectorized over (sims × drivers)**. Each lap: lap time = base pace + deg(compound, age) + fuel effect + noise. Plus a pit policy (stop when deg crosses a threshold or the mandatory compound is still unused), SC events (Bernoulli per lap from the track rate; SC compresses gaps and makes pitting cheap), DNF hazard, and an overtake model (a pass needs a pace delta above the track's difficulty threshold, otherwise the car stays behind with a dirty-air penalty). | 2,000 × 20 × ≤70 laps ≈ 2.8M cells, about 50ms of numpy work per step-pass. The target is **<300ms for 2k sims**, with a benchmark test asserting <1s. Seeded through `np.random.Generator`. |
| Degradation | `scripts/fit_degradation.py`: green-flag laps only, excluding in/out laps and SC laps. Fuel correction is 0.03 s/kg × burn rate (documented). Fit a linear `lap_time ~ tyre_age` per compound, with robust outlier trimming. Cached to `data/degradation/{circuit}_{year}.json`, with a manual override file. | As specified. Every constant is listed in `ASSUMPTIONS.md`. |
| Track parameters | `data/tracks.yaml`: pit loss (median green-flag pit-lane delta from FastF1), SC probability (share of 2018–2025 races with an SC/VSC, from `track_status`), overtaking difficulty (passes per race, normalised), and a DNF hazard (season-level rate per car-lap). | All derived from data by `scripts/build_track_params.py` and committed. |
| Event detector | Pure functions over `(prev_state, state)`, deterministic, no LLM. Importance = a base value per type + context (lead change, podium positions) × **1.5–2.0 boost for favourites**. | Tested on fixture states. |
| Facts | Stat facts are computed at build time from Jolpica plus FastF1 into `data/facts/{season}_{round}.json`, each with a `Citation`. The Wikipedia corpus (driver and team pages through the API, CC BY-SA, attributed) is chunked into a second Chroma collection. The LLM only rephrases `text_template + values`, and the numeric guard checks the output. Facts without a source are dropped at load time. A per-session "seen" set means no repeats. | Grounding is structural: the LLM never holds an unsourced fact. |
| Chat graph | Custom LangGraph: `load_context → agent ⇄ tools(ToolNode) → numeric_guard → respond`. Tools: `get_race_state`, `search_regulations`, `get_historical_stats`, `run_scenario`, `get_fact`. Each tool has a Pydantic input schema and returns structured errors instead of raising. | Every number goes through tools, so it can be audited. |
| UI refresh | `@st.fragment(run_every=tick_interval)` advances the replay one tick per run; only the fragment reruns. The sim runs **once per lap** in the session orchestrator, with results cached by `(race_id, lap, scenario)`. Scenarios run on demand. LLM explanations are generated in a small thread pool, with "explaining…" placeholders until they land. | The UI never blocks for more than one sim run (<300ms). |
| Explainer rate-limit control | Only events with importance ≥ a threshold get an LLM explanation; low-importance events use templates. Explanations are cached by `(event_id, level)`. Models and providers are set in config. Client-side rate limiter, plus Groq → Cerebras → template fallback. | Keeps a 20x replay inside free-tier rate limits. |

---

## 4. Phases (one commit each; tests green before moving on)

| # | Phase | Main files | Done when |
|---|---|---|---|
| 1 | Skeleton, config, Makefile, CI, DB layer | `pyproject.toml` (ruff, pytest), `core/config.py` (pydantic-settings that also reads `st.secrets` when present), `core/logging.py` (structlog-style JSON via stdlib `logging`), `core/models.py`, `core/profiles/repository.py`, `Makefile` (`install ingest build-data test eval run`), `.github/workflows/ci.yml`, `.env.example`, `FUTURE.md` | Repository tests pass on SQLite; CI is green |
| 2 | RaceFeed + ReplayFeed + showcase data | `core/feed/{base,replay,live_stub,loader}.py`, `scripts/build_replays.py`, `data/replays/*`, `tests/fixtures/mini_race/` | Tick sequence, speed and stepping, and seek work on the fixture; real replays are committed (**needs network**) |
| 3 | Event detector | `core/events/{detector,importance,types}.py` | Every event type is detected on hand-crafted fixture sequences |
| 4 | Profiles, quiz, levels | `core/profiles/{service,quiz}.py`, `core/prompts/{rookie,fan,expert}.md` | Quiz scoring → level; CRUD through the repo |
| 5 | Explainer + regulations RAG | `core/rag/{chunking,index,bm25,rerank,retriever}.py`, `core/llm.py` (provider factory + fallback + rate limiter), `core/agents/explainer.py`, `scripts/ingest_regs.py`, `data/raw/regulations/` | Retrieval tests (exact article path, RRF, filters) with fake embeddings; explainer tests with a mocked LLM; ablation script (**needs network**) |
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
- CI: lint and unit tests on every push. On PRs to main, a ~12-case subset is gated by `thresholds.yaml`. The `GROQ_API_KEY` and `CEREBRAS_API_KEY` repo secrets (both free) are needed. The subset is sized to stay well inside free daily quotas. Fork PRs skip the eval job.

### Memory and cold-start budget (Streamlit Cloud)
Measured in Phase 5 (RSS, Python 3.11, warm after 9 queries):

| Component | RSS |
|---|---:|
| Core + pandas + 5 showcase replays | 159 MB |
| + regulations index, Chroma, BM25, bge-small embedder, MiniLM reranker | ~600 MB |
| + LangChain / Groq client | ~603 MB |

ONNX Runtime was the surprise. The defaults (batch 64, CPU memory arena on) grew to **949 MB** and kept climbing. With batch 8, 2 threads and the arena off, RSS stays flat at about 600 MB, for about 20% more rerank latency. Streamlit adds about 100 MB on top. If Community Cloud's allowance is tight, `RERANK_ENABLED=false` saves about 130 MB; the ablation in `evals/results.md` shows what that costs in retrieval precision. Phase 13 re-measures this on the real host. FastF1 is imported lazily, only for on-demand races.

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
| Free-tier LLM rate limits (429s) during 20x replays | Missing explanations | Client-side limiter, Groq → Cerebras fallback, template fallback, importance threshold, caching, a per-session budget |
| Free tiers change terms (as Hugging Face Spaces just did) | Demo breaks | Every provider sits behind a config switch; Ollama is the local fallback; terms get re-checked at deploy time |
| Streamlit fragment + threading quirks | UI glitches | Keep all mutable state in `st.session_state`; worker threads return futures only; `AppTest` coverage |
| Jolpica volunteer API downtime | On-demand facts fail | Showcase facts are precomputed; graceful "no facts available" |

---

## 6. Open questions → resolved

| Question | Resolution |
|---|---|
| pit-window.html | Keep it and ignore it |
| Dependencies | Delegated to me. Adding `rank_bm25`, `fastembed`, `langchain-groq`, `langchain-ollama`, `sqlalchemy`, `psycopg[binary]`, `textstat`, `httpx`, `tenacity`, and `pyarrow`. Dev tools: `ruff`, `pytest-cov`, `psutil`. |
| Python | `>=3.11`; CI on 3.11 and 3.12; deploy on 3.12 |
| Models | Open-weight models on Groq, with Cerebras and Ollama as fallbacks; fastembed for embeddings and reranking |
| Hosting | Streamlit Community Cloud + Neon (both free, no card) |
| Regulation PDFs | Fetch at build time; commit only the derived chunks and embeddings |
| Showcase races | 2024 Brazil, 2024 Monza, 2025 Monaco, 2025 British, 2023 Las Vegas (final picks once data is reachable) |

**What you need to do (all free, no card):** create a Groq API key, plus an optional Cerebras key. At deploy time, create a Neon project and optionally a Langfuse Hobby project. Allow the hosts in §0.5, or let the Actions build-data workflow handle data.

## 7. Out of scope for v1 (→ `FUTURE.md`)
LiveFeed (OpenF1 paid), a NiceGUI or React frontend (needs a non-Streamlit host), leagues and leaderboards, stewards' precedent search, the 2025→2026 regulation diff, personalised debriefs, Alembic migrations, and a semantic cache.
