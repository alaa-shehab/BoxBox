PY ?= python

.PHONY: help install lint format test test-pg ingest build-data backtest eval run api

help:  ## List targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

install:  ## Install the package with dev tools
	$(PY) -m pip install -e ".[dev]"

lint:  ## Ruff lint + format check
	ruff check .
	ruff format --check .

format:  ## Auto-format and fix lint
	ruff format .
	ruff check --fix .

test:  ## Unit + integration tests (external APIs and LLMs mocked)
	$(PY) -m pytest --cov=core --cov-report=term-missing:skip-covered

test-pg:  ## Tests including Postgres (needs TEST_POSTGRES_URL)
	$(PY) -m pytest -m postgres

ingest:  ## Build the regulations index and facts corpus
	$(PY) scripts/ingest_regs.py
	$(PY) scripts/build_facts.py

build-data:  ## Build replays, degradation fits and track parameters (needs network)
	$(PY) scripts/build_replays.py
	$(PY) scripts/fit_degradation.py
	$(PY) scripts/build_track_params.py

backtest:  ## Simulator calibration backtest -> evals/sim_backtest.md
	$(PY) scripts/backtest_sim.py

eval:  ## LLM eval suite gated by evals/thresholds.yaml
	$(PY) scripts/run_evals.py

run:  ## Streamlit UI
	streamlit run streamlit_app.py

api:  ## FastAPI backend
	uvicorn api.main:app --reload
