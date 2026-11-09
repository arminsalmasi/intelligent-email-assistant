PY ?= .venv/bin/python
UV ?= uv

.PHONY: help setup test lint format eval eval-offline eval-replay eval-live demo demo-auto api engine-tunnel dashboard mlflow docker-up docker-down k8s-validate clean

help:
	@echo "setup        create .venv and install all extras"
	@echo "test         run unit + graph + API tests (offline)"
	@echo "lint         ruff check + format check"
	@echo "eval         golden eval: offline + replay with threshold gate"
	@echo "eval-live    golden eval against Gemini (records the replay cache)"
	@echo "demo         interactive CLI review session (demo-auto: approve all)"
	@echo "             pick the LLM with LLM=auto|gemini|bedrock|local|none (also api, eval-live)"
	@echo "engine-tunnel SSH tunnel to the local engine on the Mac Studio (port 11434)"
	@echo "api          FastAPI on :8000 (docs at /docs)"
	@echo "dashboard    Streamlit on :8501 (needs the API)"
	@echo "mlflow       MLflow UI on :5000"
	@echo "docker-up    api + dashboard + mlflow via docker compose"
	@echo "k8s-validate client-side validation of deploy/k8s manifests"

setup:
	$(UV) venv --python 3.12 .venv
	$(UV) pip install --python $(PY) -e ".[dev,dashboard,tracing]"
	@test -f .env || cp .env.example .env

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check email_assistant evals tests dashboard
	$(PY) -m ruff format --check email_assistant evals tests dashboard

format:
	$(PY) -m ruff format email_assistant evals tests dashboard
	$(PY) -m ruff check --fix email_assistant evals tests dashboard

eval: eval-offline eval-replay

eval-offline:
	$(PY) -m evals.run --mode offline --check

eval-replay:
	$(PY) -m evals.run --mode replay --check

# Choose the LLM backend: make demo LLM=local   (auto | gemini | bedrock | local | none)
# Only honoured on the command line, so an exported shell variable named LLM is ignored.
LLM_CLI = $(if $(filter command line,$(origin LLM)),$(LLM),)
LLM_FLAG = $(if $(LLM_CLI),--llm $(LLM_CLI),)

eval-live:
	LLM_REQUESTS_PER_MINUTE=$${LLM_REQUESTS_PER_MINUTE:-60} $(PY) -m evals.run --mode live --mlflow $(if $(LLM_CLI),--provider $(LLM_CLI),)

demo:
	$(PY) -m email_assistant.cli demo $(LLM_FLAG)

demo-auto:
	$(PY) -m email_assistant.cli demo --auto $(LLM_FLAG)

api:
	$(if $(LLM_CLI),LLM_PROVIDER=$(LLM_CLI),) $(PY) -m uvicorn email_assistant.api.main:app --reload --port 8000

engine-tunnel:
	ssh -N -L 11434:127.0.0.1:11434 studio

dashboard:
	$(PY) -m streamlit run dashboard/app.py

mlflow:
	$(PY) -m mlflow ui --backend-store-uri ./mlruns --port 5000

docker-up:
	docker compose up --build

docker-down:
	docker compose down

k8s-validate:
	kubeconform -strict -summary deploy/k8s/

clean:
	rm -rf .pytest_cache .ruff_cache data mlruns evals/reports
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
