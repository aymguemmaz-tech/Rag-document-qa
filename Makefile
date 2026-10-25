.PHONY: install dev lint format typecheck test test-db eval eval-gpt gate calibrate serve up down demo clean

install:            ## install the package
	pip install -e .

dev:                ## install with dev, eval and docs extras
	pip install -e ".[dev,eval,docs]"

lint:               ## ruff lint + format check
	ruff check . && ruff format --check .

format:             ## auto-format
	ruff format . && ruff check --fix .

typecheck:
	mypy

test:               ## unit tests (pgvector tests are skipped without a database)
	pytest

test-db:            ## all tests against the docker compose database
	RAGQA_TEST_DATABASE_URL=postgresql://ragqa:ragqa@localhost:5432/ragqa pytest

eval:               ## offline ablation grid -> results/offline/report.md
	ragqa eval run --spec configs/ablation.yaml

eval-gpt:           ## LLM ablation grid (needs OPENAI_API_KEY) -> results/gpt/report.md
	ragqa eval run --spec configs/ablation_gpt.yaml

gate:               ## CI quality gate on the offline baseline
	ragqa eval gate --config configs/quality_gate.yaml

calibrate:          ## tune the extractive abstention threshold on the dev split
	ragqa eval calibrate --spec configs/ablation.yaml --split dev

serve:              ## API + web UI on http://127.0.0.1:8000 (lite mode, no database)
	ragqa serve

up:                 ## API + pgvector with docker compose
	docker compose up --build -d && docker compose run --rm api ragqa ingest data/corpus

down:
	docker compose down

demo:               ## index the sample corpus and ask a question (lite mode)
	ragqa ingest data/corpus && ragqa ask "How long is identifiable GPS trip data kept?"

clean:
	rm -rf .ragqa .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist *.egg-info
