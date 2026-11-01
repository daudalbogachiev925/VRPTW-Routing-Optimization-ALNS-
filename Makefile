INSTANCE ?= c101

.PHONY: setup solve benchmark serve test lint clean

setup:
	python -m venv .venv
	.venv/bin/pip install -e ".[dev]"

solve:
	.venv/bin/python scripts/solve.py --instance $(INSTANCE)

benchmark:
	.venv/bin/python scripts/benchmark.py

serve:
	.venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000

test:
	.venv/bin/pytest --cov=src

lint:
	.venv/bin/ruff check src tests

clean:
	rm -rf artifacts
