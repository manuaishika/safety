.PHONY: sync lint types test check
sync:
	uv sync
lint:
	uv run ruff check . && uv run ruff format --check .
types:
	uv run mypy
test:
	uv run pytest -q
check: lint types test
