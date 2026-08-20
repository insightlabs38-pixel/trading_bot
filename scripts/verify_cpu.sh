#!/usr/bin/env bash
set -euo pipefail

uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run python -m compileall -q src tests
uv run pytest -q
