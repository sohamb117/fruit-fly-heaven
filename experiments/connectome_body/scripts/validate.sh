#!/usr/bin/env bash
set -euo pipefail
cb_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$cb_project_root"
uv sync --frozen
uv run --frozen cbbench bootstrap
uv run --frozen ruff check connectome_body tests scripts
uv run --frozen pytest -q
