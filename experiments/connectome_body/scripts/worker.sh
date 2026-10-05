#!/usr/bin/env bash
set -euo pipefail
cb_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$cb_project_root"
if [[ $# -lt 1 || $# -gt 3 ]]; then
  printf 'Usage: bash scripts/worker.sh PLAN_JSON [SHARD_INDEX [SHARD_COUNT]]\n' >&2
  exit 2
fi
exec uv run --frozen cbbench worker --plan "$1" --shard-index "${2:-0}" --shard-count "${3:-1}"
