#!/bin/zsh -l
set -eu

fish1_project_dir="${0:A:h:h}"
cd "$fish1_project_dir"
uv run --no-sync --with caveclient==8.2.1 python scripts/setup_fish1_access.py
