#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source references/emsdk/emsdk_env.sh >/dev/null 2>&1
bash packages/fly-brain-wasm/build.sh
bash packages/brain-view-wasm/build.sh
bash packages/fly-vision-wasm/build.sh
