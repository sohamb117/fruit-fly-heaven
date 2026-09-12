#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source references/emsdk/emsdk_env.sh >/dev/null 2>&1
target=packages/fly-brain-wasm/dist
mkdir -p "$target"
em++ packages/fly-brain-wasm/native/core.cpp -std=c++17 -O3 -fexceptions \
  -s DISABLE_EXCEPTION_CATCHING=0 -s MODULARIZE=1 -s EXPORT_ES6=1 \
  -s EXPORT_NAME=createCore -s ENVIRONMENT=web,worker,node \
  -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=33554432 -s MAXIMUM_MEMORY=2147483648 \
  -s FILESYSTEM=0 -s EXPORTED_RUNTIME_METHODS='["HEAPU8","HEAPU32","HEAPF64","UTF8ToString"]' \
  -s EXPORTED_FUNCTIONS='["_malloc","_free","_fb_error","_fb_graph_create","_fb_graph_destroy","_fb_brain_create","_fb_brain_destroy","_fb_inputs","_fb_currents","_fb_inject","_fb_refractory","_fb_step","_fb_read","_fb_time","_fb_spike_total","_fb_spikes"]' \
  -o "$target/core.js"
cp packages/fly-brain-wasm/src/index.js packages/fly-brain-wasm/src/index.d.ts "$target/"
if [ -f packages/fly-brain-wasm/src/worker.js ]; then cp packages/fly-brain-wasm/src/worker.js "$target/"; fi
printf 'Built %s\n' "$target/core.wasm"
