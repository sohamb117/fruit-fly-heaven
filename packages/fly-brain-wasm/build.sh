#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p dist
em++ native/core.cpp -std=c++17 -O3 -fexceptions \
  -s DISABLE_EXCEPTION_CATCHING=0 -s MODULARIZE=1 -s EXPORT_ES6=1 \
  -s EXPORT_NAME=createCore -s ENVIRONMENT=web,worker,node \
  -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=33554432 -s MAXIMUM_MEMORY=2147483648 \
  -s FILESYSTEM=0 -s EXPORTED_RUNTIME_METHODS='["HEAPU8","HEAPU32","HEAPF64","UTF8ToString"]' \
  -s EXPORTED_FUNCTIONS='["_malloc","_free","_fb_error","_fb_graph_create","_fb_graph_destroy","_fb_brain_create","_fb_brain_destroy","_fb_inputs","_fb_currents","_fb_inject","_fb_refractory","_fb_step","_fb_read","_fb_time","_fb_spike_total","_fb_spikes"]' \
  -o dist/core.js
cp src/index.js src/index.d.ts src/worker.js dist/
printf 'Built %s/dist/core.wasm\n' "$PWD"
