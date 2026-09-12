#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p dist
em++ native/core.cpp -std=c++17 -O3 -msimd128 \
  -s MODULARIZE=1 -s EXPORT_ES6=1 -s EXPORT_NAME=createCore \
  -s ENVIRONMENT=web,worker,node -s ALLOW_MEMORY_GROWTH=1 \
  -s INITIAL_MEMORY=16777216 -s MAXIMUM_MEMORY=1073741824 -s FILESYSTEM=0 \
  -s EXPORTED_RUNTIME_METHODS='["HEAPU8","HEAPU32","HEAPF32"]' \
  -s EXPORTED_FUNCTIONS='["_malloc","_free","_nv_colors","_nv_filter","_nv_pick","_nv_contour","_nv_sample"]' \
  -o dist/core.js
cp src/index.js src/index.d.ts src/worker.js dist/
printf 'Built %s/dist/core.wasm\n' "$PWD"
