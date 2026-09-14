#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p dist
em++ native/core.cpp -std=c++17 -O3 -msimd128 \
  -s MODULARIZE=1 -s EXPORT_ES6=1 -s ENVIRONMENT=web,worker,node \
  -s ALLOW_MEMORY_GROWTH=1 -s MAXIMUM_MEMORY=2147483648 -s FILESYSTEM=0 \
  -s EXPORTED_RUNTIME_METHODS='["HEAPU8","HEAPF32"]' \
  -s EXPORTED_FUNCTIONS='["_malloc","_free","_br_step","_br_step_dlm","_muscle_step","_joint_step"]' -o dist/core.js
cp src/neural.wgsl src/neural-dlm.wgsl dist/
