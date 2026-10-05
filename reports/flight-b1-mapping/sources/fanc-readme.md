# `fanc/` — FANC v1.116 reference metadata

Lightweight sister-dataset pipeline: pulls FANC (Female Adult Nerve Cord, v1.116) metadata for cross-dataset NBLAST and matching. Unlike `fafb/`, `manc/`, `malecns/`, `hemibrain/`, there is no published SJCABS-style data bundle for FANC — the metadata table is consumed only internally by `banc/nblast/banc-fanc-nblast.R` and the `franken_meta()` join.

## Scripts

- `fanc-meta.R` — Pull FANC v1.116 per-neuron metadata via `fancr`.
- `fanc-data.R` — Empty placeholder; reserved for any future FANC-release publishing.

## Executable for users?

No — needs HMS-O2 paths and FANC CAVE credentials.

See top-level [`README.md`](../README.md) and [`banc/nblast/`](../banc/nblast/) for the FANC NBLAST consumer.
