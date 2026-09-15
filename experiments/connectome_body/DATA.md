# Data sources and interchange

Audit date: 2026-09-15. URLs, versions, licenses, sizes, and SHA-256 values are recorded in [configs/sources.json](configs/sources.json). Prepared graphs include source/filter metadata and per-array hashes. A live URL without matching bytes is insufficient for reproducibility.

## Ready releases

**BANC v888 / v3.** The [documentation](https://github.com/sjcabs/fly_connectome_data_tutorial/blob/main/data/dataset_documentation/banc_data.md) and [deposit](https://doi.org/10.7910/DVN/7WTH1N) provide brain-and-ventral-nerve-cord connectivity. Exclude annotations explicitly labeled glia, trachea, or not-a-neuron and incident edges. Retain other annotations and endpoints, including isolates. Apply no additional connection-count cutoff; the published v3 table already reflects its release's synapse filtering.

The prepared snapshot contains **175,401 neurons, 13,542,180 directed edges, and raw weight sum 42,199,458**. The filter excludes 13,107 non-neuronal annotations and 78,685 incident edges. There are 12,668 isolates. These are imported-snapshot measurements, not imaging-volume headline counts. Optional transmitter-derived signs cover about 78.46% of nodes, with effect assumptions documented separately.

**MaleCNS v1.0 / minconf 0.5.** Use the official [flat-connectome downloads](https://male-cns.janelia.org/download/). The segment edge table includes fragments; treating every segment as an intact neuron would confound the experiment. Stream edges and retain both endpoints only when their annotation has `status == Traced`.

The resulting graph contains **165,122 neurons, 25,563,096 directed edges, raw weight sum 124,024,573, and 535 isolates**. Remove 101 autapses after membership filtering. No transmitter signs are imported for this release; the primary condition uses the shared seeded sign model.

Preparation may need several GB of disk and RAM. BANC can reuse local verified source files. Prepared MaleCNS is sufficient even if its redundant 1.05 GB raw edge download is no longer cached. Neither importer modifies source files.

## Pending releases

**Fish1 zebrafish.** The [CAVE documentation](https://fish1-release.storage.googleapis.com/programmatic.html) describes authenticated neuron/synapse queries and distinguishes stable soma/lore IDs from mutable segmentation roots. A benchmark export must freeze its materialization and root mapping together, state membership/proofreading coverage, aggregate neuron pairs, and checksum the resulting files. Do not join historical synapses to roots independently updated to another date.

No pinned Fish1 export has been obtained here. The registry marks it `requires_pinned_export`. An image volume, functional correlation matrix, or regional graph is not a substitute. Retain attribution under the release's [data policy](https://fish1-release.storage.googleapis.com/data_policy.html). Credentials are not part of run artifacts.

**Cockroach.** No suitable public synapse-resolved neuron graph was verified in this audit. `unverified_release` describes readiness in this suite, not a claim that no cockroach wiring data exist. A documented release can use the same importer when identified.

The full plan lists missing datasets and refuses to launch them. An explicit `--datasets banc malecns` plan scopes the campaign to fly releases; missing species are not counted as experimental failures or successes.

## Interchange format

`nodes.csv` needs unique `id` values and optionally a per-neuron `sign`: -1, 0 unknown, or +1. Include all neurons in the stated membership, including isolates. IDs must be strings or integers; floating-point IDs are rejected. Preserve leading zeros as text and never round segmentation IDs through a spreadsheet/floating-point intermediate.

`edges.csv` needs `pre`, `post`, and positive finite `weight`. Both endpoints must be in the neuron table. First aggregate individual synapses into directed neuron-pair counts. Duplicate pair rows are summed and autapses removed uniformly. Omit structural-zero edges. Column names can be overridden with `--pre`, `--post`, `--weight`, and `--node-id`.

Feather/Arrow IPC and Parquet are also supported. The importer validates structural consistency and recorded provenance; it cannot independently certify the biological interpretation of an arbitrary file.

Copy [examples/provenance.json](examples/provenance.json) and replace every placeholder. Record release, species, coverage, citation, license, membership filters, and weight meaning. For Fish1 additionally record datastack/materialization, root mapping snapshot, selected proofread population, and brain/spinal/ganglion/subset coverage.

```sh
uv run --frozen cbbench import-edges \
  --nodes /path/nodes.parquet --edges /path/edges.parquet \
  --node-id pt_root_id --pre pre_pt_root_id --post post_pt_root_id --weight synapse_count \
  --provenance /path/provenance.json --output data/graphs/fish1
uv run --frozen cbbench preflight --graph data/graphs/fish1 --output runs/fish1-preflight.json
```

Prepared directories contain `node_ids.npy`, `src.npy`, `dst.npy`, `weight.npy`, `signs.npy`, and `manifest.json`. Loading verifies fingerprints and uses memory-mapped arrays. The adapter consumes this common contract; new species do not require homologous-neuron mappings or actor changes.

## Nulls and size controls

Nulls preserve the exact invariants listed in [PROTOCOL.md](PROTOCOL.md). Their seed is independent of the training seed. Accepted swap counts and changed-target fractions are stored in run manifests and acute-rewire results.

`matched-subgraph` uniformly chooses nodes, keeps their induced edges, and uniformly thins those existing edges to the common target. It preserves surviving IDs, weights, annotations, parent fingerprint, and subset seed. It does not fabricate connections or produce an anatomically complete nervous system. Separate subset seeds remain separate graph identities; inspect conclusions across them.
