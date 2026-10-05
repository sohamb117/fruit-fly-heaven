# Exact BANC–FANC haltere annotation join

**A partial exact join exists.** Nine BANC v888 neurons have recorded FANC matches to the authors' CL3 cluster, with consistent side, haltere nerve, afferent status, and agreement between `fanc_match` and `fanc_nblast_match`. This is an auditable cross-dataset annotation transfer, not independent proof of receptor tuning or one-to-one neuron equivalence. No runtime source, wiring, or simulation was changed.

## Evidence and reproducibility

The authors' [pinned atlas CSV](https://github.com/serene-da1/Dhawan-et-al-2025-/blob/970feb58f6ab28435340958df1c429fbc031758e/Data/Annotation%20tables/final_haltere_clusters.csv) contains 372 haltere afferents, including 71 CL3 cells. The [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12872070/) identifies CAVE v840 as the analysis version.

The official FANC public bucket contains exported `cell_ids_v2` and `peripheral_nerves` tables at [v1237](https://storage.googleapis.com/lee-lab_female-adult-nerve-cord/CAVE/v1237/cell_ids_v2.parquet) and [v1444](https://storage.googleapis.com/lee-lab_female-adult-nerve-cord/CAVE/v1444/cell_ids_v2.parquet). Generation-pinned download URLs, GCS object metadata, SHA-256 hashes, and local paths are in [sources/fanc-public-tables-provenance.json](sources/fanc-public-tables-provenance.json). Arrow metadata independently declares the datastack and materialization version.

Both snapshots give the **same 332 matches**, including **64 CL3 cells**, under simultaneous exact equality of root ID, supervoxel ID, and anchor position. Stable ID, table row ID, side, and peripheral nerve also agree across the two snapshots. The remaining 40 atlas rows, including 7 CL3 rows, stay unassigned; the available data do not establish why those anchors are missing.

Run from repository root:

```sh
.venv/bin/python reports/flight-b1-mapping/analyze.py
```

The script verifies input hashes before computing results. It preserves IDs as strings or uint64, verifies all 175,401 prepared connectome IDs, and checks equality of both FANC joins. Main outputs:

- [mapping-audit.json](mapping-audit.json): machine-readable counts, checks, and nine-cell subset.
- [cl3-recorded-match-subset.csv](cl3-recorded-match-subset.csv): nine exact chains, including current root, v888 root, prepared neural index, FANC IDs, cluster, and QC columns.
- [verified-fanc-atlas-join.csv](verified-fanc-atlas-join.csv): all 332 verified FANC/atlas rows.
- [verified-banc-fanc_match-join.csv](verified-banc-fanc_match-join.csv): 57 recorded BANC matches; all are haltere afferents with consistent sides.
- [verified-banc-fanc_nblast_match-join.csv](verified-banc-fanc_nblast_match-join.csv): NBLAST-only audit, kept separate.
- [unresolved-author-atlas-rows.csv](unresolved-author-atlas-rows.csv): the 40 unmatched atlas rows.
- [provenance.json](provenance.json): all analysis input hashes and source URLs.

## Nine-cell CL3 subset

| Neural index | BANC v888 ID | BANC type | Side | FANC stable cell ID |
|---:|---|---|---|---:|
| 9524 | 720575941428193199 | SApp07 | left | 4485 |
| 30051 | 720575941461628560 | SApp07 | right | 4427 |
| 51778 | 720575941491875047 | SApp07 | right | 4422 |
| 58416 | 720575941501675826 | SApp07 | right | 4422 |
| 95036 | 720575941544353460 | SApp07 | left | 4485 |
| 99786 | 720575941550391668 | SApp07 | left | 4485 |
| 133862 | 720575941594303527 | SApp07 | right | 4422 |
| 141921 | 720575941609599078 | SApp01 | right | 4422 |
| 174042 | 720575941721004602 | SApp07 | left | 4485 |

The exact FANC root chains are 4485 → `648518346496851864`, 4422 → `648518346504488684`, and 4427 → `648518346501406051`. Each is CL3 in the pinned author table. The table's field annotation is `dF2,vF2`; it does not identify which field each individual afferent innervates.

## Limits that matter for implementation

- **Identifier semantics:** the [BANC metadata producer](https://github.com/htem/bancpipeline/blob/5c333c12f0b9e03873f88cf4e23cad34c0bb49c1/fanc/fanc-meta.R) derives `cell_id` from CAVE `id`. The FANC table documents `user_id` as the stable identifier, and [FANC Python](https://github.com/htem/FANC_auto_recon/blob/89d9769457583051e3ec9ec7b34a5f3575537bf5/fanc/lookup.py) uses that column. They are equal for every v1237 row and all 332 matched rows in both snapshots; seven unrelated v1444 rows differ. Do not assume these columns remain interchangeable.
- **Recorded matches are many-to-one:** 106 of 439 BANC haltere neurons have `fanc_match`, referring to 27 FANC IDs. The nine CL3 rows refer to only three FANC neurons. Exact identifier conversion does not independently validate that cross-animal match.
- **Do not expand by BANC type:** eight SApp07 rows map to CL3, but two other recorded SApp07 rows map to CL4b. One SApp01 maps to CL3. The data do not support an exhaustive SApp07→CL3 rule.
- **NBLAST needs separate QC:** 115 total BANC NBLAST rows join the atlas, 62 disagree on side, and one is a descending neuron without a haltere annotation. The [NBLAST producer](https://github.com/htem/bancpipeline/blob/5c333c12f0b9e03873f88cf4e23cad34c0bb49c1/banc/nblast/banc-fanc-nblast.R) documents mirrored skeleton preparation and strips the mirror marker. Opposite-side matches can be intentional morphology comparisons, but cannot directly supply the BANC receptor's side. Five haltere rows have automatic FANC AN/DN type labels; those are not used as sensory identities.
- **No physiology follows from this join:** cluster membership supplies neither preferred rotational axis/polarity, preferred wing/haltere phase, threshold, gain, temporal response, nor firing dynamics. Those require additional evidence or explicitly labeled model assumptions. No synapses were inferred or added.
- **Missing data:** completing the 40 unmatched atlas rows requires a v840 `cell_ids_v2` export containing both `id` and `user_id`, or a source-pinned graph-history mapping from their atlas anchors. The public bucket listing exposes only v1237 and v1444 exports. The author notebooks contain no cell-ID conversion table, and the public BANC franken snapshot has no FANC identifier columns. These absence checks are recorded under `sources/`; they do not prove no other public export exists.

The inspected 23.7 MB franken snapshot was removed after recording its checksum and schema because it could not provide the join. Retained artifacts are approximately 5 MB, including the useful public lookup tables.

## Existing direct B1 routes

The follow-up [b1-route-audit.json](b1-route-audit.json) reads only the two B1 incoming CSR ranges, enumerates the four electrical directions, and cross-checks chemical contact counts against batches of the pinned raw v3 edge table. All seven existing same-side chemical edges use the configured nAChR receptor and 2 ms delay.

| CL3-subset source index | Same-side B1 target ID | Raw chemical contacts | Configured integrated weight, nS·ms |
|---:|---|---:|---:|
| 9524 | 720575941521196211 | 7 | 0.7 |
| 95036 | 720575941521196211 | 5 | 0.5 |
| 99786 | 720575941521196211 | 5 | 0.5 |
| 174042 | 720575941521196211 | 1 | 0.1 |
| 30051 | 720575941549822781 | 0 | no edge |
| 51778 | 720575941549822781 | 7 | 0.7 |
| 58416 | 720575941549822781 | 17 | 1.7 |
| 133862 | 720575941549822781 | 0 | no edge |
| 141921 | 720575941549822781 | 3 | 0.3 |

Thus the existing same-side totals are 18 contacts to left B1 (index 75865, approximately 1.8 nS·ms) and 27 to right B1 (index 99458, approximately 2.7 nS·ms). The JSON stores exact Float32 values; these weights use the existing assumed 0.1 nS·ms/contact conversion, not measured conductances.

**None of the existing electrical edges touches this subset or either B1.** All four directions belong to two symmetric configured pairs: `720575941509145950` ↔ `720575941549397247` and `720575941451068597` ↔ `720575941643797624`, each approximately 0.02 nS. Their configuration explicitly labels the GF–PSI type correspondence and conductance as assumptions, not BANC-observed electrical connections. This audit supplies no basis for adding a haltere–B1 gap junction or for claiming sufficient B1 excitation.

Reproduce with `.venv/bin/python reports/flight-b1-mapping/audit-b1-routes.py`. Source hashes and both-side rows are saved in the JSON and [b1-routes.csv](b1-routes.csv). No neuron, synapse, or body was stepped.
