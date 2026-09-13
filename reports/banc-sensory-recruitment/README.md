# BANC sensory recruitment audit

The annotation fixes are implemented. They do **not** recover the requested natural feeding/flight sequence. Full-connectome input-only assays show that the current six-tarsus sugar input does not recruit the proboscis extensors, even with body inputs disconnected. The supported labellar inputs can recruit those neurons. No motor injection, neural suppression, weight change, or behavioral controller was added to production.

## Final routing and evidence

- **Seven external-sugar inputs restored:** indices 7230, 63482, 128708, 156561 (right front leg), 60605 (right middle leg), 68890 and 69032 (left labellum). The group now has 539 candidates, 516 mapped by BANC organ/side and 23 with unknown laterality that abstain. Contact remains the existing **150 Hz prior**, not a measured sugar concentration–response curve.
- **One conflicting sugar annotation explicitly excluded:** index 19757, BANC `720575941442395599`, matched FAFB `720575940628672122`. Both its BANC coarse type and its exact FlyWire match are mechanosensory BM_Taste, conflicting with BANC's detailed sugar label. It already received zero current because its BANC side is null. That side is not inferred from another dataset.
- **The two restored labellar cells are supported by exact identity evidence.** BANC labels both BM_Taste/tactile, but its detailed function says sugar/Gr64f. Their individual BANC `fafb_match` targets `720575940639259967` and `720575940621754367` are classified as gustatory sugar/water **LB3** neurons in the original FlyWire files. Generic BM_Taste type identity would have given the wrong conclusion. Original conflicting labels, matches, source records and checksums are preserved in [the evidence bundle](../../models/banc-sensory-annotation-evidence.json). Sources: [FlyWire classification v783](https://storage.googleapis.com/flywire-data/codex/data/fafb/783/classification.csv.gz), [cell types v783](https://storage.googleapis.com/flywire-data/codex/data/fafb/783/consolidated_cell_types.csv.gz), and [BANC v888 metadata](https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888/banc_888_meta.feather).
- **360 explicitly frequency-sensitive auditory cells no longer receive generic body tilt/static antenna/speed current.** All 579 other antenna chordotonal cells remain, with their existing input values. A/B vibration sensing is distinct from static deflection sensing; D and mixed groups must not be removed on a generic auditory label. [Matsuo et al. 2014, Fig. 2](https://doi.org/10.3389/fphys.2014.00179). BANC's high/low-frequency labels are preserved; no frequency tuning was invented. This removes unsupported added input, not intrinsic or recurrent neural activity.

The prepared graph, neuron indices, IO, parameters and delays are unchanged. The existing supplement URL now contains 16 missing anatomical records. A mixed console/supplement version with missing sugar identity fails visibly in the worker. The [encoder assay](encoder-assay.json) verifies all retained inputs and each new organ/side assignment. Six Python classifier tests and 30 JavaScript sensory tests pass.

## Full-neural findings

Each condition starts from the same deterministic BANC state, uses the production calibrated current mapper and original graph/physiology, and runs 200 ms at 0.5 ms with fixed hunger/AKH/insulin. No body or visual/odor simulation runs. These are causal input comparisons, not observations of successful behavior. Actual cumulative spike counts and 20 ms samples are retained alongside filtered Hz.

The first [matched assay](full-neural-assay.json) completed in 25.3 wall seconds. Removing unsupported auditory current modestly changed DLM activity under its synthetic posture. Correcting the two left labellar cells changed that isolated labellar stimulus from DLM 100.6/100.5 Hz to 0/0 Hz, with m9 rising from 0.87 to 11.43 Hz. It does not establish that the physical fly can reach a labellar contact.

The [actual-posture assay](actual-posture-neural-assay.json) ran 29 conditions in 64.7 wall seconds. Its held feedback is derived from the exact new solid-fruit capture's initial native posture. A native forward pass confirms all six tarsi contact fruit, neither labellum nor wing does, and bristle collision flags are zero because initial contacts are on claws. Recorded initial leg loads are retained. Pair trials activate every one of the 15 possible two-tarsus combinations independently; they are controlled stimulus subsets, not observed contact sequences.

| Held input, after corrections | DLM left/right, Hz | m9, Hz | m4b, Hz |
|---|---:|---:|---:|
| Six tarsi only | 88.8 / 85.6 | 0 | 0 |
| Six tarsi plus actual initial body feedback | 98.6 / 101.4 | 0 | 0 |
| Initial body feedback only | 145.0 / 150.5 | 0 | 0 |
| Six tarsi, body input excluding antenna | 61.1 / 64.1 | 0 | 0 |
| Left labellum only | 0 / 0 | 11.4 | 0 |
| Right labellum only | 106.5 / 110.0 | 147.1 | 16.2 |
| Both labella only | 89.6 / 89.3 | 126.1 | 18.3 |
| Six tarsi, both labella and body feedback | 125.3 / 127.5 | 92.7 | 0.35 |

All 15 tarsal pairs have zero m9 and m4a/m4b output. Removing position, load or antenna added input separately also leaves six-tarsus m9 at zero. Thus body feedback is not necessary for the failure to recruit PER in these 200 ms tests. These comparisons do not identify one inhibitory circuit, and do not justify disabling a sensory class. m4a remains zero in all listed conditions. Broad stimulation of every external sugar organ is a positive control and is not a natural food-contact substitute.

The final provenance cleanup removed only the zero-driven conflicted candidate 19757. [Final current parity](final-input-parity.json) checks exact index, current and requested rate against every stored condition from both GPU assays. The sugar-summary denominator changes from 540 to 539; neural input vectors do not. Both browser processes were closed after their assays.

## Physical reach and remaining flight-sensing limit

The [native posture fixture](actual-posture-fixture.json) tests 625 rostrum/haustellum configurations across their extension ranges while holding the original root and leg posture. **None contacts fruit with either labellum.** This is a sampled geometric result, not an exhaustive continuous reach proof. It shows that initial tarsal contact cannot be turned into feeding merely by extending the two currently movable mouth joints at that held posture. Coordinated body/head/leg posture and neural recruitment still require validation. The test changes isolated joint positions only; it introduces no production pose control.

The rotation adapter also remains explicitly provisional. BANC supplies 449 `mechanical_strain` afferents across 34 types: 171/157 left/right haltere, and 62/59 wing-base. The current force-gated scalar is `5*activity + 4*power*norm(angularVelocity)`, capped at 100 Hz. At fixed muscle activity it is identical for positive and negative pitch, roll or yaw. It does not consume wing phase, and native haltere kinematics/strain are not simulated. The interface therefore cannot itself encode a signed, phase-sensitive corrective rotation signal. Other sensory channels might supply information, but this adapter is not evidence of calibrated closed-loop flight. Numerical axis/sign/phase assignments cannot be obtained from the current `mechanical_strain` labels alone; measured sensillum tuning and registration to native organ geometry are still needed.

## Reproduce

`uv run --offline --with pyarrow --with numpy python scripts/prepare-banc-console.py` regenerates console annotations and the checked supplement, without rebuilding connectivity. `node scripts/audit-banc-sensory-recruitment.mjs` and `node scripts/verify-banc-assay-input-parity.mjs` run offline encoder/WASM checks. The full-neural scripts require an existing local server and an otherwise idle browser/GPU: `node scripts/probe-banc-sensory-recruitment.mjs` and the same command with `--protocol=posture`. The latter consumes the captured [posture fixture](actual-posture-fixture.json).
