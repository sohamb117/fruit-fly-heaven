# BANC contact-modality audit and scoped correction

The preparer assigned ten BANC v888 cells to mechanical leg collision because their broad class was `bristle_neuron`, despite explicit chemical function annotations. Four are sugar/low-salt cells; six are contact-pheromone cells. Actual dry native-format collision drove all ten at 60 Hz even with the taste switch disabled. None belongs to the existing 532-cell sugar group. The same broad-class rule assigned one `joint_angle` bristle (SNta35, index 42986) to collision despite unassigned joint identity and tuning. Both are input-classification errors, not changes in the connectome.

The correction excludes these exact chemical labels and the unassigned `joint_angle` bristle from mechanical preparation and adds runtime guards for older artifacts. Existing chemical group membership, organ taste mapping, graph connectivity, neuron parameters and muscle dynamics are preserved. All eleven cells retain their recurrent graph activity and receive no invented substitute sensory drive. Existing mapped position receptors remain active.

## Measured fixture result

| Production encoder / isolated WASM assay | Before | After |
| --- | ---: | ---: |
| Dry-collision requested rate per excluded chemical cell, taste off or on | 60 Hz | 0 Hz |
| Added dry-collision current per affected configured profile | 30.50938797 pA | 0 pA |
| Supported mechanical bristle cells receiving 60 Hz | 2,965 | 2,965 |
| SNta35 unassigned-position bristle collision rate | 60 Hz | 0 Hz |
| SNta35 added collision current | 30.50938797 pA | 0 pA |
| Existing mapped position receptors | 403 | 403 |
| Mapped position receptors, neutral / moved-joint fixture | 5 / 10.5 Hz | 5 / 10.5 Hz |
| Left-front sugar cells receiving 150 Hz from actual-format left-front food contact | 73 | 73 |
| Isolated configured-profile output for requested 0 / 60 / 150 Hz | 0 / 60 / 150 Hz | 0 / 60 / 150 Hz |

The fixture uses the real SensoryEncoder, prepared identities, taste mapper and production WASM profile calibration with synthetic native-format contact feedback. It is not a whole-brain or behavioral experiment. The 150 Hz food-contact value remains an unmeasured model prior. The measured current fit calibrates the configured neuron model, not biological receptor physiology.

## Source identities

The ten chemical rows below and the SNta35 row were checked against `data/raw/banc888/meta.feather` and prepared uint64 IDs. All are front-leg bristle class cells; their explicit functions distinguish ten chemical cells from the one unassigned position receptor.

| Index | BANC v888 root ID | Side | Type | Source function |
| ---: | --- | --- | --- | --- |
| 7230 | 720575941413613716 | right | LgLG4 | sugar, low_salt, Gr64f, Ir56b |
| 63482 | 720575941507085697 | right | LgLG4 | sugar, low_salt, Gr64f, Ir56b |
| 70482 | 720575941515574371 | left | LgLG8 | contact_pheromone, ppk23, ppk25 |
| 90945 | 720575941538892925 | left | LgLG8 | contact_pheromone, ppk23, ppk25 |
| 100384 | 720575941551049471 | right | LgLG5 | contact_pheromone, ppk23, ppk25 |
| 101780 | 720575941552778039 | left | LgLG8 | contact_pheromone, ppk23, ppk25 |
| 125256 | 720575941579056249 | left | LgLG8 | contact_pheromone, ppk23, ppk25 |
| 128708 | 720575941586312260 | right | LgLG4 | sugar, low_salt, Gr64f, Ir56b |
| 140900 | 720575941606157794 | right | LgLG5 | contact_pheromone, ppk23, ppk25 |
| 156561 | 720575941636252277 | right | LgLG4 | sugar, low_salt, Gr64f, Ir56b |
| 42986 | 720575941480808867 | left | SNta35 | joint_angle; specific joint and tuning unassigned |

## Preparation and validation

Only `data/prepared/banc888/console/sensory-inputs.json` changed bytes during scoped console regeneration. It now records eleven exclusions: ten chemical and one unassigned position bristle. Their indices are removed from touch channels and body transducers. Touch channels contain 1,456 left and 1,509 right cells. The second scoped regeneration changed only index 42986; every other body-transducer record and all 403 mapped position receptors are identical. Every other console file is byte-identical. `ids.bin`, `io.json`, the BANC manifest and taste-peg supplement are unchanged; the supplement's identity and IO hashes still match the manifest and actual files.

The runtime guard prevents mechanical current when an older artifact still contains these annotations. The regenerated channels also remove the indices from the legacy aggregate/assisted path. The guards use precompiled token-boundary regexes without allocating token arrays on every sensory-cell update. The position guard requires the obsolete `touch` kind plus the explicit `joint_angle` token; it does not suppress mapped position-kind receptors.

Validation: 28 ground/taste/sensory-feedback tests pass after both corrections. Raw-data classification yields exactly 2,965 retained mechanical bristles, ten chemical exclusions and one unassigned-position exclusion. Before the SNta35 correction, the actual encoder fixture measured 60 Hz collision drive; all 403 mapped position receptors measured 5 Hz at neutral angles and 10.5 Hz at the configured moved angles. The corrected fixture verifies zero SNta35 added drive in both poses, unchanged mapped-position responses, unchanged ordinary touch and organ-specific sugar input. Exact source and console hashes are recorded in the JSON evidence.

Reproduce: `node --test web/test/banc-ground-sense.test.mjs web/test/banc-taste.test.mjs web/test/sensory-feedback.test.mjs` and `node scripts/audit-banc-contact-modality.mjs after`. The historical `before` mode requires the original pre-fix source and prepared mapping; do not rerun it against the corrected source.

## Native taste interface audit

No index, leg-order, side-order, sign or stale-current error was found in the audited organ taste path. The nine supplementary taste-peg IDs match the prepared uint64 identities. Of 532 requested sugar cells, 509 have supported organ and laterality; the remaining 23 explicitly abstain. Native arrays and mapper agree on left front/middle/hind followed by right front/middle/hind. Food-contact flags reset on refresh, require actual food-surface contact, and do not use self-contact or general presence above food. Updated zero rates overwrite prior added currents. Body time is included in encoder cache invalidation.

Remaining anatomical/model gaps: whole-wing contact is a wing-margin proxy; native mouth bodies are named `labrum_left/right` and used as the labellar-contact proxy. Sensillum receptive fields are not measured. Rotation sensing remains an unsigned, force-gated prior without native haltere kinematics. All 805 motor neurons retain one 16-parameter profile; that is a physiological calibration gap, not an index error corrected here.

The SNta35 annotation mismatch is now corrected by abstention. Its actual joint identity and numerical tuning remain unresolved; no position response is invented. Earlier chemical-only audit files counted 2,966 remaining bristles because they still included SNta35. The final count is 2,965 supported mechanical mappings plus that one explicitly excluded unassigned-position cell.

## Relation to observed overturning

These are existing observer telemetry values, not new visual reviews. In the organ-taste run, frame 94 at body 0.392 s has upZ 0.977, wing power 0.713 and zero contacts. Frame 95 at 0.662 s already has upZ -0.614 and zero contacts. Frame 96 at 0.852 s has upZ -0.043 and 77 contacts. All are finite, unpaused and without recorded runtime errors. The older run's corresponding early samples at 0.392 / 0.670 / 0.938 s had wing power zero and positive upZ.

The chemical/collision bug predates both runs. Organ-specific taste also changes other neural inputs, and these observations do not isolate the cause of the changed flight behavior. The correction must not be described as solving tumbling, stable flight, food localization, landing or feeding. The live observer/control/review log was not changed by this audit.

Evidence: [before fixture](contact-modality-before.json), [runtime guard on old artifact](contact-modality-runtime-guard.json), [corrected prepared fixture](contact-modality-after.json), [chemical-only raw classifier and hash checks](contact-modality-raw-classification.json), [SNta35 before correction](contact-modality-before-position.json), [final classifier and hash checks](contact-position-classification.json), [selected existing frame telemetry](contact-modality-frame-comparison.json).
