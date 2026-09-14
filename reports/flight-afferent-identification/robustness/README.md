# Predeclared response robustness cohort

Four manual, sequential invocations test the same three anatomical populations at two conditioning durations (100/300 ms) and two maximum pulse sizes (±1.25/±2.5 Hz per afferent). Each has two identical-baseline controls and six signed pulse branches, for32 branches total. Exact commands and argument arrays are in `plan.json`; no launcher runs them automatically.

Each experiment uses the same frozen native sensory context as the historical assay. Before neural execution, `--reference-result` requires identical neural fingerprints, source-artifact hashes, encoded currents, pose/internal context, and muscle recruitment. Each branch receives an independently verified copy of the conditioned neural state. Pulse arms run only after duplicate baseline traces and full final forward-state bytes match exactly.

The 300 ms prefix deliberately changes both neural and muscle conditioning time. It is not a restored later live-flight state. Readout remains2 ms; branches remain20 ms baseline,20 ms stimulation,60 ms recovery. All449 rotation-afferent readouts and48 wing-MN readouts remain recorded even though only three populations are stimulated.

Compare direct cumulative spike counts before the50 ms rate filter, filtered rates, and muscle force separately. For amplitude comparisons at a fixed prefix, first require matching prefix-forward-state hashes, prefix muscle state, and baseline traces across invocations. Assess response sign and amplitude scaling in the first10/20/40 ms; distinguish later recurrent divergence. These experiments assign no roll/pitch/yaw tuning and do not validate live-flight controllability.

The exact historical and revised assay sources are preserved in their report directories. `--describe` validates a command and prints its branch counts without network access, output creation, or neural simulation. Four declared parser checks and an invalid-prefix rejection passed before this plan was written.
