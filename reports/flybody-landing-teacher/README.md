# Reproducible landing and walking teacher

This isolated artifact uses the released FlyBody flight and walking policies on a common articulated body. It is not BANC control. The only behavior transition is an explicitly prescribed descent, leg deployment, and a policy switch at actual native ground contact.

The regenerated run reached contact at **0.9044 s**, then completed **0.6000 s** of walking. Physical state continuity was exactly zero at the handoff. The final tilt was **4.8736°**, root height **0.13260 cm**, and externally applied force remained zero. Walking had ground contacts in 90.67% of control samples; momentary no-contact samples occur during the gait.

## Contents

- `teacher.json`: pinned source and policy hashes, original model hashes, parameters, trajectories, initial states, exact policy/action projections, transition priors, and measured outcome.
- `descent.xml`, `walking_handoff.xml`: self-contained mesh-free native models. Only noncolliding visual meshes were removed; compiled inertias were inserted. The common physical parameters are preserved exactly. A noncolliding ghost records the reference trajectory.
- `native-teacher-sequences.npz`: all **4,522 descent** and **300 walking** controls, policy inputs, and states. Policy inputs have widths **104** and **741** respectively. The full MuJoCo control vector has 65 entries in both phases.
- `native-replay-fixtures.json`: first and last ten control intervals per phase. Each has the physical state after task preparation and before its first physics substep, policy input/action, complete actuator control, and expected final state.
- `continuous-native-replay.json`: full native replay on the mesh-free models. Real-fly qpos, qvel, and activation errors are **exactly zero** across both phases. Full-state residuals of at most 1.14e-9 qpos and 2.59e-6 qvel belong exclusively to the noncolliding ghost.

NPZ keys are `<phase>/policy_input`, `<phase>/canonical_action`, `<phase>/real_action_before_task`, and `<phase>/<before_first_substep|after_control>/<time|qpos|qvel|act|ctrl|qacc_warmstart>`. Lengths are centimetres, masses grams, and time seconds. Walking resets the task clock to zero at handoff while preserving qpos, qvel and actuator activation. Add 0.9044 s for the total episode timeline.

## Reproduce

From the repository root, with `references/flybody` checked out at `d015e9bfe441bd90ae431bac24c55cb74bdbce26` and the released policy/waveform assets at the paths recorded in `teacher.json`:

```sh
uv venv --python 3.11 /tmp/flybody-baseline-venv
uv pip sync --python /tmp/flybody-baseline-venv/bin/python scripts/flybody-baseline-lock.txt
/tmp/flybody-baseline-venv/bin/python scripts/export-flybody-landing-teacher.py
/tmp/flybody-baseline-venv/bin/python scripts/verify-flybody-landing-teacher.py
```

The source revision is checked before generation. Python dependency versions are pinned in the lock file. Actual policies are loaded from the released archive at `https://ndownloader.figshare.com/files/44815195`; archive and individual policy hashes are recorded. The exporter does not train, modify the production body, change the UI, or use root forces.

For independent replay, restore the recorded state and full actuator control, then use native legacy Euler stepping: `mj_step2`, `mj_step1`, with `mj_forward` after the first substep to reproduce the sensor-binding refresh. Each flight control has four 50 μs substeps; walking has forty. Update only the noncolliding ghost to the recorded reference before each control. Preserve the actual fly state between controls. The supplied verifier demonstrates this sequence end to end.

This is a useful teacher and body-contact baseline. Its position servos, prescribed future trajectory, and contact-triggered switch are not evidence that BANC motor neurons already produce landing or walking. The separate grounded takeoff handoff tumbled and remains unresolved.
