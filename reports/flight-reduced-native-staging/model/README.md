This is a staged flight-only native model, prepared from the exact legacy bundle. It removes 44 nonwing hinge elements and 50 nonwing/adhesion actuators, leaving the free root and six wing hinges with six wing actuators. No root weld, equality constraint, runtime pose reset or added actuator is introduced. It has not been compiled or simulated by this preparation task and is not proof of flight or a promoted model.

The predicted base-model dimensions are `nq=13`, `nv=12`, `njnt=7`, `nu=6`, `neq=0`, `nbody=68` including world, `ngeom=74`, and `nsite=15`. All 67 physical bodies and inertials remain. Body transforms, geometry, sites, root free joint, wing hinge/actuator definitions, compiler settings and physical options are preserved byte-for-byte. The XML contains additional default `geom`/`site` templates, which are not compiled objects. Habitat scene insertion will add world geoms later.

The removed coordinates all have XML reference value zero. Their passive `springref` values are **not** coordinate references: for example the rostrum has `springref=0.8` but freezes at reference coordinate 0. Bodies remain in their existing XML local transforms, matching the historical positive-control reduction. Removed hinge stiffness, damping, armature and actuator behavior disappear intentionally; this is a distinct reduced plant, not a claim of identical full-body dynamics.

| Metadata field | Contract |
| --- | --- |
| `dynamics_variant` | Exactly `{schemaVersion:1, kind:'fixed-nonwing-flight-v1'}` |
| `joints` / `actuators` | Six original wing records; XML-order predicted joint ids 1–6, qpos 7–12, dofs 6–11, actuator ids 0–5 |
| `fixed_joint_poses` | 44 entries with anatomical name, neutral/range, `fixedPosition`, `fixedVelocity`, retained `bodyId`/`bodyName`, and body-local anchor; no stale `id`, `qpos` or `dof` |
| `frozen_joints` | Original 52 names followed by the 44 removed names |
| `initializeStance` | `false`; articulated stance solving is inappropriate for this model |
| `fixed_muscle_inputs` | All 135 original mapping indices, in order, with joint/kind/target/direction and explicit length/velocity inputs. 74 leg rows are `fixed-joint`; 61 original virtual rows remain `existing-virtual`. All are length 1 and shortening velocity 0 for this reference pose. |
| `muscle_mapping_contract` | Exact IO SHA, 135-row identity hash, original 28 wing mapping indices and exact 48 motor index/root-id identities |
| `native_validation` | Status `required`, predicted base dimensions, ordered body/site names and root identity; these are not a substitute for native compilation checks |

The IO data, mapping ordering, neural graph, intrinsic profiles, event priors, 27 interpreter parameters and wing transfer are not changed. `fixed_muscle_inputs` supplies kinematics only; native activation/fatigue and live motor events must still evolve normally. The declaration does not prescribe muscle forces. Frozen leg/mouth actuation and sensory posture need the separately staged adapters; loading this XML with unmodified full-body adapters is unsupported.

`build.mjs` writes `model.xml`, `metadata.json` and `ledger.json`, refusing existing artifacts. `reduce.mjs` is pure and imports only the existing native leg-sign conversion. Seven pure tests check structural counts/index predictions, reference-pose preservation, source geometry bytes, exact reproduction of the historical positive-control XML transformation, all muscle/event identities, rejection of unsupported references/root constraints, and deterministic nonmutating preparation.

```sh
node --test reports/flight-reduced-native-staging/model/model.test.mjs
node reports/flight-reduced-native-staging/model/build.mjs
```

Native follow-up must compile the exact XML, resolve joint/actuator/body/site names and addresses, check the free root, masses, geometry and preserved contact configuration, and test the adapters. The expected base `ngeom` must account for later habitat injection. Any full-body or historical results remain separate; this model cannot establish coordinated leg-assisted takeoff, landing or feeding.
