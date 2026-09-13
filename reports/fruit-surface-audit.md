# Fruit rendering and native heightfield audit

Read-only isolated geometry audit; original scene units (one millimeter under the native centimeter convention). No habitat, collision or controller edits.

| Position | Native 257 | Analytic surface | Actual FrontSide surface | Same vertices, DoubleSide | Native − analytic | Analytic − geometry |
|---|---:|---:|---:|---:|---:|---:|
| frame80 | 6.71211 | 6.71176 | -0.40000 | 6.71480 | 0.00035 | -0.00304 |
| spawn | 16.86249 | 16.86958 | 16.66792 | 16.66792 | -0.00708 | 0.20165 |

The bowl has a real rendering defect separate from the heightfield: its original lathe inner faces point downward. With the original FrontSide material, top-down rays skip the intended bowl floor and hit the lower shell. DoubleSide audit rays recover the same existing inner vertices. At frame 80 this hides a surface roughly 7.1 scene units above the lower shell, while native-vs-analytic floor error is only 0.00035. The audit uses a separate scratch material; production rendering is unchanged.

The nominal native grid spacing is 0.515625 scene units (257 points over 132 units). The 513-point comparison halves it. Native-minus-analytic isolates grid discretization; it cannot correct analytic/render shape differences.

| Sample region | Points | Mean abs native − analytic (257) | Mean abs native − analytic (513) | Mean abs analytic − geometric surface | Max abs analytic − geometric surface |
|---|---:|---:|---:|---:|---:|
| frame80 | 441 | 0.00032 | 0.00007 | 0.00591 | 0.00912 |
| spawn | 441 | 0.00742 | 0.00224 | 0.14618 | 0.29012 |
| banana_endpoint | 36 | 0.25617 | 0.02549 | 10.55232 | 20.63902 |
| apple | 12 | 0.01068 | 0.00207 | 0.10569 | 0.25093 |
| bare_bowl | 4 | 0.00020 | 0.00006 | 0.00000 | 0.00000 |

The stored 31-point banana centerlines exactly match the renderer formula (maximum coordinate error zero for all three bananas). There is no wholesale banana position/rotation mismatch.

There are real surface-model differences: the renderer uses open tube ends, while nearest-path distance adds rounded heightfield caps beyond the path endpoints; small rendered tip ellipsoids do not reproduce those caps. Apples and the bowl are coarse polygon meshes. The heightfield also cannot model under-fruit space or overhangs. These differences remain even if grid resolution is increased.

Frame 80 should be interpreted using its central sample, not the worst endpoint discrepancy elsewhere. Camera obstruction and surface mismatch are separate observations. All point samples, source hashes and the largest discrepancies are in fruit-surface-audit.json.
