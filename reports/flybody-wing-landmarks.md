The upright resting wing is a reproducible rendering mismatch. With native resting joint values `[yaw=1.5, roll=0.7, pitch=-1.0]`, both aerodynamic wing spans lie **6.19° above the thorax's horizontal plane**. The old Euler renderer shows the left span at **82.22°** and the right at **17.24°**. It reorders native joint angles into Euler XYZ and negates one side, omitting the native wing frame and joint composition.

`web/flybody-wing-pose.js` registers the existing wing ellipsoids directly to each native aerodynamic geometry's center and orientation. It preserves their original material, child position, and scale. Its returned `position` and row-major `rotation` apply to the existing `wing.hinge` group. That decorative group's origin is not the anatomical hinge; the separate `anchor` field is the true native hinge.

Create landmarks once with `createWingLandmarks(model, metadata)`, then call `sampleWingPose(data, landmarks, rootPosition, rootRotation)` beside mouth/leg sampling. It returns left/right poses in the existing original fly-mesh frame. Apply each pose to its wing parent using a quaternion from the rotation matrix. On the behavior-renderer fallback, restore the original parent position as well as its Euler angles. Clear stale physical wing poses when changing modes.

Native aerodynamic axes are `[thickness, chord, span]`, whereas the original ellipse uses `[span, thickness, chord]`. The helper resolves this permutation, the world Y/Z reflection, both anatomical sides, and the existing child-center offset. It discovers aerodynamic geometry indices from the runtime model, so adding habitat geometry does not shift them incorrectly.

Validation uses 16 independent native MuJoCo poses, including twelve arbitrary root attitudes, both wing sides, asymmetric wing angles, and an extra world geometry. MuJoCo WASM output passes through actual THREE parent/child transforms; maximum center/hinge/span-tip error is **3.97e-15 scene units**. Physical state is unchanged. Run:

```sh
.venv/bin/python scripts/inspect-flybody-wing-landmarks.py
node scripts/verify-flybody-wing-pose.mjs
```

Native fixtures are in `flybody-wing-landmarks.json`; test results are in `flybody-wing-pose-verification.json`. The preserved original span radius is 1.18 scene units, versus native 1.14; that leaves a deliberate 0.04-unit outline difference while preserving the appearance. The returned `nativeHalfExtents` can instead align the full ellipsoid if wanted.

The historical observer frames record wing power but do not include wing joint positions, so their exact past physical attitudes cannot be recovered. Zero power alone also does not imply a motionless wing because its springs and inertia remain active. The resting-pose comparison establishes the Euler bug independently of that missing historical state.
