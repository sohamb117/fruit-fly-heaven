# Native FlyBody in MuJoCo WASM

Install the pinned official engine with `npm ci --prefix packages/flybody-runtime` from the repository root. The static server exposes this package at `/body-engine/`; node_modules is not checked in.

`scripts/prepare-flybody-runtime.py` exports the pinned FlyBody asset to `models/flybody-mujoco.xml` and its metadata. Visual meshes are replaced with their compiled inertias. Fifty articulated coordinates and all six root coordinates remain; 52 coordinates are frozen. The native solver advances contacts, actuator dynamics and ellipsoid wing forces.

`web/flybody-physics.js` converts annotated BANC motor-unit rates into WASM muscle forces, then native leg/mouth/claw actuator controls and a measured wing cycle. The actuator adapter is unfitted and does not provide coordinated navigation. `web/flybody-world.js` connects that physics to the original console, sensory feedback and internal state.

Run `node scripts/verify-flybody-runtime.mjs` to compare WASM against the exported native MuJoCo trajectory and test isolated motor stimulation/disconnection. Behavioral evidence is separate: see `reports/flybody-transplant.md`.
