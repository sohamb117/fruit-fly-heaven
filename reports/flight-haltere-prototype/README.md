This folder contains an isolated **virtual haltere mechanical observer and bending-projection prior**. It preserves signed vectors and phase; it does not assign neurons, create sensory currents/spikes, add gap junctions, integrate a native body or change runtime code. Nine focused tests pass. This is a tractable mechanical input model, not validated haltere strain or a claim about flight stability.

The pure module is [virtual-haltere.mjs](virtual-haltere.mjs). Plain JSON geometry is in [geometry.json](geometry.json), extracted from the current native XML by [extract_geometry.py](extract_geometry.py). The descriptor contains each haltere's actual XML mass, body frame, attachment position and inertial COM. It also contains explicitly constructed oscillation and beam axes. The prepared native haltere bodies have **no joints**: none of the virtual oscillation is an observed native motion.

```js
validateGeometry(geometry); // returns an owned, frozen descriptor

const mechanical = evaluateVirtualHaltere(geometry, {
  omegaRootRadS: [wx, wy, wz], // native root-local angular velocity
  phaseRadians,              // supplied wing-table clock; not advanced here
  frequencyHz,
  power,                     // normalized side-specific drive, strictly [0,1]
}, {
  amplitudeMaxRadians: Math.PI / 4,
  phaseOffsetRadians: Math.PI,
  powerExponent: 1,
});

const projected = projectHaltereMoment(geometry, mechanical, {
  orientationRadians: [Math.PI/4, 3*Math.PI/4, 5*Math.PI/4, 7*Math.PI/4],
  momentScale: 1, // mandatory positive scale in the geometry's moment units
});
```

`mechanical` contains `rRoot`, `positionRoot`, `rdotRoot`, `rddotRoot`, `coriolisAccelerationRoot`, and separate `baselineForceRoot`, `coriolisForceRoot`, `totalForceRoot`, `baselineMomentRoot`, `coriolisMomentRoot`, `totalMomentRoot`. It also records `theta`, `thetaDot`, `thetaDDot`, power, frequency and oscillation phase. `rRoot` is the COM lever **from the attachment**, whereas `positionRoot` includes the attachment's root-local position. Moments are about that attachment, not the whole-fly COM.

`projected` contains `beamBendingComponents`, `signedBendingMoments`, `normalizedSigned`, nonnegative `compression`, and the explicitly named `beamFrame: 'fixed-root-attachment'`. Outputs and validated descriptors are owned copies; no input state is mutated. Unknown mechanical/projection prior keys are rejected. Nonfinite inputs, nonunit axes, invalid frames, zero mass/lever, bad normalization, invalid power/frequency and overflowing results fail explicitly.

For axis \(n\), neutral lever \(r_0\), supplied phase \(\phi\), frequency \(f\), and held normalized power \(P\),

\[
A=A_{max}P^p,\quad \psi=\phi+\delta,\quad
\theta=A\sin\psi,\quad \dot\theta=A(2\pi f)\cos\psi,\quad
\ddot\theta=-A(2\pi f)^2\sin\psi.
\]
\[
r=R(n,\theta)r_0,\quad
\dot r=\dot\theta(n\times r),\quad
\ddot r=\ddot\theta(n\times r)+\dot\theta^2[n\times(n\times r)].
\]
\[
F_0=-m\ddot r,\quad F_C=-2m\Omega\times\dot r,\qquad
M_0=r\times F_0,\quad M_C=r\times F_C,\quad M=M_0+M_C.
\]

Thus \([M(+\Omega)-M(-\Omega)]/2=M_C\) and the even component equals \(M_0\). Zero drive or zero frequency yields zero load **within this restricted model**. Taking a magnitude would discard information: in an ideal planar point-mass model the baseline and Coriolis moments can be orthogonal, making total-moment magnitude even under rotation reversal. The helper never substitutes a norm for these vectors.

For fixed beam tangent \(t\), fixed transverse normals \(b_1,b_2\), and modeled receptive orientation \(\alpha\),

\[
q=M\times t,\qquad
s_\alpha=q\cdot(b_1\cos\alpha+b_2\sin\alpha),\qquad
z_\alpha=s_\alpha/M_{scale},\qquad c_\alpha=\max(0,z_\alpha).
\]

The paired orientations differ by π, so their compression difference recovers the raw signed normalized projection. This resembles the sign pattern of axial bending stress around an ideal beam. **It is not measured strain:** cross-section geometry, elastic compliance, shear and local sensillum mechanics are absent. The tangent and normals remain fixed in the root attachment frame while the virtual COM rotates. A receptor frame that rotates with the stalk is a different physical assumption and is not implemented here.

| Descriptor/prior | Provenance and limit |
|---|---|
| Mass, attachment, body quaternion, inertial COM | Current `models/flybody-mujoco.xml`; exact raw descriptors and source hashes retained |
| Native units | g, cm, s, consistent with `mass_g` metadata and native gravity 981 cm/s²; MuJoCo itself is unit-agnostic |
| Concentrated mass | Entire native haltere mass placed at its inertial COM; native inertia tensor intentionally unused |
| Oscillation axis | Native haltere default-class joint axis transformed by normalized native body quaternion; **virtual axis prior**, because prepared halteres are fixed |
| Beam tangent | Neutral attachment-to-COM direction; not a measured stalk centerline |
| Beam normals | Oscillation axis projected perpendicular to tangent, then a right-handed cross product; not receptor-field annotations |
| Amplitude | 45° maximum, explicitly unmeasured and adjustable |
| Power scaling | Linear by default, exponent adjustable; native force is a drive proxy, not measured stroke amplitude |
| Phase offset | π relative to the supplied simulator table clock; no claim of measured antiphase alignment |
| Receptive orientations | Four opposed modeled orientations; not assigned to BANC cells or inferred from CL3/field matches |
| Normalization | Caller-supplied positive moment scale; `1` in the example is only a unit-scale example, not a calibrated receptor gain |

Scope omissions are deliberate: root translational acceleration, Euler acceleration \(\dot\Omega\times r\), body centrifugal acceleration \(\Omega\times(\Omega\times r)\), gravity, aerodynamics, distributed-inertia effects, structural elasticity and shear. Power/frequency are held for derivative evaluation; their derivatives are omitted. A discontinuous change between body blocks must not be interpreted as a modeled envelope acceleration. The function returns `heldEnvelope: true` to expose that assumption. No root force or desired attitude enters the observer.

Reproduce the bounded verification from the repository root:

```sh
.venv/bin/python reports/flight-haltere-prototype/extract_geometry.py
node --test reports/flight-haltere-prototype/virtual-haltere.test.mjs
node reports/flight-haltere-prototype/write-ledger.mjs
```

The tests check exact odd/even Coriolis decomposition, zero drive, attachment-origin conventions, finite-difference velocity/acceleration and second-order refinement, constant-radius identities, proper coordinate-rotation covariance, g/cm→kg/m unit covariance, opposed compression recovery, sampling refinement, immutability and invalid-input rejection. Unit conversion gives force factor \(10^{-5}\) and moment factor \(10^{-7}\); normalized projections remain invariant when the scale is converted too. Test output is saved in [tests.log](tests.log); [ledger.json](ledger.json) pins source/test/geometry inputs.

The time-refinement test integrates samples of this analytical observer over a fixed 2 ms interval with held Ω; it does not simulate the fly. **It does not establish sensory spike timing.** A runtime bridge must separately declare how phase is sampled between 2 ms body updates, how current is applied on 0.5 ms neural steps, and whether it holds or averages each interval. Advancing virtual phase from the most recent body clock with held Ω is causal; replaying future native observations into past neural intervals is not. No such bridge, current calibration or physiological validation is part of this prototype.

Existing supporting anatomy and its limits are summarized in [the saved anatomy review](../flight-sensory-observability/anatomy-review.md); the original primary-source tables do not supply these per-cell mechanical/recruitment parameters. The prototype intentionally leaves cell assignment and physiology to a separate review.
