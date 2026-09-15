# Native antenna structure, opt-in profile v2

`web/banc-antenna.js` now supports `banc-native-antenna-family-airflow-v2`. The existing v1 profile, defaults and requested-rate bytes remain unchanged. This report covers the antenna module and its focused checks. The [combined structural implementation](../structural-proprioception-20260915/README.md) documents environment integration and the full BANC/native-body evaluation. No parameters were fitted and no training or deployment was performed.

## What is measured, transferred, and still assumed

The existing exact BANC v888 ID/annotation join supplies side and morphological JO family. It is unchanged. Literature describes responses of separately recorded populations; it does **not** identify the tuning of every neuron in this BANC specimen.

| BANC family | New declared structural prior | Status |
| --- | --- | --- |
| C / CL, 47 cells | Positive anterior-deflection tonic response | Family-level transfer, not individual physiology |
| D / DA, 12 cells | Positive anterior tonic term plus a separate unsigned velocity term | Mixed response supported; velocity weighting is unfit and does not reproduce a measured frequency-response curve |
| E subtypes, 329 cells | Opposite posterior-deflection tonic response | Family-level transfer; subtype heterogeneity remains unidentified |
| F / FVA, 174 cells | No added airflow input | Explicit abstention because the effective stimulus/transfer function is unresolved |
| JO-mz / untyped JO, 17 cells | No added airflow input | Previous explicit abstention retained |

The profile drives 388 cells (191 left / 197 right) and zeros added drive to 191 of the original 579-channel indices. This does not silence recurrent activity in those cells. All original indices remain in the replacement vector, preventing an abstained cell from borrowing the old unsigned speed/tilt proxy. Root IDs never select an orientation or dynamic class in v2; same-family cells retain distinct anatomical identities and graph connections but use one explicitly coarse family prior.

**Primary evidence and limits:**

- [Yorozu et al., Nature 2009](https://doi.org/10.1038/nature07843) distinguished tonic, direction-selective wind populations from phasic bidirectional sound responses. [Matsuo et al., 2014](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2014.00179/full) explicitly reports anterior arista deflection activating C and D, posterior activating E, and D responses peaking around 100–200 Hz in their tested vibration conditions. The model adopts only family polarity and mixed D sensitivity; it does not claim to fit those recordings.
- [Hampel et al., eLife 2020](https://elifesciences.org/articles/59976) found no response of the tested JO-F lines to imposed pushes, pulls or tested vibrations in immobilized flies, despite behavioral evidence linking them to grooming. This motivates abstention, not a claim that JO-F is biologically inactive. The article's brief C/E push/pull attribution differs from other papers, so this implementation names its convention **anterior/posterior** and follows the direct Matsuo/Yorozu directional descriptions rather than silently treating all uses of “push” as the same sign.
- [Mamiya and Dickinson, J Neurosci 2015](https://pubmed.ncbi.nlm.nih.gov/25995481/) found flight-dependent responses associated with antenna oscillation at wingbeat frequency, with differences among recorded groups. A tonic wind prior alone is therefore not a complete flight proprioceptive model. This implementation does not invent a wing-induced flow carrier, flight-state gate or measured E-subtype transfer function.
- [Patella and Wilson, Current Biology 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC5952606/) identifies richer stimulus-response structure than a single family label. Further heterogeneity requires aligned data; the model does not manufacture it from IDs.

The inherited numerical scales remain visible, frozen hypotheses: 15 ms mechanical time constant, 100 cm/s reference flow, 0.5 rad maximum virtual deflection, 5 Hz baseline, 120 Hz/rad tonic gain, 0.3 Hz/(rad/s) D velocity gain, and 100 Hz ceiling. None were fitted to behavior or presented as measured physiology.

## Native geometry and local airflow

The sampler resolves `thorax`, `head`, `antenna_left/right`, their native local poses and the antenna collision-geometry centers from the **loaded** MuJoCo model. It requires the present frozen head/antenna topology and rejects newly movable joints. No geometry is hardcoded from a separately rendered fly.

The current root quaternion and linear/angular qvel are used directly. This avoids the possible one-step age of `xpos` after `mj_step`. For each receiver, local relative air velocity includes the rigid-body term:

```
v_receiver_root = R_root^T v_root_world + omega_root × r_receiver_root
u_antenna = R_antenna_to_root^T (R_root^T wind_world - v_receiver_root)
```

Both attachment and receiver positions/velocities are exposed. The receiver center is the native capsule center, an explicit center-of-pressure approximation. Native head +Y defines anterior; its actual quaternion maps that axis into the antenna frame. Projecting this anterior axis perpendicular to attachment→receiver gives one virtual bending tangent per antenna. The corresponding virtual hinge axis is derived from the cross product. Bilateral geometry is retained, with no random per-cell axes.

There is no actual a2/a3 joint in this reduced native body. The sampler consequently does **not** claim to read a measured joint angle. A passive critically damped virtual anterior/posterior coordinate is driven by the previous held local airflow. Its normalized quadratic-drag equilibrium is a mechanical prior. Neither native joint positions, controls nor forces are written. Gravity/inertial receiver torques, auditory carriers and wing-induced airflow remain absent.

The sampler requires declared wind to match `model.opt.wind`; a neural-only wind command is rejected. `advance` uses simulation time, rejects backward/over-50-ms updates, and integrates the previous held input. Reads never advance state. Snapshots bind configuration, exact geometry and prepared population identity and support deterministic restoration.

## Integration API

The outer `antennaFeedback` envelope remains schema 1; the **mechanics** declaration is the explicit schema-2 opt-in profile. `config-fragment.json` is a reviewable local fragment only and is not installed into any active config.

```js
import {
  ANTENNA_FAMILY_PRIOR,
  createAntennaPopulation,
  createAntennaAirflowModel,
  createNativeAntennaKinematics,
} from '../banc-antenna.js';

const population = await createAntennaPopulation({...base, ids}, sensory);
// Once per native model / fresh episode, after creating the native world:
const kinematics = createNativeAntennaKinematics({
  mj: world.mj, model: world.model, metadata: world.metadata,
});
const antenna = createAntennaAirflowModel(
  population, ANTENNA_FAMILY_PRIOR, kinematics.geometry,
);
// At every current native body observation, independent of preview cadence:
antenna.advance(kinematics.sample(body.data, {
  bodyTimeSeconds: body.time,
  windWorldCmPerSecond: config.antennaFeedback.windWorldCmPerSecond,
}));
const {indices, ratesHz, diagnostics} = antenna.sample();
// Replace all returned added-input rates before the existing rate→current map.
```

V1 callers keep their existing root-frame airflow inputs and do not need geometry. The schema-2 path must receive the native sample above; passing the old root-origin scalar/vector interface fails rather than falling back.

## Validation and reproducibility

```sh
node --test web/test/banc-antenna.test.mjs
node reports/antenna-structural-v2-20260915/prepare-evidence.mjs
```

All 14 tests passed. Evidence covers original v1 behavior plus a frozen requested-rate checksum; exact native attachment/receiver positions; agreement of predicted point velocity with central finite differences through MuJoCo free-joint integration; rotation covariance; native-state nonmutation; physical-wind agreement; moving-joint rejection; family polarity, separate D dynamics and explicit F abstention; causal/cadence behavior; snapshot ownership/replay; and transactional error handling.

`evidence.json` contains the exact selected/abstained ID join, native geometry, source checksum and explicit assumptions. Preparation verifies native XML/metadata and existing BANC identities, then performs no dynamics/training steps. These checks establish structural correctness and reproducibility, not improved flight or a calibrated biological receptor model.
