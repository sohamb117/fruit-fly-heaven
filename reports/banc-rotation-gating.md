The rotation-afferent input now uses the annotated organ and side. The previous shared wing-power gate stimulated all 449 haltere/wing-base afferents together, including sensors on an inactive organ or opposite side.

The bounded prior is `min(100, 5*activity + 4*power*|angularVelocity|)` Hz. For a haltere, power is the corresponding `halterePower[left/right]` muscle force and activity is the maximum of that force and the supported same-side steering-muscle forces. Steering alone contributes basal strain input, never the rotation-sensitive term. For wing-base sensors, both values are the corresponding `wingPowerLeft/Right` deployed-wing power. Missing organ/side/power does not borrow the mean wing signal. The constants, force normalization, max pooling and scalar magnitude response remain modeling assumptions.

Expected feedback fields:

| Field | Meaning |
|---|---|
| `halterePower` | `[left, right]`, normalized asynchronous haltere muscle forces |
| `haltereSteering` | `{left: {...}, right: {...}}`, forces keyed by `hi1_muscle`, `hi2_muscle`, `hiii2_muscle`, `haltere_basalare_muscle` |
| `wingPowerLeft`, `wingPowerRight` | Per-side power after the native wing deployment adapter |
| `angularVelocity` | Three root angular-velocity components in rad/s |

Native haltere position, velocity and strain are not simulated. Muscle force is only an activation proxy. All afferents on one side of one organ still receive the same unsigned signal. This change therefore does not establish gyroscopic stabilization, directional tuning, field identity or phase-locked spikes.

Validation: `node --test web/test/banc-ground-sense.test.mjs` checks isolated organ/side activation, bounded steering-only input, disabled/missing input, equal response to reversed rotations, and all actual BANC rotation annotations through `SensoryEncoder`: 171 left haltere, 157 right haltere, 62 left wing-base and 59 right wing-base afferents. Counterfactual input results are in `banc-rotation-gating.json`.

The next defensible calibration path is to combine measured organ motion with a phase-aware sensory model, preserving uncertainty in BANC cell correspondence:

1. Fit Drosophila haltere amplitude, phase and wing coupling before mapping force to strain. Rauscher and Fox 2024 manipulated wing/haltere synchrony and measured wing/head effects. Their released [Dryad data](https://datadryad.org/dataset/doi:10.5061/dryad.g1jwstqwj) include `untreatedhalteredata.mat`, `kinetbl.mat`, `treatedkinematics.mat` and wingbeat data; the page documents the kinematic fields. This supports a haltere mechanical assay, not a neuron-ID tuning table.
2. Use field-level activity and steering modulation as population constraints. Verbe et al. 2024 provide [data and analysis scripts](https://github.com/AnnaVerbe/Flies-tune-the-activity-of-their-multifunctional-gyroscope) for calcium recordings and haltere motion. Those recordings include visually induced changes in rigidly tethered, nonrotating flies; they must not be interpreted as a direct physical angular-velocity tuning curve. [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11338719/).
3. Establish anatomical correspondence separately. Dhawan et al. 2026 provide [FANC connectivity data and figure notebooks](https://github.com/serene-da1/Dhawan-et-al-2025-). Their morphological afferent subtypes span multiple peripheral sensillum fields. FANC materialization 840 IDs require an explicit, validated crosswalk to BANC v888; a type name or downstream motor target alone does not identify a preferred axis or spike phase. [Paper](https://doi.org/10.1016/j.cub.2025.12.024).
4. Fit phase and displacement thresholds only to electrophysiological data with the appropriate identity and species. Yarger and Fox 2018 release [single-unit spike times and mechanical stimulus amplitudes](https://datadryad.org/dataset/doi:10.5061/dryad.272gd5n), but the [experiments used Sarcophaga bullata](https://pmc.ncbi.nlm.nih.gov/articles/PMC6170812/). They support a timing/threshold model structure and an explicitly comparative prior, not measured Drosophila or BANC per-cell parameters.
5. Validate the complete sensory→VNC→motor→muscle loop against held-out physical rotations and afferent ablations. Sharma et al. 2026 report Drosophila yaw-rotation responses after silencing/ablation of sensory subsets. Their [public code](https://github.com/tarunsharma1/haltere_campaniform_data_analysis) documents sinusoidal trials reaching 350 and 500 degrees/s, with measured head and wing response amplitude/phase. The [author archive](https://authors.library.caltech.edu/records/r1ywc-zsa65) links the released data. These are useful behavioral validation conditions, not a license to impose corrective motor commands.

No verified source above supplies a complete three-axis, phase-resolved tuning function for every BANC v888 haltere or wing-base neuron. Such assignments remain unimplemented. The code only changes sensory input, retains the BANC graph and neural dynamics, and does not write motor activity or stabilize the root.
