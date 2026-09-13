"""A restrained-body measurement of the LEGACY folded-pose wing adapter.

Reproduce the pre-repair adapter (218 Hz, all angles blended from rest).
The production adapter is now checked by verify-flybody-flight.mjs instead.
Holding the root and
non-wing joints fixed isolates aerodynamic force from falls and floor impacts.
Nothing from this diagnostic is used to stabilize the production body.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import mujoco

p = argparse.ArgumentParser()
p.add_argument('--output', default='reports/flybody-flight-force-legacy-replay.json')
p.add_argument('--seconds', type=float, default=.2)
args = p.parse_args()
root = Path(__file__).resolve().parents[1]
meta = json.loads((root/'models/flybody-mujoco.json').read_text())
xml = (root/'models/flybody-mujoco.xml').read_text()
pattern = np.array(meta['wing_pattern'])
model = mujoco.MjModel.from_xml_string(xml)
wing = [j for side in ('left', 'right') for axis in ('yaw', 'roll', 'pitch')
        for j in meta['joints'] if j['name'] == f'wing_{axis}_{side}']
qa = np.array([j['qpos'] for j in wing]); va = np.array([j['dof'] for j in wing])
acts = np.array([model.actuator(j['name']).id for j in wing])
neutral = np.array([j['neutral'] for j in wing])
non_wing_q = np.array([j['qpos'] for j in meta['joints'] if j not in wing])
non_wing_v = np.array([j['dof'] for j in meta['joints'] if j not in wing])
rows = []
for pitch_degrees in (0, 47.5):
    for power in (0, .25, .5, .75, 1):
        data = mujoco.MjData(model)
        for j in meta['joints']: data.qpos[j['qpos']] = j['neutral']
        pitch = np.deg2rad(pitch_degrees)
        data.qpos[:7] = [0, 0, 2, np.cos(pitch/2), 0, -np.sin(pitch/2), 0]
        fixed = data.qpos.copy()
        phase = 0.; force = []; torque = []; angles = []; saturations = []
        for step in range(round(args.seconds/model.opt.timestep)):
            data.qpos[:7] = fixed[:7]; data.qvel[:6] = 0
            data.qpos[non_wing_q] = fixed[non_wing_q]; data.qvel[non_wing_v] = 0
            if step % 4 == 0:
                index = phase * len(pattern); i = int(index) % len(pattern); t = index % 1
                cycle = pattern[i]*(1-t) + pattern[(i+1) % len(pattern)]*t
                target = neutral + power*(np.tile(cycle, 2)-neutral)
                data.ctrl[acts] = np.clip(target-data.qpos[qa], -1, 1)
                phase = (phase+meta['wing_frequency_hz']*model.opt.timestep*4) % 1
            mujoco.mj_step(model, data)
            if step >= round(.05/model.opt.timestep):
                force.append(data.qfrc_fluid[:3].copy())
                torque.append(data.qfrc_fluid[3:6].copy())
                angles.append(data.qpos[qa].copy())
                saturations.append(np.abs(data.ctrl[acts]) >= .999)
        forces = np.array(force); angles = np.array(angles)
        row = dict(pitch_degrees=pitch_degrees, wing_power=power,
                   mean_fluid_force_body_weights=(forces.mean(0)/(meta['mass_g']*981)).tolist(),
                   mean_fluid_torque_g_cm2_s2=np.mean(torque, axis=0).tolist(),
                   actual_wing_range_rad=np.ptp(angles, axis=0).tolist(),
                   saturated_control_fraction=float(np.mean(saturations)))
        rows.append(row); print(json.dumps(row), flush=True)
report = dict(scope=__doc__, mujoco=mujoco.__version__, seconds=args.seconds,
              discard_seconds=.05, cases=rows,
              source_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                             for name in ('models/flybody-mujoco.xml', 'models/flybody-mujoco.json',
                                          'web/flybody-physics.js', 'scripts/diagnose-flybody-flight.py')})
(root/args.output).write_text(json.dumps(report, indent=2)+'\n')
