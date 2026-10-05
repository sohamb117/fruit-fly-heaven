"""Export a mesh-free FlyBody for the actual MuJoCo WASM solver.

Visual meshes are replaced by their compiled inertias, not fitted dynamics.
Keep the native articulated legs, wing joints, collisions and fluid model.
"""
import hashlib
import io
import json
import re
from pathlib import Path
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'd015e9bfe441bd90ae431bac24c55cb74bdbce26'
SOURCE = ROOT / 'references/flybody'
ASSETS = SOURCE / 'flybody/fruitfly/assets'
OUT = ROOT / 'models'
fmt = lambda values: ' '.join(format(float(v), '.17g') for v in values)


def main():
    assert subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() == REVISION
    original = ASSETS / 'fruitfly.xml'
    tree = ET.parse(original)
    root = tree.getroot()
    root.find('compiler').set('meshdir', str(ASSETS))
    full = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    # Retain seven articulated coordinates per leg. Distal tarsomeres, head,
    # abdomen, antennae and halteres are fixed at their reference pose.
    active = {f'{joint}_{segment}_{side}' for segment in ('T1', 'T2', 'T3')
              for side in ('left', 'right')
              for joint in ('coxa_abduct', 'coxa_twist', 'coxa', 'femur_twist', 'femur', 'tibia', 'tarsus')}
    active |= {f'wing_{axis}_{side}' for axis in ('yaw', 'roll', 'pitch') for side in ('left', 'right')}
    active |= {'rostrum', 'haustellum'}
    frozen = []
    for body in root.findall('.//worldbody//body'):
        bid = full.body(body.get('name')).id
        if full.body_mass[bid] > 0:
            ET.SubElement(body, 'inertial', pos=fmt(full.body_ipos[bid]), quat=fmt(full.body_iquat[bid]),
                          mass=str(full.body_mass[bid]), diaginertia=fmt(full.body_inertia[bid]))
        for joint in list(body.findall('joint')):
            if joint.get('name') not in active:
                frozen.append(joint.get('name'))
                body.remove(joint)
        for geom in list(body.findall('geom')):
            gid = full.geom(geom.get('name')).id
            if full.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
                body.remove(geom)
            elif 'fluid' in geom.get('name', ''):
                geom.set('fluidshape', 'ellipsoid')
                geom.set('fluidcoef', '1 .5 1.5 1.7 1')
                geom.set('contype', '0')
                geom.set('conaffinity', '0')
        for item in list(body):
            if item.tag in ('light', 'camera'):
                body.remove(item)
    for tag in ('asset', 'tendon', 'keyframe', 'sensor'):
        for element in root.findall(tag):
            root.remove(element)
    actuator = root.find('actuator')
    for element in list(actuator):
        if element.tag == 'adhesion' and element.get('name').startswith('adhere_claw_'):
            continue
        if element.get('joint') not in active:
            actuator.remove(element)
        elif not element.get('joint').startswith('wing_'):
            element.set('dyntype', 'filterexact')
            element.set('dynprm', '.01')
    for element in root.iter():
        element.attrib.pop('material', None)
    root.find('compiler').attrib.pop('meshdir')
    root.find('compiler').set('inertiafromgeom', 'false')
    root.find('option').set('timestep', '.00005')
    # Resolve compliant surface contact over 40 physics steps. The source's
    # 0.2 ms contacts repeatedly unloaded the tiny claws in the reduced model,
    # producing downhill creep even with all six feet planted and no MN input.
    root.find('./default/geom').set('solref', '.002 1')
    root.find('size').attrib.clear()
    root.find('size').set('memory', '4M')
    # Published flight-task actuator and hinge parameters (tasks/constants.py).
    wing = root.find(".//default[@class='wing']")
    wing.find('joint').set('damping', '.007769230')
    for axis in ('yaw', 'roll', 'pitch'):
        wing.find(f"default[@class='{axis}']/general").set('gainprm', '18')
    # The published FlyBody flight task excludes wing/leg self collisions.
    contact = root.find('contact')
    for body in root.findall('.//worldbody//body'):
        name = body.get('name')
        if any(part in name for part in ('coxa', 'femur', 'tibia', 'tarsus', 'claw')):
            for side in ('left', 'right'):
                ET.SubElement(contact, 'exclude', name=f'{name}_wing_{side}', body1=name, body2=f'wing_{side}')
    # Scene geometry is supplied from the unchanged habitat at construction.
    ET.SubElement(root, 'asset')
    path = OUT / 'flybody-mujoco.xml'
    ET.indent(root)
    tree.write(path, encoding='unicode')
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    joints = []
    for i in range(1, model.njnt):
        name = model.joint(i).name
        qadr, vadr = int(model.jnt_qposadr[i]), int(model.jnt_dofadr[i])
        neutral = float(model.qpos0[qadr])
        if name.startswith('wing_'):
            neutral = float(model.qpos_spring[qadr])
        joints.append(dict(name=name, id=i, qpos=qadr, dof=vadr, neutral=neutral, range=model.jnt_range[i].tolist()))
    actuators = [dict(name=model.actuator(i).name, id=i, joint=int(model.actuator_trnid[i, 0]) if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT else None,
                      range=model.actuator_ctrlrange[i].tolist(), gain=float(model.actuator_gainprm[i, 0])) for i in range(model.nu)]
    # Official experimental wing cycle, also used by the FlyBody WPG.
    archive = ROOT / 'data/raw/flybody-flight.zip'
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        urllib.request.urlretrieve('https://ndownloader.figshare.com/files/51196859', archive)
    with zipfile.ZipFile(archive) as z:
        pattern_bytes = z.read('wing_pattern_fmech.npy')
    pattern = np.load(io.BytesIO(pattern_bytes))
    assert pattern.ndim == 2 and pattern.shape[1] == 3 and np.isfinite(pattern).all()
    metadata = dict(schema=1, backend='MuJoCo WASM', source=dict(repository='https://github.com/TuragaLab/flybody',
        revision=REVISION, xml_sha256=hashlib.sha256(original.read_bytes()).hexdigest(), license='Apache-2.0'),
        mujoco=mujoco.__version__, timestep=model.opt.timestep, mass_g=float(model.body_mass.sum()),
        original_mass_g=float(full.body_mass.sum()), joints=joints, actuators=actuators, frozen_joints=frozen,
        root_dofs=['x', 'y', 'z', 'roll', 'pitch', 'yaw'],
        feet=[int(model.site(f'claw_{s}_{side}').id) for side in ('left', 'right') for s in ('T1', 'T2', 'T3')],
        leg_bodies=[[int(model.body(f'{part}_{s}_{side}').id) for part in ('femur', 'tibia', 'claw')]
                    for side in ('left', 'right') for s in ('T1', 'T2', 'T3')],
        body_to_leg=[(int(match[1])-1+(3 if match[2]=='right' else 0)) if (match:=re.search(r'_T([123])_(left|right)$',model.body(i).name)) else -1 for i in range(model.nbody)],
        claw_bodies=[int(model.body(f'claw_{s}_{side}').id) for side in ('left','right') for s in ('T1','T2','T3')],
        wing_bodies=[int(model.body(f'wing_{side}').id) for side in ('left','right')],
        mouth_bodies=[int(model.body(f'labrum_{side}').id) for side in ('left','right')],
        mouth_site=int(model.site('labrum_left').id) if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'labrum_left') >= 0 else None,
        wing_pattern=pattern.tolist(), wing_frequency_hz=218,
        wing_pattern_source='https://doi.org/10.25378/janelia.25309105',
        wing_pattern_sha256=hashlib.sha256(pattern_bytes).hexdigest(),
        contact_time_constant_s=.002,
        wing_actuation=json.loads((OUT / 'flybody-wing-actuation.json').read_text()),
        xml_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        limitations=['Motor-to-actuator adapter is a modeled boundary, not measured muscle physiology.',
                      'No learned navigation or walking policy is inserted in the biological VNC path.',
                      'Habitat collision heightfield approximates the existing rendered surface.'])
    assert abs(metadata['mass_g'] - metadata['original_mass_g']) < 1e-12
    (OUT / 'flybody-mujoco.json').write_text(json.dumps(metadata, separators=(',', ':'))+'\n')
    # Cross-engine fixture: the WASM validation replays these exact controls
    # against a native MuJoCo trajectory, including inertial coupling.
    reference = mujoco.MjData(model)
    reference.qpos[2] = 2
    for j in joints:
        reference.qpos[j['qpos']] = j['neutral']
    reference.ctrl[model.actuator('femur_T1_left').id] = .2
    mujoco.mj_forward(model, reference)
    frames = []
    for step in range(200):
        mujoco.mj_step(model, reference)
        if (step+1) % 20 == 0:
            frames.append(dict(step=step+1, qpos=reference.qpos.tolist(), qvel=reference.qvel.tolist()))
    (OUT / 'flybody-native-reference.json').write_text(json.dumps(dict(xml_sha256=metadata['xml_sha256'],
        mujoco=mujoco.__version__, control={'femur_T1_left':.2}, frames=frames), separators=(',', ':'))+'\n')
    print(json.dumps({k:metadata[k] for k in ('mujoco', 'mass_g', 'original_mass_g', 'timestep')}))
    print(f'{len(joints)} articulated joints, {len(frozen)} frozen, {model.nu} actuators, {model.ngeom} primitive geoms')


if __name__ == '__main__':
    main()
