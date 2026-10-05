"""Distill a pinned FlyBody teacher into local joint dynamics and leg kinematics.

This is system identification, not a locomotion policy. Contact and aerodynamic
forces in the browser are explicit approximations outside this fit.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'd015e9bfe441bd90ae431bac24c55cb74bdbce26'
LEG_JOINTS = [f'{joint}_{segment}_{side}' for segment in ('T1', 'T2', 'T3') for side in ('left', 'right') for joint in ('coxa', 'femur', 'tibia')]
ACTIVE = LEG_JOINTS + ['wing_yaw_left', 'wing_yaw_right', 'rostrum', 'haustellum']


def features(q):
    return np.array([1, *q, *(q*q), q[0]*q[1], q[0]*q[2], q[1]*q[2]])


def main():
    source = ROOT / 'references/flybody'
    if subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() != REVISION:
        raise ValueError(f'FlyBody teacher must be pinned to {REVISION}')
    original = source/'flybody/fruitfly/assets/fruitfly.xml'
    tree = ET.parse(original)
    root = tree.getroot()
    root.find('compiler').set('meshdir', '../references/flybody/flybody/fruitfly/assets')
    frozen = []
    for body in root.findall('.//worldbody//body'):
        for joint in list(body.findall('joint')):
            if joint.get('name') not in ACTIVE and joint.get('name') != 'free':
                frozen.append(joint.get('name'));body.remove(joint)
    for node in root.findall('tendon') + root.findall('keyframe'):
        root.remove(node)
    actuator = root.find('actuator')
    for node in list(actuator):
        if node.get('joint') not in ACTIVE:
            actuator.remove(node)
    # Existing spatial sensor sites and body meshes remain usable in MuJoCo.
    models = ROOT/'models';models.mkdir(exist_ok=True)
    reduced = models/'flybody-reduced.xml';tree.write(reduced, encoding='unicode')
    model = mujoco.MjModel.from_xml_path(str(reduced))
    data = mujoco.MjData(model)
    addresses = np.array([model.joint(name).qposadr[0] for name in ACTIVE])
    dofs = np.array([model.joint(name).dofadr[0] for name in ACTIVE])
    ranges = np.array([model.joint(name).range for name in ACTIVE])
    # Neutral pose is within the teacher joint limits; discarded joints fixed at qpos0.
    neutral = np.clip(model.qpos_spring[addresses], ranges[:, 0]+.02, ranges[:, 1]-.02)
    random = np.random.default_rng(888)
    samples = 640
    poses = np.clip(neutral + random.uniform(-.35,.35,(samples,len(ACTIVE))), ranges[:, 0], ranges[:, 1])
    inertia, feet = [], []
    for pose in poses:
        data.qpos[:] = model.qpos0
        data.qpos[addresses] = pose
        mujoco.mj_forward(model,data)
        mass = np.zeros((model.nv,model.nv))
        mujoco.mj_fullM(model,data,mass)
        # Root constrained during the local fit. Free-body contact is not fitted.
        inertia.append(np.diag(mass)[dofs])
        feet.append([data.site(f'claw_{s}_{side}').xpos.copy() for s in ('T1','T2','T3') for side in ('left','right')])
    inertia, feet = np.array(inertia), np.array(feet)
    training=512
    fit_inertia=inertia[:training].mean(axis=0)
    kinematics=[]
    errors=[]
    for leg in range(6):
        design=np.array([features(q[leg*3:leg*3+3]-neutral[leg*3:leg*3+3]) for q in poses])
        coefficients=np.linalg.lstsq(design[:training], feet[:training,leg], rcond=None)[0]
        errors.extend(np.linalg.norm(design[training:]@coefficients-feet[training:,leg],axis=1))
        kinematics.append({'joint_indices':list(range(leg*3,leg*3+3)),'coefficients':coefficients.tolist()})
    result={'schema':1,'source':{'repository':'https://github.com/TuragaLab/flybody','revision':REVISION,'license':'Apache-2.0',
            'xml_sha256':hashlib.file_digest(original.open('rb'),'sha256').hexdigest(),'mujoco':mujoco.__version__},
            'units':{'length':'cm','mass':'g','time':'s','angle':'rad','torque':'g cm^2 / s^2'},
            'mass_g':float(model.body_mass.sum()),'active_joints':[], 'frozen_joints':frozen,
            'root_dofs':['x','y','z','yaw'],'frozen_runtime_root_dofs':['pitch','roll'],
            'leg_kinematics':kinematics,
            'fit':{'training_samples':training,'held_out_samples':samples-training,'seed':888,'pose_radius_rad':.35,
                   'foot_position_rmse_cm':float(np.sqrt(np.mean(np.square(errors)))),
                   'inertia_relative_rmse':float(np.sqrt(np.mean(((inertia[training:]-fit_inertia)/inertia[training:])**2)))},
            'limitations':['Local diagonal inertia approximation; cross-joint inertial coupling omitted.',
                           'Kinematics fit only within 0.35 rad of neutral.',
                           'Contacts, muscle moment arms and averaged wing aerodynamics are host assumptions.',
                           'No teacher locomotion policy or success trajectory is distilled.']}
    for k,name in enumerate(ACTIVE):
        result['active_joints'].append({'name':name,'neutral':float(neutral[k]),'range':ranges[k].tolist(),
            'inertia':float(fit_inertia[k]),'damping':float(model.dof_damping[dofs[k]]),
            'stiffness':float(model.joint(name).stiffness[0])})
    (models/'flybody-reduced.json').write_text(json.dumps(result,indent=2)+'\n')
    (models/'FlyBody-LICENSE').write_text((source/'LICENSE').read_text())
    (ROOT/'reports/flybody-distillation.json').write_text(json.dumps(result['fit'],indent=2)+'\n')
    print(json.dumps(result['fit'],indent=2))


if __name__=='__main__':main()
