#!/usr/bin/env python3
"""Independent native hinge-angle / anterior-motion assay, no dynamics."""
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
import pyarrow.feather as feather

ROOT=Path(__file__).resolve().parents[1]

def main():
    paths=['models/flybody-mujoco.xml','models/flybody-mujoco.json','data/prepared/banc888/io.json',
           'scripts/prepare-banc.py','reports/flybody-leg-sign-placements.json']
    content={path:(ROOT/path).read_bytes() for path in paths}
    metadata=json.loads(content[paths[1]]);io=json.loads(content[paths[2]]);placements=json.loads(content[paths[4]])
    model=mujoco.MjModel.from_xml_path(str(ROOT/paths[0]));data=mujoco.MjData(model)
    baseline=model.qpos0.copy();baseline[:7]=[0,0,2,1,0,0,0]
    for joint in metadata['joints']:baseline[joint['qpos']]=joint['neutral']
    cases=[{'name':'neutral','qpos':baseline.tolist()}]+placements['cases']
    report={'scope':'Native kinematic sign audit; no integration or applied root forces. Angles are measured between actual adjacent joint anchors; tarsus2 body origin closes the distal segment.',
      'command':'node scripts/audit-flybody-leg-signs.mjs','mujoco':mujoco.__version__,
      'sourceHashes':{path:hashlib.sha256(value).hexdigest() for path,value in content.items()},
      'sourceRevision':metadata['source']['revision'],'definition':{
      'internalAngle':'acos(dot(upstreamAnchor-hingeAnchor, downstreamAnchor-hingeAnchor)/(lengths)); extension increases this angle, flexion decreases it before the straight-line branch.',
      'anteriorCoxaMotion':'Root-frame X displacement of femur anchor relative to thorax-coxa anchor; positive X points toward the head.',
      'perturbationRadians':.01,'rangeDerivativePerturbationRadians':.0001,
      'allRangeScope':'31 evenly spaced q values strictly inside the XML joint limits, repeated with the other coordinates of each of four tested poses held at their actual values.'},
      'cases':[],'mapping':[],'confirmedReversedFlexExtensionJoints':[],
      'compiledMuscleVelocityProbe':placements['compiledMuscleVelocityProbe'],
      'momentArm':{
        'identity':'For positive tensile force F, joint torque = -F * dl/dq. A drive with intended native torque sign s therefore requires dl/dq < 0 for s=+1 and >0 for s=-1.',
        'priorAdapterInput':'Before the native sign conversion: l_normalized = 1 + 0.2 * io.sign * (q-neutral); v_input = +0.2 * io.sign * qvel.',
        'priorAdapterCommand':'Before correction, positive io.sign increased the native position target. A native position actuator develops positive generalized torque when its filtered target is above q.',
        'priorInconsistency':'The prior normalized length derivative had the same sign as the intended generalized force. It was opposite to the virtual-work identity for a tensile muscle.',
        'reducedConsistentForm':'Let s_native map anatomical muscle action to the native coordinate. An explicit reduced moment-arm prior r>0 gives l_normalized=1-r*s_native*(q-q_reference). The existing kernel input must be positive shortening speed, v_shortening=+r*s_native*qvel=-dl_normalized/dt, because its velocity factor is 1-0.25*v.',
        'lengthKernelSymmetry':'The Gaussian force-length factor exp(-((l-1)/0.45)^2) is symmetric. Correcting the length-derivative sign alone leaves its instantaneous force factor unchanged; the native anatomical sign correction changes directional velocity modulation.',
        'scope':'The position-target adapter is not a physical tendon transmission. r and Fmax remain fitted priors; this identity fixes directional consistency only. Positive shortening speed must be distinguished explicitly from the length derivative.'}}

    def point(name,suffix):
        if name=='tarsus2':return data.xpos[model.body(name+'_'+suffix).id].copy()
        return data.xanchor[model.joint(name+'_'+suffix).id].copy()

    def value(kind,suffix):
        if kind=='coxa':
            rotation=data.xmat[model.body('thorax').id].reshape(3,3)
            return float((rotation.T@(point('femur',suffix)-point('coxa',suffix)))[0])
        sequence=['coxa','femur','tibia','tarsus','tarsus2'];index=sequence.index(kind)
        upstream=point(sequence[index-1],suffix)-point(kind,suffix)
        downstream=point(sequence[index+1],suffix)-point(kind,suffix)
        return float(np.arccos(np.clip(upstream@downstream/(np.linalg.norm(upstream)*np.linalg.norm(downstream)),-1,1)))

    def evaluate(pose,kind,suffix,angle):
        joint=model.joint(kind+'_'+suffix);data.qpos[:]=pose;data.qpos[joint.qposadr[0]]=angle
        mujoco.mj_forward(model,data)
        return value(kind,suffix)

    for case in cases:
        pose=np.asarray(case['qpos']);result={'name':case['name'],'joints':[]}
        for side in ['left','right']:
            for segment in ['T1','T2','T3']:
                suffix=f'{segment}_{side}'
                for kind in ['coxa','femur','tibia','tarsus']:
                    name=kind+'_'+suffix;joint=model.joint(name);at=float(pose[joint.qposadr[0]])
                    lo,hi=joint.range;minus=max(float(lo),at-.01);plus=min(float(hi),at+.01)
                    measured=[evaluate(pose,kind,suffix,angle) for angle in [minus,at,plus]]
                    derivative=(measured[2]-measured[0])/(plus-minus)
                    grid=np.linspace(lo+.0002,hi-.0002,31);slopes=[]
                    for angle in grid:
                        slopes.append((evaluate(pose,kind,suffix,float(angle)+.0001)-evaluate(pose,kind,suffix,float(angle)-.0001))/.0002)
                    row={'name':name,'kind':kind,'qpos':at,'limit':[float(lo),float(hi)],'testedQpos':[minus,at,plus],
                         'measured':measured,'units':'cm' if kind=='coxa' else 'radians',
                         'derivative':float(derivative),'rangeDerivativeMin':float(min(slopes)),'rangeDerivativeMax':float(max(slopes)),
                         'rangeNegativeDerivativeCount':sum(x < -1e-5 for x in slopes),'rangePositiveDerivativeCount':sum(x > 1e-5 for x in slopes)}
                    if kind!='coxa':row['measuredDegrees']=np.degrees(measured).tolist()
                    result['joints'].append(row)
        report['cases'].append(result)

    groups=[group for group in io['muscles'] if group['kind']=='leg']
    wanted={root for group in groups for root in group['root_ids']}
    raw=feather.read_table(ROOT/'data/raw/banc888/meta.feather',columns=['banc_888_id','cell_function_detailed','body_part_effector','side']).to_pylist()
    annotated={row['banc_888_id']:row for row in raw if row['banc_888_id'] in wanted}
    for group in groups:
        functions=sorted({annotated[root]['cell_function_detailed'] for root in group['root_ids']})
        report['mapping'].append({'joint':group['joint'],'target':group['target'],'ioSign':group['sign'],
          'sourceFunctions':functions,'motorNeuronCount':len(group['indices'])})
    for row in report['cases'][0]['joints']:
        if row['kind'] not in ['femur','tibia']:continue
        values=[next(j for j in case['joints'] if j['name']==row['name']) for case in report['cases']]
        if all(j['rangeDerivativeMin']>0 for j in values):
            report['confirmedReversedFlexExtensionJoints'].append({'joint':row['name'],
              'minimumExtensionDerivativeAcrossTestedRanges':min(j['rangeDerivativeMin'] for j in values),
              'nativePositiveAction':'extension','requiredNativeFlexionSign':-1,'requiredNativeExtensionSign':1,
              'currentSourceConvention':'flexion +1, extension -1; this convention must be converted at the native adapter boundary'})
    report['conclusion']={
      'femurAndTibia':'Positive q extends all 12 hinges at neutral and initialized placements across every sampled point of their native joint ranges. A correct direct adapter converts the anatomical io sign for these coordinates. This kinematic report establishes the required conversion; the separate production pulse report validates its implementation.',
      'coxa':'At neutral positive coxa q is anterior for T1, posterior for T2/T3. The T2 derivative changes sign nearby; raw anterior displacement is posture dependent. This does not establish one globally correct sign or axis for the anterior/posterior muscle adapter.',
      'tarsus':'Most tested positive-q perturbations reduce the internal tibia-tarsus angle. The nearly straight T1 right neutral posture has the opposite unsigned-angle derivative, and the derivative crosses the straight-line branch within joint limits. Do not flip all tarsal signs based on the extend_* XML class names.',
      'rootPose':'Internal hinge angles and root-frame anterior displacements are invariant under a global root transform; initialized poses include terrain-aligned root quaternions.',
      'behaviorScope':'This proves adapter-direction errors, not that correcting them alone restores coordinated walking, landing or flight.'}
    assert len(report['confirmedReversedFlexExtensionJoints'])==12
    assert all(case.get('time',0)==0 and case.get('externalForceMaximum',0)==0 for case in cases)
    output=ROOT/'reports/flybody-leg-sign-audit.json';output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'report':str(output.relative_to(ROOT)),'confirmed':report['confirmedReversedFlexExtensionJoints'],
      'placements':[case['name'] for case in cases],'samples':len(cases)*24*31},indent=2))

if __name__=='__main__':main()
