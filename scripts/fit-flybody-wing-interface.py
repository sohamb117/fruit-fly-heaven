"""Offline native actuator calibration; restrained rig is confined to this fit.

Fit each power knot against the SAME native gain, damping, wing masses and fluid
model used in the free body. Never optimize brain navigation or body feedback.
Run with: uv run --no-project --with scipy --with mujoco==3.13.0 scripts/fit-flybody-wing-interface.py
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--output',default='reports/flybody-wing-correction/candidate.json');parser.add_argument('--report',default='reports/flybody-wing-correction/fit.json');args=parser.parse_args()
OUT=ROOT/'reports/flybody-wing-correction'
OUT.mkdir(exist_ok=True)
meta=json.loads((ROOT/'models/flybody-mujoco.json').read_text())
old=json.loads((ROOT/'models/flybody-wing-actuation.json').read_text())
m=mujoco.MjModel.from_xml_path(str(ROOT/'models/flybody-mujoco.xml'))
pattern=np.array(meta['wing_pattern']);center=pattern.mean(0);frequency=old['frequency_hz']
names=[f'wing_{a}_{s}' for s in ['left','right'] for a in ['yaw','roll','pitch']]
qa=np.array([m.joint(n).qposadr[0] for n in names]);va=np.array([m.joint(n).dofadr[0] for n in names]);ids=np.array([m.actuator(n).id for n in names])
other=[j for j in meta['joints'] if j['name'] not in names]
limits=np.array([m.joint(n).range for n in names]);weight=meta['mass_g']*981
bases=[Rotation.from_quat(np.roll(m.body('wing_'+s).quat,-1)).as_matrix() for s in ['left','right']]
count=len(pattern);h=m.opt.timestep;steps=round((.04+20/frequency)/h)
sample_phase=(np.arange((steps+3)//4)*h*4*frequency)%1*count
ix=sample_phase.astype(int);fraction=sample_phase-ix
fixed=np.array(m.qpos0)
for j in meta['joints']:fixed[j['qpos']]=j['neutral']
fixed[:7]=[0,0,5,1,0,0,0]
oq=np.array([j['qpos'] for j in other]);ov=np.array([j['dof'] for j in other])

def table_for(power,params):
 plane,shift,amplitude=params[:3]
 cycle=center+(pattern-center)*[amplitude,min(1,power/.1),min(1,power/.1)]+[shift,0,0]
 rotation=Rotation.from_euler('ZXY',cycle).as_matrix()
 global_plane=Rotation.from_euler('y',plane).as_matrix()
 angles=np.unwrap(np.concatenate([Rotation.from_matrix((base.T@global_plane@base)@rotation).as_euler('ZXY') for base in bases],axis=1),axis=0)
 angles+=2*np.pi*np.round((np.array([0,0,1.7,0,0,1.7])-angles.mean(0))/(2*np.pi))
 if len(params)>3:
  # Tiny differential joint trims compensate the source body's measured
  # lateral COM offset. The neutral MN command then means zero COM moment,
  # rather than imposing perfect geometry symmetry on an asymmetric body.
  angles[:,:3]+=params[3:];angles[:,3:]-=params[3:]
 clip_fraction=float(np.mean((angles<limits[:,0])|(angles>limits[:,1])))
 return np.clip(angles,limits[:,0],limits[:,1]),clip_fraction

def measure(table):
 d=mujoco.MjData(m);d.qpos[:]=fixed
 target=table[ix]*(1-fraction[:,None])+table[(ix+1)%count]*fraction[:,None]
 sums=np.zeros(6);samples=0
 for step in range(steps):
  d.qpos[:7]=fixed[:7];d.qvel[:6]=0;d.qpos[oq]=fixed[oq];d.qvel[ov]=0
  if step%4==0:d.ctrl[ids]=np.clip(target[step//4]-d.qpos[qa],-1,1)
  mujoco.mj_step(m,d)
  if step*h>=.04:
   # Free-joint rotational generalized force is a moment about the thorax
   # origin. A hovering/free body needs zero aerodynamic moment about its
   # whole-body center of mass, which is posterior to that origin. In this
   # level fixture root axes equal world axes. MuJoCo's position/force cache
   # still describes the pre-integration state (root at fixed[:3]).
   wrench=d.qfrc_fluid[:6].copy()
   wrench[3:]-=np.cross(d.subtree_com[1]-fixed[:3],wrench[:3])
   sums+=wrench;samples+=1
 return sums/samples

report={'scope':'Offline tethered-body calibration of periodic muscle-to-wing targets; no free-flight claim. Native servo and original fluid forces are integrated, not prescribed wing qpos. Pitch moment is measured about whole-body COM, not the thorax/free-joint origin.',
 'frequency_hz':frequency,'force_target':'Preserve the original table mean aerodynamic force at each power; correct its moment to whole-body COM only.',
 'source':{'xml_sha256':hashlib.sha256((ROOT/'models/flybody-mujoco.xml').read_bytes()).hexdigest(),
 'pattern_sha256':meta['wing_pattern_sha256'],'flybody_revision':meta['source']['revision']},'fits':[]}
report['coordinate_checks']=[]
for angles in [[0,0,0],[.3,-.7,1.1],[-.9,.4,-2.0]]:
 d=mujoco.MjData(m);d.qpos[:]=fixed
 rotation=Rotation.from_euler('xyz',angles)
 d.qpos[3:7]=np.roll(rotation.as_quat(),1);mujoco.mj_forward(m,d)
 force=np.array([.3,-.2,.8]);generalized=np.zeros(m.nv)
 mujoco.mj_applyFT(m,d,force,np.zeros(3),d.subtree_com[1],1,generalized)
 predicted=rotation.inv().apply(np.cross(d.subtree_com[1]-d.qpos[:3],force))
 error=float(np.max(np.abs(generalized[3:6]-predicted)))
 assert error<1e-12
 report['coordinate_checks'].append({'root_euler_xyz':angles,'root_to_com_world':(d.subtree_com[1]-d.qpos[:3]).tolist(),
  'native_generalized_root_torque':generalized[3:6].tolist(),'predicted_root_local_cross_product':predicted.tolist(),'maximum_error':error})
tables=[];powers=[0,.05,.1,.25,.5,.75,1]
for power in powers:
 if power==0:
  tables.append(old['targets'][0]);continue
 started=time.monotonic();calls=0
 original_wrench=measure(np.array(old['targets'][old['powers'].index(power)]))
 preserved=old.get('calibration',{}).get('preserved_force_curve')
 target_force=np.array(next(row['force_body_weights'] for row in preserved if row['power']==power)) if preserved else original_wrench[:3]/weight
 def objective(params):
  global calls
  calls+=1;table,_=table_for(power,params);force=measure(table)
  return np.array([force[0]/weight-target_force[0],force[2]/weight-target_force[2],force[4]/(weight*.027)])
 x0=np.array([-1.02813087,-.217291784,np.sqrt(power)])
 before=objective(x0)
 fit=least_squares(objective,x0,bounds=([-2.2,-1.2,.025],[-.3,.8,1.5]),diff_step=2e-3,max_nfev=45,xtol=2e-5,ftol=2e-5,gtol=2e-5)
 trim=np.zeros(3)
 def differential_error(value):
  table,_=table_for(power,np.r_[fit.x,value]);force=measure(table)
  return np.array([force[1]/weight-target_force[1],force[3]/(weight*.027),force[5]/(weight*.027)])
 for _ in range(2):
  error=differential_error(trim);eps=.0005
  jac=np.column_stack([(differential_error(trim+np.eye(3)[k]*eps)-differential_error(trim-np.eye(3)[k]*eps))/(2*eps) for k in range(3)])
  trim-=np.linalg.lstsq(jac,error,rcond=1e-8)[0]
  assert np.max(np.abs(trim))<.03,'Source body asymmetry requires unexpectedly large trim'
 params=np.r_[fit.x,trim]
 table,clip=table_for(power,params);force=measure(table);tables.append(table.round(8).tolist())
 assert np.max(np.abs(force[:3]/weight-target_force))<5e-4,'Force curve changed beyond calibration tolerance'
 assert np.max(np.abs(force[3:]))<1e-4,'COM moment correction failed'
 row={'power':power,'parameters':params.tolist(),'target_force_body_weights':target_force.tolist(),
  'original_torque_about_com':original_wrench[3:].tolist(),
  'actual_force_body_weights':(force[:3]/weight).tolist(),'actual_torque_about_com':force[3:].tolist(),
  'objective':fit.fun.tolist(),'initial_objective':before.tolist(),'clipping_fraction':clip,'calls':calls,'seconds':time.monotonic()-started,'success':bool(fit.success)}
 report['fits'].append(row);print(json.dumps(row),flush=True)
 (ROOT/args.report).write_text(json.dumps(report,indent=2)+'\n')

candidate={**old,'frequency_hz':frequency,'powers':powers,'targets':tables,
 'calibration':{'kind':'Preserve original force and frequency; correct power-specific moments to whole-body COM','full_activation_target_body_weights':1.25,
 'preserved_force_curve':[{'power':row['power'],'force_body_weights':row['target_force_body_weights']} for row in report['fits']],
 'fits':report['fits']},'source':{**old['source'],'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
(ROOT/args.output).write_text(json.dumps(candidate,separators=(',',':'))+'\n')
