"""Fit muscle-specific wing-joint responses to published RoboFly directions.

The direction matrix is experimental, but its transfer to this morphology and
its gain are explicit priors. No body-state feedback or flight policy is fitted.
"""
import argparse, ast, hashlib, json
from pathlib import Path
import mujoco
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--config',default='models/flybody-wing-actuation.json');parser.add_argument('--report',default='reports/flybody-steering-calibration.json');args=parser.parse_args()
source=ROOT/'references/mpc-simulations/mpc_simulations/MPC_simulations.py'
tree=ast.parse(source.read_text())
node=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='dFT_du' for t in n.targets))
directions=np.array(ast.literal_eval(node.value.args[0]))[:,:12]
path=ROOT/args.config;config=json.loads(path.read_text());meta=json.loads((ROOT/'models/flybody-mujoco.json').read_text());m=mujoco.MjModel.from_xml_path(str(ROOT/'models/flybody-mujoco.xml'))
names=[f'wing_{a}_{s}' for s in ['left','right'] for a in ['yaw','roll','pitch']]
qa=np.array([m.jnt_qposadr[m.joint(n).id] for n in names]);va=np.array([m.jnt_dofadr[m.joint(n).id] for n in names]);ids=np.array([m.actuator(n).id for n in names]);otherq=np.array([j['qpos'] for j in meta['joints'] if j['qpos'] not in qa]);otherv=np.array([j['dof'] for j in meta['joints'] if j['dof'] not in va]);limits=np.array([m.jnt_range[m.joint(n).id] for n in names]);base=np.array(config['targets'][-1]);center=base.mean(0);weight=meta['mass_g']*981
frequency=config['frequency_hz'];steps=int((.06+16/frequency)/m.opt.timestep)
def measure(coeff):
 d=mujoco.MjData(m)
 for j in meta['joints']:d.qpos[j['qpos']]=j['neutral']
 d.qpos[:7]=[0,0,2,1,0,0,0];fixed=d.qpos.copy();phase=0.;samples=[]
 table=base.copy();table[:,:3]+=coeff[:3]+(base[:,:3]-center[:3])*coeff[3:];table=np.clip(table,limits[:,0],limits[:,1])
 for step in range(steps):
  d.qpos[:7]=fixed[:7];d.qvel[:6]=0;d.qpos[otherq]=fixed[otherq];d.qvel[otherv]=0
  if step%4==0:
   index=phase*len(table);i=int(index);t=index%1;target=table[i]*(1-t)+table[(i+1)%len(table)]*t;d.ctrl[ids]=np.clip(target-d.qpos[qa],-1,1);phase=(phase+frequency*m.opt.timestep*4)%1
  mujoco.mj_step(m,d)
  if step*m.opt.timestep>.06:
   wrench=d.qfrc_fluid[:6].copy()
   # At this level-body diagnostic pose, free-joint local rotational axes
   # coincide with world axes. Transfer moments about whole-body COM, not
   # about the posteriorly offset thorax/free-joint origin.
   wrench[3:]-=np.cross(d.subtree_com[1]-fixed[:3],wrench[:3])
   samples.append(wrench)
 f=np.mean(samples,axis=0);return np.r_[f[:3]/weight,f[3:]/(weight*.27)]
jac=np.empty((6,6));eps=.02
for i in range(6):
 c=np.zeros(6);c[i]=eps;jac[:,i]=(measure(c)-measure(-c))/(2*eps)
coeff=np.linalg.solve(jac.T@jac+.01*np.eye(6),jac.T@(.05*directions))
# A bounded local fit, not an unconstrained inverse which could ask joints to
# jump to a different wingbeat branch.
coeff*=min(1,.12/np.max(np.abs(coeff)))
keys=['b1','b2','b3','i1','i2','iii1','iii4','iii3','iv1','iv2','iv3','iv4']
config['steering']={k+'_muscle':coeff[:,i].round(8).tolist() for i,k in enumerate(keys)}
config['steering_source']={'repository':'https://github.com/FlyRanch/mpc-simulations','path':'mpc_simulations/MPC_simulations.py','sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'publication':'https://doi.org/10.1038/s41586-024-07293-4','status':'RoboFly force and COM torque directions transferred through a regularized local MuJoCo fit around the current wing table; numerical gain is a prior.','moment_reference':'whole_body_center_of_mass','generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
path.write_text(json.dumps(config,separators=(',',':'))+'\n')
report={'native_jacobian':jac.tolist(),'direction_targets':directions.tolist(),'coefficients':config['steering'],'predicted_responses':(jac@coeff).tolist(),'source':config['steering_source']}
(ROOT/args.report).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(config['steering'],indent=2))
