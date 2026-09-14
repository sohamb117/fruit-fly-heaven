// Optional observation of native aerodynamic wing-hinge moments. This is not
// sensillum strain, total joint reaction, a neural transducer, or a controller.
const SIDES=['left','right'],AXES=['yaw','roll','pitch'];
const fail=message=>{throw new Error('Wing load feedback: '+message);};
const requireThat=(ok,message)=>{if(!ok)fail(message);};
const finite=x=>typeof x==='number'&&Number.isFinite(x);
const clone=sample=>sample?{...sample,momentWorld:{left:sample.momentWorld.left.slice(),right:sample.momentWorld.right.slice()},
  diagnostics:{left:{...sample.diagnostics.left},right:{...sample.diagnostics.right}}}:null;

function solveMoment(axes,anchors,tau){
  requireThat([...axes.flat(),...anchors.flat(),...tau].every(finite),'nonfinite native load/geometry');
  const anchorSpreadCm=Math.max(...anchors.map(a=>Math.hypot(...a.map((x,j)=>x-anchors[0][j]))));
  requireThat(anchorSpreadCm<=1e-10,'wing hinge anchors are not coincident');
  requireThat(axes.every(a=>Math.abs(Math.hypot(...a)-1)<=1e-10),'hinge axis is not unit length');
  // Invert with partial pivoting, retaining an explicit condition bound. The
  // Frobenius product bounds condition₂ above; it is deliberately conservative
  // and needs no iterative decomposition in the 1 ms observation path.
  const a=axes.map((row,i)=>[...row,...[0,1,2].map(j=>Number(i===j))]);
  for(let column=0;column<3;column++){
    let pivot=column;
    for(let row=column+1;row<3;row++)if(Math.abs(a[row][column])>Math.abs(a[pivot][column]))pivot=row;
    [a[column],a[pivot]]=[a[pivot],a[column]];
    const divisor=a[column][column];requireThat(finite(divisor)&&Math.abs(divisor)>1e-15,'singular hinge axes');
    for(let j=0;j<6;j++)a[column][j]/=divisor;
    for(let row=0;row<3;row++)if(row!==column){
      const factor=a[row][column];for(let j=0;j<6;j++)a[row][j]-=factor*a[column][j];
    }
  }
  const inverse=a.map(row=>row.slice(3)),conditionUpperBound=Math.hypot(...axes.flat())*Math.hypot(...inverse.flat());
  requireThat(finite(conditionUpperBound)&&conditionUpperBound<=1e4,'ill-conditioned hinge axes');
  const moment=inverse.map(row=>row.reduce((sum,x,j)=>sum+x*tau[j],0));
  const magnitude=Math.hypot(...moment);
  const residualMaximum=Math.max(...axes.map((row,i)=>Math.abs(row.reduce((sum,x,j)=>sum+x*moment[j],0)-tau[i])));
  requireThat(moment.every(finite)&&finite(magnitude)&&finite(residualMaximum)&&
    residualMaximum<=1e-12+1e-10*Math.max(...tau.map(Math.abs)),'invalid moment reconstruction');
  return {moment,magnitude,diagnostics:{conditionUpperBound,anchorSpreadCm,residualMaximum}};
}

/** Resolve immutable native topology once. capture() reads one matching
 * forward cache and commits only after both sides pass. Native arrays are
 * never retained. read() returns owned plain arrays, independent of both the
 * native heap and subsequent observations. */
export function createWingLoadSampler({mj,model}){
  requireThat(model?.opt?.integrator===0,'only the Euler force-cache timing is supported');
  const timestep=model.opt.timestep;
  requireThat(finite(timestep)&&timestep>0,'invalid native timestep');
  const jointEnum=mj?.mjtObj?.mjOBJ_JOINT?.value;
  requireThat(Number.isInteger(jointEnum)&&typeof mj.mj_name2id==='function','native name resolution is unavailable');
  const ids=SIDES.map(side=>AXES.map(axis=>mj.mj_name2id(model,jointEnum,`wing_${axis}_${side}`)));
  requireThat(ids.flat().every(id=>Number.isInteger(id)&&id>=0&&id<model.njnt)&&new Set(ids.flat()).size===6,
    'six distinct named wing joints are required');
  const type=Int32Array.from(model.jnt_type),body=Int32Array.from(model.jnt_bodyid),dof=Int32Array.from(model.jnt_dofadr);
  const parent=Int32Array.from(model.body_parentid),count=Int32Array.from(model.body_jntnum);
  const joints=ids.map(sideIds=>{
    const wingBody=body[sideIds[0]];
    requireThat(wingBody>0&&wingBody<model.nbody&&sideIds.every(id=>type[id]===3&&body[id]===wingBody)&&
      count[wingBody]===3,'each wing must have exactly three hinges on one body');
    requireThat(!Array.from(parent).some((value,child)=>child!==wingBody&&value===wingBody),'wing body must be a leaf');
    return sideIds.map(id=>({id,dof:dof[id]}));
  });
  requireThat(body[ids[0][0]]!==body[ids[1][0]],'left and right wings must be distinct bodies');
  const dofs=joints.flat().map(j=>j.dof);
  requireThat(dofs.every(value=>Number.isInteger(value)&&value>=0&&value<model.nv)&&new Set(dofs).size===6,
    'invalid wing velocity coordinates');
  let latest=null;
  function capture(data,forceTimeSeconds){
    const bodyTimeSeconds=data.time,lag=bodyTimeSeconds-forceTimeSeconds;
    requireThat(finite(bodyTimeSeconds)&&finite(forceTimeSeconds)&&forceTimeSeconds>=0&&
      (Math.abs(lag)<=1e-12||Math.abs(lag-timestep)<=1e-12),'force timestamp must match the current forward or last Euler step');
    requireThat(!latest||forceTimeSeconds>=latest.forceTimeSeconds,'force timestamp regressed');
    const xaxis=data.xaxis,xanchor=data.xanchor,fluid=data.qfrc_fluid;
    requireThat(xaxis?.length===model.njnt*3&&xanchor?.length===model.njnt*3&&fluid?.length===model.nv,
      'native force/geometry arrays have an invalid shape');
    const sides=joints.map(js=>solveMoment(js.map(j=>Array.from(xaxis.subarray(j.id*3,j.id*3+3))),
      js.map(j=>Array.from(xanchor.subarray(j.id*3,j.id*3+3))),js.map(j=>fluid[j.dof])));
    latest={kind:'native-wing-aerodynamic-moment-v1',units:'g cm^2/s^2',forceTimeSeconds,bodyTimeSeconds,
      left:sides[0].magnitude,right:sides[1].magnitude,
      momentWorld:{left:sides[0].moment,right:sides[1].moment},
      diagnostics:{left:sides[0].diagnostics,right:sides[1].diagnostics}};
  }
  return Object.freeze({capture,read:()=>clone(latest)});
}
