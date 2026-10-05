// Demonstration-only privileged teacher. Ported from the independently tested
// native-control diagnostic; never imported by the deployed motor decoder.
// All controls use a supplied measured calibration. No body mutation or IO.
export const controlNames=Object.freeze(['power_left','power_right',
 ...['left','right'].flatMap(side=>['yaw','roll','pitch'].flatMap(axis=>
  ['constant','sin','cos'].map(basis=>`steering_${side}_${axis}_${basis}`)))]);
export const names=controlNames;
export const lower=Object.freeze(controlNames.map((_,i)=>i<2?0:-.25));
export const upper=Object.freeze(controlNames.map((_,i)=>i<2?1:.25));
export const bounds=Object.freeze(lower.map((value,i)=>Object.freeze([value,upper[i]])));
export const defaultCenter=Object.freeze(controlNames.map((_,i)=>i<2?.85:0));
export const defaultcenter=defaultCenter;
export const DEFAULT_GAINS=Object.freeze({naturalFrequency:24,dampingRatio:.9,
 omegaFilterTau:.006,heightKp:16,heightKd:8,maximumVerticalAcceleration:.4*981,
 minimumUpForLift:.5,gravity:981,lengthCm:.27,allocationPasses:80,regularization:1e-4});
const clamp=(value,min,max)=>Math.max(min,Math.min(max,value));
const finite=Number.isFinite;
function requireThat(condition,message){if(!condition)throw new TypeError('Oracle controller: '+message);}
function vector(value,n,label){requireThat(value?.length===n&&Array.from(value).every(finite),label+' must have '+n+' finite entries');}
function bounded(value){vector(value,20,'controls');requireThat(Array.from(value).every((x,i)=>x>=lower[i]&&x<=upper[i]),'controls outside coefficient bounds');}

/** Current wing boundary: two powers and six phase-dependent signed residuals.
 * Optional stats collects samples/clippedSamples/clippedChannels; it is not part
 * of physical control. Coefficient bounds alone do not bound their phase sum. */
export function synthesize(controls,phase,stats){
 bounded(controls);requireThat(finite(phase),'phase must be finite');
 const basis=[1,Math.sin(phase),Math.cos(phase)],steering=[[],[]];
 let clippedCount=0,maximumUnclippedMagnitude=0;
 for(let side=0;side<2;side++)for(let axis=0;axis<3;axis++){
  const offset=2+(side*3+axis)*3;
  const raw=basis.reduce((sum,value,k)=>sum+value*controls[offset+k],0);
  const value=clamp(raw,-.25,.25);steering[side].push(value);
  clippedCount+=Number(value!==raw);maximumUnclippedMagnitude=Math.max(maximumUnclippedMagnitude,Math.abs(raw));
 }
 if(stats){stats.samples=(stats.samples||0)+1;stats.clippedSamples=(stats.clippedSamples||0)+Number(clippedCount>0);
  stats.clippedChannels=(stats.clippedChannels||0)+clippedCount;
  stats.maximumUnclippedMagnitude=Math.max(stats.maximumUnclippedMagnitude||0,maximumUnclippedMagnitude);}
 return {power:[controls[0],controls[1]],steering,
  diagnostic:{clippedCount,sampleCount:1,maximumUnclippedMagnitude}};
}

/** Row-scaled regularized box least squares for the measured 4 x 20 Jacobian.
 * Minimize ||S(baseY+J(u-baseU)-target)||^2 + lambda||u-reference||^2.
 * Returns only the bounded coefficient vector. No random search is performed. */
export function allocate(J,baseU,baseY,target,options={}){
 requireThat(J?.length===4,'Jacobian must have four rows');J.forEach(row=>vector(row,20,'Jacobian row'));
 bounded(baseU);vector(baseY,4,'base response');vector(target,4,'target');
 const reference=options.reference??baseU,start=options.start??reference;
 bounded(reference);bounded(start);
 const passes=options.passes??80,regularization=options.regularization??1e-4;
 requireThat(Number.isInteger(passes)&&passes>0,'passes must be a positive integer');
 requireThat(finite(regularization)&&regularization>0,'regularization must be positive');
 const scale=options.rowScale??J.map(row=>1/Math.max(1e-6,Math.hypot(...row)));
 vector(scale,4,'row scale');requireThat(Array.from(scale).every(x=>x>0),'row scales must be positive');
 const A=J.map((row,i)=>Array.from(row,x=>x*scale[i]));
 const rhs=target.map((value,i)=>(value-baseY[i]+J[i].reduce((sum,b,j)=>sum+b*baseU[j],0))*scale[i]);
 const u=Array.from(start),residual=rhs.map((value,i)=>A[i].reduce((sum,a,j)=>sum+a*u[j],-value));
 for(let pass=0;pass<passes;pass++)for(let j=0;j<20;j++){
  const gradient=A.reduce((sum,row,i)=>sum+row[j]*residual[i],regularization*(u[j]-reference[j]));
  const curvature=A.reduce((sum,row)=>sum+row[j]*row[j],regularization);
  const next=clamp(u[j]-gradient/curvature,lower[j],upper[j]),delta=next-u[j];u[j]=next;
  for(let i=0;i<4;i++)residual[i]+=A[i][j]*delta;
 }
 bounded(u);return u;
}

export function createPDController({trim,J,response,inertia,massG,targetHeight,gains={}}){
 bounded(trim);requireThat(J?.length===4,'Jacobian must have four rows');J.forEach(row=>vector(row,20,'Jacobian row'));
 vector(response,4,'trim response');requireThat(inertia?.length===3,'inertia must have three rows');inertia.forEach(row=>vector(row,3,'inertia row'));
 requireThat(finite(massG)&&massG>0&&finite(targetHeight),'positive mass and finite target height required');
 const g={...DEFAULT_GAINS,...gains};
 for(const key of Object.keys(DEFAULT_GAINS))requireThat(finite(g[key])&&g[key]>=0,'invalid gain '+key);
 requireThat(g.gravity>0&&g.lengthCm>0&&g.minimumUpForLift>0&&g.regularization>0&&Number.isInteger(g.allocationPasses)&&g.allocationPasses>0,'positive normalization/allocation settings required');
 const weight=massG*g.gravity,baseU=Array.from(trim),baseY=Array.from(response),matrix=J.map(row=>Array.from(row)),I=inertia.map(row=>Array.from(row));
 const filteredOmega=[0,0,0];let controls=Array.from(trim),last=null;
 function reset(){filteredOmega.fill(0);controls=Array.from(baseU);last=null;}
 function update({quaternion,omegaRoot,height,verticalSpeed,dt}){
  vector(quaternion,4,'quaternion');vector(omegaRoot,3,'root angular velocity');
  requireThat(finite(height)&&finite(verticalSpeed)&&finite(dt)&&dt>0,'finite body measurements and positive dt required');
  const norm=Math.hypot(...quaternion);requireThat(norm>1e-12,'nonzero quaternion required');
  const q=Array.from(quaternion,x=>x/norm),alpha=g.omegaFilterTau===0?1:-Math.expm1(-dt/g.omegaFilterTau);
  for(let i=0;i<3;i++)filteredOmega[i]+=alpha*(omegaRoot[i]-filteredOmega[i]);
  const sign=q[0]<0?-1:1,n=Math.hypot(q[1],q[2],q[3]),angle=2*Math.atan2(n,Math.abs(q[0]));
  const error=q.slice(1).map(value=>n>1e-12?sign*value*angle/n:0);
  const kp=g.naturalFrequency**2,kd=2*g.dampingRatio*g.naturalFrequency;
  const acceleration=error.map((value,i)=>-kp*value-kd*filteredOmega[i]);
  const torque=I.map(row=>row.reduce((sum,value,i)=>sum+value*acceleration[i],0));
  const az=clamp(g.heightKp*(targetHeight-height)-g.heightKd*verticalSpeed,-g.maximumVerticalAcceleration,g.maximumVerticalAcceleration);
  const up=1-2*(q[1]*q[1]+q[2]*q[2]);
  const target=[(1+az/g.gravity)/Math.max(g.minimumUpForLift,up),...torque.map(value=>value/(weight*g.lengthCm))];
  controls=allocate(matrix,baseU,baseY,target,{reference:baseU,start:controls,passes:g.allocationPasses,regularization:g.regularization});
  last={target,error,angularAcceleration:acceleration,torque,up,filteredOmega:[...filteredOmega],
   saturatedCoefficients:controls.filter((value,i)=>value<=lower[i]+1e-9||value>=upper[i]-1e-9).length};
  return [...controls];
 }
 return {update,reset,gains:Object.freeze(g),readState:()=>({controls:[...controls],filteredOmega:[...filteredOmega],last})};
}


export function validateFlightTeacherCalibration(c){
 requireThat(c?.schemaVersion===1&&c.kind==='flight-state-teacher-calibration-v1','unsupported calibration artifact');
 requireThat(c.source?.kind==='native-control-calibration'&&/^[0-9a-f]{64}$/.test(c.source.sha256)&&
  typeof c.source.description==='string'&&c.source.description.length>0,'native calibration provenance required');
 requireThat(c.mechanics&&typeof c.mechanics==='object'&&!Array.isArray(c.mechanics)&&Object.keys(c.mechanics).length>0&&
  Object.entries(c.mechanics).every(([url,sha])=>url.startsWith('/')&&/^[0-9a-f]{64}$/.test(sha)),'mechanical source pins required');
 requireThat(JSON.stringify(c.controlNames)===JSON.stringify(controlNames),'control ordering mismatch');
 requireThat(c.gains&&Object.hasOwn(c.gains,'naturalFrequency')&&Object.hasOwn(c.gains,'dampingRatio')&&
  Object.keys(c.gains).every(k=>Object.hasOwn(DEFAULT_GAINS,k)),'explicit supported teacher gains required');
 createPDController({...c,targetHeight:0});
 return structuredClone(c);
}

export function createFlightTeacher({calibration,targetHeight}){
 const c=validateFlightTeacherCalibration(calibration),controller=createPDController({...c,targetHeight});
 let controls=c.trim.slice();
 return Object.freeze({
  update(observation){controls=controller.update(observation);return controls.slice();},
  sample(phase){
   const executed=synthesize(controls,phase),basis=[1,Math.sin(phase),Math.cos(phase)],steering=[[],[]];
   for(let side=0;side<2;side++)for(let axis=0;axis<3;axis++){
    const offset=2+(side*3+axis)*3;steering[side][axis]=basis.reduce((s,b,k)=>s+b*controls[offset+k],0);
   }
   return {controls:{power:executed.power,steering:executed.steering},
    targets:{power:controls.slice(0,2),steering},clippedChannels:executed.diagnostic.clippedCount};
  },
  reset(){controller.reset();controls=c.trim.slice();},readState:controller.readState
 });
}
