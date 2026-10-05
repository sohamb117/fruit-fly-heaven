// Pure observer/transducer. No MuJoCo, neural, filesystem or controller imports.
const TAU=2*Math.PI;
const finite=(x,label)=>{if(!Number.isFinite(x))throw new Error(`Nonfinite ${label}`);return x;};
const vec=(a,label)=>{if(!Array.isArray(a)||a.length!==3)throw new Error(`Expected vector ${label}`);return a.map(x=>finite(x,label));};
export const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
export const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
export const add=(a,b)=>a.map((x,i)=>x+b[i]);
export const scale=(a,s)=>a.map(x=>x*s);
const norm=a=>Math.hypot(...a);
const unit=(a,label)=>{a=vec(a,label);if(Math.abs(norm(a)-1)>1e-9)throw new Error(`Nonunit ${label}`);return a;};
const owned=a=>Object.freeze(a.slice());
export const DEFAULT_PRIOR=Object.freeze({amplitudeMaxRadians:Math.PI/4,phaseOffsetRadians:Math.PI,powerExponent:1});
export const DEFAULT_ORIENTATIONS=Object.freeze([Math.PI/4,3*Math.PI/4,5*Math.PI/4,7*Math.PI/4]);

export function validateGeometry(g){
  if(!g||!['left','right'].includes(g.side))throw new Error('Geometry side');
  if(!(finite(g.mass,'mass')>0))throw new Error('Nonpositive mass');
  if(!g.units||!['g','kg'].includes(g.units.mass)||!['cm','m'].includes(g.units.length)||g.units.time!=='s')throw new Error('Geometry units');
  const pivot=vec(g.pivotRoot,'pivot'),r=vec(g.neutralComFromPivotRoot,'neutral COM');
  if(!(norm(r)>0))throw new Error('Zero COM lever');
  const axis=unit(g.oscillationAxisRoot,'oscillation axis'),t=unit(g.beamTangentRoot,'beam tangent');
  if(!Array.isArray(g.beamNormalsRoot)||g.beamNormalsRoot.length!==2)throw new Error('Beam normals');
  const n1=unit(g.beamNormalsRoot[0],'beam normal1'),n2=unit(g.beamNormalsRoot[1],'beam normal2');
  if(Math.max(Math.abs(dot(t,n1)),Math.abs(dot(t,n2)),Math.abs(dot(n1,n2)))>1e-9||dot(cross(t,n1),n2)<1-1e-9)throw new Error('Beam frame must be right-handed orthonormal');
  return Object.freeze({side:g.side,mass:g.mass,units:Object.freeze({...g.units}),pivotRoot:owned(pivot),
    neutralComFromPivotRoot:owned(r),oscillationAxisRoot:owned(axis),beamTangentRoot:owned(t),beamNormalsRoot:Object.freeze([owned(n1),owned(n2)])});
}
function validatePrior(p={}){
  if(!p||typeof p!=='object'||Array.isArray(p)||Object.keys(p).some(k=>!Object.hasOwn(DEFAULT_PRIOR,k)))throw new Error('Unknown mechanical prior key');
  const q={...DEFAULT_PRIOR,...p};
  if(!(finite(q.amplitudeMaxRadians,'amplitude')>=0&&q.amplitudeMaxRadians<=Math.PI))throw new Error('Amplitude outside [0,pi]');
  finite(q.phaseOffsetRadians,'phase offset');
  if(!(finite(q.powerExponent,'power exponent')>0))throw new Error('Power exponent');
  return q;
}
function rotation(v,axis,angle){
  const c=Math.cos(angle),s=Math.sin(angle);
  return add(add(scale(v,c),scale(cross(axis,v),s)),scale(axis,dot(axis,v)*(1-c)));
}
function checkedVectors(result){
  for(const [key,value] of Object.entries(result))if(Array.isArray(value)&&value.some(x=>!Number.isFinite(x)))throw new Error(`Overflow in ${key}`);
  return result;
}

export function evaluateVirtualHaltere(geometry,input,prior={}){
  const g=validateGeometry(geometry),p=validatePrior(prior);
  if(!input)throw new Error('Missing haltere input');
  const omega=vec(input.omegaRootRadS,'root angular velocity');
  const phase=finite(input.phaseRadians,'phase'),f=finite(input.frequencyHz,'frequency'),power=finite(input.power,'power');
  if(f<0||f>1000)throw new Error('Frequency outside [0,1000] Hz');
  if(power<0||power>1)throw new Error('Power outside [0,1]');
  const phi=((phase+p.phaseOffsetRadians)%TAU+TAU)%TAU,omegaH=TAU*f;
  const amplitude=p.amplitudeMaxRadians*power**p.powerExponent;
  const theta=amplitude*Math.sin(phi),thetaDot=amplitude*omegaH*Math.cos(phi),thetaDDot=-amplitude*omegaH*omegaH*Math.sin(phi);
  const r=rotation(g.neutralComFromPivotRoot,g.oscillationAxisRoot,theta);
  const axisCrossR=cross(g.oscillationAxisRoot,r);
  const rdot=scale(axisCrossR,thetaDot);
  const rddot=add(scale(axisCrossR,thetaDDot),scale(cross(g.oscillationAxisRoot,axisCrossR),thetaDot*thetaDot));
  const coriolisAcceleration=scale(cross(omega,rdot),2);
  const baselineForce=scale(rddot,-g.mass),coriolisForce=scale(coriolisAcceleration,-g.mass),totalForce=add(baselineForce,coriolisForce);
  const baselineMoment=cross(r,baselineForce),coriolisMoment=cross(r,coriolisForce),totalMoment=add(baselineMoment,coriolisMoment);
  return checkedVectors({kind:'virtual-haltere-point-mass-load',side:g.side,frame:'root-local',units:{...g.units},
    phaseRadians:phase,oscillationPhaseRadians:phi,frequencyHz:f,power,amplitudeRadians:amplitude,theta,thetaDot,thetaDDot,
    rRoot:r,positionRoot:add(g.pivotRoot,r),rdotRoot:rdot,rddotRoot:rddot,coriolisAccelerationRoot:coriolisAcceleration,
    baselineForceRoot:baselineForce,coriolisForceRoot:coriolisForce,totalForceRoot:totalForce,
    baselineMomentRoot:baselineMoment,coriolisMomentRoot:coriolisMoment,totalMomentRoot:totalMoment,
    heldEnvelope:true});
}

export function projectHaltereMoment(geometry,mechanical,projectionPrior={}){
  if(!projectionPrior||typeof projectionPrior!=='object'||Array.isArray(projectionPrior)||Object.keys(projectionPrior).some(k=>!['orientationRadians','momentScale'].includes(k)))throw new Error('Unknown projection prior key');
  const {orientationRadians=DEFAULT_ORIENTATIONS,momentScale}=projectionPrior;
  const g=validateGeometry(geometry);
  if(mechanical?.kind!=='virtual-haltere-point-mass-load'||mechanical.side!==g.side||mechanical.frame!=='root-local')throw new Error('Mechanical sample identity/frame');
  if(!mechanical.units||['mass','length','time'].some(k=>mechanical.units[k]!==g.units[k]))throw new Error('Mechanical sample units');
  if(!(finite(momentScale,'moment scale')>0))throw new Error('Nonpositive moment scale');
  if(!Array.isArray(orientationRadians)||!orientationRadians.length||orientationRadians.length>64)throw new Error('Receptive orientations');
  const angles=orientationRadians.map(x=>finite(x,'receptive orientation'));
  // Axial bending-stress pattern of an ideal beam, up to an unmeasured shared
  // compliance/section factor. The beam frame is FIXED at the root attachment;
  // it does not rotate with virtual stalk theta. This is a declared base-frame
  // prior, not measured sensillum kinematics and not an actual strain tensor.
  const bending= cross(vec(mechanical.totalMomentRoot,'total moment'),g.beamTangentRoot);
  const components=g.beamNormalsRoot.map(n=>dot(bending,n));
  const signed=angles.map(a=>Math.cos(a)*components[0]+Math.sin(a)*components[1]);
  const normalized=signed.map(x=>x/momentScale);
  if([...signed,...normalized].some(x=>!Number.isFinite(x)))throw new Error('Projection overflow');
  return {kind:'haltere-bending-projection-prior',side:g.side,frame:'root-local',beamFrame:'fixed-root-attachment',units:{...g.units},
    beamBendingComponents:components,orientationRadians:angles,signedBendingMoments:signed,
    momentScale,normalizedSigned:normalized,compression:normalized.map(x=>Math.max(0,x))};
}

// Version 2 is an opt-in, one-way mechanical observer. It never applies force
// to the native body. Its coefficients are explicit priors, not measurements
// from Drosophila or BANC. The original stateless functions above are unchanged.
const COUPLED_KIND='ipsilateral-wing-thorax-haltere-v2';
const GRID_SECONDS=.0005,INTERNAL_SECONDS=.00005,TIME_EPS=1e-9;
const copy=value=>structuredClone(value);
const exactKeys=(value,keys,label)=>{
  if(!value||typeof value!=='object'||Array.isArray(value)||
    Object.keys(value).length!==keys.length||keys.some(key=>!Object.hasOwn(value,key)))throw new Error(`Invalid ${label} keys`);
};
const inRange=(value,lo,hi,label)=>{finite(value,label);if(value<lo||value>hi)throw new Error(`Invalid ${label} range`);return value;};
const wrap=value=>((value%TAU)+TAU)%TAU;

export function validateCoupledHaltereModel(config){
  exactKeys(config,['schemaVersion','kind','coefficientStatus','naturalFrequencyHz','dampingRatio',
    'ownDriveAmplitudeRadians','ownDrivePhaseRadians','couplingStiffnessRatio','couplingDampingRatio','sides'],'coupled haltere model');
  if(config.schemaVersion!==2||config.kind!==COUPLED_KIND||config.coefficientStatus!=='unmeasured-structural-prior')throw new Error('Explicit version 2 mechanical prior required');
  inRange(config.naturalFrequencyHz,1,1000,'natural frequency');
  inRange(config.dampingRatio,.01,2,'damping ratio');
  inRange(config.ownDriveAmplitudeRadians,0,Math.PI/4,'own drive amplitude');
  finite(config.ownDrivePhaseRadians,'own drive phase');
  inRange(config.couplingStiffnessRatio,0,2,'coupling stiffness');
  inRange(config.couplingDampingRatio,0,2,'coupling damping');
  exactKeys(config.sides,['left','right'],'mechanical sides');
  for(const side of ['left','right']){
    const s=config.sides[side];
    exactKeys(s,['joint','nativeSign','centerRadians','wingToHaltereRatio'],'side coupling');
    if(!['yaw','roll','pitch'].some(axis=>s.joint===`wing_${axis}_${side}`))throw new Error('Coupling must use its ipsilateral wing joint');
    if(s.nativeSign!==1&&s.nativeSign!==-1)throw new Error('Explicit native joint sign required');
    finite(s.centerRadians,'native joint center');
    inRange(s.wingToHaltereRatio,0,2,'wing to haltere ratio');
  }
  return copy(config);
}

function motionLoad(g,omega,state,acceleration,extra){
  const r=rotation(g.neutralComFromPivotRoot,g.oscillationAxisRoot,state.angleRadians),a=cross(g.oscillationAxisRoot,r);
  const rdot=scale(a,state.angularVelocityRadS),rddot=add(scale(a,acceleration),scale(cross(g.oscillationAxisRoot,a),state.angularVelocityRadS**2));
  const coriolisAcceleration=scale(cross(omega,rdot),2),baselineForce=scale(rddot,-g.mass),coriolisForce=scale(coriolisAcceleration,-g.mass);
  const baselineMoment=cross(r,baselineForce),coriolisMoment=cross(r,coriolisForce);
  return checkedVectors({kind:'virtual-haltere-point-mass-load',side:g.side,frame:'root-local',units:{...g.units},
    ...extra,theta:state.angleRadians,thetaDot:state.angularVelocityRadS,thetaDDot:acceleration,
    rRoot:r,positionRoot:add(g.pivotRoot,r),rdotRoot:rdot,rddotRoot:rddot,coriolisAccelerationRoot:coriolisAcceleration,
    baselineForceRoot:baselineForce,coriolisForceRoot:coriolisForce,totalForceRoot:add(baselineForce,coriolisForce),
    baselineMomentRoot:baselineMoment,coriolisMomentRoot:coriolisMoment,totalMomentRoot:add(baselineMoment,coriolisMoment),heldEnvelope:false});
}

export function createCoupledVirtualHalteres(geometries,configuration){
  const config=validateCoupledHaltereModel(configuration),sides=['left','right'];
  if(!Array.isArray(geometries)||geometries.length!==2)throw new Error('Two haltere geometries required');
  const gs=geometries.map(validateGeometry),geometry=Object.fromEntries(gs.map(g=>[g.side,g]));
  if(sides.some(side=>!geometry[side]))throw new Error('Both haltere sides required');
  const contract=JSON.stringify({config,geometry}),w=TAU*config.naturalFrequencyHz;
  let tick=null,held=null,states;
  function reset(){tick=null;held=null;states=sides.map(()=>({angleRadians:0,angularVelocityRadS:0,ownPhaseRadians:wrap(config.ownDrivePhaseRadians)}));}
  reset();
  function normalize(sample){
    if(!sample||typeof sample!=='object')throw new Error('Missing native mechanical sample');
    const time=finite(sample.bodyTimeSeconds,'body time'),offset=finite(sample.elapsedSeconds??0,'sample offset');
    if(time<0||offset<0||offset>.002+TIME_EPS)throw new Error('Invalid body time or held offset');
    const blockTick=Math.round(time/.002),offsetTick=Math.round(offset/GRID_SECONDS);
    if(Math.abs(time-blockTick*.002)>TIME_EPS||Math.abs(offset-offsetTick*GRID_SECONDS)>TIME_EPS)throw new Error('Haltere inputs require native 2 ms boundaries and 0.5 ms sample grid');
    const result={bodyTimeSeconds:blockTick*.002,elapsedSeconds:offsetTick*GRID_SECONDS,
      wingPhaseRadians:finite(sample.wingPhaseRadians,'wing phase'),wingFrequencyHz:inRange(sample.wingFrequencyHz,0,1000,'wing frequency'),
      omegaRootRadS:vec(Array.from(sample.omegaRootRadS??[]),'native root angular velocity'),halterePower:[],wingPower:[],wingMotion:{}};
    for(const key of ['halterePower','wingPower']){
      const values=sample[key];
      if(!(Array.isArray(values)||ArrayBuffer.isView(values))||values.length!==2)throw new Error(`Both actual ${key} values required`);
      result[key]=Array.from(values,(value)=>inRange(value,0,1,key));
    }
    for(const side of sides){
      const m=sample.wingMotion?.[side],p=config.sides[side];
      if(m?.joint!==p.joint||m.source!=='native-joint'||m.frame!=='native-joint-coordinate')throw new Error('Native ipsilateral wing motion identity/frame required');
      result.wingMotion[side]={joint:m.joint,source:m.source,frame:m.frame,
        angleRadians:finite(m.angleRadians,'wing angle'),angularVelocityRadS:finite(m.angularVelocityRadS,'wing angular velocity')};
    }
    return result;
  }
  function driver(sample,side,time){
    const m=sample.wingMotion[side],p=config.sides[side],dt=time-sample.bodyTimeSeconds,
      frequency=sample.wingPower[sides.indexOf(side)]>0?TAU*sample.wingFrequencyHz:0;
    // Local harmonic continuation is a sampling prior, not a future native
    // observation. It is reconstructed from this side's actual q/qdot.
    const q=m.angleRadians-p.centerRadians,v=m.angularVelocityRadS;
    const angle=frequency===0?q+v*dt:q*Math.cos(frequency*dt)+v/frequency*Math.sin(frequency*dt);
    const velocity=frequency===0?v:-q*frequency*Math.sin(frequency*dt)+v*Math.cos(frequency*dt);
    return {angle:p.nativeSign*p.wingToHaltereRatio*angle,velocity:p.nativeSign*p.wingToHaltereRatio*velocity};
  }
  function derivative(state,sample,i,time){
    const d=driver(sample,sides[i],time),power=sample.halterePower[i];
    // Forced torsional oscillator. Own drive and transmitted same-side motion
    // remain separate; zero own power does not erase passive coupling.
    const own=2*config.dampingRatio*w*w*config.ownDriveAmplitudeRadians*power*Math.cos(state.ownPhaseRadians);
    const coupling=config.couplingStiffnessRatio*w*w*(d.angle-state.angleRadians)+
      2*config.couplingDampingRatio*w*(d.velocity-state.angularVelocityRadS);
    return [state.angularVelocityRadS,own+coupling-2*config.dampingRatio*w*state.angularVelocityRadS-w*w*state.angleRadians,w];
  }
  const shifted=(s,d,h)=>({angleRadians:s.angleRadians+d[0]*h,angularVelocityRadS:s.angularVelocityRadS+d[1]*h,ownPhaseRadians:s.ownPhaseRadians+d[2]*h});
  function integrate(state,sample,i,time,h){
    const a=derivative(state,sample,i,time),b=derivative(shifted(state,a,h/2),sample,i,time+h/2),
      c=derivative(shifted(state,b,h/2),sample,i,time+h/2),d=derivative(shifted(state,c,h),sample,i,time+h);
    const next=shifted(state,a.map((x,k)=>(x+2*b[k]+2*c[k]+d[k])/6),h);next.ownPhaseRadians=wrap(next.ownPhaseRadians);
    if(Object.values(next).some(x=>!Number.isFinite(x))||Math.abs(next.angleRadians)>Math.PI)throw new Error('Coupled haltere state outside declared angular domain');
    return next;
  }
  function sample(input){
    const current=normalize(input),nextTick=Math.round((current.bodyTimeSeconds+current.elapsedSeconds)/GRID_SECONDS);
    if(tick===null&&nextTick!==0)throw new Error('Initialize haltere dynamics at time zero or restore a snapshot');
    if(tick!==null&&(nextTick<tick||nextTick>tick+1))throw new Error('Haltere time must advance exactly 0.5 ms');
    if(nextTick===tick&&JSON.stringify(current)!==JSON.stringify(held))throw new Error('Conflicting repeated haltere timestamp');
    let next=copy(states);
    if(tick!==null&&nextTick>tick){
      // Integrate with the PREVIOUS observed body context. New observations may
      // change endpoint acceleration but must never act backwards in time.
      const n=Math.round(GRID_SECONDS/INTERNAL_SECONDS),h=GRID_SECONDS/n;
      for(let k=0;k<n;k++)next=next.map((s,i)=>integrate(s,held,i,tick*GRID_SECONDS+k*h,h));
    }
    const loads={};
    for(const [i,side]of sides.entries()){
      const state=next[i],acceleration=derivative(state,current,i,nextTick*GRID_SECONDS)[1];
      const amplitude=Math.hypot(state.angleRadians,state.angularVelocityRadS/w),phase=amplitude===0?null:wrap(Math.atan2(state.angleRadians,state.angularVelocityRadS/w));
      loads[side]=motionLoad(geometry[side],current.omegaRootRadS,state,acceleration,{mechanicalModel:COUPLED_KIND,
        sampleTimeSeconds:nextTick*GRID_SECONDS,power:current.halterePower[i],actualMusclePower:current.halterePower[i],
        ipsilateralWingPower:current.wingPower[i],nativeWingMotion:copy(current.wingMotion[side]),
        amplitudeRadians:amplitude,oscillationPhaseRadians:phase,ownDrivePhaseRadians:state.ownPhaseRadians,
        naturalFrequencyHz:config.naturalFrequencyHz,wingPhaseRadians:current.wingPhaseRadians,
        coefficientStatus:config.coefficientStatus,wingMotionContinuation:'causal-local-harmonic',
        amplitudeDefinition:'hypot(theta,thetaDot/naturalOmega); not measured stroke amplitude'});
    }
    states=next;tick=nextTick;held=current;
    return {kind:COUPLED_KIND,sampleTimeSeconds:tick*GRID_SECONDS,sides:loads};
  }
  function snapshot(){return copy({schemaVersion:2,kind:COUPLED_KIND,contract,tick,held,states});}
  function restore(value){
    exactKeys(value,['schemaVersion','kind','contract','tick','held','states'],'haltere snapshot');
    if(value.schemaVersion!==2||value.kind!==COUPLED_KIND||value.contract!==contract)throw new Error('Haltere snapshot contract mismatch');
    if(value.tick!==null&&(!Number.isSafeInteger(value.tick)||value.tick<0))throw new Error('Invalid haltere snapshot time');
    if(!Array.isArray(value.states)||value.states.length!==2)throw new Error('Invalid haltere snapshot sides');
    const next=copy(value.states);
    for(const s of next){exactKeys(s,['angleRadians','angularVelocityRadS','ownPhaseRadians'],'haltere snapshot state');
      inRange(s.angleRadians,-Math.PI,Math.PI,'snapshot angle');finite(s.angularVelocityRadS,'snapshot velocity');inRange(s.ownPhaseRadians,0,TAU,'snapshot phase');}
    const context=value.held===null?null:normalize(value.held);
    if((value.tick===null)!==(context===null)||context&&Math.round((context.bodyTimeSeconds+context.elapsedSeconds)/GRID_SECONDS)!==value.tick)throw new Error('Haltere snapshot clock mismatch');
    if(value.tick===null&&JSON.stringify(next)!==JSON.stringify(sides.map(()=>({angleRadians:0,angularVelocityRadS:0,ownPhaseRadians:wrap(config.ownDrivePhaseRadians)}))))throw new Error('Nonzero uninitialized haltere state');
    states=next;tick=value.tick;held=context;
  }
  return Object.freeze({kind:COUPLED_KIND,sample,reset,snapshot,restore});
}
