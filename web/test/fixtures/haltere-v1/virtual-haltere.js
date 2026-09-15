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
