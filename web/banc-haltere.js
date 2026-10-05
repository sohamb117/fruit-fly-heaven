import {validateGeometry,evaluateVirtualHaltere,projectHaltereMoment,createCoupledVirtualHalteres,validateCoupledHaltereModel} from './virtual-haltere.js';

const TAU=2*Math.PI,HALTERE_COUNT=328;
const SIDES=Object.freeze(['left','right']);
const finite=(value,name)=>{if(!Number.isFinite(value))throw new Error(`Invalid ${name}`);return value;};
const freeze=value=>{if(value&&typeof value==='object'){for(const child of Object.values(value))freeze(child);Object.freeze(value);}return value;};
const rootId=value=>typeof value==='bigint'&&value>0n?value.toString():typeof value==='string'&&/^[1-9][0-9]*$/.test(value)?value:null;
export const HALTERE_CURRENT_PRIOR=Object.freeze({maxCurrentPa:800,referenceFrequencyHz:236,
  referenceAmplitudeRadians:Math.PI/4,status:'declared sensitivity prior; not measured receptor physiology',
  formula:'I_pA = maxCurrentPa * c / (1 + c); c = max(0, signedBending / momentScale)',
  scale:'mass * norm(neutralComFromPivotRoot)^2 * (pi/4) * (2*pi*236)^2'});

export function validateHaltereCurrentProfile(profile,{sensoryManifest,preparedIds}={}){
  if(profile?.schemaVersion===2)return validatePopulationProfile(profile,{sensoryManifest,preparedIds});
  if(profile?.schemaVersion!==1||profile.kind!=='banc-haltere-mechanical-current-profile'||profile.dataset!=='BANC'||profile.materialization!==888)throw new Error('Expected an explicit BANC v888 haltere current profile');
  if(sensoryManifest?.schema_version!==1||!Number.isInteger(sensoryManifest.neuron_count)||sensoryManifest.neuron_count<HALTERE_COUNT||profile.neuronCount!==sensoryManifest.neuron_count)throw new Error('Haltere profile neuron count mismatch');
  if(!(preparedIds instanceof BigUint64Array)&&!Array.isArray(preparedIds))throw new Error('Prepared root IDs must be exact bigint or decimal strings');
  if(preparedIds.length!==profile.neuronCount)throw new Error('Prepared root ID count mismatch');
  if(!Array.isArray(sensoryManifest.body_transducers))throw new Error('Missing prepared body transducers');
  const annotated=new Map(),bodyIndices=new Set(),annotatedRoots=new Set();
  for(const cell of sensoryManifest.body_transducers){
    if(!Number.isInteger(cell.index)||cell.index<0||cell.index>=profile.neuronCount||bodyIndices.has(cell.index))throw new Error('Invalid or duplicate prepared body transducer');
    bodyIndices.add(cell.index);
    if(cell.organ!=='haltere')continue;
    if(cell.kind!=='rotation'||cell.annotation!=='haltere_campaniform_sensillum_neuron'||!SIDES.includes(cell.side)||typeof cell.cell_type!=='string'||!cell.cell_type)throw new Error('Invalid annotated haltere identity/side/type');
    const id=rootId(preparedIds[cell.index]);
    if(!id||annotatedRoots.has(id))throw new Error('Invalid or duplicate prepared haltere root ID');
    annotatedRoots.add(id);annotated.set(cell.index,{...cell,root_id:id});
  }
  if(annotated.size!==HALTERE_COUNT||!Array.isArray(profile.cells)||profile.cells.length!==HALTERE_COUNT)throw new Error('Haltere profile must cover exactly all 328 annotated transducers');
  if(profile.orientationAssignment?.status!=='exchangeable-orientation-prior'||profile.orientationAssignment.anatomicalKnowledge!==false)throw new Error('Orientations must be explicitly labeled an exchangeable prior');
  const seenIndices=new Set(),seenRoots=new Set(),cells=[];
  for(const cell of profile.cells){
    const original=annotated.get(cell.index);
    if(!original||seenIndices.has(cell.index)||seenRoots.has(cell.root_id))throw new Error('Missing, duplicate, or unannotated haltere profile identity');
    if(typeof cell.root_id!=='string'||cell.root_id!==original.root_id)throw new Error('Haltere index/root ID mismatch');
    if(cell.organ!=='haltere'||cell.side!==original.side||cell.cell_type!==original.cell_type)throw new Error('Haltere organ/side/type mismatch');
    if(!Number.isFinite(cell.orientationRadians)||cell.orientationRadians<0||cell.orientationRadians>=TAU)throw new Error('Explicit haltere orientation must be in [0,2pi)');
    seenIndices.add(cell.index);seenRoots.add(cell.root_id);
    cells.push(Object.freeze({index:cell.index,root_id:cell.root_id,organ:cell.organ,side:cell.side,cell_type:cell.cell_type,orientationRadians:cell.orientationRadians}));
  }
  return Object.freeze(cells);
}

export function createHaltereCurrentMapper({enabled=false,profile,sensoryManifest,preparedIds,geometries,maxCurrentPa=HALTERE_CURRENT_PRIOR.maxCurrentPa,mechanicalModel}={}){
  if(profile?.schemaVersion===2)return createPopulationMapper({enabled,profile,sensoryManifest,preparedIds,geometries,maxCurrentPa,mechanicalModel});
  if(mechanicalModel!==undefined)throw new Error('Coupled haltere dynamics require an explicit version 2 sensory profile');
  if(enabled!==true)throw new Error('Haltere mechanical current requires explicit enabled:true');
  const cells=validateHaltereCurrentProfile(profile,{sensoryManifest,preparedIds});
  if(!Array.isArray(geometries)||geometries.length!==2)throw new Error('Both side-specific haltere geometries are required');
  const geometry=new Map();
  for(const descriptor of geometries){const g=validateGeometry(descriptor);if(geometry.has(g.side))throw new Error('Duplicate haltere geometry side');geometry.set(g.side,g);}
  if(SIDES.some(side=>!geometry.has(side)))throw new Error('Missing side-specific haltere geometry');
  if(!(finite(maxCurrentPa,'maximum haltere current')>=0&&maxCurrentPa<=3.4028234663852886e38))throw new Error('Haltere maximum current is outside finite Float32 range');
  const neuronCount=profile.neuronCount,indices=Object.freeze(cells.map(cell=>cell.index));
  const scales=new Map(SIDES.map(side=>{const g=geometry.get(side),r=Math.hypot(...g.neutralComFromPivotRoot);
    const value=g.mass*r*r*HALTERE_CURRENT_PRIOR.referenceAmplitudeRadians*(TAU*HALTERE_CURRENT_PRIOR.referenceFrequencyHz)**2;
    if(!(value>0&&Number.isFinite(value)))throw new Error('Invalid haltere moment scale');return [side,value];}));
  const projections=cells.map(cell=>({side:cell.side,cos:Math.cos(cell.orientationRadians),sin:Math.sin(cell.orientationRadians)}));
  const validateTarget=target=>{if(!(target instanceof Float32Array)||target.length!==neuronCount)throw new Error('Expected a full Float32 neural-current vector');};

  function compute(sample){
    if(!sample||typeof sample!=='object'||Array.isArray(sample))throw new Error('Missing haltere mechanical sample');
    const omega=sample.omegaRootRadS;
    if(!(Array.isArray(omega)||ArrayBuffer.isView(omega))||omega.length!==3||Array.from(omega).some(value=>!Number.isFinite(value)))throw new Error('Expected native root-local angular velocity, rad/s');
    const frequency=finite(sample.wingFrequencyHz,'wing frequency'),phase=finite(sample.wingPhaseRadians,'wing phase');
    const elapsed=finite(sample.elapsedSeconds??0,'elapsed block offset');
    if(frequency<0||frequency>1000)throw new Error('Wing frequency outside [0,1000] Hz');
    if(elapsed<0||elapsed>.002)throw new Error('Haltere sample offset outside the held 2ms body block');
    const powerInput=sample.halterePower;
    if(powerInput!==undefined&&powerInput!==null&&(!(Array.isArray(powerInput)||ArrayBuffer.isView(powerInput))||!Number.isInteger(powerInput.length)||powerInput.length>2))throw new Error('Expected side-specific halterePower [left,right]');
    const samplePhase=phase+TAU*frequency*elapsed;
    if(!Number.isFinite(samplePhase))throw new Error('Haltere phase overflow');
    const sideSamples={},components={};
    for(const [i,side]of SIDES.entries()){
      const raw=powerInput?.[i],missing=raw===undefined||raw===null,power=missing?0:finite(raw,side+' haltere power');
      if(power<0||power>1)throw new Error('Haltere power outside [0,1]');
      const g=geometry.get(side),mechanical=evaluateVirtualHaltere(g,{omegaRootRadS:Array.from(omega),phaseRadians:samplePhase,frequencyHz:frequency,power});
      const projection=projectHaltereMoment(g,mechanical,{orientationRadians:[0,Math.PI/2],momentScale:scales.get(side)});
      components[side]=projection.beamBendingComponents;
      sideSamples[side]={powerStatus:missing?'missing-zeroed':'present',power,momentScale:scales.get(side),
        baselineMomentRoot:mechanical.baselineMomentRoot,coriolisMomentRoot:mechanical.coriolisMomentRoot,
        totalMomentRoot:mechanical.totalMomentRoot,beamBendingComponents:projection.beamBendingComponents,
        activeCells:0,maxCurrentPa:0};
    }
    const values=new Float32Array(cells.length);
    for(let k=0;k<cells.length;k++){
      const p=projections[k],b=components[p.side],signed=p.cos*b[0]+p.sin*b[1],c=Math.max(0,signed/scales.get(p.side));
      const current=maxCurrentPa*(c/(1+c));
      if(!Number.isFinite(c)||!Number.isFinite(current)||current<0)throw new Error('Nonfinite haltere transduction');
      values[k]=current;
      if(!Number.isFinite(values[k]))throw new Error('Haltere current exceeds Float32 range');
      if(values[k]>0)sideSamples[p.side].activeCells++;
      sideSamples[p.side].maxCurrentPa=Math.max(sideSamples[p.side].maxCurrentPa,values[k]);
    }
    const diagnostics=freeze({kind:'haltere-mechanical-current-prior',units:'pA',frame:'root-local',elapsedSeconds:elapsed,
      wingPhaseRadians:phase,samplePhaseRadians:samplePhase,wingFrequencyHz:frequency,omegaRootRadS:Array.from(omega),maxCurrentPa,
      sensitivityStatus:HALTERE_CURRENT_PRIOR.status,orientationStatus:'exchangeable-orientation-prior',sides:sideSamples});
    return {values,diagnostics};
  }
  function currents(sample){const result=compute(sample);Object.defineProperty(result.values,'diagnostics',{value:result.diagnostics});return result.values;}
  function writeInto(target,sample){
    validateTarget(target);const result=compute(sample);
    // Compute/validate all 328 values before touching the caller's vector.
    for(let k=0;k<indices.length;k++)target[indices[k]]=result.values[k];
    return result.diagnostics;
  }
  function apply(base,sample){validateTarget(base);const target=base.slice(),diagnostics=writeInto(target,sample);
    Object.defineProperty(target,'haltereDiagnostics',{value:diagnostics});return target;}
  return Object.freeze({kind:'banc-haltere-mechanical-current-mapper',indices,cells,neuronCount,maxCurrentPa,currents,writeInto,apply});
}

function validatePopulationProfile(profile,{sensoryManifest,preparedIds}){
  if(profile.kind!=='banc-haltere-mechanical-current-profile'||profile.dataset!=='BANC'||profile.materialization!==888||
    profile.orientationAssignment?.status!=='population-tuning-explicit'||profile.orientationAssignment.anatomicalKnowledge!==false)
    throw new Error('Version 2 requires explicit population tuning with no inferred anatomical directions');
  if(!Array.isArray(profile.cells)||profile.cells.some(c=>Object.hasOwn(c,'orientationRadians')))throw new Error('Version 2 uses population tuning, not per-root orientation cycling');
  // Reuse the exact prepared 328-cell join. Dummy angles are used only for this
  // identity validation and are never returned or used by the transducer.
  const identities=validateHaltereCurrentProfile({...profile,schemaVersion:1,
    orientationAssignment:{status:'exchangeable-orientation-prior',anatomicalKnowledge:false},
    cells:profile.cells.map(c=>({...c,orientationRadians:0}))},{sensoryManifest,preparedIds});
  if(!Array.isArray(profile.tuningPopulations)||profile.tuningPopulations.length===0)throw new Error('Explicit tuning populations required');
  const populations=new Map(),used=new Set(),annotations=new Map(sensoryManifest.body_transducers.map(c=>[c.index,c]));
  for(const p of profile.tuningPopulations){
    const keys=['id','side','cell_type','field','fieldStatus','orientationStatus','orientationRadians','evidence'];
    if(!p||Object.keys(p).length!==keys.length||keys.some(k=>!Object.hasOwn(p,k))||typeof p.id!=='string'||!p.id||populations.has(p.id)||
      !SIDES.includes(p.side)||typeof p.cell_type!=='string'||!p.cell_type)throw new Error('Invalid tuning population identity');
    if(!Array.isArray(p.evidence)||p.evidence.some(e=>!e||Object.keys(e).length!==2||typeof e.source!=='string'||!e.source||typeof e.note!=='string'||!e.note))throw new Error('Explicit tuning evidence records required');
    if(p.fieldStatus==='unknown'){if(p.field!==null)throw new Error('Unknown receptor field must remain null');}
    else if(p.fieldStatus!=='prepared-annotation'||typeof p.field!=='string'||!p.field||!p.evidence.length)throw new Error('Receptor field needs an exact prepared annotation and evidence');
    if(p.orientationStatus==='unknown'){if(p.orientationRadians!==null)throw new Error('Unknown orientation must remain null');}
    else if(p.orientationStatus!=='declared-prior'||!Number.isFinite(p.orientationRadians)||p.orientationRadians<0||p.orientationRadians>=TAU||!p.evidence.length)
      throw new Error('Orientation must be unknown or an explicitly evidenced declared prior');
    populations.set(p.id,freeze(structuredClone(p)));
  }
  const groups=new Map();
  const cells=identities.map((identity,k)=>{
    const id=profile.cells[k].populationId,p=populations.get(id),a=annotations.get(identity.index);
    if(!p||p.side!==identity.side||p.cell_type!==identity.cell_type)throw new Error('Tuning population must match annotated side and cell type');
    const field=a.receptor_field??null;
    if((p.fieldStatus==='prepared-annotation'&&field!==p.field)||(p.fieldStatus==='unknown'&&field!==null))throw new Error('Receptor field annotation mismatch');
    // No arbitrary split of indistinguishable cells into preferred directions.
    const key=JSON.stringify([identity.side,identity.cell_type,field]);
    if(groups.has(key)&&groups.get(key)!==id)throw new Error('Unannotated within-population orientation split');
    groups.set(key,id);used.add(id);
    const {orientationRadians,...cell}=identity;
    return freeze({...cell,populationId:id,tuning:p});
  });
  if(used.size!==populations.size)throw new Error('Unused tuning population');
  return Object.freeze(cells);
}

function createPopulationMapper({enabled,profile,sensoryManifest,preparedIds,geometries,maxCurrentPa,mechanicalModel}){
  if(enabled!==true)throw new Error('Haltere mechanical current requires explicit enabled:true');
  const config=validateCoupledHaltereModel(mechanicalModel),cells=validatePopulationProfile(profile,{sensoryManifest,preparedIds});
  const motion=createCoupledVirtualHalteres(geometries,config),geometry=new Map(geometries.map(g=>{const v=validateGeometry(g);return [v.side,v];}));
  if(!(finite(maxCurrentPa,'maximum haltere current')>=0&&maxCurrentPa<=3.4028234663852886e38))throw new Error('Haltere maximum current is outside finite Float32 range');
  const indices=Object.freeze(cells.map(c=>c.index)),neuronCount=profile.neuronCount;
  const scales=Object.fromEntries(SIDES.map(side=>{const g=geometry.get(side),r=Math.hypot(...g.neutralComFromPivotRoot),s=g.mass*r*r*HALTERE_CURRENT_PRIOR.referenceAmplitudeRadians*(TAU*HALTERE_CURRENT_PRIOR.referenceFrequencyHz)**2;
    if(!Number.isFinite(s)||s<=0)throw new Error('Invalid haltere moment scale');return [side,s];}));
  const contract=JSON.stringify({profile,config,geometries:SIDES.map(s=>geometry.get(s)),maxCurrentPa});
  const validateTarget=t=>{if(!(t instanceof Float32Array)||t.length!==neuronCount)throw new Error('Expected a full Float32 neural-current vector');};
  function compute(sample){
    const before=motion.snapshot();
    try{
      const measured=motion.sample(sample),sides={},components={};
      for(const side of SIDES){
        const m=measured.sides[side],p=projectHaltereMoment(geometry.get(side),m,{orientationRadians:[0,Math.PI/2],momentScale:scales[side]});
        components[side]=p.beamBendingComponents;
        sides[side]={actualMusclePower:m.actualMusclePower,power:m.actualMusclePower,powerStatus:'present',
          ipsilateralWingPower:m.ipsilateralWingPower,nativeWingMotion:m.nativeWingMotion,
          amplitudeRadians:m.amplitudeRadians,oscillationPhaseRadians:m.oscillationPhaseRadians,ownDrivePhaseRadians:m.ownDrivePhaseRadians,
          theta:m.theta,thetaDot:m.thetaDot,thetaDDot:m.thetaDDot,
          baselineMomentRoot:m.baselineMomentRoot,coriolisMomentRoot:m.coriolisMomentRoot,totalMomentRoot:m.totalMomentRoot,
          beamBendingComponents:p.beamBendingComponents,momentScale:scales[side],activeCells:0,maxCurrentPa:0};
      }
      const values=new Float32Array(cells.length);
      for(let k=0;k<cells.length;k++){
        const cell=cells[k],p=cell.tuning,b=components[cell.side];
        // Unknown orientation means an isotropic population expectation, not
        // an invented preferred direction: E[max(0,B cos alpha)] = |B|/pi.
        const load=p.orientationStatus==='unknown'?Math.hypot(...b)/Math.PI:
          Math.max(0,Math.cos(p.orientationRadians)*b[0]+Math.sin(p.orientationRadians)*b[1]);
        const c=load/scales[cell.side],value=maxCurrentPa*c/(1+c);
        if(!Number.isFinite(c)||!Number.isFinite(value))throw new Error('Nonfinite haltere transduction');
        values[k]=value;if(!Number.isFinite(values[k]))throw new Error('Haltere current exceeds Float32 range');
        if(values[k]>0)sides[cell.side].activeCells++;
        sides[cell.side].maxCurrentPa=Math.max(sides[cell.side].maxCurrentPa,values[k]);
      }
      return {values,diagnostics:freeze({kind:'haltere-coupled-population-current-prior',schemaVersion:2,units:'pA',frame:'root-local',
        sampleTimeSeconds:measured.sampleTimeSeconds,bodyTimeSeconds:sample.bodyTimeSeconds,elapsedSeconds:sample.elapsedSeconds??0,
        wingPhaseRadians:sample.wingPhaseRadians,wingFrequencyHz:sample.wingFrequencyHz,omegaRootRadS:Array.from(sample.omegaRootRadS),
        maxCurrentPa,sensitivityStatus:HALTERE_CURRENT_PRIOR.status,coefficientStatus:config.coefficientStatus,
        orientationStatus:'population-tuning-explicit',unknownOrientationRule:'uniform-orientation-mean-positive-bending',
        mechanicalModel:config.kind,wingMotionContinuation:'causal-local-harmonic',sides})};
    }catch(error){motion.restore(before);throw error;}
  }
  function currents(sample){const result=compute(sample);Object.defineProperty(result.values,'diagnostics',{value:result.diagnostics});return result.values;}
  function writeInto(target,sample){validateTarget(target);const result=compute(sample);for(let k=0;k<indices.length;k++)target[indices[k]]=result.values[k];return result.diagnostics;}
  function apply(base,sample){validateTarget(base);const result=base.slice(),diagnostics=writeInto(result,sample);Object.defineProperty(result,'haltereDiagnostics',{value:diagnostics});return result;}
  function snapshot(){return structuredClone({schemaVersion:2,kind:'banc-haltere-coupled-mapper-state',contract,motion:motion.snapshot()});}
  function restore(value){if(value?.schemaVersion!==2||value.kind!=='banc-haltere-coupled-mapper-state'||value.contract!==contract)throw new Error('Haltere mapper snapshot contract mismatch');motion.restore(value.motion);}
  return Object.freeze({kind:'banc-haltere-coupled-population-current-mapper',indices,cells,neuronCount,maxCurrentPa,
    mechanicalModel:freeze(config),currents,writeInto,apply,snapshot,restore,reset:()=>motion.reset()});
}
