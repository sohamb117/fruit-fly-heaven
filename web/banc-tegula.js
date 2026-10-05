// BANC v888 identities: reports/flight-tegula-inputs/anatomy.json.
// Organ/type annotation identifies tegula campaniform afferents, not their
// receptive fields. Aerodynamic hinge-moment magnitude is an explicit proxy
// for strain; rate scale, equal within-side drive and saturation are priors.
export const TEGULA_PROFILE='tegula-fluid-load-v1';
export const TEGULA_CELLS=Object.freeze([
 [3327,'720575941402023280','SNpp28','right'],
 [6425,'720575941411902609','SNpp28','left'],
 [9154,'720575941425239892','SNpp28','left'],
 [16314,'720575941438938924','SNpp38','left'],
 [17789,'720575941440235711','SNpp28','right'],
 [26657,'720575941456225520','SNpp28','left'],
 [31631,'720575941464136500','SNpp28','left'],
 [33284,'720575941468617335','SNpp28','left'],
 [39519,'720575941477379121','SNpp38','right'],
 [49643,'720575941488986828','SNpp37','left'],
 [65457,'720575941508718370','SNpp37','right'],
 [65478,'720575941508742354','SNpp37','left'],
 [71570,'720575941516591064','SNpp28','left'],
 [92757,'720575941540812669','SNpp28','left'],
 [95465,'720575941544991164','SNpp28','right'],
 [101109,'720575941551787326','SNpp28','right'],
 [103417,'720575941554420507','SNpp37','right'],
 [108586,'720575941560365806','SNpp28','right'],
 [110137,'720575941562060712','SNpp37','left'],
 [118587,'720575941571867656','SNpp28','left'],
 [130295,'720575941589569022','SNpp38','left'],
 [141398,'720575941608300874','SNpp28','right'],
 [143846,'720575941612358950','SNpp28','right'],
 [148849,'720575941623377354','SNpp37','left'],
 [157367,'720575941640342389','SNpp38','right'],
 [173342,'720575941696308186','SNpp28','right'],
].map(([index,root_id,cell_type,side])=>Object.freeze({index,root_id,cell_type,side})));
const sides=['left','right'],types=new Set(['SNpp28','SNpp37','SNpp38']);
const expected=new Map(TEGULA_CELLS.map(cell=>[cell.index,cell]));
const record=x=>x!==null&&typeof x==='object'&&!Array.isArray(x)&&!ArrayBuffer.isView(x);
const fail=message=>{throw new Error('Tegula input: '+message);};
function requireThat(condition,message){if(!condition)fail(message);}
export function validateTegulaConfig(config){
 if(config?.schema===2)return validateTegulaStrainConfig(config);
 const keys=['schema','profile','maxRateHz','halfLoadNative'];
 requireThat(record(config)&&Reflect.ownKeys(config).length===keys.length&&keys.every(key=>Object.hasOwn(config,key)), 'invalid config schema');
 requireThat(config.schema===1&&config.profile===TEGULA_PROFILE&&config.maxRateHz===100&&
  typeof config.halfLoadNative==='number'&&Number.isFinite(config.halfLoadNative)&&config.halfLoadNative>0,'unsupported profile or load/rate scale');
 return Object.freeze({...config});
}

// Validate the augmented sensory declaration again at the encoder boundary.
// This prevents a wing_strain row from silently using an old broad fallback.
export function validateTegulaSensoryManifest(sensory){
 const rows=(sensory.body_transducers||[]).filter(row=>row.kind==='wing_strain');
 const channels=(sensory.channels||[]).filter(row=>/^tegula_/.test(row.key));
 if(sensory.tegula_model===undefined){
  requireThat(rows.length===0&&channels.length===0,'wing strain requires an explicit tegula model');return null;
 }
 const config=validateTegulaConfig(sensory.tegula_model);
 requireThat(rows.length===26&&new Set(rows.map(row=>row.index)).size===26,'expected 26 unique wing-strain transducers');
 for(const row of rows){
  const cell=expected.get(row.index);
  requireThat(cell&&row.root_id===cell.root_id&&row.cell_type===cell.cell_type&&row.side===cell.side&&
   row.organ==='wing_tegula'&&row.function==='mechanical_strain','wing-strain identity mismatch');
 }
 requireThat(channels.length===2,'expected two tegula channels');
 for(const side of sides){
  const matches=channels.filter(channel=>channel.key==='tegula_'+side),ids=TEGULA_CELLS.filter(cell=>cell.side===side).map(cell=>cell.index);
  requireThat(matches.length===1&&matches[0].indices?.length===ids.length&&matches[0].indices.every((id,k)=>id===ids[k]),'tegula channel identity mismatch');
 }
 requireThat(!(sensory.body_transducer_exclusions||[]).some(row=>expected.has(row.index)),'tegula input overlaps an exclusion');
 return config;
}

/** Augment an owned clone only; never edit prepared annotations or the graph.
 * baseModel.ids must be the verified ids.bin BigUint64Array. Prepared sensory
 * IO omits root IDs, so matching only its integer indices is insufficient.
 */
export function createTegulaSensoryManifest(baseModel,sensory,config){
 const checked=validateTegulaConfig(config),m=baseModel?.manifest,ids=baseModel?.ids;
 requireThat(m?.dataset==='BANC'&&m.materialization===888&&m.neuron_count===175401&&
  sensory?.schema_version===1&&sensory.neuron_count===m.neuron_count,'expected BANC v888 sensory/model identity');
 requireThat(ids instanceof BigUint64Array&&ids.length===m.neuron_count,'verified ids.bin BigUint64Array is required');
 requireThat(Array.isArray(baseModel.io?.sensory),'prepared sensory IO is required');
 const selected=baseModel.io.sensory.filter(row=>types.has(row.cell_type));
 requireThat(selected.length===26&&new Set(selected.map(row=>row.index)).size===26,'expected the 26 annotated tegula cells');
 for(const row of selected){
  const cell=expected.get(row.index);
  requireThat(cell&&String(ids[row.index])===cell.root_id&&row.cell_type===cell.cell_type&&row.side===cell.side&&
   row.kind==='load'&&row.body_part==='wing_tegula'&&(row.root_id===undefined||row.root_id===cell.root_id),'prepared tegula index/root/type/side/organ join mismatch');
 }
 requireThat(sensory.tegula_model===undefined,'sensory manifest already has a tegula model');
 requireThat(Array.isArray(sensory.channels)&&Array.isArray(sensory.vision?.receptors),'missing sensory channels');
 const occupied=new Set([...sensory.channels.flatMap(channel=>channel.indices),...sensory.vision.receptors.map(row=>row.index),
  ...(sensory.body_transducers||[]).map(row=>row.index),...(sensory.body_transducer_exclusions||[]).map(row=>row.index)]);
 requireThat(TEGULA_CELLS.every(cell=>!occupied.has(cell.index))&&!sensory.channels.some(channel=>/^tegula_/.test(channel.key)),'tegula inputs overlap existing sensory declarations');
 const result=structuredClone(sensory);
 result.tegula_model={...checked};
 for(const side of sides)result.channels.push({key:'tegula_'+side,label:'Tegula '+side,indices:TEGULA_CELLS.filter(cell=>cell.side===side).map(cell=>cell.index)});
 result.body_transducers??=[];
 result.body_transducers.push(...TEGULA_CELLS.map(cell=>({...cell,kind:'wing_strain',organ:'wing_tegula',function:'mechanical_strain',
  annotation:'wing_tegula_campaniform_sensillum_neuron',tuning_status:checked.schema===2?
   'Signed local aerodynamic moment through a virtual elastic hinge; receptive projections and compliance are declared priors, not measured sensillum strain.':
   'Unsigned aerodynamic hinge-moment proxy; equal within-side rates. Gain and saturation are unmeasured priors; no axis, polarity or preferred phase is assigned.'})));
 validateTegulaSensoryManifest(result);return result;
}

export function createTegulaInputMapper(config){
 if(config?.schema===2)return createTegulaStrainMapper(config);
 const p=validateTegulaConfig(config);
 const rate=load=>{
  if(load<=p.halfLoadNative){const ratio=load/p.halfLoadNative;return p.maxRateHz*ratio/(1+ratio);}
  return p.maxRateHz/(1+p.halfLoadNative/load);
 };
 return Object.freeze({config:p,rates(feedback,enabled=true,bodyTimeSeconds){
  if(!enabled)return {tegula_left:0,tegula_right:0};
  const load=feedback?.wingLoad;
  requireThat(record(load)&&load.kind==='native-wing-aerodynamic-moment-v1'&&load.units==='g cm^2/s^2','missing native wing-load feedback');
  requireThat([load.left,load.right,load.forceTimeSeconds,load.bodyTimeSeconds,bodyTimeSeconds].every(x=>typeof x==='number'&&Number.isFinite(x)&&x>=0), 'invalid native wing-load value or clock');
  requireThat(Math.abs(load.bodyTimeSeconds-bodyTimeSeconds)<=1e-8&&load.forceTimeSeconds<=load.bodyTimeSeconds+1e-8,'native wing-load clock mismatch');
  // The declared observer reads the final Euler step's cached force: exactly
  // one 50 us native step old, except the fresh mj_forward sample at time zero.
  requireThat(bodyTimeSeconds===0?load.forceTimeSeconds===0:
   Math.abs(load.bodyTimeSeconds-load.forceTimeSeconds-.00005)<=1e-8,'stale or mismatched native wing-load force time');
  return {tegula_left:rate(load.left),tegula_right:rate(load.right)};
 }});
}

export const TEGULA_STRAIN_PROFILE='tegula-local-deformation-v2';
const deepFreeze=value=>{if(value&&typeof value==='object'){Object.values(value).forEach(deepFreeze);Object.freeze(value);}return value;};
const finite=value=>typeof value==='number'&&Number.isFinite(value);
const vec3=value=>Array.isArray(value)&&value.length===3&&value.every(finite);
const exactKeys=(value,keys)=>record(value)&&Object.keys(value).length===keys.length&&keys.every(key=>Object.hasOwn(value,key));

/** The shared tegula field is known; its local axes/compliance are not. This
 * usable, mirrored root-axis prior is explicitly labeled and can be replaced
 * by registered per-cell projections. A null projection abstains. No ID hash,
 * arbitrary split into opponents, or cell-type-to-axis claim is made. */
export function createTegulaStrainPrior({halfLoadNative}={}){
 requireThat(finite(halfLoadNative)&&halfLoadNative>0,'positive reference moment required');
 return validateTegulaStrainConfig({schema:2,profile:TEGULA_STRAIN_PROFILE,
  complianceRadiansPerNativeMoment:[.01/halfLoadNative,.01/halfLoadNative,.01/halfLoadNative],
  relaxationSeconds:.002,maxDeflectionRadians:.2,maxRateHz:100,baselineRateHz:0,
  halfStrain:.01,velocityWeightSeconds:0,
  fields:TEGULA_CELLS.map(cell=>({...cell,projection:cell.side==='left'?[1,0,0]:[-1,0,0],
   status:'engineering-prior',evidence:'Common tegula field; mirrored thorax-X bending projection is unmeasured.'}))});
}

export function validateTegulaStrainConfig(config){
 requireThat(exactKeys(config,['schema','profile','complianceRadiansPerNativeMoment','relaxationSeconds',
  'maxDeflectionRadians','maxRateHz','baselineRateHz','halfStrain','velocityWeightSeconds','fields'])&&
  config.schema===2&&config.profile===TEGULA_STRAIN_PROFILE,'invalid local-strain config');
 requireThat(vec3(config.complianceRadiansPerNativeMoment)&&config.complianceRadiansPerNativeMoment.every(x=>x>0&&x<=1e6),
  'invalid positive diagonal compliance');
 for(const key of ['relaxationSeconds','maxDeflectionRadians','maxRateHz','halfStrain'])
  requireThat(finite(config[key])&&config[key]>0,'invalid '+key);
 requireThat(config.relaxationSeconds>=.00005&&config.relaxationSeconds<=1&&config.maxDeflectionRadians<=1&&
  config.maxRateHz<=200&&finite(config.baselineRateHz)&&config.baselineRateHz>=0&&config.baselineRateHz<=config.maxRateHz&&
  finite(config.velocityWeightSeconds)&&config.velocityWeightSeconds>=0&&config.velocityWeightSeconds<=1,'unsupported strain dynamics');
 requireThat(Array.isArray(config.fields)&&config.fields.length===TEGULA_CELLS.length,'expected26 explicit receptive fields');
 const seen=new Set();
 for(const field of config.fields){
  const cell=expected.get(field?.index);
  requireThat(exactKeys(field,['index','root_id','cell_type','side','projection','status','evidence'])&&cell&&!seen.has(field.index)&&
   field.root_id===cell.root_id&&field.cell_type===cell.cell_type&&field.side===cell.side,'receptive-field identity mismatch');
  seen.add(field.index);
  requireThat(typeof field.evidence==='string'&&field.evidence.trim().length>0,'field provenance required');
  requireThat(field.projection===null?field.status==='unassigned':
   vec3(field.projection)&&Math.abs(Math.hypot(...field.projection)-1)<1e-8&&
    ['engineering-prior','measured'].includes(field.status),'invalid field projection or status');
 }
 return deepFreeze(structuredClone(config));
}

/** A causal observer, not a physical hinge actuator. The signed vector is
 * retained through diagonal elastic compliance and Kelvin-Voigt relaxation.
 * Projected virtual deformation stands in for local cuticle strain. It omits
 * inertial/contact/actuator reaction loads and cannot be called measured
 * sensillum physiology. State evolves under the previous held native sample. */
function createTegulaStrainMapper(config){
 const p=validateTegulaStrainConfig(config),canonical=JSON.stringify(p);
 let time=0,deformation=[[0,0,0],[0,0,0]],held=[[0,0,0],[0,0,0]],lastSource=null;
 const snapshot=()=>({schema:1,profile:TEGULA_STRAIN_PROFILE,config:structuredClone(p),time,
  deformation:deformation.map(v=>v.slice()),held:held.map(v=>v.slice()),lastSource:lastSource&&structuredClone(lastSource)});
 const reset=()=>{time=0;deformation=[[0,0,0],[0,0,0]];held=[[0,0,0],[0,0,0]];lastSource=null;return snapshot();};
 const restore=state=>{
  requireThat(record(state)&&state.schema===1&&state.profile===TEGULA_STRAIN_PROFILE&&JSON.stringify(state.config)===canonical&&
   finite(state.time)&&state.time>=0&&[state.deformation,state.held].every(v=>Array.isArray(v)&&v.length===2&&v.every(vec3))&&
   state.deformation.flat().every(x=>Math.abs(x)<=p.maxDeflectionRadians),'invalid strain snapshot');
  if(state.lastSource===null)requireThat(state.time===0&&[state.deformation,state.held].flat(2).every(x=>x===0),'uninitialized strain snapshot has state');
  else{
   requireThat(exactKeys(state.lastSource,['bodyTimeSeconds','identity'])&&state.lastSource.bodyTimeSeconds===state.time&&
    typeof state.lastSource.identity==='string','invalid snapshot source');
   let source;try{source=JSON.parse(state.lastSource.identity);}catch{requireThat(false,'invalid snapshot source encoding');}
   requireThat(exactKeys(source,['input','forceTimeSeconds'])&&JSON.stringify(source.input)===JSON.stringify(state.held)&&
    finite(source.forceTimeSeconds)&&source.forceTimeSeconds>=0&&
    (state.time===0?source.forceTimeSeconds===0:Math.abs(state.time-source.forceTimeSeconds-.00005)<1e-8),
   'snapshot held load or force time differs from its source');
  }
  time=state.time;deformation=state.deformation.map(v=>v.slice());held=state.held.map(v=>v.slice());lastSource=state.lastSource&&structuredClone(state.lastSource);
 };
 function rates(feedback,enabled=true,bodyTimeSeconds){
  requireThat(typeof enabled==='boolean','invalid bodySense flag');
  if(!enabled)return {tegula_left:0,tegula_right:0,cellRates:new Map(TEGULA_CELLS.map(c=>[c.index,0]))};
  const load=feedback?.wingLoad;
  requireThat(record(load)&&load.kind==='native-wing-aerodynamic-moment-v1'&&load.units==='g cm^2/s^2'&&
   load.localFrame==='native-thorax'&&sides.every(side=>vec3(load.momentThorax?.[side])),'signed native thorax moments required');
  requireThat([bodyTimeSeconds,load.bodyTimeSeconds,load.forceTimeSeconds,load.localFrameTimeSeconds].every(finite)&&
   bodyTimeSeconds>=time&&Math.abs(load.bodyTimeSeconds-bodyTimeSeconds)<=1e-8&&
   load.localFrameTimeSeconds===load.forceTimeSeconds&&load.forceTimeSeconds>=0&&
   (bodyTimeSeconds===0?load.forceTimeSeconds===0:Math.abs(bodyTimeSeconds-load.forceTimeSeconds-.00005)<1e-8),
  'local strain source clock mismatch');
  const input=sides.map(side=>load.momentThorax[side].slice()),identity=JSON.stringify({input,forceTimeSeconds:load.forceTimeSeconds});
  requireThat(!lastSource||bodyTimeSeconds!==time||identity===lastSource.identity,'conflicting same-time strain sample');
  const dt=bodyTimeSeconds-time;
  requireThat(dt<=.05+1e-10,'strain sampling gap exceeds50ms');
  if(dt>0){
   const decay=Math.exp(-dt/p.relaxationSeconds);
   deformation=deformation.map((v,s)=>v.map((theta,k)=>{
    const target=p.maxDeflectionRadians*Math.tanh(held[s][k]*p.complianceRadiansPerNativeMoment[k]/p.maxDeflectionRadians);
    return target+(theta-target)*decay;
   }));
  }
  time=bodyTimeSeconds;held=input;lastSource={bodyTimeSeconds,identity};
  const velocity=deformation.map((v,s)=>v.map((theta,k)=>
   (p.maxDeflectionRadians*Math.tanh(held[s][k]*p.complianceRadiansPerNativeMoment[k]/p.maxDeflectionRadians)-theta)/p.relaxationSeconds));
  const cellRates=new Map(),strains=[],totals=[0,0],counts=[0,0];
  for(const field of p.fields){
   const s=field.side==='left'?0:1,d=field.projection;
   const strain=d?d.reduce((sum,x,k)=>sum+x*(deformation[s][k]+p.velocityWeightSeconds*velocity[s][k]),0):0;
   const compression=Math.max(0,strain),fraction=compression/(p.halfStrain+compression);
   const rate=d?p.baselineRateHz+(p.maxRateHz-p.baselineRateHz)*fraction:0;
   requireThat(finite(rate),'nonfinite strain rate');cellRates.set(field.index,rate);strains.push(strain);totals[s]+=rate;counts[s]++;
  }
  return {tegula_left:totals[0]/counts[0],tegula_right:totals[1]/counts[1],cellRates,
   diagnostics:{profile:TEGULA_STRAIN_PROFILE,bodyTimeSeconds,forceTimeSeconds:load.forceTimeSeconds,
    momentThorax:structuredClone(load.momentThorax),deformationRadians:deformation.map(v=>v.slice()),
    strainProxy:strains,unassigned:p.fields.filter(f=>f.projection===null).length,
    status:'Virtual aerodynamic-load deformation; compliance and receptive fields require calibration.'}};
 }
 return Object.freeze({config:p,rates,snapshot,restore,reset});
}
