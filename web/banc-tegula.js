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
  annotation:'wing_tegula_campaniform_sensillum_neuron',tuning_status:'Unsigned aerodynamic hinge-moment proxy; equal within-side rates. Gain and saturation are unmeasured priors; no axis, polarity or preferred phase is assigned.'})));
 validateTegulaSensoryManifest(result);return result;
}

export function createTegulaInputMapper(config){
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
