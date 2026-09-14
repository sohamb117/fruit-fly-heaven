// Explicit opt-in intrinsic model. Mathematical source: Hürkey et al., Nature
// (2023), doi:10.1038/s41586-023-06099-0 (paper CC BY 4.0). No author-code import.
export const DLM_PROFILE='dlm-snl-2023-v1';
export const DLM_MAX_TIME_MS=2**18; // 3-ULP timestamp neighbourhoods cease to be disjoint above this clock.
export const DLM_CELLS=Object.freeze([
 [12322,'720575941432491145'],[31143,'720575941463144336'],[41465,'720575941479034563'],
 [74497,'720575941519783640'],[84317,'720575941533190337'],[127563,'720575941583644382'],
 [135445,'720575941596050112'],[157417,'720575941640465439'],[160138,'720575941645403000'],
 [173140,'720575941692400603'],
].map(([index,root_id])=>Object.freeze({index,root_id})));
function keys(x,expected,label){
 if(!x||typeof x!=='object'||Array.isArray(x)||Reflect.ownKeys(x).length!==expected.length||expected.some(k=>!Object.hasOwn(x,k)))throw new Error('Invalid '+label+' schema');
}
export function intrinsicLayout(model){
 const n=model.manifest.neuron_count,config=model.manifest.intrinsic_models;
 const base={count:0,cells:Object.freeze([]),packedLength:n*17+27,kineticsLength:n*19,eventContract:undefined};
 if(config===undefined)return Object.freeze(base);
 keys(config,['schema','profile','cells','ionic_step_ms','initial_gates','event_policy'],'intrinsic model');
 keys(config.initial_gates,['h','b'],'initial gates');
 if(config.schema!==1||config.profile!==DLM_PROFILE||config.ionic_step_ms!==.1||config.initial_gates.h!==.146||config.initial_gates.b!==.146||config.event_policy!=='threshold-10ms-guard')throw new Error('Unsupported DLM intrinsic profile');
 if(model.manifest.dt_ms!==.5||n>=2**24)throw new Error('DLM model requires 0.5 ms graph ticks and exact slot indices');
 if(!Array.isArray(config.cells)||!config.cells.length||config.cells.length>10)throw new Error('Invalid DLM cell list');
 const fixture=model.manifest.fixture===true;
 if(fixture&&n>64)throw new Error('DLM fixture exemption is limited to at most 64 neurons');
 const cells=config.cells.map((c,k)=>{
  keys(c,['index','root_id'],'DLM cell');
  if(!Number.isInteger(c.index)||c.index<0||c.index>=n||typeof c.root_id!=='string'||(k&&c.index<=config.cells[k-1].index))throw new Error('DLM cells must be sorted, unique and in bounds');
  if(model.params[c.index*16+8]>.5)throw new Error('DLM cells must be declared spiking');
  if(model.offsets[n+1+c.index]!==model.offsets[n+2+c.index])throw new Error('DLM profile does not yet support incoming gap junctions');
  if(fixture&&!/^fixture[0-9]+$/.test(c.root_id))throw new Error('DLM fixture cells require explicit synthetic root IDs');
  return Object.freeze({index:c.index,root_id:c.root_id});
 });
 if(!fixture&&(cells.length!==10||cells.some((c,k)=>c.index!==DLM_CELLS[k].index||c.root_id!==DLM_CELLS[k].root_id)))throw new Error('Only the ten registered BANC v888 DLM cells may use this profile');
 if(!fixture){
  const io=model.io;
  if(!Array.isArray(io?.motor_neurons)||!Array.isArray(io?.muscles))throw new Error('DLM profile requires annotated model IO');
  for(const c of cells){
   const matches=io.motor_neurons.filter(row=>row.index===c.index);
   const expectedType=[135445,173140].includes(c.index)?'DLM5':'DLM1-4';
   if(matches.length!==1||matches[0].root_id!==c.root_id||matches[0].cell_type!==expectedType||matches[0].super_class!=='motor'||matches[0].region!=='ventral_nerve_cord'||matches[0].peripheral_target_type!=='dorsal_longitudinal_muscle')throw new Error('DLM motor-neuron identity mismatch');
  }
  const muscles=io.muscles.filter(row=>row.target==='dorsal_longitudinal_muscle');
  if(muscles.length!==2||new Set(muscles.map(row=>row.joint)).size!==2)throw new Error('DLM bilateral muscle mappings required');
  const joined=[];
  for(const row of muscles){
   if(row.kind!=='asynchronous_wing'||!['wing_power_left','wing_power_right'].includes(row.joint)||row.sign!==1||!Array.isArray(row.indices)||row.indices.length!==5||!Array.isArray(row.root_ids)||row.root_ids.length!==5)throw new Error('Invalid DLM muscle mapping');
   row.indices.forEach((index,k)=>{
    const c=cells.find(c=>c.index===index),motor=io.motor_neurons.find(m=>m.index===index);
    if(!c||c.root_id!==row.root_ids[k]||row.joint!=='wing_power_'+motor.side)throw new Error('DLM muscle/root-ID join mismatch');
    joined.push(index);
   });
  }
  if(joined.length!==10||new Set(joined).size!==10)throw new Error('DLM mappings must cover each selected cell once');
 }
 const result=Object.freeze({...base,count:cells.length,cells:Object.freeze(cells),packedLength:n*18+27,kineticsLength:n*19+cells.length*4,
  eventContract:Object.freeze({schema:1,profile:DLM_PROFILE,indices:Object.freeze(cells.map(c=>c.index))})});
 return result;
}

// The event reader and effector consume this immutable, model-derived declaration.
// The detector guard is an observation convention, not a voltage reset/refractory.
export function validateMotorEventContract(contract){
 if(contract===undefined)return Object.freeze([]);
 keys(contract,['schema','profile','indices'],'motor event contract');
 if(contract.schema!==1||contract.profile!==DLM_PROFILE||!Array.isArray(contract.indices)||!contract.indices.length||contract.indices.length>10||contract.indices.some((i,k)=>!Number.isInteger(i)||i<0||i>=2**24||(k&&i<=contract.indices[k-1])))throw new Error('Invalid DLM motor event contract');
 return Object.freeze([...contract.indices]);
}
// Decode bounded float32 arithmetic (including WGSL division) to a unique
// decimal substep. Accept at most three float32 ULP and require disjoint error
// neighbourhoods: 6 ULP < 0.1 ms. This rejects precision exhaustion and halfway
// ambiguity. Effector packets/snapshots use exact canonical JS time, not raw f32.
export function decodeDlmEventTime(raw,{stored=true}={}){
 if(!Number.isFinite(raw)||raw<0)throw new RangeError('Invalid DLM event time');
 const tick=Math.round(raw*10),canonical=tick/10;
 const rounded=Math.fround(canonical);
 const ulp=2**(Math.floor(Math.log2(Math.max(canonical,.1)))-23);
 const neighbourUlp=2**(Math.floor(Math.log2(Math.max(canonical+.1,.1)))-23);
 if(!Number.isSafeInteger(tick)||tick>=2**24||6*neighbourUlp>=.1||
   (stored?(raw!==Math.fround(raw)||Math.abs(raw-rounded)>3*ulp):raw!==canonical))throw new RangeError('DLM event time must uniquely represent a 0.1 ms substep');
 return canonical;
}
export function validateIntrinsicState(layout,values){
 if(!(values instanceof Float32Array)||values.length!==layout.count*4)throw new Error('Invalid intrinsic readback layout');
 for(let k=0;k<layout.count;k++){
  const h=values[k*4],b=values[k*4+1],guard=values[k*4+2],failed=values[k*4+3];
  if(!Number.isFinite(h)||!Number.isFinite(b)||h<0||h>1||b<0||b>1||!Number.isInteger(guard)||guard<0||guard>100||failed!==0)throw new Error('DLM ionic integration failed at cell '+layout.cells[k].index);
 }
}
