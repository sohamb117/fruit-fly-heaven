import {intrinsicLayout} from './cell-models.js';
export const STATE_STRIDE=8, PARAM_STRIDE=16;
export const SPIKE_CAPACITY=16384;
export function decodeSpikeHistory(words){
  const floats=new Float32Array(words.buffer,words.byteOffset,words.length),count=words[1],start=(words[0]-count)>>>0,spikes=[];
  for(let k=0;k<count;k++){const slot=((start+k)>>>0)%SPIKE_CAPACITY;spikes.push([floats[2+slot*2],words[3+slot*2]]);}
  return spikes.sort((a,b)=>a[0]-b[0]||a[1]-b[1]);
}
const layouts=new WeakMap();
// Exact storage reduction: retain only the delay history that can be addressed.
// The manifest's 32 slots remain the supported maximum, not a per-fly allocation.
export function historySlots(model){
  if(!layouts.has(model)){
    let maximum=1;
    for(let e=0;e<model.manifest.chemical_edges;e++)maximum=Math.max(maximum,model.edges[e*4+3]);
    layouts.set(model,maximum+1);
  }
  return layouts.get(model);
}
export function validateModel(model) {
  const {manifest:m,offsets,edges,params}=model,n=m.neuron_count;
  if(m.dataset!=='BANC'||m.materialization!==888||m.schema!==1)throw new Error('Expected a BANC v888 model');
  if(!Number.isInteger(n)||n<1||!Number.isFinite(m.dt_ms)||m.dt_ms<=0||m.dt_ms>1)throw new Error('Invalid neuron count or time step');
  if(m.delay_slots!==32||m.receptors?.length!==9)throw new Error('Invalid kinetics layout');
  if(!(offsets instanceof Uint32Array)||offsets.length!==2*(n+1)||!(edges instanceof Uint32Array)||edges.length%4||params.length!==n*16)throw new Error('Invalid model buffer sizes');
  const chemical=m.chemical_edges,total=edges.length/4;
  if(offsets[0]!==0||offsets[n]!==chemical||offsets[n+1]!==chemical||offsets.at(-1)!==total||offsets.some((v,i)=>v>total||(i&&v<offsets[i-1])))throw new Error('Invalid incoming CSR offsets');
  const ef=new Float32Array(edges.buffer,edges.byteOffset,edges.length);
  for(let e=0;e<total;e++)if(edges[e*4]>=n||!Number.isFinite(ef[e*4+1])||ef[e*4+1]<0||edges[e*4+2]>=9||(e<chemical&&(edges[e*4+3]<1||edges[e*4+3]>=32)))throw new Error(`Invalid edge ${e}`);
  if(!params.every(Number.isFinite))throw new Error('Nonfinite physiology');
  for(let i=0;i<n;i++)if([0,1,6,10].some(k=>params[i*16+k]<=0))throw new Error('Nonpositive physiology constant');
  for(const r of m.receptors)if(!(r.rise_ms>0&&r.decay_ms>0&&Number.isFinite(r.reversal_mv)))throw new Error('Invalid receptor kinetics');
  intrinsicLayout(model);
  return model;
}
export function initialBuffers(model) {
  const {manifest:m,params}=model,n=m.neuron_count;
  const layout=intrinsicLayout(model),packed=new Float32Array(layout.packedLength);packed.set(params);
  for(let k=0;k<layout.count;k++)packed[n*17+27+layout.cells[k].index]=k+1;
  m.receptors.forEach((r,i)=>packed.set([r.rise_ms,r.decay_ms,r.reversal_mv],n*17+i*3));
  const state=new Float32Array(n*8);
  for(let i=0;i<n;i++)state[i*8]=params[i*16+2];
  const kinetics=new Float32Array(layout.kineticsLength);kinetics.fill(-1e30,n*18,n*19);
  for(let k=0;k<layout.count;k++)kinetics.set([.146,.146,0,0],n*19+k*4);
  return {packed,state,history:new Float32Array(n*historySlots(model)),kinetics};
}
export function validateStep(steps,input,n,internal) {
  if(!Number.isInteger(steps)||steps<1||steps>128)throw new Error('steps must be 1..128');
  if(!(input instanceof Float32Array)||input.length!==n||!input.every(Number.isFinite))throw new Error('Invalid input currents');
  for(const key of ['hunger','insulin','akh'])if(!Number.isFinite(internal[key])||internal[key]<0||internal[key]>1)throw new Error(`Invalid ${key}`);
}
export async function loadBancModel(base='/banc-data/') {
  const get=async name=>{const r=await fetch(new URL(name,new URL(base,globalThis.location?.href)));if(!r.ok)throw new Error(`BANC data unavailable (${name}); run scripts/prepare-banc.py --download`);return r;};
  const manifest=await (await get('manifest.json')).json();
  if(manifest.dataset!=='BANC'||manifest.materialization!==888)throw new Error('BANC v888 data required');
  const files=await Promise.all(['offsets.bin','edges.bin','params.bin','io.json'].map(async name=>{
    const buffer=await (await get(name)).arrayBuffer(),entry=manifest.files[name];
    if(buffer.byteLength!==entry.bytes)throw new Error(`Truncated ${name}`);
    const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',buffer)),b=>b.toString(16).padStart(2,'0')).join('');
    if(hash!==entry.sha256)throw new Error(`Checksum mismatch: ${name}`);
    return buffer;
  }));
  const [offsets,edges,params,io]=files;
  return validateModel({manifest,offsets:new Uint32Array(offsets),edges:new Uint32Array(edges),params:new Float32Array(params),io:JSON.parse(new TextDecoder().decode(io))});
}
