// Isolated configured-cell assay using the production WasmBrain public API.
// This validates the Hz -> current interface, not BANC biological physiology.
import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {resolve} from 'node:path';
import {createWasmCore,WasmBrain} from '../packages/banc-runtime/src/wasm.js';
import {createBancSensoryCurrentMapper} from '../web/banc-sensory-current.js';

const root=new URL('../',import.meta.url);
export const requestedRates=[0,2,5,7,10,17,20,33,50,73,100,107,150,163,191,200,235];
export const legacyCurrent=(p,hz)=>hz<=0?0:p[8]>.5?hz*.8:p[1]*(p[3]-p[2])*(1-Math.exp(-hz/10))+hz*p[0]*(p[3]-p[4])/1000;

// Separate from the mapper's calibration harness: exercise model validation,
// packed parameters, stepping and readState through the public runtime API.
export function measurePublicRuntime(core,parameters,currents,dtMs,{settleMs=2000,measureMs=8000,internal={hunger:0,insulin:0,akh:0}}={}){
 const n=parameters.length,model={
  manifest:{dataset:'BANC',materialization:888,schema:1,neuron_count:n,dt_ms:dtMs,delay_slots:32,chemical_edges:0,receptors:Array.from({length:9},()=>({rise_ms:1,decay_ms:1,reversal_mv:0}))},
  params:Float32Array.from(parameters.flatMap(p=>Array.from(p))),offsets:new Uint32Array(2*(n+1)),edges:new Uint32Array(0)
 };
 const brain=new WasmBrain(core,model),input=Float32Array.from(currents);
 const advance=ms=>{let steps=Math.ceil(ms/dtMs);while(steps){const count=Math.min(128,steps);brain.step(count,input,internal,false);steps-=count;}};
 try{
  advance(settleMs);const before=brain.readState();advance(measureMs);const after=brain.readState(),duration=Math.ceil(measureMs/dtMs)*dtMs;
  return parameters.map((p,i)=>p[8]>.5?after[i*8+4]:(after[i*8+3]-before[i*8+3])*1000/duration);
 }finally{brain.dispose();}
}

export async function runSensoryCurrentAssay(){
 const configBytes=await fs.readFile(new URL('configs/banc-physiology.json',root)),config=JSON.parse(configBytes);
 const names=Object.keys(config.profiles),model={params:Float32Array.from(names.flatMap(name=>config.profiles[name])),manifest:{dt_ms:config.dt_ms,neuron_count:names.length}};
 const core=await createWasmCore(),start=performance.now(),mapper=await createBancSensoryCurrentMapper(core,model),calibrationMs=performance.now()-start;
 const currents=[],parameters=[];
 for(let i=0;i<names.length;i++)for(const hz of requestedRates){const p=config.profiles[names[i]];parameters.push(p,p);currents.push(legacyCurrent(p,hz),mapper.current(i,hz));}
 const actual=measurePublicRuntime(core,parameters,currents,config.dt_ms),results=[];let offset=0;
 for(let i=0;i<names.length;i++){
  const profile=mapper.profiles[i],limits=mapper.limits(i),samples=requestedRates.map(requestedHz=>{
   const legacyHz=actual[offset++],measuredHz=actual[offset++],targetHz=requestedHz<=0?profile.baselineHz??0:Math.max(limits.minimumHz,Math.min(limits.maximumHz,requestedHz));
   return {requestedHz,targetHz,addedCurrentPa:mapper.current(i,requestedHz),legacyHz,measuredHz};
  });
  const metric=key=>samples.reduce((sum,s)=>sum+Math.abs(s[key]-s.targetHz),0)/samples.length;
  results.push({name:names[i],...profile,limits,samples,legacyMeanAbsoluteErrorHz:metric('legacyHz'),meanAbsoluteErrorHz:metric('measuredHz'),maximumErrorHz:Math.max(...samples.map(s=>Math.abs(s.measuredHz-s.targetHz)))});
 }
 const preparedBytes=await fs.readFile(new URL('data/prepared/banc888/params.bin',root)),prepared=new Float32Array(preparedBytes.buffer,preparedBytes.byteOffset,preparedBytes.length/4),io=JSON.parse(await fs.readFile(new URL('data/prepared/banc888/io.json',root)));
 const hashes=Object.fromEntries(await Promise.all(['packages/banc-runtime/dist/core.wasm','configs/banc-physiology.json','web/banc-sensory-current.js'].map(async path=>[path,createHash('sha256').update(await fs.readFile(new URL(path,root))).digest('hex')])));
 const knownProfiles=new Map(names.map(name=>[JSON.stringify(Array.from(Float32Array.from(config.profiles[name]))),name])),preparedProfileCounts={};
 for(let i=0;i<prepared.length;i+=16){const name=knownProfiles.get(JSON.stringify(Array.from(prepared.subarray(i,i+16))))??'unrecognized';preparedProfileCounts[name]=(preparedProfileCounts[name]||0)+1;}
 return {createdAt:new Date().toISOString(),scope:'Isolated cells using the actual configured production WASM neuron kernel; no network edges, motor output, body stabilization or full-fly behavioral claim.',dtMs:config.dt_ms,settleMs:2000,measureMs:8000,calibrationMs,hashes,preparedProfileCounts,preparedSensoryCount:io.sensory.length,requestedRates,results};
}

if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 const report=await runSensoryCurrentAssay(),path=new URL('reports/banc-sensory-current-calibration.json',root);
 await fs.writeFile(path,JSON.stringify(report,null,2)+'\n');
 console.table(report.results.map(p=>({profile:p.name,legacyMAE:p.legacyMeanAbsoluteErrorHz,newMAE:p.meanAbsoluteErrorHz,maxError:p.maximumErrorHz})));
 console.log(`Calibration: ${report.calibrationMs.toFixed(1)} ms; report: ${fileURLToPath(path)}`);
}
