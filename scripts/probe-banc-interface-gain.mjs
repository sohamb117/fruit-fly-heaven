// Diagnostic copies only. No production files, muscles, or native body are changed.
import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
const directory='reports/banc-interface-audit',read=async f=>JSON.parse(await fs.readFile(f,'utf8'));
const after={sensory:await read('data/prepared/banc888/console/sensory-inputs.json'),groups:await read('data/prepared/banc888/console/groups.json'),supplement:await read('models/banc-taste-peg-annotations.json')};
const payload={after,posture:await read('reports/banc-sensory-recruitment/actual-posture-fixture.json'),expected:await read('reports/banc-sensory-recruitment/actual-posture-neural-assay.json'),tolerance:{filteredRateAbsoluteHz:1e-4,cumulativeSpikes:0}};
const sourcePaths=['scripts/probe-banc-interface-gain.mjs','scripts/probe-banc-sensory-recruitment.mjs','web/sensory-encoder.js','web/banc-taste.js','web/banc-ground-sense.js','web/banc-sensory-current.js','packages/banc-runtime/src/webgpu.js','packages/banc-runtime/src/model.js','packages/banc-runtime/src/neural.wgsl','packages/banc-runtime/dist/core.wasm','data/prepared/banc888/manifest.json','data/prepared/banc888/params.bin','data/prepared/banc888/edges.bin','data/prepared/banc888/io.json','data/prepared/banc888/console/groups.json','data/prepared/banc888/console/sensory-inputs.json','models/banc-taste-peg-annotations.json','reports/banc-sensory-recruitment/actual-posture-fixture.json','reports/banc-sensory-recruitment/actual-posture-neural-assay.json'];
const hashes=async()=>Object.fromEntries(await Promise.all(sourcePaths.map(async p=>[p,createHash('sha256').update(await fs.readFile(p)).digest('hex')]))),sourceSha256=await hashes();
const{chromium}=await import(process.env.PLAYWRIGHT_MODULE||'/tmp/fruit-fly-browser-tests/node_modules/playwright/index.mjs');
const browser=await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,args:['--enable-unsafe-webgpu']});
const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.stack));page.on('console',m=>{if(m.type()==='log')console.log(m.text());});
let result,timer;
try{
 await page.goto('http://127.0.0.1:7842/banc/verify.html');
 result=await Promise.race([page.evaluate(async({after,posture,expected,tolerance})=>{
  const{loadBancModel,WebGPUBrain,createWasmCore}=await import('/banc-engine/src/index.js');
  const{SensoryEncoder}=await import('/sensory-encoder.js');
  const{createBancTasteMapper}=await import('/banc-taste.js');
  const{createBancSensoryCurrentMapper}=await import('/banc-sensory-current.js');
  const base=await loadBancModel(),core=await createWasmCore(),internal={hunger:.65,akh:.65,insulin:0},started=performance.now();
  const environment={odor:()=>0,surface:()=>({y:0,contact:false})};
  const make=()=>new SensoryEncoder(after.sensory,after.groups,environment,{tasteMapper:createBancTasteMapper([...base.io.sensory,...after.supplement.annotations],after.groups.sweet)});
  const indices=make().indices,mapper=await createBancSensoryCurrentMapper(core,base,{indices}),sweet=new Set(after.groups.sweet);
  const motorIndices=base.io.motor_neurons.map(m=>m.index),readIds=Uint32Array.from(new Set([...motorIndices,...indices])),lookup=new Map(Array.from(readIds,(i,k)=>[i,k*8]));
  const muscles=base.io.muscles.filter(m=>m.joint.startsWith('wing_power')||m.joint==='proboscis'||m.kind==='pump'||m.target==='iii1_muscle');
  const conditions=[{name:'body_taste150',body:true,taste:150,legs:true,baseline:'six_tarsi_body'},
   {name:'taste150',body:false,taste:150,legs:true,baseline:'six_tarsi_only'},
   {name:'labellar150',body:false,taste:150,mouth:true,baseline:'both_labellum_only'},
   {name:'zero',body:false,taste:0},{name:'body_only',body:true,taste:0,baseline:'actual_body_only'},
   {name:'body_taste25',body:true,taste:25,legs:true},{name:'body_taste75',body:true,taste:75,legs:true}];
  const results=[],baselineChecks=[],gainCopies=[];
  const sha=async a=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',a)),x=>x.toString(16).padStart(2,'0')).join('');
  const electricalStart=base.manifest.chemical_edges*4,electrical=base.edges.subarray(electricalStart);
  let failure=null;
  try{for(const gain of [1,.5,2]){
   const edges=base.edges.slice(),weights=new Float32Array(edges.buffer,edges.byteOffset,edges.length);
   for(let e=0;e<base.manifest.chemical_edges;e++)weights[e*4+1]*=gain;
   const tail=edges.subarray(electricalStart);
   if(!tail.every((v,i)=>v===electrical[i]))throw new Error('Electrical edge mutation');
   gainCopies.push({gain,chemicalWeightPerContactNsMs:.1*gain,edgesSha256:await sha(edges),electricalTailSha256:await sha(tail),electricalUnchanged:true});
   const model={...base,edges};let anchor;
   try{for(const condition of conditions){
    const brain=await WebGPUBrain.create(model,anchor?{shared:anchor}:{});anchor??=brain;
    const encoder=make(),feedback=structuredClone(posture.feedback);feedback.legFoodContact.fill(condition.legs?1:0);feedback.mouthFoodContact.fill(condition.mouth?1:0);feedback.wingFoodContact.fill(0);
    const pose={x:0,y:0,z:0,heading:0,bodyTime:0,contact:false,feedback},input=new Float32Array(model.manifest.neuron_count),samples=[];
    let inputs,drivenIndices,tasteIndices;
    try{for(let ms=0;ms<300;ms+=20){
     pose.bodyTime=ms/1000;const on=ms<200,encoded=encoder.update(pose,null,{odor:false,vision:false,taste:on&&condition.taste>0,bodySense:on&&condition.body});
     const cells=[];
     for(let k=0;k<encoded.indices.length;k++){
      const index=encoded.indices[k],requestedHz=sweet.has(index)&&encoded.ratesHz[k]>0?condition.taste:encoded.ratesHz[k];
      input[index]=mapper.current(index,requestedHz);if(ms===0&&input[index]!==0)cells.push({index,requestedHz,currentPa:input[index],taste:sweet.has(index)});
     }
     if(ms===0){inputs=cells;drivenIndices=cells.map(x=>x.index);tasteIndices=cells.filter(x=>x.taste).map(x=>x.index);}
     await brain.step(20/model.manifest.dt_ms,input,internal,true);const state=await brain.readState(readIds);
     if(!state.every(Number.isFinite))throw new Error('Nonfinite neural state');
     const values=(ids,k)=>ids.map(i=>state[lookup.get(i)+k]);
     const selected=muscles.map(m=>({joint:m.joint,target:m.target,indices:m.indices,meanRateHz:m.indices.reduce((s,i)=>s+state[lookup.get(i)+4],0)/m.indices.length,spikes:values(m.indices,3),ratesHz:values(m.indices,4),voltageMv:values(m.indices,0),conductanceNs:values(m.indices,6),currentPa:values(m.indices,7)}));
     const sample={ms:brain.timeMs,stimulusOn:on,totalSpikes:state.totalSpikes,activeEver:state.activeEver,drivenSpikes:values(drivenIndices,3),drivenRatesHz:values(drivenIndices,4),motorSpikes:values(motorIndices,3),motorRatesHz:values(motorIndices,4),muscles:selected,
      drivenTaste:{cells:tasteIndices.length,cumulativeSpikes:values(tasteIndices,3).reduce((s,x)=>s+x,0),meanFilteredRateHz:values(tasteIndices,4).reduce((s,x)=>s+x,0)/Math.max(1,tasteIndices.length)}};
     samples.push(sample);
     if(gain===1&&condition.baseline&&brain.timeMs===200){
      const old=expected.results.find(x=>x.name===condition.baseline).samples.find(x=>x.ms===200);let maxRateError=0,maxSpikeError=0;
      for(const m of old.muscles){const found=selected.find(x=>x.joint===m.joint&&x.target===m.target);if(!found)throw new Error('Missing baseline motor');
       m.ratesHz.forEach((v,k)=>maxRateError=Math.max(maxRateError,Math.abs(v-found.ratesHz[k])));m.spikes.forEach((v,k)=>maxSpikeError=Math.max(maxSpikeError,Math.abs(v-found.spikes[k])));}
      const check={condition:condition.name,reference:condition.baseline,maxRateErrorHz:maxRateError,maxCumulativeSpikeError:maxSpikeError,totalSpikeError:sample.totalSpikes-old.totalSpikes,passed:maxRateError<=tolerance.filteredRateAbsoluteHz&&maxSpikeError<=tolerance.cumulativeSpikes};
      baselineChecks.push(check);console.log(JSON.stringify({baselineCheck:check}));if(!check.passed)throw new Error('Baseline reproduction failed; sensitivity interpretation forbidden');
     }
    }}finally{if(brain!==anchor)brain.dispose();}
    const row={gain,...condition,inputs,drivenIndices,tasteIndices,feedback,samples};results.push(row);
    console.log(JSON.stringify({completed:results.length,total:21,gain,condition:condition.name,elapsedSeconds:(performance.now()-started)/1000,taste:samples.find(x=>x.ms===200).drivenTaste,motors:samples.find(x=>x.ms===200).muscles.filter(m=>m.target==='proboscis_m9_muscle'||m.target==='proboscis_m4b_muscle'||m.target==='dorsal_longitudinal_muscle').map(m=>({joint:m.joint,target:m.target,hz:m.meanRateHz}))}));
   }}finally{anchor?.dispose();}
  }}catch(error){failure=error.stack;}
  return {scope:'Neural-only sensitivity assay with in-memory chemical-weight copies. Original graph topology/parameters/electrical weights and production files unchanged. No body/muscles/controller runs.',
   protocol:{pulseMs:200,offMs:100,sampleMs:20,dtMs:base.manifest.dt_ms,internal,gains:[1,.5,2],conditions,postureSource:'reports/banc-sensory-recruitment/actual-posture-fixture.json',off:'All added inputs off, intrinsic/recurrent dynamics retained.',baselineTolerance:tolerance},
   manifest:base.manifest,motorIndices,readIds:Array.from(readIds),gainCopies,baselineChecks,results,elapsedSeconds:(performance.now()-started)/1000,failure,passed:!failure&&results.length===21&&baselineChecks.length===4&&baselineChecks.every(x=>x.passed)};
 },payload),new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error('210 second assay deadline exceeded')),210000);})]);
 assert.deepEqual(errors,[]);assert.deepEqual(await hashes(),sourceSha256,'Executed source changed during assay');
 result={...result,sourceSha256,sourceHashesUnchangedAtEnd:true,errors,date:new Date().toISOString(),browser:browser.version()};if(!result.passed)process.exitCode=1;
}catch(error){result={...result,passed:false,error:error.stack,sourceSha256,errors,date:new Date().toISOString()};process.exitCode=1;}
finally{clearTimeout(timer);await browser.close();await fs.writeFile(directory+'/gain-assay.json',JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify({passed:result.passed,output:directory+'/gain-assay.json',error:result.error||result.failure,browserClosed:true}));}
