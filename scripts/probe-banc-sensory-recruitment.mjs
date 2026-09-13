// Input-only matched full-connectome assay. Run without competing browser or
// benchmark jobs. No muscle/body model, desired action, or neural suppression.
import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
const opts=Object.fromEntries(process.argv.slice(2).map(v=>v.replace(/^--/,'').split('=')));
const directory=opts.output||'reports/banc-sensory-recruitment',durationMs=Number(opts.ms||200);
assert(durationMs>=100&&durationMs<=400&&durationMs%20===0);
const read=async path=>JSON.parse(await fs.readFile(path,'utf8'));
const payload={durationMs,before:{sensory:await read(directory+'/before-sensory-inputs.json'),groups:await read(directory+'/before-groups.json'),
 supplement:await read(directory+'/before-taste-supplement.json')},after:{sensory:await read('data/prepared/banc888/console/sensory-inputs.json'),
 groups:await read('data/prepared/banc888/console/groups.json'),supplement:await read('models/banc-taste-peg-annotations.json')}};
if(opts.protocol==='posture'){
 payload.posture=await read(directory+'/actual-posture-fixture.json');
 payload.conditions=[];
 const add=(name,legs,body,extra={})=>payload.conditions.push({name,version:'after',legs,body,taste:true,...extra});
 for(let left=0;left<6;left++)for(let right=left+1;right<6;right++)add(`tarsal_pair_${left}_${right}`,[left,right],false);
 add('six_tarsi_only',[0,1,2,3,4,5],false);
 add('six_tarsi_before',[0,1,2,3,4,5],false,{version:'before'});
 add('six_tarsi_body',[0,1,2,3,4,5],true);
 add('actual_body_only',[],true,{taste:false});
 for(const kind of ['position','load','antenna']){
  add('six_tarsi_without_'+kind,[0,1,2,3,4,5],true,{excludeBodyKind:kind});
  add('six_tarsi_with_only_'+kind,[0,1,2,3,4,5],true,{onlyBodyKind:kind});
 }
 add('left_labellum_only',[],false,{mouth:[0]});add('right_labellum_only',[],false,{mouth:[1]});
 add('both_labellum_only',[],false,{mouth:[0,1]});
 add('six_tarsi_both_labellum_body',[0,1,2,3,4,5],true,{mouth:[0,1]});
}
const resultFile=opts.protocol==='posture'?'actual-posture-neural-assay.json':'full-neural-assay.json';
const sourcePaths=['web/sensory-encoder.js','web/banc-world-worker.js','web/banc-taste.js','web/banc-ground-sense.js','web/banc-sensory-current.js',
 'packages/banc-runtime/src/webgpu.js','packages/banc-runtime/src/neural.wgsl','data/prepared/banc888/manifest.json',
 'data/prepared/banc888/console/groups.json','data/prepared/banc888/console/sensory-inputs.json','models/banc-taste-peg-annotations.json'];
const hashes=async()=>Object.fromEntries(await Promise.all(sourcePaths.map(async path=>[path,createHash('sha256').update(await fs.readFile(path)).digest('hex')])));
const sources=await hashes(),errors=[];
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'/tmp/fruit-fly-browser-tests/node_modules/playwright/index.mjs');
const browser=await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,args:['--enable-unsafe-webgpu']});
const page=await browser.newPage();page.on('console',m=>{if(m.type()==='log')console.log(m.text());});page.on('pageerror',e=>errors.push(e.stack));
let result,timer;
try{
 await page.goto((opts.url||'http://127.0.0.1:7842')+'/banc/verify.html');
 result=await Promise.race([page.evaluate(async({before,after,durationMs,posture,conditions:requestedConditions})=>{
  const {loadBancModel,WebGPUBrain,createWasmCore}=await import('/banc-engine/src/index.js');
  const {SensoryEncoder}=await import('/sensory-encoder.js');
  const {createBancTasteMapper}=await import('/banc-taste.js');
  const {createBancSensoryCurrentMapper}=await import('/banc-sensory-current.js');
  const model=await loadBancModel(),core=await createWasmCore(),internal={hunger:.65,akh:.65,insulin:0};
  const environment={odor:()=>0,surface:()=>({y:0,contact:false})};
  const make=version=>{const p=version==='before'?before:after;return new SensoryEncoder(p.sensory,p.groups,environment,
   {tasteMapper:createBancTasteMapper([...model.io.sensory,...p.supplement.annotations],p.groups.sweet)});};
  const allIndices=Uint32Array.from(new Set([...make('before').indices,...make('after').indices]));
  const mapper=await createBancSensoryCurrentMapper(core,model,{indices:allIndices});
  const motors=model.io.motor_neurons,readIds=Uint32Array.from(new Set([...motors.map(row=>row.index),...allIndices]));
  const lookup=new Map(Array.from(readIds,(index,k)=>[index,k]));
  const muscles=model.io.muscles.filter(m=>m.joint.startsWith('wing_power')||m.joint==='proboscis'||m.kind==='pump');
  const conditions=requestedConditions||[{name:'no_external',version:'after',body:false,taste:false},
   ...['body','front_sugar','labellar_sugar','body_front_sugar'].flatMap(kind=>['before','after'].map(version=>
    ({name:kind+'_'+version,version,body:kind.startsWith('body'),taste:kind!=='body',organ:kind.includes('labellar')?'mouth':'front'}))),
   {name:'all_external_sugar_after',version:'after',body:false,taste:true,organ:'all'}];
  let anchor;const results=[],started=performance.now();
  try{for(const condition of conditions){
   const brain=await WebGPUBrain.create(model,anchor?{shared:anchor}:{});anchor??=brain;
   const encoder=make(condition.version),input=new Float32Array(model.manifest.neuron_count),samples=[];
   // Deliberately fixed native-format fixture, NOT an observed trajectory or
   // biomechanical antennal model. Same exact physical values in each pair.
   const feedback=posture?structuredClone(posture.feedback):{legs:Array.from({length:6},()=>({loadBodyWeights:1/6,collision:1,tibiaAngle:.3,coxaAngle:.2,tibiaVelocity:0,vibration:0})),
    antennae:[{angle:.4,speed:0},{angle:-.3,speed:0}],speed:0,tilt:.7,angularVelocity:[0,0,0],
    wingPowerLeft:0,wingPowerRight:0,halterePower:[0,0],legFoodContact:Array(6).fill(0),mouthFoodContact:[0,0],wingFoodContact:[0,0]};
   if(posture){feedback.legFoodContact.fill(0);feedback.mouthFoodContact.fill(0);feedback.wingFoodContact.fill(0);
    for(const slot of condition.legs??[])feedback.legFoodContact[slot]=1;
    for(const slot of condition.mouth??[])feedback.mouthFoodContact[slot]=1;}
   if(condition.organ==='front')feedback.legFoodContact[3]=1;
   if(condition.organ==='mouth')feedback.mouthFoodContact[0]=1;
   if(condition.organ==='all'){feedback.legFoodContact.fill(1);feedback.mouthFoodContact.fill(1);feedback.wingFoodContact.fill(1);}
   const pose={x:0,y:0,z:0,heading:0,bodyTime:0,contact:false,feedback};
   let encoded;
   const manifest=(condition.version==='before'?before:after).sensory;
   const bodyKinds=new Map(manifest.body_transducers.map(row=>[row.index,row.kind]));
   for(const channel of manifest.channels)if(channel.key.startsWith('antenna_'))for(const index of channel.indices)bodyKinds.set(index,'antenna');
   try{for(let ms=0;ms<durationMs;ms+=20){
    pose.bodyTime=ms/1000;encoded=encoder.update(pose,null,{odor:false,vision:false,taste:condition.taste,bodySense:condition.body});
    for(let k=0;k<encoded.indices.length;k++){
     const index=encoded.indices[k],kind=bodyKinds.get(index);
     const excluded=kind&&(condition.excludeBodyKind===kind||(condition.onlyBodyKind&&condition.onlyBodyKind!==kind));
     input[index]=mapper.current(index,excluded?0:encoded.ratesHz[k]);
    }
    await brain.step(20/model.manifest.dt_ms,input,internal,true);
    const state=await brain.readState(readIds);
    if(!state.every(Number.isFinite))throw new Error('Nonfinite neural state');
    const selected=muscles.map(m=>{const rows=m.indices.map(i=>lookup.get(i)*8);return {joint:m.joint,target:m.target,indices:m.indices,
     meanRateHz:rows.reduce((sum,k)=>sum+state[k+4],0)/rows.length,spikes:rows.map(k=>state[k+3]),ratesHz:rows.map(k=>state[k+4])};});
    const sensory=Object.fromEntries(['sweet',...before.sensory.channels.map(c=>c.key)].map(key=>{
     const indices=key==='sweet'?(condition.version==='before'?before:after).groups.sweet:before.sensory.channels.find(c=>c.key===key).indices;
     const rows=indices.map(index=>lookup.get(index)*8);return [key,{cells:rows.length,meanRateHz:rows.reduce((sum,k)=>sum+state[k+4],0)/rows.length,
      spikes:rows.reduce((sum,k)=>sum+state[k+3],0)}];}));
    samples.push({ms:brain.timeMs,totalSpikes:state.totalSpikes,activeEver:state.activeEver,muscles:selected,sensory});
   }}finally{if(brain!==anchor)brain.dispose();}
   const inputs=Array.from(encoded.indices,(index,k)=>({index,requestedHz:encoded.ratesHz[k],currentPa:input[index]})).filter(row=>row.currentPa!==0);
   const item={...condition,inputCount:inputs.length,inputs,feedback,samples};results.push(item);
   console.log(JSON.stringify({condition:condition.name,elapsedSeconds:(performance.now()-started)/1000,
    inputCount:inputs.length,motors:samples.at(-1).muscles.filter(m=>m.joint.startsWith('wing_power')||m.target.includes('_m9_')||m.target.includes('_m4')).map(({joint,target,meanRateHz})=>({joint,target,hz:meanRateHz}))}));
  }}finally{anchor?.dispose();}
  return {scope:'Fresh full BANC input-only runs; synthetic fixed posture/contact, no body, visual input, odor input, controller, neuron suppression or synaptic/physiological retuning.',
   protocol:{durationMs,dtMs:model.manifest.dt_ms,internal,posture,initialization:'Identical deterministic initial buffers; model has no random seed input.',samplesEveryMs:20,
    caveat:'Intrinsic graded release and recurrent activity remain active even when added sensory current is zero. Filtered MN Hz is reported with actual cumulative spike counts.'},
   manifest:model.manifest,adapter:anchor?.adapterInfo,elapsedSeconds:(performance.now()-started)/1000,results};
 },payload),new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error('240 second full-neural assay deadline exceeded')),240000);})]);
 assert.deepEqual(errors,[]);assert.deepEqual(await hashes(),sources,'Assay source changed during run');
 result={...result,passed:true,sourceSha256:sources,errors,date:new Date().toISOString(),browser:browser.version()};
}catch(error){result={...result,passed:false,error:error.stack,sourceSha256:sources,errors,date:new Date().toISOString()};process.exitCode=1;}
finally{clearTimeout(timer);await browser.close();await fs.writeFile(directory+'/'+resultFile,JSON.stringify(result,null,2)+'\n');
 console.log(JSON.stringify({passed:result.passed,output:directory+'/'+resultFile,error:result.error,browserClosed:true}));}
