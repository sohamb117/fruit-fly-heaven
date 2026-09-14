// Real browser/worker controls with a synthetic environment and HTTP fixtures.
// No model is downloaded and no result is sent to a real training service.
import assert from 'node:assert/strict';
import {readFile,mkdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {wasmConfigFixture} from './fixtures/training-native-config.mjs';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'/tmp/fruit-fly-browser-tests/node_modules/playwright/index.mjs');
const web=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),origin='https://wasm.example';
const config=wasmConfigFixture(JSON.parse(await readFile(path.join(web,'training/config.json'))));
config.stage='maintained_flight';config.stages=[{...config.stages.at(-1),id:config.stage,durationSeconds:5}];config.durationSeconds=5;
const bytes=JSON.stringify(config),configHash=createHash('sha256').update(bytes).digest('hex');
const execution={backend:'wasm',neuralEngine:'wasm',wasmExecution:config.optimizer.acceptance.nativeExecution};
const checkpoint={schemaVersion:1,algorithm:config.algorithm,modelFingerprint:config.modelFingerprint,configHash,generation:1,stage:config.stage,
  parameterNames:config.parameters.map(p=>p.name),parameters:config.parameters.map(p=>p.initial)};
const job={jobId:'browser-wasm-fixture',leaseToken:'fixture-token-123456',generation:1,pairId:'g1-p0',sign:1,seed:888,
  modelFingerprint:config.modelFingerprint,configHash,parameters:checkpoint.parameters,parametersHash:'a'.repeat(64),stage:config.stage,durationSeconds:5};
const pose={position:[0,0,.15],quaternion:[1,0,0,0],stage:config.stage,time:0,simSeconds:0,
  bowl:{radiusCm:2,floor:{baseCm:0,radialCoefficientPerCm:0,capRadiusCm:1}}};
const brainSample={schemaVersion:1,dataset:'BANC v888',modelFingerprint:config.modelFingerprint,preparedIdsSha256:'a'.repeat(64),neuronCount:175401,sampleCount:8,
  indices:[0,1,2,3,4,5,6,7],positions:[-100,-20,0,-50,30,0,0,0,0,50,30,0,100,-20,0,0,80,0,0,150,0,0,230,0],
  ids:Array.from({length:8},(_,i)=>String(1000+i)),labels:Array.from({length:8},(_,i)=>'Cell '+i)};
const fixtureObserver=`import {createTrainingWorkerController} from './worker.js';
self.__brainObservation={enabled:false,indices:[]};
const controller=createTrainingWorkerController({postMessage:value=>self.postMessage(value),createEnvironment:async()=> (await import('./environment.js')).createTrainingEnvironment()});
self.onmessage=event=>{if(event.data.type==='observe-brain')self.__brainObservation=event.data;else controller.handle(event.data);};`;
const fixtureEnvironment=`
const config=${JSON.stringify(config)},identity=${JSON.stringify({configHash,modelFingerprint:config.modelFingerprint})},execution=${JSON.stringify(execution)},pose=${JSON.stringify(pose)};
export async function createTrainingEnvironment(){return {
  async ready(){return {...identity,...execution,frame:pose};},dispose(){},
  async evaluate(job,{checkpoint,onFrame,onProgress,getBudget}){
    let lastBrain=-Infinity;
    for(let i=0;i<100;i++){
      await checkpoint();await new Promise(r=>setTimeout(r,20));
      onFrame({...pose,time:i*.002,simSeconds:i*.002,metrics:{return:-1}});
      onProgress({message:'fixture duty '+getBudget().dutyCycle});
      const observation=self.__brainObservation;
      if(observation?.enabled&&performance.now()-lastBrain>=500){lastBrain=performance.now();self.postMessage({type:'brain',snapshot:{jobId:job.jobId,neuralTimeMs:i*2,
        indices:observation.indices,voltage:observation.indices.map((_,k)=>-60+k),rates:observation.indices.map(()=>5),lastSpikeMs:observation.indices.map(()=>-1)}});}
    }
    const provenance={...identity,...execution,environmentVersion:config.environmentVersion,bodyBackend:'mujoco-wasm',dtMs:config.dtMs,bodyBlockMs:config.bodyBlockMs,
      stage:job.stage,seed:job.seed,generation:job.generation,pairId:job.pairId,sign:job.sign,parametersHash:job.parametersHash};
    return {...provenance,provenance,parameters:job.parameters,return:-1,success:false,terminated:true,cancelled:false,
      simSeconds:.2,steps:100,reason:'excessive_rotation',metrics:{wallSeconds:2}};
  }
};}
`;
const browser=await chromium.launch({executablePath:process.env.CHROME_PATH||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true});
const context=await browser.newContext(),requests=[],results=[],errors=[];let leased=false,offline=false;
await context.addInitScript(()=>{
  globalThis.__tabHidden=false;Object.defineProperty(document,'hidden',{configurable:true,get:()=>__tabHidden});
  globalThis.__workersStarted=0;const NativeWorker=globalThis.Worker;
  globalThis.Worker=class extends NativeWorker{constructor(...args){super(...args);globalThis.__workersStarted++;}};
});
await context.route('**/*',async route=>{
  const url=new URL(route.request().url());requests.push(url.pathname);
  if(url.pathname==='/training/observer-worker.js'){await route.fulfill({contentType:'text/javascript',body:fixtureObserver});return;}
  if(url.pathname==='/training/brain-sample.json'){await route.fulfill({contentType:'application/json',body:JSON.stringify(brainSample)});return;}
  if(url.pathname==='/training/config.json'){await route.fulfill({contentType:'application/json',body:bytes});return;}
  if(url.pathname==='/training/environment.js'){await route.fulfill({contentType:'text/javascript',body:fixtureEnvironment});return;}
  if(url.pathname.startsWith('/api/training/')){
    if(offline){await route.fulfill({status:503,contentType:'application/json',body:JSON.stringify({error:'Coordinator unavailable'})});return;}
    let body;
    if(url.pathname.endsWith('/status'))body={configHash,modelFingerprint:config.modelFingerprint,generation:1,stage:config.stage,checkpoint,acceptedResults:results.length};
    else if(url.pathname.endsWith('/lease')){body={job:leased?null:job,waitMs:200};leased=true;}
    else if(url.pathname.endsWith('/result')){results.push(route.request().postDataJSON());body={accepted:true};}
    else if(url.pathname.endsWith('/release'))body={released:true};
    else throw new Error('Unexpected browser fixture API '+url.pathname);
    await route.fulfill({contentType:'application/json',body:JSON.stringify(body)});return;
  }
  const contentType={'.html':'text/html','.js':'text/javascript','.css':'text/css','.json':'application/json'}[path.extname(url.pathname)]||'application/octet-stream';
  try{await route.fulfill({contentType,body:await readFile(path.join(web,url.pathname))});}catch{await route.fulfill({status:404,body:'Fixture asset missing'});}
});
const page=await context.newPage();page.on('pageerror',error=>errors.push(error.message));
const output=path.resolve(process.env.TRAINING_WASM_UI_REPORT_DIR||'reports/training-wasm-ui');await mkdir(output,{recursive:true});
async function assertCopy(){
  const copy=await page.evaluate(()=>[document.body.innerText,...[...document.querySelectorAll('[title],[aria-label]')].map(n=>`${n.title||''} ${n.getAttribute('aria-label')||''}`)].join('\n'));
  assert.doesNotMatch(copy,/\b(shared|coordinator|BANC|FlyBody|WebGPU|WASM|MuJoCo|unverified|fingerprint|lease)\b/i);
  assert.doesNotMatch(copy,/connected trainer|recorded preview|https?:\/\//i);
}
try{
  await page.goto(origin+'/train.html');await page.waitForFunction(()=>globalThis.heavenTraining?.state.coordinator.connected);
  assert.equal(await page.textContent('#run-status'),'Ready');assert.equal(await page.textContent('#preview-heading'),'Live preview');
  assert.equal(await page.isDisabled('#start-training'),false);assert.equal(await page.isDisabled('#computer-budget'),false);
  assert.equal(await page.isHidden('#trainer-note'),true);assert.equal(await page.evaluate(()=>__workersStarted),0);
  assert.equal(requests.some(p=>p==='/training/worker.js'||p==='/api/training/lease'),false);await assertCopy();
  assert.equal(requests.includes('/training/brain-sample.json'),false,'Brain positions are lazy');
  assert.equal(await page.locator('#parameter-rows tr').count(),32);assert.equal(await page.locator('#parameter-rows tr td:last-child').first().textContent(),'—');
  assert.equal(await page.locator('.parameters-panel th').allTextContents().then(x=>x.join('|')),'Name|Value');
  await page.click('#show-brain');await page.waitForFunction(()=>!!heavenTraining.brainPreview.renderer);
  assert.equal(await page.evaluate(()=>__workersStarted),0,'Inspecting positions never starts training');
  assert.equal(await page.evaluate(()=>!!heavenTraining.preview.renderer),false);assert.equal(requests.filter(p=>p==='/training/brain-sample.json').length,1);
  await page.evaluate(()=>{globalThis.__frames=0;heavenTraining.client.addEventListener('frame',()=>__frames++);});
  await page.click('#start-training');await page.waitForFunction(()=>heavenTraining.state.phase==='training'&&__frames>2);
  assert.equal(await page.textContent('#run-status'),'Running');assert.equal(await page.evaluate(()=>__workersStarted),1);
  assert.equal(await page.textContent('#stage-title'),'Maintain flight');
  await page.waitForFunction(()=>!!heavenTraining.brainPreview.snapshot);
  assert.equal(await page.evaluate(()=>heavenTraining.brainPreview.snapshot.jobId),job.jobId);
  assert.equal(await page.locator('#parameter-rows tr td:last-child').first().textContent(),job.parameters[0].toFixed(5));
  assert.equal(await page.evaluate(()=>Object.hasOwn(heavenTraining.state.activeJob,'leaseToken')),false);
  await page.evaluate(()=>{
    globalThis.__initialStageNode=document.querySelector('#stage-list li');globalThis.__progressEvents=0;
    heavenTraining.client.addEventListener('state',event=>{if(event.detail.message?.startsWith('fixture duty'))__progressEvents++;});
  });
  await page.waitForFunction(()=>__progressEvents>=3);
  assert.equal(await page.evaluate(()=>__initialStageNode===document.querySelector('#stage-list li')),true,'Actual worker progress retains unchanged task nodes');
  await page.click('#pause-training');assert.equal(await page.textContent('#run-status'),'Paused');assert.equal(await page.textContent('#pause-training'),'Resume');
  await page.waitForTimeout(80);const frames=await page.evaluate(()=>__frames);await page.waitForTimeout(150);
  assert.equal(await page.evaluate(()=>__frames),frames,'Pausing blocks the real worker controller');
  assert.equal(await page.textContent('#preview-status'),'Paused · last sample');
  const neuronPoint=await page.evaluate(async()=>{
    const {Vector3}=await import('/vendor/three.module.js'),brain=heavenTraining.brainPreview,point=new Vector3().fromArray(brain.positions,6);
    brain.updateCamera();point.project(brain.camera);const box=document.getElementById('training-brain').getBoundingClientRect(),scale=Math.min(box.width/480,box.height/270);
    return {x:box.left+(box.width-480*scale)/2+(point.x+1)*240*scale,y:box.top+(box.height-270*scale)/2+(1-point.y)*135*scale};
  });
  await page.mouse.click(neuronPoint.x,neuronPoint.y);assert.match(await page.textContent('#brain-reading'),/Cell 2.*-58.0 mV.*5.0 Hz/);
  await page.evaluate(()=>{__tabHidden=true;document.dispatchEvent(new Event('visibilitychange'));});
  assert.equal(await page.evaluate(()=>heavenTraining.client.brainObservation.enabled),false);assert.equal(await page.evaluate(()=>heavenTraining.client.paused),true);
  await page.evaluate(()=>{__tabHidden=false;document.dispatchEvent(new Event('visibilitychange'));});
  assert.equal(await page.evaluate(()=>heavenTraining.client.brainObservation.enabled),true);
  await page.evaluate(()=>heavenTraining.client.worker.dispatchEvent(new MessageEvent('message',{data:{type:'brain-error',message:'Fixture readout unavailable'}})));
  assert.equal(await page.evaluate(()=>heavenTraining.client.state.phase),'paused');assert.match(await page.textContent('#preview-empty'),/Brain view unavailable/);
  await page.click('#show-brain');assert.equal(await page.evaluate(()=>heavenTraining.client.brainObservation.enabled),true);
  for(const viewport of [{width:1440,height:1000},{width:390,height:844}]){
    await page.setViewportSize(viewport);await assertCopy();assert(await page.evaluate(()=>document.documentElement.scrollWidth)<=viewport.width);
    await page.screenshot({path:path.join(output,`brain-control-fixture-${viewport.width}.png`),fullPage:true});
  }
  const inspection=await page.evaluate(()=>{
    const h=heavenTraining,node=document.querySelector('#parameter-rows tr'),vector=h.state.activeJob.parameters.slice();
    h.client.emit({parameters:h.state.parameters.map(()=>0)});
    const result={retained:node===document.querySelector('#parameter-rows tr'),current:h.state.activeJob.parameters,brainTime:h.brainPreview.snapshot.neuralTimeMs};
    h.client.dispatchEvent(new CustomEvent('frame',{detail:{...h.frame,time:999}}));result.clock=document.getElementById('preview-time').textContent;
    h.client.emit({parameters:vector});return result;
  });
  assert.equal(inspection.retained,true,'Checkpoint changes do not rebuild current parameter rows');assert.deepEqual(inspection.current,job.parameters);assert.notEqual(inspection.clock,'999 s');
  await page.click('#parameter-next');assert.equal(await page.textContent('#parameter-page'),'2 / 21');
  await page.locator('#parameter-search').fill('yaw');assert(await page.locator('#parameter-rows tr').count()>0);
  assert((await page.locator('#parameter-rows').textContent()).includes('yaw'));await page.locator('#parameter-search').fill('');
  await page.click('#show-fly');assert.equal(await page.evaluate(()=>heavenTraining.client.brainObservation.enabled),false);
  assert.equal(await page.evaluate(()=>heavenTraining.brainPreview.active),false);await page.waitForFunction(()=>!!heavenTraining.preview.renderer);
  await page.locator('#computer-budget').fill('25');assert.equal(await page.evaluate(()=>heavenTraining.client.options.dutyCycle),.25);
  await page.click('#pause-training');await page.waitForFunction(()=>heavenTraining.state.message==='fixture duty 0.25');
  await page.waitForFunction(()=>heavenTraining.state.contributedEpisodes===1);
  assert.equal(results.length,1);assert.equal(results[0].metrics.terminated,true);assert.equal(results[0].metrics.cancelled,false);
  assert.deepEqual(results[0].provenance.wasmExecution,execution.wasmExecution);assert.equal(results[0].provenance.backend,'wasm');
  assert.equal(await page.textContent('#uploaded-count'),'1');assert.equal(await page.textContent('#evaluations-count'),'1');
  await page.waitForFunction(()=>heavenTraining.state.activity==='waiting');
  assert.equal(await page.textContent('#run-status'),'Waiting for work');assert.equal(await page.isDisabled('#pause-training'),false);
  assert.equal(await page.isDisabled('#stop-training'),false);
  const memoization=await page.evaluate(()=>{
    const client=heavenTraining.client,$=id=>document.getElementById(id),first=id=>$(id).firstElementChild;
    const saved=Object.fromEntries(['history','lastReturn','completedEpisodes','contributedEpisodes','wallSeconds','stage','curriculum','message','episodeSimSeconds'].map(key=>[key,client.state[key]]));
    const ids=['stage-list','reward-dots','reward-grid','candidate-rows'],nodes=ids.map(first);
    const observer=new MutationObserver(()=>{});for(const id of ids)observer.observe($(id),{childList:true,subtree:true,attributes:true,characterData:true});
    for(let i=0;i<3;i++)client.emit({message:'Fixture progress',episodeSimSeconds:.01*(i+1),wallSeconds:3,contributedEpisodes:2});
    const unchanged={nodes:ids.every((id,i)=>first(id)===nodes[i]),mutations:observer.takeRecords().length,
      wall:$('work-time').textContent,uploaded:$('uploaded-count').textContent};observer.disconnect();
    client.emit({history:[...saved.history,{episode:2,stage:'maintained_flight',return:.5,success:true}],completedEpisodes:2,lastReturn:.5});
    const added={rows:$('candidate-rows').children.length,dots:$('reward-dots').children.length,
      score:$('latest-reward').textContent,result:first('candidate-rows').lastElementChild.textContent,
      rebuilt:first('candidate-rows')!==nodes[3]&&first('reward-dots')!==nodes[1]};
    const row=first('candidate-rows');client.emit({history:client.state.history.map(entry=>entry.episode===2?{...entry,return:.75,success:false}:entry),lastReturn:.75});
    const corrected={rebuilt:row!==first('candidate-rows'),score:first('candidate-rows').children[2].textContent,result:first('candidate-rows').lastElementChild.textContent};
    client.emit({stage:'flight',curriculum:[{id:'maintained_flight',status:'complete'},{id:'flight',status:'training'}]});
    const advanced={rebuilt:first('stage-list')!==nodes[0],title:$('stage-title').textContent,progress:$('stage-progress-value').textContent,
      current:$('stage-list').querySelector('.current span').textContent};
    client.emit({history:[],lastReturn:null,completedEpisodes:0});
    const empty={rows:$('candidate-rows').children.length,chartHidden:$('reward-chart').hasAttribute('hidden'),emptyHidden:$('history-empty').hidden};
    client.emit(saved);
    return {unchanged,added,corrected,advanced,empty};
  });
  assert.deepEqual(memoization,{
    unchanged:{nodes:true,mutations:0,wall:'3.0 s',uploaded:'2'},
    added:{rows:2,dots:2,score:'0.500',result:'Success',rebuilt:true},
    corrected:{rebuilt:true,score:'0.750',result:'Not yet'},
    advanced:{rebuilt:true,title:'Stay airborne',progress:'1 / 2',current:'Stay airborne'},
    empty:{rows:0,chartHidden:true,emptyHidden:false},
  });
  await page.click('#pause-training');assert.equal(await page.textContent('#run-status'),'Paused');
  await page.click('#pause-training');assert.equal(await page.textContent('#run-status'),'Waiting for work');
  for(const viewport of [{width:1440,height:1000},{width:390,height:844}]){
    await page.setViewportSize(viewport);await assertCopy();assert(await page.evaluate(()=>document.documentElement.scrollWidth)<=viewport.width);
    await page.screenshot({path:path.join(output,`browser-control-fixture-${viewport.width}.png`),fullPage:true});
  }
  await page.click('#stop-training');assert.equal(await page.textContent('#run-status'),'Stopped');
  await page.click('#show-brain');assert.equal(await page.textContent('#preview-status'),'Last sample');
  assert.deepEqual(await page.evaluate(()=>heavenTraining.state.activeJob.parameters),job.parameters);
  assert.equal(await page.evaluate(()=>!!heavenTraining.client.worker),false);
  offline=true;await page.click('#start-training');await page.waitForFunction(()=>heavenTraining.state.phase==='error');
  assert.equal(await page.evaluate(()=>__workersStarted),1,'Offline retry must not launch another worker');await assertCopy();
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({passed:true,scope:'Synthetic environment with real worker controls: lazy brain geometry/activity/picking, hidden observation pause, isolated brain error/retry, current-only parameter paging, mandatory upload, unchanged DOM retention; no simulation evidence',memoization,viewports:[1440,390]}));
}finally{await browser.close();}
