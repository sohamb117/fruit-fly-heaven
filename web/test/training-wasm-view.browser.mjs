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
const fixtureEnvironment=`
const config=${JSON.stringify(config)},identity=${JSON.stringify({configHash,modelFingerprint:config.modelFingerprint})},execution=${JSON.stringify(execution)},pose=${JSON.stringify(pose)};
export async function createTrainingEnvironment(){return {
  async ready(){return {...identity,...execution,frame:pose};},dispose(){},
  async evaluate(job,{checkpoint,onFrame,onProgress,getBudget}){
    for(let i=0;i<100;i++){
      await checkpoint();await new Promise(r=>setTimeout(r,20));
      onFrame({...pose,time:i*.002,simSeconds:i*.002,metrics:{return:-1}});
      onProgress({message:'fixture duty '+getBudget().dutyCycle});
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
  globalThis.__workersStarted=0;const NativeWorker=globalThis.Worker;
  globalThis.Worker=class extends NativeWorker{constructor(...args){super(...args);globalThis.__workersStarted++;}};
});
await context.route('**/*',async route=>{
  const url=new URL(route.request().url());requests.push(url.pathname);
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
async function assertCopy(){
  const copy=await page.evaluate(()=>[document.body.innerText,...[...document.querySelectorAll('[title],[aria-label]')].map(n=>`${n.title||''} ${n.getAttribute('aria-label')||''}`)].join('\n'));
  assert.doesNotMatch(copy,/\b(shared|coordinator|BANC|FlyBody|WebGPU|WASM|MuJoCo|parameters|unverified|fingerprint|lease)\b/i);
  assert.doesNotMatch(copy,/connected trainer|recorded preview|https?:\/\//i);
}
try{
  await page.goto(origin+'/train.html');await page.waitForFunction(()=>globalThis.heavenTraining?.state.coordinator.connected);
  assert.equal(await page.textContent('#run-status'),'Ready');assert.equal(await page.textContent('#preview-heading'),'Live preview');
  assert.equal(await page.isDisabled('#start-training'),false);assert.equal(await page.isDisabled('#computer-budget'),false);
  assert.equal(await page.isHidden('#trainer-note'),true);assert.equal(await page.evaluate(()=>__workersStarted),0);
  assert.equal(requests.some(p=>p==='/training/worker.js'||p==='/api/training/lease'),false);await assertCopy();
  await page.evaluate(()=>{globalThis.__frames=0;heavenTraining.client.addEventListener('frame',()=>__frames++);});
  await page.click('#start-training');await page.waitForFunction(()=>heavenTraining.state.phase==='training'&&__frames>2);
  assert.equal(await page.textContent('#run-status'),'Running');assert.equal(await page.evaluate(()=>__workersStarted),1);
  assert.equal(await page.textContent('#stage-title'),'Maintain flight');
  await page.click('#pause-training');assert.equal(await page.textContent('#run-status'),'Paused');assert.equal(await page.textContent('#pause-training'),'Resume');
  await page.waitForTimeout(80);const frames=await page.evaluate(()=>__frames);await page.waitForTimeout(150);
  assert.equal(await page.evaluate(()=>__frames),frames,'Pausing blocks the real worker controller');
  await page.locator('#computer-budget').fill('25');assert.equal(await page.evaluate(()=>heavenTraining.client.options.dutyCycle),.25);
  await page.click('#pause-training');await page.waitForFunction(()=>heavenTraining.state.message==='fixture duty 0.25');
  await page.waitForFunction(()=>heavenTraining.state.contributedEpisodes===1);
  assert.equal(results.length,1);assert.equal(results[0].metrics.terminated,true);assert.equal(results[0].metrics.cancelled,false);
  assert.deepEqual(results[0].provenance.wasmExecution,execution.wasmExecution);assert.equal(results[0].provenance.backend,'wasm');
  assert.equal(await page.textContent('#uploaded-count'),'1');assert.equal(await page.textContent('#evaluations-count'),'1');
  const output=path.resolve(process.env.TRAINING_WASM_UI_REPORT_DIR||'reports/training-wasm-ui');await mkdir(output,{recursive:true});
  for(const viewport of [{width:1440,height:1000},{width:390,height:844}]){
    await page.setViewportSize(viewport);await assertCopy();assert(await page.evaluate(()=>document.documentElement.scrollWidth)<=viewport.width);
    await page.screenshot({path:path.join(output,`browser-control-fixture-${viewport.width}.png`),fullPage:true});
  }
  await page.click('#stop-training');assert.equal(await page.textContent('#run-status'),'Stopped');
  assert.equal(await page.evaluate(()=>!!heavenTraining.client.worker),false);
  offline=true;await page.click('#start-training');await page.waitForFunction(()=>heavenTraining.state.phase==='error');
  assert.equal(await page.evaluate(()=>__workersStarted),1,'Offline retry must not launch another worker');await assertCopy();
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({passed:true,scope:'Browser UI and real worker control with synthetic environment: idle, Start, Pause/Resume, intensity, local preview, mandatory upload, Stop, fail-closed retry; no simulation evidence',viewports:[1440,390]}));
}finally{await browser.close();}
