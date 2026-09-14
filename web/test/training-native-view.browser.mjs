// UI fixtures only: no native simulation, real progress or real trainer used.
import assert from 'node:assert/strict';
import {mkdir,readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {nativeConfigFixture} from './fixtures/training-native-config.mjs';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'/tmp/fruit-fly-browser-tests/node_modules/playwright/index.mjs');
const web=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const config=nativeConfigFixture(JSON.parse(await readFile(path.join(web,'training/config.json'))));
config.stage='maintained_flight';config.stages=[{...config.stages.at(-1),id:config.stage}];
const configBytes=JSON.stringify(config),configHash=createHash('sha256').update(configBytes).digest('hex'),origin='https://flies.example';
const output=path.resolve(process.env.TRAINING_NATIVE_UI_REPORT_DIR||'reports/training-native-ui');
const checkpoint=generation=>({schemaVersion:1,algorithm:config.algorithm,configHash,modelFingerprint:config.modelFingerprint,
  parameterNames:config.parameters.map(p=>p.name),parameters:config.parameters.map((p,i)=>p.initial+(i===0?generation/100:0)),generation,stage:config.stage,status:'unverified'});
const pose={position:[0,0,.1],quaternion:[1,0,0,0],time:.6,stage:config.stage,
  feet:[[.08,.07,.01],[0,.09,.01],[-.08,.07,.01],[.08,-.07,.01],[0,-.09,.01],[-.08,-.07,.01]],
  legs:[[0,0,.09],[0,0,.09],[0,0,.09],[0,0,.09],[0,0,.09],[0,0,.09]].map((p,i)=>[p,[i<3?.07:-.07,i%3*.03-.03,.05],[i<3?.11:-.11,i%3*.05-.05,.01]]),
  bowl:{radiusCm:2,floor:{baseCm:0,radialCoefficientPerCm:0,capRadiusCm:1}},contacts:{legs:[true,true,true,true,true,true]}};
const browser=await chromium.launch({executablePath:process.env.CHROME_PATH||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true});
await mkdir(output,{recursive:true});
const context=await browser.newContext({acceptDownloads:true}),requests=[],errors=[];
let generation=1,offline=false,mismatch=false,includeFrame=false,frameStale=false;
await context.addInitScript(()=>{
  globalThis.__workersStarted=0;globalThis.Worker=class{constructor(){globalThis.__workersStarted++;throw new Error('No browser simulation in native-only UI');}};
  globalThis.__tabHidden=false;
  Object.defineProperty(document,'hidden',{get:()=>globalThis.__tabHidden});
});
await context.route('**/*',async route=>{
  const url=new URL(route.request().url());requests.push({path:url.pathname+url.search,method:route.request().method()});
  if(url.pathname==='/training/config.json'){await route.fulfill({contentType:'application/json',body:configBytes});return;}
  if(url.pathname.startsWith('/api/training/')){
    if(offline){await route.fulfill({status:503,contentType:'application/json',body:JSON.stringify({error:'Coordinator unavailable'})});return;}
    if(url.pathname==='/api/training/status'){
      const status={configHash:mismatch?'previous-build':configHash,modelFingerprint:config.modelFingerprint,generation,acceptedResults:generation*8,
        jobs:{leased:generation===1?0:1},stage:config.stage,checkpoint:checkpoint(generation),recentWallSeconds:25*generation,
        recentTrials:[{episode:generation*8,generation,stage:config.stage,return:.1*generation,success:false,simSeconds:1,wallSeconds:25,completedAt:1700000000}],
        lastParameterUpdate:{generation,changedCount:generation,parameterCount:672,completedAt:1700000000},
        latestFrame:includeFrame?{jobId:'fixture-job',generation,serverReceivedAt:Date.now()/1000,recorded:true,stale:frameStale,frame:pose}:null};
      await route.fulfill({contentType:'application/json',body:JSON.stringify(status)});return;
    }
    if(url.pathname==='/api/training/checkpoint'){await route.fulfill({contentType:'application/json',body:JSON.stringify(checkpoint(generation))});return;}
    throw new Error('Native monitor must not request '+url.pathname);
  }
  const contentType={'.html':'text/html','.js':'text/javascript','.css':'text/css','.json':'application/json'}[path.extname(url.pathname)]||'application/octet-stream';
  try{await route.fulfill({contentType,body:await readFile(path.join(web,url.pathname))});}
  catch{await route.fulfill({status:404,body:'No UI fixture asset'});}
});
const page=await context.newPage();page.on('pageerror',error=>errors.push(error.message));
const assertCopy=async()=>{
  const copy=await page.evaluate(()=>[document.body.innerText,...[...document.querySelectorAll('[title],[aria-label]')].map(n=>`${n.title||''} ${n.getAttribute('aria-label')||''}`)].join('\n'));
  assert.doesNotMatch(copy,/\b(shared|coordinator|BANC|FlyBody|WebGPU|WASM|MuJoCo|parameters|unverified|fingerprint|lease)\b/i);
  assert.doesNotMatch(copy,/https?:\/\//);
};
try{
  await page.goto(origin+'/train.html');await page.waitForFunction(()=>globalThis.heavenTraining?.state.coordinator.connected);
  assert.equal(await page.textContent('#run-status'),'Watching');assert.equal(await page.textContent('#preview-heading'),'Recorded preview');
  assert.equal(await page.textContent('#stage-title'),'Maintain flight');assert.equal(await page.textContent('#stage-description'),'Stay airborne and in control.');
  assert.equal(await page.locator('#candidate-rows td:nth-child(2)').innerText(),'Maintain flight');
  assert.equal(await page.textContent('#trainer-note'),'This run is using the connected trainer.');
  for(const id of ['start-training','pause-training','stop-training','computer-budget'])assert.equal(await page.isDisabled('#'+id),true);
  assert.equal(await page.evaluate(()=>heavenTraining.client.parameters.length),672);assert.equal(await page.textContent('#evaluations-count'),'8');
  assert.equal(await page.textContent('#uploaded-count'),'1 / 672');assert.equal(await page.textContent('#latest-reward'),'0.100');
  assert.equal(await page.evaluate(()=>!!heavenTraining.preview.renderer),false);assert.equal(await page.evaluate(()=>__workersStarted),0);await assertCopy();
  assert.equal(requests.filter(r=>r.path==='/training/config.json').length,1,'Only one uncached config source');

  // The actual 15-second timer picks up another trainer's progress.
  generation=2;await page.waitForFunction(()=>heavenTraining.state.coordinator.generation===2,{},{timeout:20000});
  assert.equal(await page.textContent('#run-status'),'Training');assert.equal(await page.textContent('#evaluations-count'),'16');
  assert.equal(await page.textContent('#uploaded-count'),'2 / 672');assert.equal(await page.textContent('#latest-reward'),'0.200');
  assert(requests.some(r=>r.path==='/api/training/status?compact=1'));
  assert.equal(await page.evaluate(()=>heavenTraining.state.completedEpisodes),0,'The browser must not invent local trials');

  // Hidden tabs stop polling; returning refreshes immediately.
  await page.evaluate(()=>{__tabHidden=true;document.dispatchEvent(new Event('visibilitychange'));});
  const before=requests.filter(r=>r.path.startsWith('/api/training/status')).length;
  await new Promise(resolve=>setTimeout(resolve,15500));
  assert.equal(requests.filter(r=>r.path.startsWith('/api/training/status')).length,before);
  generation=3;await page.evaluate(()=>{__tabHidden=false;document.dispatchEvent(new Event('visibilitychange'));});
  await page.waitForFunction(()=>heavenTraining.state.coordinator.generation===3);

  offline=true;await page.evaluate(()=>heavenTraining.client.refreshCoordinatorStatus().catch(()=>{}));
  assert.equal(await page.textContent('#run-status'),'Connection lost');
  assert.equal(await page.textContent('#training-error'),'Connection lost. Reconnecting automatically.');
  assert.equal(await page.isDisabled('#start-training'),true);assert.equal(await page.evaluate(()=>__workersStarted),0);await assertCopy();
  offline=false;mismatch=true;await page.evaluate(()=>heavenTraining.client.refreshCoordinatorStatus().catch(()=>{}));
  assert.equal(await page.textContent('#training-error'),'This page is out of date. Reload and try again.');
  mismatch=false;await page.evaluate(()=>heavenTraining.client.refreshCoordinatorStatus());assert.equal(await page.isHidden('#training-error'),true);

  const downloadPromise=page.waitForEvent('download');await page.click('#save-checkpoint');const download=await downloadPromise;
  assert.equal(download.suggestedFilename(),'heaven-checkpoint-generation-3.json');
  assert.equal(JSON.parse(await readFile(await download.path())).parameters.length,672);

  // Supplied recorded geometry uses the existing low-resolution renderer.
  includeFrame=true;await page.evaluate(()=>heavenTraining.client.refreshCoordinatorStatus());
  await page.waitForFunction(()=>!!heavenTraining.preview.renderer);
  assert.equal(await page.textContent('#preview-candidate'),'—');assert.match(await page.textContent('#preview-status'),/^Updated /);
  assert.equal(await page.evaluate(()=>heavenTraining.preview.frame.time),.6);
  assert.equal(await page.textContent('#preview-label'),'Maintain flight');
  for(const viewport of [{width:1440,height:1000},{width:390,height:844},{width:320,height:740}]){
    await page.setViewportSize(viewport);await assertCopy();
    assert(await page.evaluate(()=>document.documentElement.scrollWidth)<=viewport.width);
    await page.screenshot({path:path.join(output,`recorded-fixture-${viewport.width}.png`),fullPage:true});
  }
  frameStale=true;await page.evaluate(()=>heavenTraining.client.refreshCoordinatorStatus());
  assert.equal(await page.textContent('#preview-status'),'Last received frame');
  assert.equal(await page.evaluate(()=>__workersStarted),0);
  assert.equal(requests.some(r=>/\/(lease|result|heartbeat)$|\.wasm|connectome\.bin|\/worker\.js|\/body-model\//.test(r.path)),false);
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({passed:true,scope:'Native monitor UI fixtures only; real 15s polling, hidden pause, mismatch recovery, fresh download, recorded pose rendering; zero workers or leases',viewports:[1440,390,320]}));
}finally{await browser.close();}
