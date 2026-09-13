// One actual shared evaluation on the deployed site, followed by cancellation
// and release of the next real lease. Never injects scores or replaces physics.
// Run only after TLS is ready. Normal certificate verification stays enabled.
import assert from 'node:assert/strict';
import {mkdir,readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import path from 'node:path';

const base=process.env.TRAINING_URL||'https://flytrain.morisoba.moe/train.html',origin=new URL(base).origin;
assert.equal(new URL(base).protocol,'https:','Live deployment verification requires HTTPS');
const directory=path.resolve(process.env.REPORT_DIR||'reports/gcp-ui-cleanup');
const deadlineMs=Number(process.env.TRAINING_TIMEOUT_MS||600000);
assert(Number.isFinite(deadlineMs)&&deadlineMs>0&&deadlineMs<=1200000,'Use a bounded timeout of at most 20 minutes');
await mkdir(directory,{recursive:true});
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'/tmp/fruit-fly-browser-tests/node_modules/playwright/index.mjs');
const report={date:new Date().toISOString(),url:base,scope:'Automatic metadata connection and fresh checkpoint download without compute, blocked training during an intercepted coordinator outage, then one actual shared BANC/FlyBody episode from the public HTTPS site and release of a second cancelled lease. This verifies deployment, not learned task success.',certificateVerification:'Normal browser and Node TLS validation; no insecure flags or overrides.',requests:[],apiResponses:[],consoleErrors:[],pageErrors:[],networkErrors:[],viewports:[],passed:false};
const sensitive=new Set(['leaseToken']);
const scrub=value=>Array.isArray(value)?value.map(scrub):value&&typeof value==='object'?Object.fromEntries(Object.entries(value).filter(([key])=>!sensitive.has(key)).map(([key,item])=>[key,scrub(item)])):value;
const digest=bytes=>createHash('sha256').update(bytes).digest('hex');
async function fetchJSON(endpoint){const response=await fetch(origin+endpoint,{signal:AbortSignal.timeout(20000)});assert.equal(response.ok,true,`${endpoint} returned ${response.status}`);return response.json();}
const heavy=url=>/\.wasm(?:\?|$)|\.bin(?:\?|$)|\/worker\.js(?:\?|$)|\/body-model\/|\/banc-data\//.test(url);
const removedControls='#share-opt-in,#connect-coordinator,#disconnect-coordinator,#coordinator-url,#coordinator-help,#coordinator-status,#training-stage,#load-checkpoint,#validate-checkpoint';
async function verifyPublicCopy(target){
  const copy=await target.evaluate(()=>[document.title,document.body.innerText,...Array.from(document.querySelectorAll('[aria-label]'),element=>element.getAttribute('aria-label'))].join('\n'));
  assert.doesNotMatch(copy,/\b(shared|coordinator|webgpu|wasm|mujoco|unverified|fingerprint|lease)\b|configuration hash|config hash|native frame|parameter vector/i,'Public UI should use plain training language');
}
function instrumentPage(){
  globalThis.__deploymentAudit={workers:0,webglContexts:0,frames:[],states:[],pausedAfterResult:false,releaseCheck:false,pausedForRelease:false};
  globalThis.Worker=new Proxy(globalThis.Worker,{construct(target,args,newTarget){__deploymentAudit.workers++;return Reflect.construct(target,args,newTarget);}});
  const getContext=HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext=function(type,...args){if(type==='webgl'||type==='webgl2')__deploymentAudit.webglContexts++;return getContext.call(this,type,...args);};
}
let browser,page,timer,phase='setup';const pendingResponses=new Set();
const watch=promise=>{pendingResponses.add(promise);promise.finally(()=>pendingResponses.delete(promise));};
try{
  // These explicit operator checks are separate from page-originated requests.
  report.before=await fetchJSON('/api/training/status');
  const configResponse=await fetch(origin+'/training/config.json',{signal:AbortSignal.timeout(20000)});
  assert.equal(configResponse.ok,true);const configBytes=Buffer.from(await configResponse.arrayBuffer()),config=JSON.parse(configBytes);
  report.configHash=digest(configBytes);report.modelFingerprint=config.modelFingerprint;
  assert.equal(report.configHash,report.before.configHash);assert.equal(config.modelFingerprint,report.before.modelFingerprint);
  report.localConfigHash=digest(await readFile('web/training/config.json'));
  assert.equal(report.configHash,report.localConfigHash,'Deployed configuration differs from the reviewed local build');
  browser=await chromium.launch({executablePath:process.env.CHROME_PATH||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,args:['--enable-unsafe-webgpu']});
  report.browser=browser.version();
  // A separate browser context blocks only coordinator traffic. It never sends
  // fake status, checkpoints, jobs or scores to the live coordinator.
  phase='offline-coordinator';
  const outageContext=await browser.newContext({viewport:{width:1440,height:1000},deviceScaleFactor:1});
  const offline={requests:[],blocked:[],pageErrors:[],expectedNetworkErrors:[]};report.offlineCoordinator=offline;
  try{
    await outageContext.addInitScript(instrumentPage);
    await outageContext.route(origin+'/api/training/**',async route=>{
      offline.blocked.push({url:route.request().url(),method:route.request().method()});await route.abort('internetdisconnected');
    });
    const outagePage=await outageContext.newPage();
    outagePage.on('request',request=>offline.requests.push({url:request.url(),method:request.method()}));
    outagePage.on('pageerror',error=>offline.pageErrors.push(error.message));
    outagePage.on('requestfailed',request=>offline.expectedNetworkErrors.push({url:request.url(),error:request.failure()?.errorText}));
    await outagePage.goto(base,{waitUntil:'networkidle',timeout:30000});
    await outagePage.waitForFunction(()=>globalThis.heavenTraining?.client?.config&&!document.querySelector('#training-error').hidden);
    assert.equal(await outagePage.locator(removedControls).count(),0);
    assert.equal(offline.blocked.length,1,'Page should attempt one automatic metadata connection');
    assert.equal(new URL(offline.blocked[0].url).pathname,'/api/training/status');
    assert.equal(await outagePage.isDisabled('#start-training'),false,'Start must allow retrying the shared connection');
    await outagePage.click('#start-training');
    await outagePage.waitForFunction(()=>heavenTraining.state.phase==='error'&&!heavenTraining.client.starting);
    offline.state=await outagePage.evaluate(()=>({phase:heavenTraining.state.phase,connected:heavenTraining.state.coordinator.connected,sharedOnly:heavenTraining.client.sharedOnly,mode:heavenTraining.client.options.mode,workers:__deploymentAudit.workers,webglContexts:__deploymentAudit.webglContexts,worker:!!heavenTraining.client.worker,renderer:!!heavenTraining.preview.renderer,episodes:heavenTraining.state.completedEpisodes,error:document.querySelector('#training-error').textContent}));
    assert.equal(offline.blocked.length,2,'Start should retry metadata without starting local training');
    assert(offline.blocked.every(request=>request.method==='GET'&&new URL(request.url).pathname==='/api/training/status'));
    assert.deepEqual({...offline.state,error:undefined},{phase:'error',connected:false,sharedOnly:true,mode:'shared',workers:0,webglContexts:0,worker:false,renderer:false,episodes:0,error:undefined});
    assert(offline.state.error.length>0);assert.equal(offline.requests.some(request=>heavy(request.url)),false);
    await verifyPublicCopy(outagePage);
    assert(offline.expectedNetworkErrors.every(request=>new URL(request.url).pathname==='/api/training/status'));
    assert.deepEqual(offline.pageErrors,[]);offline.passed=true;
  }finally{await outageContext.close();}
  const context=await browser.newContext({viewport:{width:1440,height:1000},deviceScaleFactor:1});
  await context.addInitScript(instrumentPage);
  page=await context.newPage();
  page.on('pageerror',error=>report.pageErrors.push({phase,message:error.message}));
  page.on('console',message=>{if(message.type()==='error')report.consoleErrors.push({phase,text:message.text()});});
  page.on('request',request=>report.requests.push({phase,url:request.url(),method:request.method(),resourceType:request.resourceType()}));
  page.on('requestfailed',request=>report.networkErrors.push({phase,url:request.url(),error:request.failure()?.errorText}));
  page.on('response',response=>watch((async()=>{
    const url=response.url();
    if(new URL(url).pathname.startsWith('/api/training/')){
      let body;try{body=scrub(await response.json());}catch{body=null;}
      report.apiResponses.push({phase,url,status:response.status(),body});
    }
    if(response.status()>=400&&!new URL(url).pathname.endsWith('/favicon.ico'))report.networkErrors.push({phase,url,status:response.status()});
  })().catch(error=>report.networkErrors.push({phase,message:error.message}))));
  phase='idle';const documentResponse=await page.goto(base,{waitUntil:'networkidle',timeout:30000});
  assert.equal(documentResponse.status(),200);report.tls=await documentResponse.securityDetails();
  await page.waitForFunction(()=>globalThis.heavenTraining?.state.coordinator.connected||!document.querySelector('#training-error').hidden);
  assert.equal(await page.evaluate(()=>heavenTraining.state.phase),'ready',await page.textContent('#training-error'));
  assert.equal(await page.locator(removedControls).count(),0);
  assert.equal(await page.evaluate(()=>heavenTraining.client.requiredCoordinatorUrl),origin);
  await verifyPublicCopy(page);
  report.idle=await page.evaluate(()=>({secureContext:isSecureContext,workers:__deploymentAudit.workers,webglContexts:__deploymentAudit.webglContexts,worker:!!heavenTraining.client.worker,renderer:!!heavenTraining.preview.renderer,episodes:heavenTraining.state.completedEpisodes,connected:heavenTraining.state.coordinator.connected,sharedOnly:heavenTraining.client.sharedOnly,mode:heavenTraining.client.options.mode}));
  assert.deepEqual(report.idle,{secureContext:true,workers:0,webglContexts:0,worker:false,renderer:false,episodes:0,connected:true,sharedOnly:true,mode:'shared'});
  assert.equal(report.requests.some(request=>heavy(request.url)),false);
  const idleAPI=report.requests.filter(request=>new URL(request.url).pathname.startsWith('/api/training/'));
  assert.equal(idleAPI.length,1);assert.equal(new URL(idleAPI[0].url).pathname,'/api/training/status');assert.equal(idleAPI[0].method,'GET');
  assert.equal(await page.locator('#computer-budget').getAttribute('type'),'range');
  await page.locator('#computer-budget').press('End');assert.equal(await page.inputValue('#computer-budget'),'100');
  await page.selectOption('#preview-quality','low');
  assert.equal(await page.textContent('#start-training'),'Start');
  report.connected=await page.evaluate(()=>({workers:__deploymentAudit.workers,worker:!!heavenTraining.client.worker,renderer:!!heavenTraining.preview.renderer,configHash:heavenTraining.client.configHash,modelFingerprint:heavenTraining.state.modelFingerprint}));
  assert.equal(report.connected.workers,0);assert.equal(report.connected.worker,false);assert.equal(report.connected.renderer,false);
  assert.equal(report.connected.configHash,report.configHash);assert.equal(report.connected.modelFingerprint,report.modelFingerprint);
  assert.equal(report.requests.some(request=>heavy(request.url)),false);
  phase='checkpoint-download';
  assert.equal(await page.textContent('#save-checkpoint'),'Download checkpoint');
  const checkpointResponsePromise=page.waitForResponse(response=>new URL(response.url()).pathname==='/api/training/checkpoint'&&response.request().method()==='GET');
  const downloadPromise=page.waitForEvent('download');await page.click('#save-checkpoint');
  const [checkpointResponse,download]=await Promise.all([checkpointResponsePromise,downloadPromise]);
  assert.equal(checkpointResponse.status(),200);const checkpointHeaders=await checkpointResponse.allHeaders(),servedCheckpoint=await checkpointResponse.json();
  assert.equal(checkpointHeaders['cache-control'],'no-store');
  assert.equal(checkpointHeaders['content-disposition'],`attachment; filename="heaven-checkpoint-generation-${servedCheckpoint.generation}.json"`);
  const checkpointPath=path.join(directory,'downloaded-checkpoint.json');await download.saveAs(checkpointPath);
  const downloadedCheckpoint=JSON.parse(await readFile(checkpointPath,'utf8'));
  assert.deepEqual(downloadedCheckpoint,servedCheckpoint,'Downloaded checkpoint must preserve the exact freshly fetched server metadata and parameters');
  assert.doesNotMatch(download.suggestedFilename(),/shared/i);
  assert.equal(download.suggestedFilename(),`heaven-checkpoint-generation-${servedCheckpoint.generation}.json`);
  report.checkpointDownload={filename:download.suggestedFilename(),path:checkpointPath,headers:checkpointHeaders,servedCheckpoint,downloadedCheckpoint};
  assert.deepEqual(await page.evaluate(()=>({workers:__deploymentAudit.workers,worker:!!heavenTraining.client.worker,renderer:!!heavenTraining.preview.renderer,episodes:heavenTraining.state.completedEpisodes})),{workers:0,worker:false,renderer:false,episodes:0});
  assert.equal(report.requests.some(request=>heavy(request.url)),false);
  await verifyPublicCopy(page);
  await page.evaluate(()=>{
    heavenTraining.client.addEventListener('frame',event=>{
      const f=event.detail;if(__deploymentAudit.frames.length<2000)__deploymentAudit.frames.push(structuredClone(f));
    });
    heavenTraining.client.addEventListener('state',event=>{
      const s=event.detail;
      if(__deploymentAudit.states.length<3000)__deploymentAudit.states.push({phase:s.phase,message:s.message,episode:s.episode,completedEpisodes:s.completedEpisodes,contributedEpisodes:s.contributedEpisodes,simSeconds:s.simSeconds,backend:s.backend});
      if(s.phase==='training'&&s.contributedEpisodes>=1&&!__deploymentAudit.pausedAfterResult){
        __deploymentAudit.pausedAfterResult=true;heavenTraining.client.pause('Deployment verification: one real result accepted');
      }else if(__deploymentAudit.releaseCheck&&!__deploymentAudit.pausedForRelease&&s.phase==='training'&&s.episode===2&&heavenTraining.client.activeLease){
        __deploymentAudit.pausedForRelease=true;heavenTraining.client.pause('Deployment verification: release next lease without a score');
      }
    });
  });
  phase='training';const started=Date.now();await page.click('#start-training');
  timer=setInterval(()=>{page.evaluate(()=>({phase:heavenTraining.state.phase,message:heavenTraining.state.message,episodes:heavenTraining.state.completedEpisodes,accepted:heavenTraining.state.contributedEpisodes,backend:heavenTraining.state.backend,frames:__deploymentAudit.frames.length})).then(state=>console.log(JSON.stringify({elapsedSeconds:Math.round((Date.now()-started)/1000),...state}))).catch(()=>{});},30000);
  await page.waitForFunction(()=>__deploymentAudit.pausedAfterResult||heavenTraining.state.phase==='error'||!document.querySelector('#training-error').hidden,null,{timeout:deadlineMs});
  report.episode=await page.evaluate(()=>heavenTraining.state);
  assert.notEqual(report.episode.phase,'error',report.episode.error);assert.equal(report.episode.phase,'paused',await page.textContent('#training-error'));
  assert.equal(report.episode.completedEpisodes,1);assert.equal(report.episode.contributedEpisodes,1);assert.equal(report.episode.backend,'webgpu');
  assert.equal(report.episode.history.length,1);assert.equal(report.episode.history[0].role,'contribution');assert(Number.isFinite(report.episode.history[0].return));
  report.frames=await page.evaluate(()=>__deploymentAudit.frames);
  assert(report.frames.length>=2,'No progressing native preview frames');
  assert(report.frames.every(frame=>frame.position.length===3&&frame.position.every(Number.isFinite)&&frame.quaternion.length===4&&frame.quaternion.every(Number.isFinite)));
  assert(report.frames.some(frame=>frame.simSeconds>0&&frame.neuralMs>0&&frame.legs.length===6&&frame.wings.length===2));
  const initial=report.frames.find(frame=>frame.simSeconds===0)||report.frames[0];
  assert(report.frames.some(frame=>frame.simSeconds>0&&frame.position.some((value,index)=>Math.abs(value-initial.position[index])>1e-8)),'No native body movement was observed');
  report.frameInspection={count:report.frames.length,firstTime:report.frames[0].simSeconds,lastTime:report.frames.at(-1).simSeconds,firstPosition:report.frames[0].position,lastPosition:report.frames.at(-1).position,positiveNeuralActivity:report.frames.some(frame=>frame.neuralSpikes>0),nativeLegs:report.frames.at(-1).legs.length,nativeWings:report.frames.at(-1).wings.length,vision:false};
  phase='paused';await page.waitForFunction(()=>heavenTraining.preview.renderer&&document.querySelector('#training-preview canvas'));
  await verifyPublicCopy(page);
  const pauseBefore=await page.evaluate(()=>({frames:__deploymentAudit.frames.length,simSeconds:heavenTraining.frame.simSeconds}));await page.waitForTimeout(500);
  assert.deepEqual(await page.evaluate(()=>({frames:__deploymentAudit.frames.length,simSeconds:heavenTraining.frame.simSeconds})),pauseBefore);
  for(const viewport of [{width:1440,height:1000},{width:390,height:844}]){
    await page.setViewportSize(viewport);await page.evaluate(()=>document.activeElement?.blur());await page.mouse.move(0,0);await page.waitForTimeout(100);
    const dimensions=await page.evaluate(()=>{const canvas=document.querySelector('#training-preview canvas');return {viewport:innerWidth,scrollWidth:document.documentElement.scrollWidth,canvasWidth:canvas.width,canvasHeight:canvas.height,previewVisible:!canvas.hidden};});
    assert(dimensions.scrollWidth<=viewport.width,'Horizontal overflow');assert.equal(dimensions.canvasWidth,320);assert.equal(dimensions.canvasHeight,180);assert.equal(dimensions.previewVisible,true);
    const screenshot=path.join(directory,`live-training-${viewport.width}.png`);await page.screenshot({path:screenshot,fullPage:true});report.viewports.push({...viewport,...dimensions,screenshot});
  }
  report.afterEpisode=await fetchJSON('/api/training/status');
  assert(report.afterEpisode.acceptedResults>=report.before.acceptedResults+1);assert.equal(report.afterEpisode.configHash,report.configHash);
  // Resume normal client work, immediately pause at the next assigned episode,
  // and Stop. This checks release of an outstanding real lease without scoring it.
  phase='lease-release';await page.evaluate(()=>{__deploymentAudit.releaseCheck=true;});await page.click('#pause-training');
  await page.waitForFunction(()=>__deploymentAudit.pausedForRelease||heavenTraining.state.phase==='error',null,{timeout:30000});
  report.beforeStop=await page.evaluate(()=>({phase:heavenTraining.state.phase,completedEpisodes:heavenTraining.state.completedEpisodes,contributedEpisodes:heavenTraining.state.contributedEpisodes,lease:heavenTraining.client.activeLease?{jobId:heavenTraining.client.activeLease.jobId,stage:heavenTraining.client.activeLease.stage,seed:heavenTraining.client.activeLease.seed}:null}));
  assert.equal(report.beforeStop.phase,'paused');assert(report.beforeStop.lease);assert.equal(report.beforeStop.completedEpisodes,1);
  const released=page.waitForResponse(response=>new URL(response.url()).pathname==='/api/training/release'&&response.request().method()==='POST',{timeout:20000});
  await page.click('#stop-training');const releaseResponse=await released;report.release={status:releaseResponse.status(),body:await releaseResponse.json()};
  assert.equal(report.release.status,200);assert.equal(report.release.body.released,true);
  report.stopped=await page.evaluate(()=>({phase:heavenTraining.state.phase,worker:!!heavenTraining.client.worker,lease:!!heavenTraining.client.activeLease,heartbeat:!!heavenTraining.client.heartbeat,completedEpisodes:heavenTraining.state.completedEpisodes,contributedEpisodes:heavenTraining.state.contributedEpisodes}));
  assert.deepEqual(report.stopped,{phase:'stopped',worker:false,lease:false,heartbeat:false,completedEpisodes:1,contributedEpisodes:1});
  await verifyPublicCopy(page);
  report.after=await fetchJSON('/api/training/status');report.elapsedSeconds=(Date.now()-started)/1000;
  await Promise.allSettled([...pendingResponses]);
  assert.equal(report.apiResponses.filter(response=>new URL(response.url).pathname==='/api/training/result'&&response.body?.accepted===true).length,1);
  assert.deepEqual(report.pageErrors,[]);assert.deepEqual(report.networkErrors,[]);
  const substantiveConsoleErrors=report.consoleErrors.filter(item=>!item.text.includes('favicon.ico'));
  assert.deepEqual(substantiveConsoleErrors,[]);
  report.passed=true;
}catch(error){report.failure={message:error.message,stack:error.stack};process.exitCode=1;}
finally{
  clearInterval(timer);
  if(page&&!page.isClosed()){
    try{report.finalAudit=await page.evaluate(()=>({workers:__deploymentAudit?.workers,webglContexts:__deploymentAudit?.webglContexts,states:__deploymentAudit?.states}));await page.evaluate(()=>globalThis.heavenTraining?.client.stop());}catch(error){report.cleanupError=error.message;}
  }
  if(browser)await browser.close();await Promise.allSettled([...pendingResponses]);
  await writeFile(path.join(directory,'browser-verification.json'),JSON.stringify(report,null,2)+'\n');
}
console.log(JSON.stringify({passed:report.passed,episodes:report.episode?.completedEpisodes,accepted:report.episode?.contributedEpisodes,backend:report.episode?.backend,frames:report.frameInspection?.count,report:path.join(directory,'browser-verification.json'),failure:report.failure?.message}));
