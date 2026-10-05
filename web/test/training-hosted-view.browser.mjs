// UI-only checks: coordinator responses are fixtures and Worker is forbidden.
// No native pose, reward, model, or real coordinator is used by this regression.
import assert from 'node:assert/strict';
import {mkdir,readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import path from 'node:path';

const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'/tmp/fruit-fly-browser-tests/node_modules/playwright/index.mjs');
const web=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const configBytes=await readFile(path.join(web,'training/config.json'));
const config=JSON.parse(configBytes),configHash=createHash('sha256').update(configBytes).digest('hex');
const sharedOrigin='https://flytrain.morisoba.moe';
const checkpoint=generation=>({schemaVersion:1,algorithm:config.algorithm,configHash,modelFingerprint:config.modelFingerprint,
  parameterNames:config.parameters.map(p=>p.name),parameters:config.parameters.map(p=>p.initial),generation,stage:config.stage,status:'unverified'});
const browser=await chromium.launch({executablePath:process.env.CHROME_PATH||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true});
const cases=[
  {origin:'https://flies.example',hosted:true},
  {origin:sharedOrigin,hosted:true},
  {origin:'http://127.0.0.1:7842',hosted:false},
  {origin:'https://localhost',hosted:false},
  {origin:'https://lab.localhost',hosted:false},
  {origin:'https://127.0.0.2',hosted:false},
  {origin:'https://[::1]',hosted:false},
];
const passed=[],viewports=[],output=path.resolve(process.env.TRAINING_UI_REPORT_DIR||'reports/training-simple-ui');
const forbidden=/\b(shared|coordinator|BANC|FlyBody|VNC|WebGPU|WASM|MuJoCo|unverified|fingerprint|lease|evolution strategies|model route)\b/i;
async function assertPublicCopy(page){
  const exposed=await page.evaluate(()=>[document.body.innerText,...[...document.querySelectorAll('[aria-label],[title]')].map(node=>`${node.getAttribute('aria-label')||''} ${node.getAttribute('title')||''}`)].join('\n'));
  assert.doesNotMatch(exposed,forbidden);
  assert.doesNotMatch(exposed,/https?:\/\//);
  assert.equal(await page.locator('header h1').innerText(),'Training');
}
await mkdir(output,{recursive:true});
try{
  for(const {origin,hosted}of cases){
    const requests=[],errors=[],context=await browser.newContext({acceptDownloads:true});
    let offline=false,settingsUnavailable=false,invalidCheckpoint=false,downloadGeneration=2;
    await context.addInitScript(()=>{
      globalThis.__workersStarted=0;globalThis.__webglContexts=0;
      globalThis.Worker=class{constructor(){globalThis.__workersStarted++;throw new Error('UI-only check must not create workers');}};
      const getContext=HTMLCanvasElement.prototype.getContext;
      HTMLCanvasElement.prototype.getContext=function(type,...args){if(type==='webgl'||type==='webgl2')globalThis.__webglContexts++;return getContext.call(this,type,...args);};
    });
    await context.route('**/*',async route=>{
      const request=route.request(),url=new URL(request.url());requests.push({url:url.href,method:request.method()});
      if(settingsUnavailable&&url.pathname==='/training/config.json'){
        await route.fulfill({status:503,contentType:'application/json',body:'{}'});return;
      }
      if(url.pathname.startsWith('/api/training/')){
        const headers={'Access-Control-Allow-Origin':origin,'Cache-Control':'no-store'};
        if(offline){await route.fulfill({status:503,headers,contentType:'application/json',body:JSON.stringify({error:'Shared coordinator unavailable'})});return;}
        if(url.pathname==='/api/training/status'){
          await route.fulfill({headers,contentType:'application/json',body:JSON.stringify({modelFingerprint:config.modelFingerprint,configHash,generation:1,stage:config.stage,completedJobs:0,totalJobs:8,contributors:0,checkpoint:checkpoint(1)})});return;
        }
        if(url.pathname==='/api/training/checkpoint'){
          const value=checkpoint(downloadGeneration++);if(invalidCheckpoint)value.modelFingerprint='wrong-model';
          await route.fulfill({headers,contentType:'application/json',body:JSON.stringify(value)});return;
        }
        throw new Error(`Unexpected API call in metadata-only regression: ${url.pathname}`);
      }
      const extension=path.extname(url.pathname),contentType={'.html':'text/html','.js':'text/javascript','.css':'text/css','.json':'application/json'}[extension]||'application/octet-stream';
      try{await route.fulfill({contentType,body:await readFile(path.join(web,url.pathname))});}
      catch{await route.fulfill({status:404,body:'Not found in UI-only fixture'});}
    });
    const page=await context.newPage();page.on('pageerror',error=>errors.push(error.message));
    await page.goto(origin+'/train.html');
    await page.waitForFunction(()=>globalThis.heavenTraining?.state.coordinator.connected||!document.querySelector('#training-error').hidden);
    assert.equal(await page.evaluate(()=>heavenTraining.state.phase),'ready',`${origin}: ${await page.textContent('#training-error')}`);
    assert.equal(await page.evaluate(()=>heavenTraining.state.coordinator.connected),true);
    const expected=hosted?origin:sharedOrigin;
    assert.equal(await page.evaluate(()=>heavenTraining.client.requiredCoordinatorUrl),expected);
    assert.equal(await page.locator('#share-opt-in,#connect-coordinator,#disconnect-coordinator,#load-checkpoint,#validate-checkpoint,#coordinator-url,#coordinator-help,#coordinator-status,#training-stage').count(),0);
    assert.equal(await page.textContent('#start-training'),'Start');
    assert.equal(await page.textContent('#save-checkpoint'),'Download checkpoint');
    assert.equal(await page.textContent('#run-status'),'Ready');
    assert.equal(await page.getAttribute('#computer-budget','type'),'range');
    await page.locator('#computer-budget').fill('35');
    assert.equal(await page.evaluate(()=>heavenTraining.client.options.dutyCycle),.35);
    assert.equal(await page.textContent('#budget-value'),'35%');
    await page.locator('#computer-budget').fill('60');
    await assertPublicCopy(page);
    assert.equal(await page.locator('a[href="/"]').count(),0,'standalone training must not promise an absent observation habitat');
    assert.deepEqual(requests.filter(request=>request.url.includes('/api/training/')),[{url:expected+'/api/training/status',method:'GET'}]);

    // Downloads must freshly fetch shared metadata, even when the tab has a
    // different cached checkpoint. The existing local export path is forbidden.
    await page.evaluate(()=>{
      heavenTraining.client.exportCheckpoint=()=>{throw new Error('A shared download must never export local parameters');};
      heavenTraining.client.sharedCheckpoint={...heavenTraining.client.sharedCheckpoint,generation:999};
    });
    let downloads=0;page.on('download',()=>downloads++);
    for(const generation of [2,3]){
      const downloadPromise=page.waitForEvent('download');await page.click('#save-checkpoint');
      const download=await downloadPromise,value=JSON.parse(await readFile(await download.path(),'utf8'));
      assert.equal(value.generation,generation);assert.equal(value.status,'unverified');assert.equal(value.configHash,configHash);
      assert.equal(download.suggestedFilename(),`heaven-checkpoint-generation-${generation}.json`);
      await page.waitForFunction(()=>!document.querySelector('#save-checkpoint').disabled);
      await assertPublicCopy(page);
    }
    assert.equal(requests.filter(request=>request.url===expected+'/api/training/checkpoint').length,2);
    invalidCheckpoint=true;
    await page.click('#save-checkpoint');
    await page.waitForFunction(()=>!document.querySelector('#training-error').hidden);
    assert.equal(await page.textContent('#training-error'),'This page is out of date. Reload and try again.');
    assert.equal(downloads,2,'incompatible shared checkpoints must not download');invalidCheckpoint=false;

    // Coordinator failure exercises the real client start path. It must not
    // create a worker or fall back to a local run, including after a failed load.
    offline=true;await page.click('#start-training');
    await page.waitForFunction(()=>heavenTraining.state.phase==='error');
    assert.equal(await page.textContent('#training-error'),'Connection lost. Check your connection and press Start to retry.');
    assert.equal(await page.textContent('#run-status'),'Connection lost');
    await assertPublicCopy(page);
    assert.equal(await page.isDisabled('#start-training'),false,'Start remains available to retry');
    assert.equal(await page.evaluate(()=>__workersStarted),0);
    await assertPublicCopy(page);
    await page.reload();
    await page.waitForFunction(()=>globalThis.heavenTraining?.state.phase==='ready'&&!document.querySelector('#training-error').hidden);
    assert.equal(await page.evaluate(()=>heavenTraining.state.coordinator.connected),false);
    await page.click('#start-training');await page.waitForFunction(()=>heavenTraining.state.phase==='error');
    assert.equal(await page.evaluate(()=>__workersStarted),0);

    // Handler contract only: successful simulation itself is covered separately
    // by the live native-worker check. This spy cannot produce results or jobs.
    await page.evaluate(()=>{heavenTraining.client.start=async options=>{globalThis.__startOptions=options;};});
    await page.click('#start-training');
    assert.deepEqual(await page.evaluate(()=>__startOptions),{mode:'shared',coordinatorUrl:expected,dutyCycle:.6,previewHz:6});
    offline=false;await page.reload();await page.waitForFunction(()=>globalThis.heavenTraining?.state.coordinator.connected);
    if(origin==='https://flies.example'){
      for(const viewport of [{width:1440,height:1000},{width:390,height:844},{width:320,height:740}]){
        await page.setViewportSize(viewport);
        const geometry=await page.evaluate(()=>({scrollWidth:document.documentElement.scrollWidth,
          unlabeled:[...document.querySelectorAll('button,input,select')].filter(e=>!e.getAttribute('aria-label')&&!e.labels?.length&&!e.textContent?.trim()).map(e=>e.id)}));
        assert.ok(geometry.scrollWidth<=viewport.width,`Horizontal overflow at ${viewport.width}: ${geometry.scrollWidth}`);assert.deepEqual(geometry.unlabeled,[]);
        await page.screenshot({path:path.join(output,`idle-${viewport.width}.png`),fullPage:true});viewports.push(viewport.width);
      }
      // These rendering fixtures only verify that worker messages cannot leak
      // internal vocabulary into the public interface. No result is submitted.
      await page.evaluate(()=>heavenTraining.client.dispatchEvent(new CustomEvent('state',{detail:{...heavenTraining.client.state,phase:'training',message:'BANC coordinator shared parameters unverified',history:[{episode:1,stage:'posture',return:-.25,success:false}],completedEpisodes:1,lastReturn:-.25}})));
      assert.equal(await page.textContent('#run-status'),'Running');await assertPublicCopy(page);
      await page.setViewportSize({width:1440,height:1000});await page.screenshot({path:path.join(output,'running-copy-fixture-1440.png'),fullPage:true});
      await page.evaluate(()=>heavenTraining.client.dispatchEvent(new CustomEvent('state',{detail:{...heavenTraining.client.state,phase:'paused',message:'shared coordinator paused'}})));
      assert.equal(await page.textContent('#run-status'),'Paused');await assertPublicCopy(page);
      await page.evaluate(()=>heavenTraining.client.dispatchEvent(new CustomEvent('state',{detail:heavenTraining.client.state})));
    }
    assert.deepEqual(await page.evaluate(()=>({workers:__workersStarted,webgl:__webglContexts,worker:!!heavenTraining.client.worker,renderer:!!heavenTraining.preview.renderer,episodes:heavenTraining.state.completedEpisodes})),{workers:0,webgl:0,worker:false,renderer:false,episodes:0});
    assert.equal(requests.some(request=>/\.wasm|connectome\.bin|\/worker\.js|\/body-model\//.test(request.url)),false);
    assert.deepEqual(errors,[]);
    if(origin==='https://flies.example'){
      settingsUnavailable=true;await page.reload();
      await page.waitForFunction(()=>!document.querySelector('#training-error').hidden);
      assert.equal(await page.textContent('#training-error'),'Could not load training. Check your connection and reload.');
      assert.equal(await page.isDisabled('#start-training'),true);
      assert.equal(await page.evaluate(()=>__workersStarted),0);await assertPublicCopy(page);
    }
    passed.push(origin);await context.close();
  }
}finally{await browser.close();}
console.log(JSON.stringify({passed:true,scope:'UI-only automatic metadata connection, mandatory sharing, no compute before Start or on coordinator failure, fresh validated checkpoint downloads; no simulation',origins:passed,viewports}));
