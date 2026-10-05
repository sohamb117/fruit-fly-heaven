import {mkdir,writeFile,readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
import path from 'node:path';

const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'/tmp/fruit-fly-browser-tests/node_modules/playwright/index.mjs');
const base=process.env.TRAINING_URL||'http://127.0.0.1:7842/train.html';
const output=path.resolve(process.env.REPORT_DIR||'reports/training-ui-idle');
await mkdir(output,{recursive:true});
const browser=await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true});
const report={url:base,browser:browser.version(),scope:'Shared-only idle UI and explicit reward-chart UI fixtures. Metadata connection and checkpoint download are allowed; no Start, worker, native pose injection, or score submission. Fixture scores test rendering only.',viewports:[],errors:[],requests:[],sourceHashes:{}};
try{
  const context=await browser.newContext({viewport:{width:1440,height:1000},deviceScaleFactor:1,acceptDownloads:true});
  await context.addInitScript(()=>{
    globalThis.__idleAudit={workers:0,webglContexts:0};
    globalThis.Worker=new Proxy(globalThis.Worker,{construct(target,args,newTarget){globalThis.__idleAudit.workers++;return Reflect.construct(target,args,newTarget);}});
    const getContext=HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext=function(type,...args){if(type==='webgl'||type==='webgl2')globalThis.__idleAudit.webglContexts++;return getContext.call(this,type,...args);};
  });
  const page=await context.newPage();
  page.on('pageerror',error=>report.errors.push(error.message));
  page.on('request',request=>report.requests.push(new URL(request.url()).pathname));
  await page.goto(base,{waitUntil:'networkidle'});
  await page.waitForFunction(()=>globalThis.heavenTraining?.state?.coordinator?.connected);
  assert.equal(await page.locator('#start-training').isEnabled(),true);
  assert.equal(await page.locator('#pause-training').isDisabled(),true);
  assert.equal(await page.locator('#stop-training').isDisabled(),true);
  assert.equal(await page.locator('#connect-coordinator,#share-opt-in,#disconnect-coordinator,#load-checkpoint,#validate-checkpoint,#coordinator-url,#coordinator-help,#coordinator-status,#training-stage').count(),0);
  assert.equal(await page.locator('header h1').innerText(),'Training');
  assert.equal(await page.locator('#start-training').innerText(),'Start');
  assert.equal(await page.locator('#save-checkpoint').innerText(),'Download checkpoint');
  assert.equal(await page.locator('#computer-budget').getAttribute('type'),'range');
  assert.equal(await page.evaluate(()=>heavenTraining.client.sharedOnly),true);
  assert.equal(await page.locator('#history-empty').isVisible(),true);
  assert.equal(await page.locator('#reward-chart').isVisible(),false);
  // Explicit UI fixture: bypass only the EventTarget view boundary, never the
  // evaluator. These values are not behavioral evidence and are not persisted.
  await page.evaluate(()=>heavenTraining.client.dispatchEvent(new CustomEvent('state',{detail:{...structuredClone(heavenTraining.client.state),
    message:'UI regression fixture — no training evaluation',history:[
      {episode:1,stage:heavenTraining.client.config.stage,return:-.25,success:false,role:'ui-regression-fixture'},
      {episode:2,stage:heavenTraining.client.config.stage,return:.75,success:false,role:'ui-regression-fixture'}],completedEpisodes:2,lastReturn:.75,bestReturn:.75}})));
  const populatedChart=await page.locator('#reward-chart').evaluate(svg=>({hiddenAttribute:svg.hasAttribute('hidden'),display:getComputedStyle(svg).display,
    width:svg.getBoundingClientRect().width,height:svg.getBoundingClientRect().height,path:svg.querySelector('#reward-path').getAttribute('d'),dots:svg.querySelectorAll('#reward-dots circle').length}));
  assert.equal(await page.locator('#reward-chart').isVisible(),true);assert.equal(await page.locator('#history-empty').isVisible(),false);
  assert.equal(populatedChart.hiddenAttribute,false);assert.notEqual(populatedChart.display,'none');assert.ok(populatedChart.width>0&&populatedChart.height>0);
  assert.match(populatedChart.path,/^M [\d.]+ [\d.]+ L [\d.]+ [\d.]+$/);assert.equal(populatedChart.dots,2);
  await page.evaluate(()=>heavenTraining.client.dispatchEvent(new CustomEvent('state',{detail:structuredClone(heavenTraining.client.state)})));
  assert.equal(await page.locator('#reward-chart').isVisible(),false);assert.equal(await page.locator('#history-empty').isVisible(),true);
  assert.equal(await page.locator('#reward-chart').getAttribute('hidden'),'');
  report.chartRegression={scope:'Two synthetic UI fixture points only; never evaluated or saved.',populatedChart,hiddenAgain:true,passed:true};
  await page.selectOption('#preview-quality','off');
  assert.match(await page.locator('#preview-empty').innerText(),/Preview off/);
  assert.equal(await page.evaluate(()=>heavenTraining.client.options.previewHz),0);
  await page.selectOption('#preview-quality','balanced');
  assert.match(await page.locator('#preview-empty').innerText(),/Press Start/);
  await page.locator('#computer-budget').fill('35');
  assert.equal(await page.evaluate(()=>heavenTraining.client.options.dutyCycle),.35);
  await page.locator('#computer-budget').fill('60');
  // A fresh server checkpoint download is metadata-only and starts no worker.
  const checkpointResponse=page.waitForResponse(response=>new URL(response.url()).pathname==='/api/training/checkpoint');
  const downloadPromise=page.waitForEvent('download');
  await page.click('#save-checkpoint');
  const download=await downloadPromise,checkpoint=JSON.parse(await readFile(await download.path(),'utf8'));
  assert.deepEqual(checkpoint,await (await checkpointResponse).json());
  report.sharedCheckpointDownload={parameterCount:checkpoint.parameters.length,stage:checkpoint.stage,generation:checkpoint.generation,filename:download.suggestedFilename()};
  for(const viewport of [{width:1440,height:1000},{width:390,height:844},{width:320,height:740}]){
    await page.setViewportSize(viewport);await page.evaluate(()=>document.activeElement?.blur());await page.mouse.move(0,0);
    const geometry=await page.evaluate(()=>({innerWidth,scrollWidth:document.documentElement.scrollWidth,bodyWidth:document.body.getBoundingClientRect().width,
      unlabeled:[...document.querySelectorAll('button,input,select')].filter(e=>!e.getAttribute('aria-label')&&!e.labels?.length&&!e.textContent?.trim()).map(e=>e.id),
      tinyTargets:[...document.querySelectorAll('button,select,input:not([type="checkbox"]):not([type="file"])')].filter(e=>e.getBoundingClientRect().height<30).map(e=>e.id)}));
    assert.ok(geometry.scrollWidth<=viewport.width,`Horizontal overflow at ${viewport.width}: ${geometry.scrollWidth}`);
    assert.deepEqual(geometry.unlabeled,[]);assert.deepEqual(geometry.tinyTargets,[]);
    const exposed=await page.evaluate(()=>[document.body.innerText,...[...document.querySelectorAll('[aria-label],[title]')].map(node=>`${node.getAttribute('aria-label')||''} ${node.getAttribute('title')||''}`)].join('\n'));
    assert.doesNotMatch(exposed,/\b(shared|coordinator|BANC|FlyBody|VNC|WebGPU|WASM|MuJoCo|unverified|fingerprint|lease|evolution strategies|parameters|model route)\b/i);
    assert.doesNotMatch(exposed,/https?:\/\//);
    const screenshot=path.join(output,`idle-${viewport.width}.png`);await page.screenshot({path:screenshot,fullPage:true});
    report.viewports.push({...viewport,...geometry,screenshot});
  }
  report.idle=await page.evaluate(()=>({...__idleAudit,phase:heavenTraining.state.phase,frame:heavenTraining.frame,episodes:heavenTraining.state.completedEpisodes,contributed:heavenTraining.state.contributedEpisodes,worker:!!heavenTraining.client.worker,renderer:!!heavenTraining.preview.renderer,coordinator:heavenTraining.state.coordinator.connected}));
  assert.equal(report.idle.workers,0);assert.equal(report.idle.webglContexts,0);assert.equal(report.idle.worker,false);assert.equal(report.idle.renderer,false);
  assert.equal(report.idle.frame,null);assert.equal(report.idle.episodes,0);assert.equal(report.idle.coordinator,true);assert.deepEqual(report.errors,[]);
  assert.equal(report.requests.some(p=>/\.wasm$|connectome\.bin|\/worker\.js$|\/body-model\//.test(p)),false);
  const historySource=path.resolve(process.env.TRAINING_HISTORY_REPORT||'reports/training-ui-validation/result.json');
  let historyBytes;try{historyBytes=await readFile(historySource);}catch(error){if(error.code!=='ENOENT')throw error;}
  if(historyBytes){
    const recorded=JSON.parse(historyBytes),local=recorded.local;
    assert.ok(Array.isArray(local?.history)&&local.history.length,'Recorded report contains no local evaluation history');
    assert.ok(local.history.every(entry=>Number.isFinite(entry.return)),'Recorded reward must be finite');
    await page.setViewportSize({width:1440,height:1000});
    await page.evaluate(local=>{
      const detail={...structuredClone(heavenTraining.client.state),history:local.history,completedEpisodes:local.completedEpisodes,
        generation:local.generation,lastReturn:local.lastReturn,bestReturn:local.bestReturn,stage:local.stage,
        message:'Recorded local evaluation history replay — no compute is running'};
      heavenTraining.client.dispatchEvent(new CustomEvent('state',{detail}));
      document.querySelector('#reward-range').textContent='RECORDED LOCAL RUN · '+document.querySelector('#reward-range').textContent;
    },local);
    assert.equal(await page.locator('#reward-chart').isVisible(),true);
    assert.equal(await page.locator('#reward-dots circle').count(),Math.min(120,local.history.length));
    const screenshot=path.join(output,'recorded-local-history.png');await page.locator('.timeline-panel').screenshot({path:screenshot});
    report.recordedHistoryReplay={source:historySource,sourceSha256:createHash('sha256').update(historyBytes).digest('hex'),
      scope:'Rendering-only replay of actual recorded local evaluation history; no model initialization or new evaluations.',
      sourceRunPassed:recorded.passed,historyCount:local.history.length,completedEpisodes:local.completedEpisodes,generation:local.generation,
      history:local.history,screenshot,passed:true};
    await page.evaluate(()=>heavenTraining.client.dispatchEvent(new CustomEvent('state',{detail:structuredClone(heavenTraining.client.state)})));
  }else report.recordedHistoryReplay={pending:true,source:historySource};
  assert.deepEqual(await page.evaluate(()=>__idleAudit),{workers:0,webglContexts:0});
  report.passed=true;
}catch(error){report.passed=false;report.failure={message:error.message,stack:error.stack};process.exitCode=1;}
finally{
  await browser.close();
  for(const file of ['web/train.html','web/training/train.css','web/training/preview.js','web/training/view.js','web/training/client.js','web/training/config.json','web/console-shell.js'])report.sourceHashes[file]=createHash('sha256').update(await readFile(file)).digest('hex');
  await writeFile(path.join(output,'result.json'),JSON.stringify(report,null,2)+'\n');
  process.stdout.write(JSON.stringify({passed:report.passed,idle:report.idle,viewports:report.viewports.map(v=>v.width),failure:report.failure?.message,report:path.join(output,'result.json')})+'\n');
}
