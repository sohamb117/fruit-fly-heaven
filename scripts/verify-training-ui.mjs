// Real complete episodes and coordinator updates; no replacement neural worker.
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'playwright');
const directory='reports/training-ui-validation';await fs.mkdir(directory,{recursive:true});
const report={date:new Date().toISOString(),scope:'Real one-fly BANC/FlyBody browser training, complete local ES generation, shared contributions from independent browser profiles. This verifies training infrastructure, not learned task success.',errors:[],clients:[]};
const browser=await chromium.launch({executablePath:process.env.CHROME_PATH||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,args:['--enable-unsafe-webgpu']});
const base='http://127.0.0.1:7842/train.html',coordinator='http://127.0.0.1:7850';
async function pageFor(name){
 const context=await browser.newContext({viewport:{width:1440,height:1000}}),page=await context.newPage();
 page.on('pageerror',e=>report.errors.push({client:name,message:e.message}));
 page.on('response',r=>{if(r.status()>=400&&!r.url().endsWith('/favicon.ico'))report.errors.push({client:name,url:r.url(),status:r.status()});});
 await page.goto(base);await page.waitForFunction(()=>window.heavenTraining?.state.phase==='ready');
 assert.equal(await page.evaluate(()=>heavenTraining.client.worker),null);
 await page.selectOption('#computer-budget','high');await page.selectOption('#preview-quality','low');
 await page.evaluate(()=>{window.observedFrames=[];heavenTraining.client.addEventListener('frame',e=>{const f=e.detail;observedFrames.push({time:f.simSeconds,stage:f.stage,position:f.position,quaternion:f.quaternion,return:f.metrics?.return,contacts:f.contacts,neuralSpikes:f.neuralSpikes});});});
 return {context,page};
}
async function until(page,condition,timeout=300000){
 await page.waitForFunction(condition,null,{timeout});
 const state=await page.evaluate(()=>heavenTraining.state);if(state.phase==='error')throw new Error(state.error);return state;
}
try{
 const {context,page}=await pageFor('local');
 await page.evaluate(()=>heavenTraining.client.addEventListener('state',e=>{if(e.detail.generation>=1&&e.detail.phase==='training')heavenTraining.client.pause('Validation capture after one complete generation');}));
 await page.click('#start-training');
 await until(page,()=>heavenTraining.state.completedEpisodes>=1||heavenTraining.state.phase==='error');
 await page.click('#pause-training');
 await page.waitForFunction(()=>heavenTraining.state.phase==='paused');await page.waitForTimeout(300);
 const paused=await page.evaluate(()=>({frames:observedFrames.length,sim:heavenTraining.frame?.simSeconds}));await page.waitForTimeout(500);
 const pausedAfter=await page.evaluate(()=>({frames:observedFrames.length,sim:heavenTraining.frame?.simSeconds}));assert.deepEqual(pausedAfter,paused);report.pause={passed:true,...paused};
 await page.click('#pause-training');
 report.local=await until(page,()=>heavenTraining.state.generation>=1||heavenTraining.state.phase==='error');
 assert(report.local.completedEpisodes>=12);assert.equal(report.local.backend,'webgpu');
 await page.waitForTimeout(350);await page.screenshot({path:directory+'/local-training.png',fullPage:true});
 report.localFrames=await page.evaluate(()=>observedFrames);report.local.checkpoint=await page.evaluate(()=>heavenTraining.client.exportCheckpoint());
 console.log(JSON.stringify({phase:'local-generation',generation:report.local.generation,episodes:report.local.completedEpisodes,return:report.local.bestReturn,status:report.local.checkpointStatus}));
 const [download]=await Promise.all([page.waitForEvent('download'),page.click('#save-checkpoint')]);await download.saveAs(directory+'/local-checkpoint.json');
 await page.click('#stop-training');await page.setInputFiles('#load-checkpoint',directory+'/local-checkpoint.json');
 await page.waitForFunction(()=>heavenTraining.state.checkpointStatus==='unverified');assert.equal(await page.evaluate(()=>heavenTraining.client.worker),null);
 await page.reload();await page.waitForFunction(()=>heavenTraining.state.phase==='ready');assert.equal(await page.evaluate(()=>heavenTraining.client.worker),null);
 report.restored=await page.evaluate(()=>({generation:heavenTraining.state.generation,completedEpisodes:heavenTraining.state.completedEpisodes,parameters:heavenTraining.state.parameters}));assert.deepEqual(report.restored.parameters,report.local.checkpoint.parameters);
 await context.close();
 report.coordinatorBefore=await (await fetch(coordinator+'/api/training/status')).json();
 for(let i=0;i<2;i++){
  const {context,page}=await pageFor('contributor-'+i);
  await page.fill('#coordinator-url',coordinator);await page.check('#share-opt-in');await page.click('#connect-coordinator');
  await page.waitForFunction(()=>heavenTraining.state.coordinator.connected);assert.equal(await page.evaluate(()=>heavenTraining.client.worker),null);
  await page.evaluate(()=>heavenTraining.client.addEventListener('state',e=>{if(e.detail.contributedEpisodes>=4&&e.detail.phase==='training')heavenTraining.client.pause('Contribution capture after four accepted jobs');}));
  await page.click('#start-training');
  const state=await until(page,()=>heavenTraining.state.contributedEpisodes>=4||heavenTraining.state.phase==='error');
  await page.waitForTimeout(350);await page.screenshot({path:directory+`/contributor-${i}.png`,fullPage:true});
  report.clients.push({...state,contributorId:await page.evaluate(()=>heavenTraining.client.contributorId),frames:await page.evaluate(()=>observedFrames)});
  assert.equal(state.checkpointStatus,'shared-unverified');
  await page.click('#stop-training');assert.equal(await page.evaluate(()=>heavenTraining.client.worker),null);await context.close();
  console.log(JSON.stringify({phase:'shared-contributor',client:i,accepted:state.contributedEpisodes,generation:state.generation}));
 }
 assert.notEqual(report.clients[0].contributorId,report.clients[1].contributorId);
 report.coordinatorAfter=await (await fetch(coordinator+'/api/training/status')).json();
 assert.equal(report.coordinatorAfter.acceptedResults-report.coordinatorBefore.acceptedResults,8);assert.equal(report.coordinatorAfter.generation,report.coordinatorBefore.generation+1);
 assert.equal(report.coordinatorAfter.checkpoint.status,'unverified');assert.equal(report.errors.length,0);
 report.passed=true;
}catch(error){report.passed=false;report.failure=error.stack;process.exitCode=1;}
finally{await browser.close();await fs.writeFile(directory+'/result.json',JSON.stringify(report,null,2)+'\n');}
console.log(JSON.stringify({passed:report.passed,localEpisodes:report.local?.completedEpisodes,sharedEpisodes:report.clients.map(c=>c.contributedEpisodes),report:directory+'/result.json',failure:report.failure}));
