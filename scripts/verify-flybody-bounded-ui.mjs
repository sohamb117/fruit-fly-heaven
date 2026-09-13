// Independently watch the default bounded reference in the original live UI.
// Own browser only; does not attach to Safari or the hour-long observer.
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'/tmp/fruit-fly-browser-tests/node_modules/playwright/index.mjs');
const browser=await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,args:['--enable-unsafe-webgpu']});
const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[],sourceHashes={};
const report={date:new Date().toISOString(),scope:'Independent live observation of one bounded published-policy reference in the original 3D UI. BANC remains an observer. No scripted physical advancement, pose override, or change to the separate hour observer.',browser:browser.version(),errors,sourceHashes,checks:[]};
page.on('pageerror',error=>errors.push(error.message));
page.on('response',async response=>{
 const path=new URL(response.url()).pathname;
 if(['/app.js','/flybody-reference-world.js','/flybody-flight-reference.js','/flybody-reference-trajectory.js','/flybody-policy.js'].includes(path)){
  try{sourceHashes[path]=createHash('sha256').update(await response.body()).digest('hex');}catch{}
 }
});
const read=()=>page.evaluate(()=>{
 const h=window.heaven,w=h.bodyWorld,f=h.state.flies[0];
 return {bodyTime:w.time,neuralTimeMs:h.state.time_ms,complete:w.complete,paused:h.state.paused,
   sample:w.controller.sample(),worldPosition:[f.x,f.y,f.z],worldRadius:Math.hypot(f.x,f.z),
   backend:f.physicsBackend,phase:f.task.phase,reference:w.reference,kind:w.metadata.prescribedTrajectory?.kind,
   renderedCalls:h.renderer.info.render.calls,canvasSize:[h.renderer.domElement.width,h.renderer.domElement.height],
   pendingSeconds:w.accumulator};
});
try{
 await page.addInitScript(()=>{
  localStorage.setItem('fruit-fly-population','7');localStorage.setItem('fruit-fly-movement-mode','behavior');
  localStorage.setItem('fruit-fly-neural-body-clock','neural');localStorage.setItem('fruit-fly-flight','true');
 });
 await page.goto('http://127.0.0.1:7842/?controller=flybody-reference&population=7&follow=1');
 await page.waitForFunction(()=>window.heaven?.ready&&window.heaven.bodyWorld.time>=.03,null,{timeout:180000});
 assert.equal(await page.evaluate(()=>window.heaven.state.flies.length),1);
 assert.equal(await page.locator('#movement-mode').isDisabled(),true);
 assert.equal(await page.locator('#body-clock').inputValue(),'live');
 for(const id of ['habitat','subject','controls','optics','cortex','circuits','model'])assert.equal(await page.locator('#window-'+id).count(),1);
 assert.match(await page.locator('#movement-note').textContent(),/bounded hover/);
 assert.match(await page.locator('#movement-note').textContent(),/BANC does not control/);
 report.checks.push('Default reference selects brake-hover despite stale local storage; one fly; original seven UI windows and controls retained; learned control explicitly labeled');
 report.start=await read();assert.equal(report.start.kind,'brake-hover');assert(report.start.renderedCalls>0);
 await page.screenshot({path:'reports/flybody-bounded-ui-start.png',timeout:10000});
 await page.evaluate(()=>{
  window.__boundedUITestTrace=[];
  window.__boundedUITestTimer=setInterval(()=>{
   const w=window.heaven.bodyWorld,f=window.heaven.state.flies[0],sample=w.controller.sample();
   window.__boundedUITestTrace.push({bodyTime:w.time,neuralTimeMs:window.heaven.state.time_ms,worldPosition:[f.x,f.y,f.z],worldRadius:Math.hypot(f.x,f.z),sample});
  },100);
 });
 const wallStart=performance.now();
 for(const target of [1,2,4,6,8]){
  await page.waitForFunction(t=>window.heaven.bodyWorld.time>=t||window.heaven.state.paused||window.heaven.bodyWorld.complete,target,{timeout:240000});
  const current=await read();assert(current.bodyTime>=target&&!current.complete&&!current.paused,JSON.stringify(current));
  console.log(JSON.stringify({observedSeconds:current.bodyTime,worldRadius:current.worldRadius,referenceErrorCm:current.sample.referenceError}));
  if(target===4)await page.screenshot({path:'reports/flybody-bounded-ui-mid.png',timeout:10000});
 }
 report.end=await read();report.observationWallSeconds=(performance.now()-wallStart)/1000;
 await page.locator('#pause').click();await page.waitForTimeout(200);
 const frozen=await read();await page.waitForTimeout(300);assert.equal((await read()).bodyTime,frozen.bodyTime);
 report.checks.push('Original pause control stops reference physics');
 report.trace=await page.evaluate(()=>{clearInterval(window.__boundedUITestTimer);return window.__boundedUITestTrace;});
 report.maximumWorldRadius=Math.max(...report.trace.map(t=>t.worldRadius),report.start.worldRadius);
 report.maximumReferenceErrorCm=Math.max(...report.trace.map(t=>t.sample.referenceError));
 report.minimumHeightCm=Math.min(...report.trace.map(t=>t.sample.position[2]));
 report.maximumAppliedForce=Math.max(...report.trace.map(t=>t.sample.maximumAppliedForce));
 report.maximumRootActuatorForce=Math.max(...report.trace.map(t=>t.sample.maximumRootActuatorForce));
 assert(report.end.bodyTime>=8&&report.end.neuralTimeMs>report.start.neuralTimeMs);
 assert(report.maximumWorldRadius<61,'Fly must remain within the displayed habitat radius');
 assert(report.maximumReferenceErrorCm<.08&&report.minimumHeightCm>.9);
 assert.equal(report.maximumAppliedForce,0);assert.equal(report.maximumRootActuatorForce,0);
 report.checks.push('At least eight simulated seconds advanced live; fly stayed within habitat; original BANC observer advanced; actual wing forces maintained flight with zero injected root forces');
 await page.screenshot({path:'reports/flybody-bounded-ui-end.png',timeout:10000});
 assert.equal(await page.locator('#error').textContent(),'');assert.deepEqual(errors,[]);
 report.passed=true;
}catch(error){report.passed=false;report.failure=error.stack;process.exitCode=1;}
finally{
 await browser.close();
 await fs.writeFile('reports/flybody-bounded-ui.json',JSON.stringify(report,null,2)+'\n');
 console.log(JSON.stringify({...report,trace:report.trace?.length},null,2));
}
