// Run deliberately AFTER the active observer/performance experiment finishes.
// Original UI, original live brain; response routing adds observation hooks.
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';

const options=Object.fromEntries(process.argv.slice(2).map(x=>x.replace(/^--/,'').split('=')));
const directory=options.output||'reports/flybody-onset-causal',seconds=Number(options.seconds||.4);
assert(seconds>0&&seconds<=1&&Math.abs(seconds/.002-Math.round(seconds/.002))<1e-8);
await fs.mkdir(directory,{recursive:true});
const hooks=await fs.readFile('scripts/flybody-onset-capture-hooks.mjs','utf8');
const patches={
 '/flybody-physics.js':source=>`import {installOnsetCapture} from '/__onset/hooks.mjs';\n${source}\ninstallOnsetCapture(FlyBodyPhysics);\n`,
 '/flybody-world.js':source=>{
  const marker='const scene=flybodyScene(xml,this.habitat);';
  assert.equal(source.split(marker).length,2,'Expected exactly one scene creation');
  return `import {onsetScene} from '/__onset/hooks.mjs';\n${source.replace(marker,marker+'onsetScene(this,scene,metadata,io);')}`;
 }};
const sourceHashes={},sourceCopies={},pending=[],errors=[];
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'/tmp/fruit-fly-browser-tests/node_modules/playwright/index.mjs');
const browser=await chromium.launch({executablePath:options.chrome||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:options.headless!=='false',args:['--enable-unsafe-webgpu']});
const page=await browser.newPage({viewport:{width:1440,height:1000}});
let report;
try{
 await page.addInitScript(({seconds,fast})=>{
  localStorage.setItem('fruit-fly-population','1');localStorage.setItem('fruit-fly-flight','true');
  localStorage.setItem('fruit-fly-neural-body-clock','neural');localStorage.setItem('fruit-fly-movement-mode','direct');
  localStorage.setItem('fruit-fly-fast-mode',String(fast));
  globalThis.__flybodyOnset={enabled:true,kind:'actual-original-ui-onset',seconds,frames:[]};
 },{seconds,fast:options.fast==='true'});
 await page.route('**/__onset/hooks.mjs',route=>route.fulfill({contentType:'text/javascript',body:hooks}));
 for(const [pathname,patch] of Object.entries(patches))await page.route(`**${pathname}*`,async route=>{
  const response=await route.fetch(),original=await response.text(),instrumented=patch(original);
  sourceHashes[pathname]={original:hash(original),instrumented:hash(instrumented)};
  sourceCopies[pathname]=original;
  await route.fulfill({response,contentType:'text/javascript',body:instrumented});
 });
 page.on('pageerror',error=>errors.push(error.stack||error.message));
 page.on('response',response=>{
  const pathname=new URL(response.url()).pathname;
  if(!/\.(js|mjs|json|xml|wasm)$/.test(pathname)||pathname.startsWith('/banc-data/anatomy/'))return;
  const task=(async()=>{try{
   const bytes=await response.body();
   if(!sourceHashes[pathname])sourceHashes[pathname]={served:hash(bytes)};
   if(/\.(js|mjs)$/.test(pathname)&&!pathname.includes('/vendor/'))sourceCopies[pathname]??=bytes.toString();
  }catch{}})();pending.push(task);
 });
 const base=options.url||'http://127.0.0.1:7842';
 await page.goto(`${base}/?population=1&movement=direct&clock=neural&follow=1`,{waitUntil:'domcontentloaded'});
 await page.waitForFunction(()=>globalThis.__flybodyOnset?.done||window.heaven?.state?.runtimeError,null,{timeout:Number(options.timeout||300000)});
 report=await page.evaluate(async()=>{
  const {exportOnsetCapture}=await import('/__onset/hooks.mjs');
  const h=window.heaven;
  return {...exportOnsetCapture(),ui:{ready:h.ready,population:h.state.flies.length,backend:h.state.backend,
   reference:h.bodyWorld.reference??false,movementMode:h.bodyWorld.movementMode,paused:h.state.paused,
   error:document.getElementById('error')?.textContent?.trim()||'',runtimeError:h.state.runtimeError??null}};
 });
 await page.screenshot({path:path.join(directory,'onset-end.png'),timeout:15000});
 assert(report.done&&report.frames.length===Math.round(seconds/.002));
 assert(report.ui.ready&&report.ui.population===1&&report.ui.reference===false&&report.ui.movementMode==='direct');
 assert.equal(report.initial.native.time,0);assert.equal(report.ui.runtimeError,null);assert.equal(report.ui.error,'');assert.deepEqual(errors,[]);
 report.capturePassed=true;
}catch(error){report??={};report.capturePassed=false;report.failure=error.stack;process.exitCode=1;}
finally{
 await Promise.allSettled(pending);
 report={...report,date:new Date().toISOString(),browser:browser.version(),sourceHashes,errors,
  recorder:{scriptSha256:hash(await fs.readFile('scripts/capture-flybody-onset.mjs')),hooksSha256:hash(hooks),
   instrumentation:'Only route-added observer hooks; original stepping, inputs and dynamics remain unchanged. UI Pause clicked after final captured step.'}};
 await browser.close();
 for(const [pathname,source] of Object.entries(sourceCopies)){
  const destination=path.join(directory,'sources',pathname.slice(1));await fs.mkdir(path.dirname(destination),{recursive:true});await fs.writeFile(destination,source);
 }
 await fs.writeFile(path.join(directory,'capture.json'),JSON.stringify(report)+'\n');
 console.log(JSON.stringify({passed:report.capturePassed,frames:report.frames?.length,seconds:report.frames?.at(-1)?.after.native.time,
  output:path.join(directory,'capture.json'),failure:report.failure},null,2));
}
