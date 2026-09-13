// End-to-end browser comparison, with actual rendered eyes and inspection.
// PLAYWRIGHT_MODULE=/path/to/playwright/index.mjs node scripts/benchmark-transplant.mjs
import fs from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import os from 'node:os';
import {createHash} from 'node:crypto';
import {sampleValidity,compareSamples} from './benchmark-transplant-metrics.mjs';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'playwright');
const options=Object.fromEntries(process.argv.slice(2).map(s=>s.replace(/^--/,'').split('=')));
const populations=(options.populations||'1,100').split(',').map(Number),modes=(options.modes||'reference,fast').split(',');
const datasets=(options.datasets||'flywire,banc').split(','),seconds=Number(options.seconds||30),trials=Number(options.trials||1);
const output=options.output||'reports/transplant-performance.json';
if(!populations.every(n=>Number.isInteger(n)&&n>0)||!modes.every(m=>['reference','fast'].includes(m))||!datasets.every(d=>['flywire','banc'].includes(d))||!Number.isFinite(seconds)||seconds<=0||!Number.isInteger(trials)||trials<1)throw new Error('Invalid benchmark options');
const report={date:new Date().toISOString(),platform:process.platform,arch:process.arch,cpu:os.cpus()[0].model,ramBytes:os.totalmem(),
  scope:'Whole-product throughput with the common original 3D console, rendered binocular RGB, FlyVis, color, all senses, direct movement, neural body clock and open anatomy. Intended physics differ: original kinematic FlyWire body versus BANC muscles/MuJoCo with synchronized feedback. This is not an equal numerical workload or behavioral-parity test.',
  requested:{populations,modes,datasets,seconds,trials},frameMetric:'frameHz counts requestAnimationFrame callbacks, not guaranteed displayed or distinct scene frames; renderer/anatomy counters are reported separately.',
  memory:'Sum of RSS of Chrome descendant processes, including shared pages counted more than once; not unique physical memory.',samples:[]};
await fs.mkdir('reports',{recursive:true});
report.sourceSha256=Object.fromEntries(await Promise.all(['web/app.js','web/wasm-world-worker.js','web/banc-world-worker.js','web/body-world.js','web/flybody-world.js','web/flybody-physics.js','web/flybody-wings.js','models/flybody-mujoco.xml','models/flybody-mujoco.json','packages/flybody-runtime/package-lock.json','web/brain-view.js','packages/banc-runtime/src/neural.wgsl','packages/banc-runtime/src/webgpu.js','packages/banc-runtime/dist/core.wasm','packages/fly-brain-wasm/dist/core.wasm','configs/banc-physiology.json'].map(async file=>[file,createHash('sha256').update(await fs.readFile(file)).digest('hex')])));
report.graphs={flywire:JSON.parse(await fs.readFile('data/prepared/metadata.json','utf8')).prepared_sha256,banc:JSON.parse(await fs.readFile('data/prepared/banc888/manifest.json','utf8')).files};
const bancManifest=JSON.parse(await fs.readFile('data/prepared/banc888/manifest.json','utf8'));
report.effectiveModelSettings={flywire:{reference:{dtMs:.1,precision:'float64',body:'original kinematic BodyWorld',bodyStepSeconds:1/60},fast:{dtMs:1,precision:'float32',body:'original kinematic BodyWorld',bodyStepSeconds:1/60}},
  banc:{reference:{dtMs:bancManifest.dt_ms,precision:'float32',body:'native MuJoCo',bodyFeedbackBlockMs:2},fast:{dtMs:1,precision:'float32',body:'native MuJoCo',bodyFeedbackBlockMs:2}},
  provenance:'Effective mode settings derived from the captured worker/source configuration; runtime UI controls, brain backend and body backend are recorded at both measurement endpoints.'};
for(const file of ['scripts/benchmark-transplant.mjs','scripts/benchmark-transplant-metrics.mjs','web/simulation-modes.js','web/flybody-habitat-collision.js','web/flybody-contact-environment.js','models/flybody-wing-actuation.json','data/prepared/banc888/console/groups.json'])report.sourceSha256[file]=createHash('sha256').update(await fs.readFile(file)).digest('hex');
for(const file of ['web/flybody-stance.js','web/flybody-leg-actuation.js','web/fruit-camera-occlusion.js','web/banc-ground-sense.js','web/banc-sensory-current.js','web/banc-proboscis.js','web/banc-taste.js','web/flybody-mouth-pose.js','web/flybody-wing-pose.js','web/banc/embodiment.js','web/sensory-encoder.js','models/banc-taste-peg-annotations.json','data/prepared/banc888/console/sensory-inputs.json'])report.sourceSha256[file]=createHash('sha256').update(await fs.readFile(file)).digest('hex');
function memory(){
  const rows=execFileSync('ps',['-axo','pid,ppid,rss'],{encoding:'utf8'}).trim().split('\n').slice(1).map(s=>s.trim().split(/\s+/).map(Number));
  const descendants=new Set([process.pid]);let changed=true;
  while(changed){changed=false;for(const [pid,ppid]of rows)if(descendants.has(ppid)&&!descendants.has(pid)){descendants.add(pid);changed=true;}}
  return rows.reduce((sum,[pid,,rss])=>sum+(descendants.has(pid)&&pid!==process.pid?rss*1024:0),0);
}
for(let trial=1;trial<=trials;trial++)for(const population of populations)for(const mode of modes)for(const dataset of datasets){
  const browser=await chromium.launch({executablePath:process.env.CHROME_PATH||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,args:['--enable-unsafe-webgpu']});
  const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.addInitScript(({population,mode})=>{
    localStorage.setItem('fruit-fly-population',String(population));localStorage.setItem('fruit-fly-fast-mode',String(mode==='fast'));
    localStorage.setItem('fruit-fly-movement-mode','direct');localStorage.setItem('fruit-fly-neural-body-clock','neural');
    window.benchmarkFrames=0;const frame=()=>{window.benchmarkFrames++;requestAnimationFrame(frame);};requestAnimationFrame(frame);
  },{population,mode});
  const boot=Date.now();let sample={trial,population,mode,dataset};
  try{
    await page.goto(`http://127.0.0.1:7842/?dataset=${dataset}`);
    await page.waitForFunction(()=>window.heaven?.ready||document.querySelector('#error')?.textContent.trim(),null,{timeout:180000});
    const startError=await page.locator('#error').textContent();if(startError.trim())throw new Error(startError);
    sample.startupMs=Date.now()-boot;
    await page.locator('#brain-view').click();
    await page.waitForFunction(()=>window.heaven.state.time_ms>=20||document.querySelector('#error')?.textContent.trim(),null,{timeout:180000});
    await page.waitForTimeout(2000);
    const read=()=>page.evaluate(()=>({wall:performance.now(),neural:window.heaven.state.time_ms,frames:window.benchmarkFrames,
      paused:window.heaven.state.paused,population:window.heaven.state.flies.length,
      configuration:{movementMode:window.heaven.bodyWorld.movementMode,movementControl:document.querySelector('#movement-mode').value,bodyClock:document.querySelector('#body-clock').value,
        fast:document.querySelector('#fast-mode').checked,flightEnabled:window.heaven.bodyWorld.flightEnabled,motorCoupling:window.heaven.bodyWorld.motorCoupling,
        sensorySwitches:Object.fromEntries(['odor','taste','vision','body-sense'].map(id=>[id,document.getElementById(id).checked])),
        bodyBackend:window.heaven.bodyWorld.backend||'kinematic',referenceController:!!window.heaven.bodyWorld.reference,
        precisionLabel:document.querySelector('#precision-label').textContent,precisionDescription:document.querySelector('#precision-description').textContent},
      bodies:window.heaven.state.flies.map(f=>({id:f.id,time:f.bodyTime,neuralMs:f.brain.time_ms,position:[f.x,f.y,f.z],eyeSequence:f.sensory?.vision?.sequence})),
      body:(()=>{const h=window.heaven,f=h.state.flies[0];return {time:f.bodyTime,position:[f.x,f.y,f.z],airborne:f.airborne,onFood:f.contact,motion:f.motion,takeoffs:h.bodyWorld.takeoffs,landings:h.bodyWorld.landings};})(),
      rendererFrames:window.heaven.renderer.info.render.frame,backend:window.heaven.state.backend,allocatedBytes:window.heaven.state.heapBytes,
      anatomy:window.heaven.anatomy,
      min:Math.min(...window.heaven.state.flies.map(f=>f.brain.time_ms)),max:Math.max(...window.heaven.state.flies.map(f=>f.brain.time_ms)),
      eyesReady:window.heaven.state.flies.filter(f=>f.sensory?.vision?.ready).length,
      gradedReady:window.heaven.state.flies.filter(f=>f.sensory?.vision?.graded?.ready).length,
      colorReady:window.heaven.state.flies.filter(f=>f.sensory?.vision?.color?.ready).length,
      errors:[document.querySelector('#error').textContent,document.querySelector('#anatomy-error').textContent].filter(Boolean)}));
    const before=await read();let peakRSS=memory();
    for(let elapsed=0;elapsed<seconds;elapsed+=5){await page.waitForTimeout(Math.min(5,seconds-elapsed)*1000);peakRSS=Math.max(peakRSS,memory());}
    const after=await read();sample={...sample,before,after,wallMs:after.wall-before.wall,neuralMs:after.neural-before.neural,
      cohortSpeed:(after.neural-before.neural)/(after.wall-before.wall),frameHz:(after.frames-before.frames)*1000/(after.wall-before.wall),peakRSS,errors:[...errors,...after.errors]};
    sample.validityIssues=sampleValidity(sample,{seconds});sample.valid=sample.validityIssues.length===0;
    // Image capture is supplementary evidence. A capture timeout must not
    // discard completed timing, sensory and runtime measurements.
    try{await page.screenshot({path:`/tmp/transplant-${dataset}-${population}-${mode}.png`,timeout:5000});}
    catch(error){sample.screenshotWarning=error.message;}
  }catch(error){sample={...sample,valid:false,error:error.message,errors};}
  finally{await browser.close();}
  report.samples.push(sample);await fs.writeFile(output,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(sample));
}
report.sourceHashesUnchangedAtEnd=true;report.changedSourceFiles=[];
for(const [file,digest]of Object.entries(report.sourceSha256))if(createHash('sha256').update(await fs.readFile(file)).digest('hex')!==digest){report.sourceHashesUnchangedAtEnd=false;report.changedSourceFiles.push(file);}
report.comparisons=populations.flatMap(population=>modes.map(mode=>compareSamples(report.samples,{population,mode,trials})));
if(!report.sourceHashesUnchangedAtEnd)for(const comparison of report.comparisons){comparison.eligible=false;comparison.withinOneOrderOfMagnitude=null;comparison.everyPairedTrialWithinOneOrderOfMagnitude=null;comparison.slowdown=null;comparison.sourceChanged=true;}
await fs.writeFile(output,JSON.stringify(report,null,2)+'\n');
