// One-fly perturbation: no external sensory input, normal input, then no input.
// This isolates drive; it does not fit parameters or command any actuators.
import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'playwright');
const browser=await chromium.launch({executablePath:process.env.CHROME_PATH||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,args:['--enable-unsafe-webgpu']});
const page=await browser.newPage(),report={date:new Date().toISOString(),population:1,initialCondition:'off-food',phases:[],errors:[],
  scope:'One direct BANC fly, Reference mode, neural body clock. Sensory flags are false from worker initialization; intrinsic graded release and internal-state currents remain active.'};
page.on('pageerror',error=>report.errors.push(error.message));
try{
  report.sourceSha256=Object.fromEntries(await Promise.all(['web/app.js','web/banc-world-worker.js','web/banc/embodiment.js','web/banc-body-world.js','configs/banc-physiology.json','packages/banc-runtime/src/neural.wgsl'].map(async file=>[file,createHash('sha256').update(await fs.readFile(file)).digest('hex')])));
  await page.addInitScript(()=>{
    localStorage.setItem('fruit-fly-fast-mode','false');
    localStorage.setItem('fruit-fly-movement-mode','direct');
    localStorage.setItem('fruit-fly-neural-body-clock','neural');
    localStorage.setItem('fruit-fly-flight','true');
    window.__diagnosticInitializations=[];
    const NativeWorker=window.Worker;
    window.Worker=class extends NativeWorker{
      postMessage(data,...rest){
        if(data.type==='init'){
          data={...data,odor:false,taste:false,vision:false,bodySense:false};
          window.__diagnosticInitializations.push({flies:data.flies.length,odor:data.odor,taste:data.taste,vision:data.vision,bodySense:data.bodySense});
        }
        return super.postMessage(data,...rest);
      }
    };
  });
  await page.route('http://127.0.0.1:7842/habitat.json',async route=>{
    const response=await route.fetch(),habitat=await response.json();
    habitat.flies[0]={...habitat.flies[0],x:47,z:-15,heading:Math.PI/2,contact:false};
    await route.fulfill({response,json:habitat});
  });
  await page.goto('http://127.0.0.1:7842/?population=1');
  await page.waitForFunction(()=>window.heaven?.ready||document.querySelector('#error')?.textContent,null,{timeout:120000});
  if(await page.locator('#error').textContent())throw new Error(await page.locator('#error').textContent());
  report.initializations=await page.evaluate(()=>window.__diagnosticInitializations);
  if(report.initializations.length!==1||report.initializations[0].flies!==1)throw new Error('Expected exactly one fly');
  for(const {name,enabled,duration} of [{name:'sensory-off-from-start',enabled:false,duration:150},{name:'sensory-on',enabled:true,duration:200},{name:'sensory-off-again',enabled:false,duration:250}]){
    // Dispatch even when already unchecked so worker and controls agree.
    await page.evaluate(enabled=>{
      for(const id of ['odor','taste','vision','body-sense']){const el=document.getElementById(id);el.checked=enabled;el.dispatchEvent(new Event('change'));}
    },enabled);
    const phase={name,startMs:await page.evaluate(()=>window.heaven.state.flies[0].brain.time_ms),samples:[]};
    report.phases.push(phase);
    const deadline=Date.now()+90000;
    while(Date.now()<deadline){
      const sample=await page.evaluate(()=>{
        const h=window.heaven,f=h.state.flies[0],b=h.bodyWorld.bodies.get(f.id);
        return {ms:f.brain.time_ms,backend:f.brain.backend,position:[f.x,f.y,f.z],airborne:f.airborne,motor:f.brain.motor,wingPower:b.wingPower,foodIntake:b.internal.ingested,spikes:f.brain.spikes,active:f.brain.active_ever,sensory:f.sensory,error:document.getElementById('error').textContent};
      });
      phase.samples.push(sample);
      console.log(JSON.stringify({phase:name,ms:sample.ms,wingLeftHz:sample.motor?.wing_left,wingRightHz:sample.motor?.wing_right,wingPower:sample.wingPower,airborne:sample.airborne}));
      if(sample.error)throw new Error(sample.error);
      if(sample.ms>=phase.startMs+duration){phase.completed=true;break;}
      await page.waitForTimeout(500);
    }
    if(!phase.completed)throw new Error('Phase timed out: '+name);
  }
}catch(error){report.failure=error.stack;process.exitCode=1;}
finally{await browser.close();await fs.writeFile('reports/banc-wing-drive-perturbation.json',JSON.stringify(report,null,2)+'\n');}
