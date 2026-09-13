// Real-wall-clock observation of one fly. Images are fresh browser screenshots;
// version changes/restarts are recorded, never hidden in the trajectory.
import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'playwright');
const options=Object.fromEntries(process.argv.slice(2).map(x=>x.replace(/^--/,'').split('=')));
const directory=options.output||'reports/observation-60min',count=Number(options.frames||200),duration=Number(options.minutes||60)*60000;
if(count<2||duration<=0)throw new Error('Need at least two real frames and positive duration');
await fs.mkdir(path.join(directory,'frames'),{recursive:true});
const controlPath=path.join(directory,'control.json');
try{await fs.access(controlPath);}catch{await fs.writeFile(controlPath,JSON.stringify({id:'straight-reference-failure',url:'http://127.0.0.1:7842/?controller=flybody-reference&follow=1'}));}
const browser=await chromium.launch({executablePath:process.env.CHROME_PATH||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,args:['--enable-unsafe-webgpu']});
const context=await browser.newContext({viewport:{width:1440,height:1000}}),page=await context.newPage();
await page.addInitScript(()=>{localStorage.setItem('fruit-fly-population','1');localStorage.setItem('fruit-fly-fast-mode','false');localStorage.setItem('fruit-fly-movement-mode','direct');localStorage.setItem('fruit-fly-neural-body-clock','neural');});
const errors=[],frames=[],changes=[];page.on('pageerror',e=>errors.push({at:new Date().toISOString(),message:e.message}));
let activeId=null,start=null;
const sourceFiles=['web/app.js','web/flybody-world.js','web/flybody-physics.js','web/flybody-wings.js','web/flybody-reference-world.js','web/flybody-flight-reference.js','web/banc-world-worker.js','web/banc-ground-sense.js','configs/banc-physiology.json'];
const hashes=async()=>Object.fromEntries(await Promise.all(sourceFiles.map(async file=>[file,createHash('sha256').update(await fs.readFile(file)).digest('hex')])));
async function control(){
 const requested=JSON.parse(await fs.readFile(controlPath,'utf8'));
 if(requested.id===activeId)return;
 if(!/^http:\/\/127\.0\.0\.1:7842\//.test(requested.url))throw new Error('Observer only opens the local authorized simulator');
 activeId=requested.id;changes.push({id:activeId,url:requested.url,at:new Date().toISOString(),sourceSha256:await hashes()});
 await fs.writeFile(path.join(directory,'changes.json'),JSON.stringify(changes,null,2));
 await page.goto(requested.url);
 await page.waitForFunction(()=>window.heaven?.ready||document.querySelector('#error')?.textContent.trim(),null,{timeout:120000});
 if(requested.paused)await page.locator('#pause').click();
 console.log(JSON.stringify({change:activeId,at:new Date().toISOString()}));
}
async function observe(){
 return page.evaluate(()=>{
  const h=window.heaven;
  if(!h)return {ready:false,error:document.querySelector('#error')?.textContent};
  const w=h.bodyWorld,f=h.state.flies[0],b=w.bodies?.get(f.id)||w.controller,d=b?.data;
  const root=d?Array.from(d.qpos.slice(0,7)):null;
  return {ready:h.ready,population:h.state.flies.length,reference:!!w.reference,backend:w.backend,paused:h.state.paused,
   neuralTimeMs:h.state.time_ms,bodyTime:w.time,position:[f.x,f.y,f.z],physicalRoot:root,velocity:[f.vx,f.vy,f.vz],motion:f.motion,
   upZ:root?1-2*(root[4]**2+root[5]**2):null,airborne:f.airborne,onFood:f.contact,
   wingPower:b?.wingPower,contacts:b?.contactCount,legLoads:b?.legLoads?Array.from(b.legLoads):null,
   mouthContact:b?.mouthContact,proboscis:b?.proboscis,pump:b?.pump,internal:f.internal||b?.internal,
   task:f.task,complete:w.complete,terminationReason:b?.terminationReason,actuators:f.actuators,
   rates:f.brain?.motorNeuronRates?Array.from(f.brain.motorNeuronRates):null,
   source:f.controllerSource||'BANC motor neurons',errors:[document.querySelector('#error')?.textContent,document.querySelector('#anatomy-error')?.textContent].filter(Boolean),
   appliedForce:d?Math.max(...Array.from(d.xfrc_applied,Math.abs),...Array.from(d.qfrc_applied,Math.abs)):null,
   bodyFinite:root?.every(Number.isFinite),insideHabitat:Math.hypot(f.x,f.z)<=64};
 });
}
try{
 await control();start=Date.now();
 await fs.writeFile(path.join(directory,'session.json'),JSON.stringify({startedAt:new Date(start).toISOString(),scheduledEnd:new Date(start+duration).toISOString(),durationMs:duration,frames:count,browser:browser.version(),scope:'Real wall-clock iterative visual observation; build changes recorded separately. One fly per run. Not a claim of successful autonomous behavior.'},null,2));
 for(let index=0;index<count;index++){
  const due=start+duration*index/(count-1);
  while(Date.now()<due){await control();await new Promise(resolve=>setTimeout(resolve,Math.min(1000,due-Date.now())));}
  await control();
  const before=Date.now(),state=await observe(),file=`frames/${String(index+1).padStart(3,'0')}.png`;
  await page.screenshot({path:path.join(directory,file),timeout:15000});
  const frame={index:index+1,at:new Date(before).toISOString(),elapsedSeconds:(before-start)/1000,latenessMs:before-due,version:activeId,file,state};
  frames.push(frame);
  await fs.writeFile(path.join(directory,'latest.json'),JSON.stringify(frame,null,2));
  await fs.appendFile(path.join(directory,'frames.jsonl'),JSON.stringify(frame)+'\n');
  if(index%5===0)console.log(JSON.stringify({frame:frame.index,elapsed:frame.elapsedSeconds,version:activeId,bodyTime:state.bodyTime,motion:state.motion,inside:state.insideHabitat}));
 }
 const result={startedAt:new Date(start).toISOString(),endedAt:new Date().toISOString(),actualDurationSeconds:(Date.now()-start)/1000,requestedDurationSeconds:duration/1000,capturedFrames:frames.length,changes,errors,complete:true};
 await fs.writeFile(path.join(directory,'result.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
}finally{await browser.close();}
