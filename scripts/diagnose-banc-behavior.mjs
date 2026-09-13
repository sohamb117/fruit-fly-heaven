// Observe the actual neural/body loop in the restored console, without commands
// to neurons or actuators. This is a behavior diagnostic, not a success test.
import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'playwright');
const args=Object.fromEntries(process.argv.slice(2).map(s=>s.replace(/^--/,'').split('=')));
const targetMs=Number(args.ms||2000),population=args.population===undefined?null:Number(args.population),disconnectMs=Number(args['disconnect-ms']||0);
const frameDir=args['frames-dir'],wallLimitMs=Number(args['wall-ms']||240000),intervalMs=Number(args['interval-ms']||2000);
const initialCondition=args.start||'original';
if(!['original','off-food'].includes(initialCondition))throw new Error('Unknown initial condition');
if(frameDir)await fs.mkdir(frameDir,{recursive:true});
const browser=await chromium.launch({executablePath:process.env.CHROME_PATH||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,args:['--enable-unsafe-webgpu']});
const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
page.on('pageerror',e=>errors.push(e.message));
const report={date:new Date().toISOString(),population,mode:args.mode||'reference',targetMs,disconnectMs,initialCondition,errors,samples:[],
  scope:'Original 3D console, rendered eyes, direct BANC motor neurons, neural body clock. Initial pose is recorded explicitly; the original on-fruit pose cannot validate food localization or approach.'};
report.sourceSha256=Object.fromEntries(await Promise.all(['web/app.js','web/flybody-world.js','web/flybody-physics.js','web/flybody-wings.js','web/flybody-stance.js','web/banc-ground-sense.js','web/sensory-encoder.js','web/banc/embodiment.js','web/banc-world-worker.js','packages/banc-runtime/src/webgpu.js','packages/banc-runtime/src/neural.wgsl','packages/banc-runtime/src/wasm.js','models/flybody-mujoco.json','models/flybody-mujoco.xml','configs/banc-physiology.json','data/prepared/banc888/console/sensory-inputs.json'].map(async name=>[name,createHash('sha256').update(await fs.readFile(name)).digest('hex')])));
for(const name of ['web/body-world.js','web/flybody-habitat-collision.js','web/flybody-contact-environment.js','web/flybody-leg-actuation.js','web/banc-taste.js','models/banc-taste-peg-annotations.json','models/flybody-wing-actuation.json','data/prepared/banc888/console/groups.json'])report.sourceSha256[name]=createHash('sha256').update(await fs.readFile(name)).digest('hex');
try{
  if(initialCondition==='off-food')await page.route('http://127.0.0.1:7842/habitat.json',async route=>{
    const response=await route.fetch(),habitat=await response.json();
    habitat.flies[0]={...habitat.flies[0],x:47,z:-15,heading:Math.PI/2,contact:false};
    report.initialPose=habitat.flies[0];
    await route.fulfill({response,json:habitat});
  });
  await page.addInitScript(({population,mode})=>{
    if(population!==null)localStorage.setItem('fruit-fly-population',String(population));
    localStorage.setItem('fruit-fly-fast-mode',String(mode==='fast'));
    localStorage.setItem('fruit-fly-movement-mode','direct');
    localStorage.setItem('fruit-fly-neural-body-clock','neural');
  },{population,mode:report.mode});
  await page.goto('http://127.0.0.1:7842/');
  await page.waitForFunction(()=>window.heaven?.ready||document.querySelector('#error')?.textContent,null,{timeout:120000});
  if(await page.locator('#error').textContent())throw new Error(await page.locator('#error').textContent());
  report.actualPopulation=await page.evaluate(()=>window.heaven.state.flies.length);
  if(report.actualPopulation!==(population??1))throw new Error('Unexpected population default');
  if(args['flight-trace']==='true')await page.evaluate(()=>{
    const h=window.heaven,w=h.bodyWorld,b=w.bodies.get(h.state.flies[0].id),refresh=b.refresh.bind(b),weight=b.metadata.mass_g*981;
    const geomNames=Array.from({length:w.model.ngeom},(_,i)=>w.mj.mj_id2name(w.model,5,i));
    window.flightTrace=[];let last=-1;
    b.refresh=()=>{
      refresh();if(b.time===last)return;last=b.time;
      const contacts=[],vector=b.data.ncon?b.data.contact:null;
      try{for(let i=0;i<b.data.ncon;i++){
        const c=vector.get(i);
        try{contacts.push({geoms:Array.from(c.geom,id=>geomNames[id]),distance:c.dist});}finally{c.delete();}
      }}finally{vector?.delete();}
      window.flightTrace.push({t:b.time,position:[b.x,b.y,b.z],velocity:[b.vx,b.vy,b.vz],
        tilt:Math.acos(Math.max(-1,Math.min(1,1-2*(b.quaternion[1]**2+b.quaternion[2]**2)))),
        angularSpeed:Math.hypot(...b.data.qvel.slice(3,6)),wings:[b.wingPowerLeft,b.wingPowerRight],
        wingDrive:[b.wingDriveLeft,b.wingDriveRight],wingDeployment:Array.from(b.wings.deployment),
        aerodynamicForceBodyWeights:Array.from(b.data.qfrc_fluid.slice(0,3),x=>x/weight),
        constraintForceBodyWeights:Array.from(b.data.qfrc_constraint.slice(0,3),x=>x/weight),
        legLoads:Array.from(b.legLoads),airborne:b.airborne,contacts});
    };
  });
  if(args.follow==='true')await page.locator('#follow').click();
  const start=Date.now();let disconnectedAt=null,lastFrameMs=-Infinity;
  while(Date.now()-start<wallLimitMs){
    const sample=await page.evaluate(()=>{
      const h=window.heaven,f=h.state.flies[0],b=h.bodyWorld.bodies.get(f.id),rates=f.brain.motorNeuronRates||[];
      const joints=b.metadata?.joints||b.model.active_joints;
      const muscleRates=b.mappings.map(m=>m.indices.reduce((s,id)=>s+(rates[h.bodyWorld.io.motor_neurons.findIndex(c=>c.index===id)]||0),0)/m.indices.length);
      return {wall:performance.now(),neuralMs:f.brain.time_ms,bodySeconds:f.bodyTime,position:[f.x,f.y,f.z],heading:f.heading,speed:f.velocity,
        atCeiling:f.y>=h.bodyWorld.habitat.ceiling-.01,atWall:Math.hypot(f.x,f.z)>=60.9,
        motion:f.motion,airborne:f.airborne,foodContact:b.onFood,mouthContact:b.mouthContact,proboscis:b.proboscis,pump:b.pump,wingPower:b.wingPower,
        spikes:f.brain.spikes,activeEver:f.brain.active_ever,motor:f.brain.motor,motorRates:rates,muscleRates,
        muscleActivation:Array.from(b.muscleState).filter((_,i)=>i%3===0),jointState:Array.from(b.jointState),
        muscleFatigue:Array.from(b.muscleState).filter((_,i)=>i%3===1),muscleForce:Array.from(b.muscleState).filter((_,i)=>i%3===2),
        jointLimits:joints.map((j,i)=>Math.abs(b.jointState[i*2]-(b.metadata?j.range[0]:Math.max(j.range[0],j.neutral-.35)))<1e-5||Math.abs(b.jointState[i*2]-(b.metadata?j.range[1]:Math.min(j.range[1],j.neutral+.35)))<1e-5),
        physics:h.bodyWorld.backend||'reduced',quaternion:b.quaternion,contacts:b.contactCount,actuatorControls:b.data?Array.from(b.data.ctrl):null,
        physicsPosition:f.physicsPosition,groundFeedback:f.feedback,legLoads:Array.from(b.legLoads||[]),angularVelocity:b.data?Array.from(b.data.qvel.slice(3,6)):null,
        senses:f.senses,bodySense:f.sensory?.body,internal:{...b.internal},events:b.monitor.events,error:document.querySelector('#error').textContent};
    });
    sample.coupled=disconnectedAt===null;report.samples.push(sample);
    if(frameDir&&(sample.neuralMs-lastFrameMs>=Number(args['frame-ms']||500)||sample.neuralMs===0)){
      sample.frame=`${frameDir}/${String(Math.round(sample.neuralMs)).padStart(6,'0')}.png`;
      await page.screenshot({path:sample.frame});lastFrameMs=sample.neuralMs;
    }
    console.log(JSON.stringify({wallSeconds:(Date.now()-start)/1000,neuralMs:sample.neuralMs,body:sample.bodySeconds,position:sample.position,speed:sample.speed,motion:sample.motion,coupled:sample.coupled,atLimit:sample.jointLimits.filter(Boolean).length,events:sample.events.map(e=>e.stage)}));
    if(sample.error)throw new Error(sample.error);
    if(disconnectedAt!==null){if(sample.neuralMs>=disconnectedAt+disconnectMs)break;}
    else if(sample.neuralMs>=targetMs){
      if(!disconnectMs)break;
      disconnectedAt=sample.neuralMs;report.disconnectedAtMs=disconnectedAt;
      await page.locator('#motor-coupling').uncheck();
    }
    await page.waitForTimeout(intervalMs);
  }
  report.reachedTarget=report.samples.at(-1).neuralMs>=(disconnectedAt===null?targetMs:disconnectedAt+disconnectMs);
  report.browser=browser.version();
  if(args['flight-trace']==='true')report.flightTrace=await page.evaluate(()=>window.flightTrace);
  await page.screenshot({path:args.screenshot||'/tmp/banc-behavior.png'});
}catch(e){report.failure=e.stack;process.exitCode=1;}
finally{
  report.changedSourceFiles=[];
  for(const [name,digest]of Object.entries(report.sourceSha256))if(createHash('sha256').update(await fs.readFile(name)).digest('hex')!==digest)report.changedSourceFiles.push(name);
  report.sourceHashesUnchangedAtEnd=report.changedSourceFiles.length===0;
  await browser.close();await fs.writeFile(args.output||'reports/banc-behavior-diagnostic.json',JSON.stringify(report,null,2)+'\n');
}
