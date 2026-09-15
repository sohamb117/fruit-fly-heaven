import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {installFeedbackRuntime,root,sha} from './flight-feedback-runtime.mjs';
const args=Object.fromEntries(process.argv.slice(2).map(a=>{const m=/^--([a-z-]+)=(.+)$/.exec(a);if(!m)throw new Error('Use --name=value');return [m[1],m[2]];}));
const variant=args.variant??'vision-airflow',name=args.name??variant;
assert(['off','vision','airflow','vision-airflow'].includes(variant));assert(/^[a-z0-9-]+$/.test(name));
const directory=path.resolve(root,args['output-dir']??'reports/sensory-feedback-20260915');
assert(directory.startsWith(path.join(root,'reports')+path.sep),'Results must stay inside reports');
await fs.mkdir(directory,{recursive:true});
const runtime=await installFeedbackRuntime(args.bundle?path.resolve(root,args.bundle):path.join(directory,variant+'.bundle.json'));
const {config,identity}=runtime,parameters=config.parameters.map(p=>p.initial),seed=Number(args.seed??2590888);
const capture=args.capture==='true';
const captureTimes=(args['capture-at-ms']??'500,600').split(',').map(Number);
assert(captureTimes.length>0&&captureTimes.length<=4&&captureTimes.every((x,i)=>Number.isInteger(x)&&x>=500&&x%2===0&&(!i||x>captureTimes[i-1])));
const assayModule=capture?await import('./flight-feedback-assay.mjs'):null;
const snapshots=[],frames=[],observations=[],sensorySamples=[],images=[];
let framesSeen=0,frameStride=1,lastObservedFrame=null;
const recordFrame=frame=>{
  lastObservedFrame=structuredClone(frame);
  if(framesSeen++%frameStride===0)frames.push(lastObservedFrame);
  if(frames.length>199){frames.splice(0,frames.length,...frames.filter((_,i)=>i%2===0));frameStride*=2;}
};
const write=(suffix,x)=>fs.writeFile(path.join(directory,name+suffix),JSON.stringify(x,null,2)+'\n',{flag:'wx'});
await write('.reservation.json',{startedAt:new Date().toISOString(),variant,seed,identity,capture});
const {createTrainingEnvironment}=await import('../web/training/environment.js');
let environment,interrupted=false,lastProgress=0,provenance=null;
process.on('SIGINT',()=>interrupted=true);process.on('SIGTERM',()=>interrupted=true);
const progress=p=>{if(Date.now()-lastProgress>5000){lastProgress=Date.now();console.log(JSON.stringify({event:'progress',variant,...p}));}};
const saveImage=context=>{
  const frame=context.sensoryFeedback?.lastFrame;
  if(!frame||images.some(x=>x.sequence===frame.sequence)||images.length>=3)return;
  if(images.length&&context.body.time<.7+images.length*.15)return;
  images.push({sequence:frame.sequence,bodyTime:frame.bodyTime,width:frame.width,height:frame.height,rgb:Buffer.from(frame.rgb),summary:frame.summary});
};
const captureState=context=>{
  saveImage(context);
  if(capture&&snapshots.length<captureTimes.length&&context.body.time*1000+1e-5>=captureTimes[snapshots.length])
    snapshots.push(assayModule.captureWasmFlightState({...context,provenance:context.provenance??provenance}));
};
const neuralSensorySummary=context=>{
  const populations={vision:context.sensoryFeedback.motion?.indices,antenna:context.sensoryFeedback.antenna?.sample()?.indices,
    ...(context.sensoryFeedback.legs?{legs:context.sensoryFeedback.legs.metadata.cells.filter(c=>c.enabled).map(c=>c.index)}:{}),
    ...(context.encoder.tegula?.config.schema===2?{wingStrain:context.encoder.tegula.config.fields.map(c=>c.index)}:{})},out={};
  for(const [name,indices]of Object.entries(populations))if(indices?.length){
    const state=context.brain.readState(indices,{includeStatistics:false,includeSpikeHistory:false});
    let sum=0,max=0,active=0,current=0;
    for(let k=0;k<indices.length;k++){const rate=state[k*8+4];sum+=rate;max=Math.max(max,rate);active+=state[k*8+3]>0;
      current+=context.brain.core.HEAPF32[context.brain.params/4+context.brain.n*16+indices[k]];}
    out[name]={cells:indices.length,meanActualRateHz:sum/indices.length,maxActualRateHz:max,everSpiked:active,meanAppliedCurrentPa:current/indices.length};
  }
  return out;
};
try{
  environment=await createTrainingEnvironment(config,{onProgress:progress});await environment.ready();
  const started=performance.now();
  const result=await environment.evaluate({seed,stage:'maintained_flight',durationSeconds:5,parameters},{dutyCycle:1,previewHz:1,onProgress:progress,
    checkpoint:async()=>{await new Promise(r=>setImmediate(r));if(interrupted){const e=new Error('Interrupted');e.name='AbortError';throw e;}},
    onInitialState:c=>{assert.equal(c.phase,'scored-release');provenance=c.provenance;captureState(c);},
    onPhysicsStep:c=>{
      if(c.phase!=='scored')return;
      captureState(c);
      observations.push({time:c.body.time,position:Array.from(c.body.data.qpos.slice(0,3)),quaternion:Array.from(c.body.data.qpos.slice(3,7)),
        velocity:Array.from(c.body.data.qvel.slice(0,6)),contacts:c.body.environmentContactCount,power:[c.body.wingDriveLeft,c.body.wingDriveRight]});
      if(Math.round(c.body.time*1000)%20===0)sensorySamples.push({time:c.body.time,vision:c.fly.sensory.vision,antenna:c.fly.sensory.antennaFeedback,
        ...(c.fly.sensory.legProprioception?{legs:c.fly.sensory.legProprioception}:{}),
        ...(c.fly.sensory.wingStrain?{wingStrain:c.fly.sensory.wingStrain}:{}),
        ...(c.fly.sensory.haltereCurrent?{haltere:c.fly.sensory.haltereCurrent}:{}),
        antennaRates:[c.fly.sensory.body.rates.antenna_left,c.fly.sensory.body.rates.antenna_right],neural:neuralSensorySummary(c)});
    },
    onFrame:recordFrame});
  assert.equal(result.cancelled,false);assert.notEqual(result.reason,'simulation_error',result.metrics.error);assert.deepEqual(result.parameters,parameters);
  if(lastObservedFrame&&frames.at(-1)!==lastObservedFrame)frames.push(lastObservedFrame);
  assert(frames.length<=200);
  await runtime.verify();
  const summary={variant,seed,identity,completed:true,wallSeconds:(performance.now()-started)/1000,score:result.return,success:result.success,
    duration:result.simSeconds,bestFlightSeconds:result.metrics.bestFlightSeconds,reason:result.reason,sensory:result.metrics.sensoryFeedback,
    snapshots:snapshots.length,parametersChanged:false,cloudWrites:false,
    recordedFrames:{seen:framesSeen,saved:frames.length,stride:frameStride,includesFinalFrame:true}};
  if(variant==='off'&&seed===2590888){
    const prior=JSON.parse(await fs.readFile(path.join(root,'reports/live-decoder-diagnosis-20260915/fitted-identity-b-2590888.evaluation.json')));
    for(const key of ['return','simSeconds','success','reason'])assert.deepEqual(result[key],prior[key],'Disabled regression: '+key);
    assert.deepEqual(result.metrics.finalObservation,prior.metrics.finalObservation,'Disabled final physical state');
    summary.disabledParity=true;
  }
  await write('.evaluation.json',result);await write('.frames.json',frames);await write('.observations.json',observations);await write('.sensory.json',sensorySamples);
  for(let i=0;i<images.length;i++){
    const image=images[i],n=image.width*image.height;
    for(let side=0;side<2;side++)await fs.writeFile(path.join(directory,`${name}-eye-${i}-${side===0?'left':'right'}.ppm`),
      Buffer.concat([Buffer.from(`P6\n${image.width} ${image.height}\n255\n`),image.rgb.subarray(side*n*3,(side+1)*n*3)]));
  }
  await write('.summary.json',summary);console.log(JSON.stringify({event:'evaluation-complete',...summary}));
  if(capture){
    const {createHaltereCurrentMapper}=await import('../web/banc-haltere.js');
    const haltereBundle=JSON.parse(await fs.readFile(path.join(root,'reports/flight-haltere-live/mechanical/bundle.json')));
    const haltereConfig=JSON.parse(haltereBundle.configText).haltereFeedback;
    const sensory=JSON.parse(await runtime.read('/banc-data/console/sensory-inputs.json'));
    const idBytes=await runtime.read('/banc-data/ids.bin'),preparedIds=new BigUint64Array(idBytes.buffer.slice(idBytes.byteOffset,idBytes.byteOffset+idBytes.byteLength));
    for(let i=0;i<snapshots.length;i++){
      const report=await assayModule.runFlightFeedbackAssay(snapshots[i],{
        haltereMapper:({maxCurrentPa})=>createHaltereCurrentMapper({enabled:true,...haltereConfig,maxCurrentPa,sensoryManifest:sensory,preparedIds}),
        maxCurrentPa:[100,200,400,800],phaseOffsetsRadians:[0,Math.PI],
        checkpoint:async()=>{await new Promise(r=>setImmediate(r));if(interrupted)throw new Error('Assay interrupted');},
        onArm:(arm,settings)=>progress({arm:arm.name,...settings})});
      await write(`.assay-${i}.json`,report);console.log(JSON.stringify({event:'assay-complete',snapshot:i,summary:report.summary}));
    }
    await runtime.verify();
  }
}finally{for(const snapshot of snapshots)snapshot.dispose();environment?.dispose();runtime.dispose();}
