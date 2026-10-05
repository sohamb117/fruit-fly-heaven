// One real native fly at a time. Controlled MN events characterize the motor
// bridge; these are not live BANC outputs or a free-flight success claim.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import loadMujoco from '../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {createWasmCore,WasmMuscles} from '../packages/banc-runtime/src/wasm.js';
import {FlyBodyPhysics} from '../web/flybody-physics.js';
import {buildMotorDecoderContract,createMotorDecoder} from '../web/motor-decoder.js';
import {worldComWrench,readInstabilityWarnings} from './motor-wing-calibration-helpers.mjs';

const output=process.argv[2]||'reports/motor-decoder-v1/native.json';
assert(path.resolve(output).startsWith(path.resolve('reports')+path.sep));
const sha=x=>createHash('sha256').update(x).digest('hex');
const files=['web/motor-decoder.js','web/flybody-physics.js','web/flybody-wings.js',
 'web/flybody-wing-event-excitation.js','models/flybody-mujoco.xml','models/flybody-mujoco.json',
 'data/prepared/banc888/io.json','scripts/test-motor-decoder-native.mjs'];
const sourceHashes=Object.fromEntries(await Promise.all(files.map(async file=>[file,sha(await fs.readFile(file))])));
const [mj,core,xml,metadata,io]=await Promise.all([loadMujoco(),createWasmCore(),fs.readFile(files[4],'utf8'),
 fs.readFile(files[5],'utf8').then(JSON.parse),fs.readFile(files[6],'utf8').then(JSON.parse)]);
assert.equal(sha(xml),metadata.xml_sha256);
const contract=buildMotorDecoderContract(io),initial=contract.parameters.map(p=>p.initial);
const wingMappings=io.muscles.filter(m=>['asynchronous_wing','wing_steering_assumption'].includes(m.kind));
const steering=wingMappings.filter(m=>m.kind==='wing_steering_assumption').flatMap(m=>m.indices).sort((a,b)=>a-b);
const power=new Set(wingMappings.filter(m=>m.kind==='asynchronous_wing').flatMap(m=>m.indices));
const b1=side=>wingMappings.find(m=>m.target==='b1_muscle'&&m.joint==='wing_steer_'+side).indices[0];
const selected=b1('left'),other=b1('right');
const coefficient=(index,axis,lag=0,basis=0)=>24+steering.indexOf(index)*27+axis*9+lag*3+basis;
const chosen=coefficient(selected,0),phaseChosen=coefficient(selected,0,0,1);
const trials=[
 {name:'silent',power:false,unit:null,gain:0},
 {name:'power_only',power:true,unit:null,gain:0},
 {name:'positive',power:true,unit:selected,gain:.02},
 {name:'negative',power:true,unit:selected,gain:-.02},
 {name:'wrong_neuron',power:true,unit:other,gain:.02},
 {name:'phase',power:true,unit:selected,gain:.02,phase:true},
 {name:'phase_shifted_events',power:true,unit:selected,gain:.02,phase:true,shift:2},
];
const durationMs=160,measurementStartMs=100,results=[];
for(const trial of trials){
 const vector=initial.slice();vector[trial.phase?phaseChosen:chosen]=trial.gain;
 const model=mj.MjModel.from_xml_string(xml),rootBody=model.jnt_bodyid[0];
 let samples=0,restraintWrites=0,maxActuatorForce=0,maxWingSpeed=0,clipped=0;
 const sum=new Float64Array(6),controlEnergy=new Float64Array(6);
 const proxy=new Proxy(mj,{get(target,key){
  if(key!=='mj_step')return target[key];
  return (m,d)=>{
   // This isolated calibration rig restrains only the root at every physics
   // step. All native joints, actuator torques and aerodynamics still evolve.
   d.qpos.set([0,0,3,1,0,0,0]);d.qvel.fill(0,0,6);restraintWrites++;
   mj.mj_step(m,d);
   if(d.time*1000>=measurementStartMs){
    const wrench=worldComWrench(Array.from(d.qfrc_fluid.slice(0,6)),Array.from(d.xmat.slice(rootBody*9,rootBody*9+9)),
     Array.from(d.xpos.slice(rootBody*3,rootBody*3+3)),Array.from(d.subtree_com.slice(rootBody*3,rootBody*3+3)));
    for(let k=0;k<6;k++)sum[k]+=wrench[k];samples++;
   }
  };
 }});
 const body=new FlyBodyPhysics(proxy,model,metadata,io,n=>new WasmMuscles(core,n),
  {surface:()=>0,odor:()=>0,foodAt:()=>null});
 try{
  body.data.qpos.set([0,0,3,1,0,0,0]);body.data.qvel.fill(0);mj.mj_forward(model,body.data);body.refresh();
  body.enableWingMotorEvents();body.enableMotorDecoder(vector);
  const counts=Array(48).fill(0),ratesHz=Array(48).fill(0),indices=Array.from(contract.indices);
  body.acceptWingMotorEvents({initialized:true,fromTimeMs:null,timeMs:0,indices,counts:counts.slice(),ratesHz,events:[]});
  const frames=[],start=performance.now();
  for(let end=2;end<=durationMs;end+=2){
   const events=[];
   for(let tick=(end-2)*2+1;tick<=end*2;tick++){
    const timeMs=tick/2;
    if(trial.power&&tick%20===1)for(const index of power)events.push({index,timeMs});
    if(trial.unit!==null&&((tick-2*(trial.shift||0))%10+10)%10===1)events.push({index:trial.unit,timeMs});
   }
   events.sort((a,b)=>a.timeMs-b.timeMs||a.index-b.index);
   for(const event of events)counts[indices.indexOf(event.index)]++;
   body.acceptWingMotorEvents({initialized:false,fromTimeMs:end-2,timeMs:end,indices,counts:counts.slice(),ratesHz,events});
   body.step(new Map(),.002);
   assert.equal(body.motorDecoder.timeMs,end);
   assert(body.data.qpos.every(Number.isFinite)&&body.data.qvel.every(Number.isFinite));
   assert(body.data.qfrc_applied.every(v=>v===0)&&body.data.xfrc_applied.every(v=>v===0));
   const applied=body.wings.decodedControls.appliedResidual.flat();
   for(let k=0;k<6;k++)controlEnergy[k]+=applied[k]**2;
   for(const a of body.wings.actuators){
    maxActuatorForce=Math.max(maxActuatorForce,Math.abs(body.data.actuator_force[a.id]));
    clipped+=Number(body.data.ctrl[a.id]===a.range[0]||body.data.ctrl[a.id]===a.range[1]);
   }
   for(const j of body.wings.joints)maxWingSpeed=Math.max(maxWingSpeed,Math.abs(body.data.qvel[j.dof]));
   if(end%10===0)frames.push({timeMs:end,power:Array.from(body.wings.power),appliedResidual:applied,
    wingAngles:body.wings.joints.map(j=>body.data.qpos[j.qpos]),wingActuatorForces:body.wings.actuators.map(a=>body.data.actuator_force[a.id])});
  }
  results.push({...trial,meanWorldCOMWrench:Array.from(sum,v=>v/samples),controlEnergy:Array.from(controlEnergy),
   maxActuatorForce,maxWingSpeed,clippedControlSamples:clipped,restraintWrites,elapsedWallSeconds:(performance.now()-start)/1000,
   warnings:readInstabilityWarnings(body.data.warning,mj.mjtWarning),finalPower:Array.from(body.wings.power),frames});
 }finally{body.dispose();model.delete();}
 console.log(JSON.stringify({trial:trial.name,...Object.fromEntries(['meanWorldCOMWrench','controlEnergy','elapsedWallSeconds'].map(k=>[k,results.at(-1)[k]]))}));
}
const byName=Object.fromEntries(results.map(r=>[r.name,r])),base=byName.power_only.meanWorldCOMWrench;
const delta=r=>r.meanWorldCOMWrench.map((v,i)=>v-base[i]);
const positive=delta(byName.positive),negative=delta(byName.negative);
const gates={
 silentPower:byName.silent.finalPower.every(v=>v===0),
 silentActiveSteering:byName.silent.controlEnergy.every(v=>v===0),
 correctAnatomicalChannel:byName.positive.controlEnergy[0]>0&&byName.positive.controlEnergy.slice(1).every(v=>v===0),
 wrongNeuronHasNoEffect:byName.wrong_neuron.controlEnergy.every(v=>v===0)&&byName.wrong_neuron.meanWorldCOMWrench.every((v,i)=>v===base[i]),
 nonzeroNativeResponse:Math.hypot(...positive)>1e-8,
 opposingNativeResponse:positive.reduce((s,v,i)=>s+v*negative[i],0)<0,
 timingAffectsMechanics:Math.hypot(...byName.phase.meanWorldCOMWrench.map((v,i)=>v-byName.phase_shifted_events.meanWorldCOMWrench[i]))>1e-8,
 noNativeWarnings:results.every(r=>r.warnings.length===0),
};
// Benchmark just the small learned bridge, excluding the neural and body engines.
const decoder=createMotorDecoder(io,initial),input=new Float64Array(48).fill(.3),bench=performance.now();
for(let i=0;i<100000;i++){if(i%5===0)decoder.advance(input);decoder.sample(i*.296);}
const report={schemaVersion:1,scope:'Controlled synthetic MN events through the actual one-fly native body; restrained calibration, no live BANC or free-flight claim.',
 createdAt:new Date().toISOString(),sourceHashes,parameterCount:contract.parameters.length,selectedMotorNeuron:selected,
 units:{wrench:'g cm/s^2 and g cm^2/s^2',time:'ms',residual:'native actuator control'},results,gates,
 decoderMicrosecondsPerSample:(performance.now()-bench)*1000/100000,passed:Object.values(gates).every(Boolean)};
for(const [file,digest]of Object.entries(sourceHashes))assert.equal(sha(await fs.readFile(file)),digest,'Source changed during native assay');
await fs.mkdir(path.dirname(output),{recursive:true});await fs.writeFile(output,JSON.stringify(report,null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({output,passed:report.passed,gates,decoderMicrosecondsPerSample:report.decoderMicrosecondsPerSample}));
if(!report.passed)process.exitCode=1;
