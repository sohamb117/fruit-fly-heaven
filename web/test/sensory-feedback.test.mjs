import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {BodyWorld,createHabitat,jointPose} from '../body-world.js';
import {SensoryEncoder,bodyInputRates,validateSensoryManifest} from '../sensory-encoder.js';
import {createBrainModule} from '../../packages/fly-brain-wasm/dist/index.js';

const data=JSON.parse(await readFile(new URL('../habitat.json',import.meta.url)));
const actual=JSON.parse(await readFile(new URL('../sensory-inputs.json',import.meta.url)));
const motors=JSON.parse(await readFile(new URL('../motor-outputs.json',import.meta.url)));
const habitat=createHabitat(data.fruit),keys=actual.channels.map(c=>c.key);
const fixture={schema_version:1,neuron_count:16,vision:{width:32,height:16,receptors:[
  {index:3,side:'left',u:0,v:0},{index:4,side:'left',u:1,v:1},{index:5,side:'right',u:0,v:0},{index:6,side:'right',u:1,v:1}]},
  channels:keys.map((key,i)=>({key,indices:[7+i]}))};
const groups={odor_left:[0],odor_right:[1],sweet:[2]};
const make=()=>new SensoryEncoder(fixture,groups,habitat);
const world=()=>new BodyWorld(data.fruit,[{id:1,x:50,z:0,heading:Math.PI,brain:{time_ms:10,motor:{}}}]);
const eye=(left=0,right=0,sequence=0,bodyTime=0)=>({sequence,bodyTime,pixels:Uint8Array.from({length:1024},(_,i)=>i<512?left:right)});
const offFood={odor:false,taste:false};
const step=(w,t=1)=>{for(let i=0;i<t*60;i++)w.advance(1/60);};

test('visual columns keep left/right and spatially distinct samples separate',()=>{
  const e=make(),w=world(),frame=eye(0,0);frame.pixels[0]=255;frame.pixels[1023]=128;
  const out=e.update(w.poses()[0],frame,offFood),r=out.ratesHz;
  assert.equal(r[3],20);assert.equal(r[4],2);assert.equal(r[5],2);assert.ok(r[6]>10&&r[6]<12);
  assert.equal(out.sample.vision.sequence,0);assert.equal(out.sample.vision.contrast,0);
});

test('vision and body-sense disconnection zero only their own input channels',()=>{
  const e=make(),w=world(),p=w.poses()[0],f=eye(220,180);
  const all=e.update(p,f),before=all.ratesHz.slice();assert.ok(before.slice(3,7).every(n=>n>0));assert.ok(before.slice(7).some(n=>n>0));
  const blind=e.update(p,f,{vision:false});assert.ok(blind.ratesHz.slice(3,7).every(n=>n===0));assert.deepEqual(blind.ratesHz.slice(7),before.slice(7));assert.deepEqual(blind.ratesHz.slice(0,3),before.slice(0,3));
  const numb=e.update(p,f,{vision:false,bodySense:false});assert.ok(numb.ratesHz.slice(3).every(n=>n===0));
  const restored=e.update(p,f);assert.deepEqual(restored.ratesHz,before);
});

test('contrast responds to image changes, adapts in body time, and holds between samples',()=>{
  const e=make(),p=world().poses()[0];e.update(p,eye(50,50));
  const changed=e.update(p,eye(200,50,1,.05));assert.ok(changed.sample.vision.leftHz>changed.sample.vision.rightHz+10);
  assert.ok(changed.sample.vision.contrast>0);
  assert.equal(e.update(p,eye(200,50,1,.05)),null);
  for(let i=2;i<30;i++)e.update(p,eye(200,50,i,i*.05));
  assert.ok(e.sample.vision.contrast<.001);
  assert.ok(e.ratesHz.every(n=>Number.isFinite(n)&&n>=0&&n<=150));
});

test('body sensing follows realized joint poses, support, turns and landing impact',()=>{
  const w=world(),f=w.flies[0];assert.deepEqual(f.joints,jointPose(f));assert.equal(f.feedback.legs.every(l=>l.speed===0),true);
  f.brain.motor={forward:42,turn_left:20,antenna_left:22};step(w,.3);
  assert.deepEqual(f.joints,jointPose(f));assert.ok(f.feedback.speed>0);assert.ok(f.feedback.yaw!==0);
  assert.ok(f.feedback.legs.some(l=>l.speed>0));assert.ok(f.feedback.antennae[0].speed>0);
  const moving=bodyInputRates(f.feedback);assert.ok(moving.self_motion_left>8);assert.ok(moving.antenna_left>moving.antenna_right);
  w.setMotorCoupling(false);step(w,2);assert.equal(f.feedback.legs.every(l=>l.speed===0),true);assert.equal(f.feedback.speed,0);assert.equal(f.feedback.yaw,0);
  w.setMotorCoupling(true);f.brain.motor={takeoff:42};step(w,.2);assert.ok(f.airborne);assert.ok(f.feedback.legs.every(l=>l.support===0));
  let impact=false;for(let i=0;i<100;i++){w.advance(1/60);impact ||= f.feedback.impact>0;}
  assert.ok(impact);assert.equal(f.airborne,false);assert.doesNotThrow(()=>structuredClone(w.poses()));
});

test('all mapped input IDs are unique and bounded; invalid mappings fail closed',()=>{
  assert.doesNotThrow(()=>validateSensoryManifest(actual,138639));
  assert.equal(actual.vision.receptors.length,6244);assert.equal(actual.channels.length,8);
  for(const r of actual.vision.receptors)assert.match(r.root_id,/^\d+$/);
  assert.throws(()=>validateSensoryManifest(actual,1),/match/);
  const duplicate=structuredClone(fixture);duplicate.vision.receptors[1].index=3;
  assert.throws(()=>new SensoryEncoder(duplicate,groups,habitat),/duplicate/);
  const outputs=new Set(motors.channels.flatMap(c=>c.indices));
  assert.ok([...actual.vision.receptors.map(r=>r.index),...actual.channels.flatMap(c=>c.indices)].every(i=>!outputs.has(i)));
});

for(const precision of ['float64','float32'])test(`encoded senses drive independent WASM spike states in ${precision}`,async()=>{
  const runtime=await createBrainModule({precision});
  const graph=runtime.createConnectome({neuronCount:16,rowOffsets:new Uint32Array(17),targets:new Uint32Array(),weights:new Float32Array()});
  const brains=graph.createPopulation(100,{seed:783});
  try{
    for(let i=0;i<100;i++){
      const e=make(),input=e.update(world().poses()[0],eye(255,100),{...offFood,vision:i%2===0,bodySense:false});
      brains[i].setPoissonInputs(input).step(1000);
      const spikes=brains[i].readActivations({field:'spikeCount'});
      if(i%2)assert.equal(brains[i].totalSpikes,0);
      else{assert.ok(spikes.slice(3,7).some(n=>n>0));assert.ok([...spikes.slice(0,3),...spikes.slice(7)].every(n=>n===0));}
    }
  }finally{brains.forEach(b=>b.dispose());graph.dispose();}
});
