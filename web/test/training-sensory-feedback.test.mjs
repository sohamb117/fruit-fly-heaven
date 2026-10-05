import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {createTrainingSensoryResources,worldVectorToRoot} from '../training/sensory-feedback.js';
import {SensoryEncoder} from '../sensory-encoder.js';
import {ANTENNA_PRIOR} from '../banc-antenna.js';
const read=async p=>JSON.parse(await fs.readFile(new URL('../../data/prepared/banc888/'+p,import.meta.url)));
const [manifest,io,sensory,groups,projection,bytes]=await Promise.all([
  read('manifest.json'),read('io.json'),read('console/sensory-inputs.json'),read('console/groups.json'),read('console/visual-projections.json'),
  fs.readFile(new URL('../../data/prepared/banc888/ids.bin',import.meta.url))]);
const ids=new BigUint64Array(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength)),base={manifest,io};
function fixture(){
  const body={time:0,data:{qpos:Float64Array.from([0,0,3,1,0,0,0]),qvel:new Float64Array(6)}},
    fly={id:1,bodyTime:0,x:0,y:3,z:0,heading:0,contact:false,feedback:{speed:7,tilt:.4}},
    world={habitat:{odor:()=>0,ceiling:100,fruit:[{kind:'apple',x:20,y:3,z:0,radius:5,remaining:10}]}};
  return {body,fly,world};
}
const options={base,sensory,groups,projection,ids};
test('world velocity is rotated into signed body coordinates',()=>{
  const q=[Math.SQRT1_2,0,0,Math.SQRT1_2],v=worldVectorToRoot([1,0,0],q);
  assert(Math.abs(v[0])<1e-12);assert(Math.abs(v[1]+1)<1e-12);assert.equal(v[2],0);
});
test('absent feedback configuration preserves the existing sensory vector exactly',async()=>{
  const c=fixture(),resources=await createTrainingSensoryResources({...options,config:{bodyBlockMs:2,vision:false}}),actual=resources.create(c);
  const expected=new SensoryEncoder(sensory,groups,c.world.habitat).update(c.fly,null,{vision:false});
  assert.deepEqual(actual.update().ratesHz,expected.ratesHz);assert.equal(actual.summary().frameCount,0);actual.dispose();
});
test('high-resolution retina uses simulation time and only the declared T4/T5 injection boundary',async()=>{
  const c=fixture(),config={bodyBlockMs:2,vision:true,visionFeedback:{schema:1,profile:'compact-retinal-motion-v1',width:256,height:128,frameIntervalMs:20}};
  const resources=await createTrainingSensoryResources({...options,config}),system=resources.create(c);
  const assertActiveEyeSummary=()=>{
    const eye=system.encoder.sample.vision,motion=system.motion.summary;
    assert.equal(eye.leftHz,motion.leftHz??0);assert.equal(eye.rightHz,motion.rightHz??0);
    assert.equal(eye.ready,motion.ready);assert.equal(eye.rateSource,'requested T4/T5 rates');
    assert.equal(eye.luminanceInputEnabled,false);
    assert.equal(eye.luminanceSummary.leftHz,0);assert.equal(eye.luminanceSummary.rightHz,0);
    assert.equal(Object.hasOwn(eye.luminanceSummary,'luminanceSummary'),false);
  };
  system.update();assert.equal(system.lastFrame.pixels.length,2*256*128);assert.equal(system.lastFrame.rgb.length,6*256*128);
  assertActiveEyeSummary();
  const sequence=system.lastFrame.sequence,firstSummary=structuredClone(system.encoder.sample.vision);
  system.update();assert.equal(system.lastFrame.sequence,sequence);assert.deepEqual(system.encoder.sample.vision,firstSummary);
  c.body.time=.01;c.fly.bodyTime=.01;system.update();assert.equal(system.lastFrame.sequence,sequence);
  c.body.time=.02;c.fly.bodyTime=.02;c.body.data.qpos[0]=.03;const encoded=system.update();
  assert.equal(system.lastFrame.sequence,sequence+1);assert.equal(system.encoder.sample.vision.injectionBoundary,'T4/T5 only');
  for(const cell of sensory.vision.receptors){const at=Array.from(encoded.indices).indexOf(cell.index);assert.equal(encoded.ratesHz[at],0);}
  assert(system.motion.mapping.cells.every(c=>/^T[45][abcd]$/.test(c.type)));assert.equal(system.summary().frameCount,2);
  assertActiveEyeSummary();assert(system.encoder.sample.vision.leftHz>0);assert(system.encoder.sample.vision.rightHz>0);
  const activeSummary=structuredClone(system.encoder.sample.vision);system.update();assertActiveEyeSummary();
  assert.deepEqual(system.encoder.sample.vision,activeSummary);system.dispose();
});
test('airflow replaces every legacy antenna rate without changing unrelated inputs',async()=>{
  const c=fixture(),config={bodyBlockMs:2,vision:false,antennaFeedback:{schema:1,windWorldCmPerSecond:[0,0,0],mechanics:ANTENNA_PRIOR}};
  const resources=await createTrainingSensoryResources({...options,config}),system=resources.create(c);
  const legacy=new SensoryEncoder(sensory,groups,c.world.habitat).update(c.fly,null,{vision:false});
  const actual=system.update(),antIds=new Set(sensory.channels.filter(c=>c.key.startsWith('antenna_')).flatMap(c=>c.indices));
  let differences=0;for(let k=0;k<actual.indices.length;k++)if(!antIds.has(actual.indices[k]))assert.equal(actual.ratesHz[k],legacy.ratesHz[k]);else differences+=actual.ratesHz[k]!==legacy.ratesHz[k];
  assert(differences>0);const before=system.antenna.snapshot();system.summary();assert.deepEqual(system.antenna.snapshot(),before);system.dispose();
});
