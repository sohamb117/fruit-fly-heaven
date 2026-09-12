// Full-data regression: real WASM readouts drive the restored controller.
// Requires the prepared connectome; no added motor stimulation or graph edits.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createBrainModule} from '../../packages/fly-brain-wasm/dist/index.js';
import {BodyWorld} from '../body-world.js';
import {SensoryEncoder} from '../sensory-encoder.js';
import {SIMULATION_MODES} from '../simulation-modes.js';

const root=new URL('../../',import.meta.url),json=path=>JSON.parse(fs.readFileSync(new URL(path,root)));
const available=fs.existsSync(new URL('data/prepared/indptr.bin',root));
const read=(name,Type)=>{const b=fs.readFileSync(new URL('data/prepared/'+name,root));return new Type(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));};

for(const [name,mode]of Object.entries(SIMULATION_MODES))test(`full ${name} WASM brain moves through original behavior readouts`,{skip:available?false:'Prepare the local connectome dataset first'},async t=>{
  const data=json('web/habitat.json'),groups=json('data/prepared/groups.json'),motor=json('web/motor-outputs.json'),sensory=json('web/sensory-inputs.json');
  const runtime=await createBrainModule({precision:mode.precision});
  const graph=runtime.createConnectome({neuronCount:motor.neuron_count,rowOffsets:read('indptr.bin',Uint32Array),targets:read('targets.bin',Uint32Array),weights:read('weights.bin',Float32Array)});
  const brain=graph.createBrain({...mode.parameters,seed:20360915});
  try{
    const world=new BodyWorld(data.fruit,[structuredClone(data.flies[0])],{movementMode:'behavior'});
    // Shadow body receives exactly the same readouts, isolating the decoder.
    const direct=new BodyWorld(data.fruit,[structuredClone(data.flies[0])],{movementMode:'direct'});
    const fly=world.flies[0],start={x:fly.x,z:fly.z},encoder=new SensoryEncoder(sensory,groups,world.habitat);
    const outputIds=new Set(motor.channels.flatMap(c=>c.indices));
    assert.ok([...encoder.indices].every(i=>!outputIds.has(i)));
    brain.setRefractoryPeriod(Uint32Array.from([...groups.odor_left,...groups.odor_right,...groups.sweet]),0);
    const names=['walk_hz','left_hz','right_hz','feed_hz','antenna_hz'];
    const readGroups=[groups.walk,groups.steer_left,groups.steer_right,groups.feed,groups.antenna,...motor.channels.map(c=>c.indices)];
    const ids=Uint32Array.from(readGroups.flat()),rates=new Float64Array(readGroups.length);
    let previous=new Float64Array(ids.length),distance=0,directDistance=0;
    for(let ms=0;ms<600;ms+=2){
      // Uniform light is a controlled test image; the browser uses habitat renders.
      const frame={sequence:Math.floor(ms/50),bodyTime:Math.floor(ms/50)*.05,pixels:new Uint8Array(1024).fill(195)};
      const input=encoder.update(world.poses()[0],frame);if(input)brain.setPoissonInputs(input);
      brain.step(2);
      const counts=brain.readActivations({field:'spikeCount',indices:ids}),stats={time_ms:brain.timeMs,motor:{}};
      let offset=0;
      for(let k=0;k<readGroups.length;k++){
        let delta=0;for(let j=0;j<readGroups[k].length;j++,offset++)delta+=counts[offset]-previous[offset];
        rates[k]+=(delta*500/readGroups[k].length-rates[k])*(1-Math.exp(-2/100));
        if(k<names.length)stats[names[k]]=rates[k];else stats.motor[motor.channels[k-names.length].key]=rates[k];
      }
      previous=counts;fly.brain=stats;direct.flies[0].brain=stats;
      const x=fly.x,z=fly.z,dx=direct.flies[0].x,dz=direct.flies[0].z;
      world.advance(.002);direct.advance(.002);
      distance+=Math.hypot(fly.x-x,fly.z-z);directDistance+=Math.hypot(direct.flies[0].x-dx,direct.flies[0].z-dz);
    }
    assert.ok(brain.totalSpikes>0);assert.ok(fly.brain.left_hz+fly.brain.right_hz>0);
    assert.ok(distance>.05);assert.ok(distance>directDistance+.05);
    assert.ok(Math.hypot(fly.x-start.x,fly.z-start.z)>.05);
    assert.ok(fly.feedback.speed>0&&fly.feedback.legs.some(l=>l.speed>0));
    const neural=brain.readActivations(),time=brain.timeMs;
    world.setMovementMode('direct');world.setMovementMode('behavior');
    assert.equal(brain.timeMs,time);assert.deepEqual(brain.readActivations(),neural);
    t.diagnostic(JSON.stringify({neurons:graph.neuronCount,neuralMs:time,walkingHz:fly.brain.walk_hz,leftHz:fly.brain.left_hz,rightHz:fly.brain.right_hz,feedingHz:fly.brain.feed_hz,behaviorDistance:distance,directDistance}));
  }finally{brain.dispose();graph.dispose();}
});
