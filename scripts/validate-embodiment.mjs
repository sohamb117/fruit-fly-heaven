// Full-connectome integration validation. Controlled images are test stimuli;
// the running habitat supplies real renders. No motor cells are stimulated.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import {createBrainModule} from '../packages/fly-brain-wasm/dist/index.js';
import {createVisionModule} from '../packages/fly-vision-wasm/dist/index.js';
import {compileVisualModel,compileVisualProjection,GradedVision} from '../web/graded-vision.js';
import {BodyWorld} from '../web/body-world.js';
import {SensoryEncoder} from '../web/sensory-encoder.js';
import {SIMULATION_MODES} from '../web/simulation-modes.js';

const root=new URL('../',import.meta.url),json=p=>JSON.parse(fs.readFileSync(new URL(p,root)));
const groups=json('data/prepared/groups.json'),motors=json('web/motor-outputs.json'),sense=json('web/sensory-inputs.json'),habitat=json('web/habitat.json'),mapping=json('web/visual-projections.json'),metadata=json('data/prepared/metadata.json');
const visualSpec=json('web/visual-model.json'),compiled=compileVisualModel(visualSpec),vr=await createVisionModule(),vm=vr.createModel(compiled.csr),projection=compileVisualProjection(compiled,mapping);
const read=(name,T)=>{const b=fs.readFileSync(new URL('data/prepared/'+name,root));assert.equal(crypto.createHash('sha256').update(b).digest('hex'),metadata.prepared_sha256[name]);return new T(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));};
const report={protocol:'Both full brain kernels. 800 ms paired vision on/off probes with matched seeds, fixed pose, odor/taste/body inputs, and a moving retinal grating. Then 10 neural seconds of behavior-mode embodiment with actual body feedback and a controlled uniform 195/255 retinal image. Vision maps only into annotated visual cells; all original graph edges remain. Flight programs are host assumptions. These tests do not establish physiological fidelity or obstacle avoidance.',
  visualModel:visualSpec.model,visualCheckpointSha256:visualSpec.checkpoint_sha256,graphSha256:metadata.prepared_sha256,results:[]};
const names=['walk_hz','left_hz','right_hz','feed_hz'];
const readGroups=[groups.walk,groups.steer_left,groups.steer_right,groups.feed,...motors.channels.map(c=>c.indices)];
const readIds=Uint32Array.from(readGroups.flat());
const motorIds=new Set(motors.channels.flatMap(c=>c.indices)),visualIds=Uint32Array.from(mapping.cells.filter(c=>/^T[45]/.test(c.type)).map(c=>c.index));
assert.ok(mapping.cells.every(c=>!motorIds.has(c.index)));
const sum=a=>a.reduce((s,n)=>s+n,0);
try{
  for(const [modeName,mode]of Object.entries(SIMULATION_MODES)){
    const bm=await createBrainModule({precision:mode.precision}),graph=bm.createConnectome({neuronCount:138639,rowOffsets:read('indptr.bin',Uint32Array),targets:read('targets.bin',Uint32Array),weights:read('weights.bin',Float32Array)});
    const setup=()=>{
      const brain=graph.createBrain({...mode.parameters,seed:20360915});
      brain.setRefractoryPeriod(Uint32Array.from([...groups.odor_left,...groups.odor_right,...groups.sweet]),0);
      const world=new BodyWorld(habitat.fruit,[structuredClone(habitat.flies[0])],{movementMode:'behavior'});
      return {brain,world,encoder:new SensoryEncoder(sense,groups,world.habitat,{visualMapping:mapping}),graded:new GradedVision(vm,compiled,mapping,projection)};
    };
    const result={mode:modeName,paired:[],flight:null};
    try{
      for(const enabled of [false,true]){
        const {brain,world,encoder,graded}=setup();
        try{
          for(let ms=0;ms<800;ms+=20){
            const frame={sequence:ms/20,bodyTime:ms/1000,pixels:Uint8Array.from({length:1024},(_,i)=>255*(.5+.45*Math.sin(2*Math.PI*((i%32)/31*3-ms/500))))};
            graded.update(frame,ms,enabled);brain.setPoissonInputs(encoder.update(world.poses()[0],frame,{vision:enabled,graded}));brain.step(20);
          }
          const motor=Object.fromEntries(motors.channels.map(c=>[c.key,sum(brain.readActivations({field:'spikeCount',indices:Uint32Array.from(c.indices)}))]));
          result.paired.push({vision:enabled,totalSpikes:brain.totalSpikes,motionPopulationSpikes:sum(brain.readActivations({field:'spikeCount',indices:visualIds})),motorSpikes:motor});
        }finally{brain.dispose();graded.dispose();}
      }
      assert.ok(result.paired[1].motionPopulationSpikes>result.paired[0].motionPopulationSpikes+1000);
      assert.notDeepEqual(result.paired[1].motorSpikes,result.paired[0].motorSpikes,'Visual stimulation must reach downstream motor populations through the full graph');
      const {brain,world,encoder,graded}=setup(),rates=new Float64Array(readGroups.length);
      let previous=new Float64Array(readIds.length),maxHeight=0,airborneSamples=0;const trace=[],started=performance.now();
      try{
        for(let ms=0;ms<10000;ms+=2){
          const fly=world.flies[0],frame={sequence:Math.floor(ms/50),bodyTime:Math.floor(ms/50)*.05,pixels:new Uint8Array(1024).fill(195)};
          graded.update(frame,ms);const input=encoder.update(world.poses()[0],frame,{graded});if(input)brain.setPoissonInputs(input);brain.step(2);
          const counts=brain.readActivations({field:'spikeCount',indices:readIds}),stats={time_ms:brain.timeMs,motor:{}};let offset=0;
          for(let k=0;k<readGroups.length;k++){
            let delta=0;for(let j=0;j<readGroups[k].length;j++,offset++)delta+=counts[offset]-previous[offset];
            rates[k]+=(delta*500/readGroups[k].length-rates[k])*(1-Math.exp(-2/100));
            if(k<4)stats[names[k]]=rates[k];else stats.motor[motors.channels[k-4].key]=rates[k];
          }
          previous=counts;fly.brain=stats;world.advance(.002);maxHeight=Math.max(maxHeight,fly.altitude);airborneSamples+=fly.airborne?1:0;
          assert.ok(Number.isFinite(fly.x+fly.y+fly.z)&&fly.y>=world.habitat.surface(fly.x,fly.z).y-1e-8);
          if(ms%1000===0)trace.push({neuralMs:ms,phase:fly.flightProgram.phase,height:fly.altitude,wingCommand:fly.actuators.wing,wingHz:stats.motor.wing_left,preparation:fly.flightProgram.preparation});
        }
        assert.ok(world.takeoffs>0&&world.landings>0&&maxHeight>3&&airborneSamples>500);
        result.flight={neuralMs:brain.timeMs,bodySeconds:world.time,takeoffs:world.takeoffs,landings:world.landings,maxHeight,airborneSeconds:airborneSamples*.002,wallMs:performance.now()-started,trace};
      }finally{brain.dispose();graded.dispose();}
      report.results.push(result);fs.writeFileSync(new URL('reports/embodiment-validation.json',root),JSON.stringify(report,null,2)+'\n');
      console.log(JSON.stringify({mode:modeName,paired:result.paired,flight:{...result.flight,trace:undefined}}));
    }finally{graph.dispose();}
  }
}finally{vm.dispose();}
