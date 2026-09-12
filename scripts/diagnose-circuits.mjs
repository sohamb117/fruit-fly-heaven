// Controlled full-connectome probes, separate from the live habitat. No graph edits.
import fs from 'node:fs';
import crypto from 'node:crypto';
import {createBrainModule} from '../packages/fly-brain-wasm/dist/index.js';
import {CircuitDiagnostics} from '../web/circuit-diagnostics.js';
import {decodeMotorOutput} from '../web/body-world.js';

const root=new URL('../',import.meta.url),json=name=>JSON.parse(fs.readFileSync(new URL(name,root)));
const durationMs=Number(process.argv[2]??1000),seedCount=Number(process.argv[3]??3),warmupMs=200;
if(!Number.isInteger(durationMs)||durationMs<300||durationMs%50||!Number.isInteger(seedCount)||seedCount<1||seedCount>20)throw new Error('Usage: node scripts/diagnose-circuits.mjs <duration ms, multiple of 50, >=300> <1–20 seeds>');
const probe=json('web/circuit-probe.json'),sensory=json('web/sensory-inputs.json'),food=json('data/prepared/groups.json');
const meta=json('data/prepared/metadata.json');
const read=(name,T)=>{const b=fs.readFileSync(new URL('data/prepared/'+name,root));
  if(crypto.createHash('sha256').update(b).digest('hex')!==probe.source_sha256['data/prepared/'+name])throw new Error('Probe does not match '+name);
  return new T(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));};
const module=await createBrainModule(),graph=module.createConnectome({neuronCount:probe.neuron_count,rowOffsets:read('indptr.bin',Uint32Array),targets:read('targets.bin',Uint32Array),weights:read('weights.bin',Float32Array)});
read('ids.bin',BigInt64Array);
const foodIds=[...food.odor_left,...food.odor_right,...food.sweet];
const inputIds=[...foodIds,...sensory.vision.receptors.map(r=>r.index),...sensory.channels.flatMap(c=>c.indices)];
const defaults={eye:16,body:true,sugar:150,odor:56};
const conditions=[
  {name:'baseline',...defaults},
  {name:'no_sugar',...defaults,sugar:0},
  {name:'no_odor',...defaults,odor:0},
  {name:'no_body',...defaults,body:false},
  {name:'food_only',...defaults,eye:0,body:false},
  {name:'vision_only',...defaults,body:false,odor:0,sugar:0},
  {name:'bright_stress',...defaults,eye:60},
  {name:'moving_grating',...defaults,grating:true},
  {name:'silence_CB0890',...defaults,silence:['CB0890']},
  {name:'silence_top_forward_inhibitors',...defaults,silence:['CB0890','CB0677','DNge054']},
  {name:'motor_positive_control',...defaults,stimulateMotorHz:80},
];
const report={schema_version:1,engine:'Full graph / Float64 reference WASM',parameters:meta.parameters,
  durationMs,warmupMs,measurementMs:durationMs-warmupMs,seedCount,conditions,
  protocol:'Independent brains and identical per-neuron seed streams. Controlled inputs: bilateral odor 56 Hz, sugar 150 Hz, uniform photoreceptors 16 Hz, self-motion proxy 15 Hz and antennal proxy 2 Hz; other body inputs zero. Input removals and interventions as listed. No rendered eyes, body movement, or live closed-loop feedback in these probes. Moving grating updates a sinusoidal retinal pattern every 50 neural ms through the existing column positions, with 2–60 Hz drive; this is a stress stimulus, not calibrated optics.',
  interpretation:'Inhibitory-source silencing and direct motor stimulation are artificial diagnostic interventions on fresh brains, never applied to the habitat. Weighted presynaptic spike rates rank candidate inputs; they do not measure delivered current, account for refractory losses, or alone establish causation. Sampled voltages are endpoints, not a physiological calibration.',
  graphSha256:meta.prepared_sha256,visualSignAudit:probe.visual_sign_audit,
  probeSha256:crypto.createHash('sha256').update(fs.readFileSync(new URL('web/circuit-probe.json',root))).digest('hex'),results:[]};
const output=new URL('reports/circuit-diagnosis.json',root);
try{
  for(const condition of conditions)for(let trial=0;trial<seedCount;trial++){
    const seed=20360915+trial,brain=graph.createBrain({seed});
    try{
      brain.setRefractoryPeriod(Uint32Array.from(foodIds),0);
      const ids=[...inputIds],rates=[...food.odor_left.map(()=>condition.odor),...food.odor_right.map(()=>condition.odor),...food.sweet.map(()=>condition.sugar),
        ...sensory.vision.receptors.map(()=>condition.eye),...sensory.channels.flatMap(c=>c.indices.map(()=>condition.body?(c.key.startsWith('self_motion')?15:c.key.startsWith('antenna')?2:0):0))];
      if(condition.stimulateMotorHz){ids.push(...probe.channels.flatMap(c=>c.indices));rates.push(...probe.channels.flatMap(c=>c.indices.map(()=>condition.stimulateMotorHz)));}
      const stimulus={indices:Uint32Array.from(ids),ratesHz:Float32Array.from(rates)};
      brain.setPoissonInputs(stimulus);
      const silenced=Object.values(probe.cells).filter(c=>condition.silence?.includes(c.type)).map(c=>c.index);
      if(silenced.length)brain.setCurrentInputs(Uint32Array.from(silenced),new Float32Array(silenced.length).fill(-1000));
      const diagnostic=new CircuitDiagnostics(probe,{windowMs:durationMs-warmupMs}),trace=[],start=performance.now();
      for(let ms=0;ms<durationMs;ms+=50){
        if(ms===warmupMs)diagnostic.reset(ms,brain.readActivations({field:'spikeCount'}));
        if(condition.grating){
          sensory.vision.receptors.forEach((r,k)=>{stimulus.ratesHz[foodIds.length+k]=31+29*Math.sin(2*Math.PI*(r.u*3-ms/500));});
          brain.setPoissonInputs(stimulus);
        }
        brain.step(50);
        trace.push({timeMs:brain.timeMs,voltageMv:Array.from(brain.readActivations({indices:diagnostic.motorIds})),spikeCount:Array.from(brain.readActivations({field:'spikeCount',indices:diagnostic.motorIds}))});
      }
      const counts=brain.readActivations({field:'spikeCount'}),voltage=brain.readActivations();
      const silenceResults=silenced.map(i=>({...probe.cells[i],measuredSpikes:counts[i]-diagnostic.previous[i]}));
      const sample=diagnostic.update(brain.timeMs,counts,voltage,brain.readActivations({field:'synapticDrive',indices:diagnostic.motorIds}));
      const motor=Object.fromEntries(sample.channels.map(c=>[c.key,c.meanHz]));
      const result={condition:condition.name,seed,wallMs:performance.now()-start,totalSpikes:brain.totalSpikes,sample,
        actuatorFromWindowMean:decodeMotorOutput({time_ms:brain.timeMs,motor}),silenced:silenceResults,trace};
      report.results.push(result);fs.writeFileSync(output,JSON.stringify(report,null,2)+'\n');
      console.log(JSON.stringify({condition:condition.name,seed,forwardHz:motor.forward,wingLeftHz:motor.wing_left,wingRightHz:motor.wing_right,motionHz:sample.groups.find(g=>g.key==='motion').meanHz,forwardMv:sample.channels[0].meanMv}));
    }finally{brain.dispose();}
  }
}finally{graph.dispose();}
