// The BANC physiology engine implements the original 3D console worker contract.
// Eyes, FlyVis, color processing and body feedback are the same host components.
import {loadBancModel,WebGPUBrain,createWasmCore,WasmBrain} from '/banc-engine/src/index.js';
import {createVisionModule} from '/vision-engine/index.js';
import {createColorModule,loadFlyColorModel} from '/color-engine/index.js';
import {ColorVision} from './color-vision.js';
import {compileVisualModel,compileVisualProjection,GradedVision} from './graded-vision.js';
import {createHabitat} from './body-world.js';
import {SensoryEncoder} from './sensory-encoder.js';
import {CircuitDiagnostics} from './circuit-diagnostics.js';
import {createBancSensoryCurrentMapper} from './banc-sensory-current.js';
import {createBancTasteMapper} from './banc-taste.js';

const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const names=['odor_left_hz','odor_right_hz','sweet_hz','walk_hz','left_hz','right_hz','feed_hz','antenna_hz'];
let model,brains=[],byId=new Map(),poses=[],eyes=new Map(),selected=1,selectedNeuron=0;
let paused=false,switches={odor:true,taste:true,vision:true,bodySense:true},probe,motorChannels,readGroups,readIds,allIds;
let diagnostics,diagnosticFly=null,dt=.5,running=false,requestedInspection=false,sensoryCurrent;
let bodySynchronized=false,lastFeedbackMs=0,resolveFeedback=null;

async function init(args){
  bodySynchronized=args.bodySynchronized??false;
  paused=args.paused??false;selected=args.selected??1;selectedNeuron=args.selectedNeuron??0;
  for(const key of Object.keys(switches))switches[key]=args[key]??true;
  model=await loadBancModel();
  if(args.mode==='fast'){
    const scale=model.manifest.dt_ms;
    model={...model,manifest:{...model.manifest,dt_ms:1},edges:model.edges.slice()};
    for(let e=0;e<model.manifest.chemical_edges;e++)model.edges[e*4+3]=Math.max(1,Math.round(model.edges[e*4+3]*scale));
  }
  dt=model.manifest.dt_ms;
  const n=model.manifest.neuron_count;
  if(args.visualMapping.neuron_count!==n||args.visualMapping.ids_sha256!==model.manifest.files['ids.bin'].sha256)throw new Error('BANC visual mapping mismatch');
  const [visionRuntime,colorRuntime,colorModel,tasteSupplement]=await Promise.all([createVisionModule(),createColorModule(),loadFlyColorModel(),
    fetch('/body-model/banc-taste-peg-annotations.json').then(r=>{if(!r.ok)throw new Error('Missing verified external-sugar annotations');return r.json();})]);
  const compiled=compileVisualModel(args.visualModel),visionModel=visionRuntime.createModel(compiled.csr);
  const projection=compileVisualProjection(compiled,args.visualMapping),colorMapper=colorRuntime.createMapper(colorModel);
  if(colorModel.metadata.lut_sha256!==args.colorMapping.lut_sha256)throw new Error('Color LUT mismatch');
  const environment=createHabitat(args.fruit),groups=args.groups;
  if(tasteSupplement.prepared_ids_sha256!==model.manifest.files['ids.bin'].sha256||tasteSupplement.prepared_io_sha256!==model.manifest.files['io.json'].sha256)throw new Error('Taste annotation identity mismatch');
  const tasteMapper=createBancTasteMapper([...model.io.sensory,...tasteSupplement.annotations],groups.sweet);
  if(tasteMapper.coverage.unmapped.some(row=>row.reason==='missing sensory annotation'))throw new Error('Incomplete external-sugar annotations; reload matching console data and supplement');
  probe=args.circuitProbe;motorChannels=args.motorOutputs.channels;
  readGroups=[groups.odor_left,groups.odor_right,groups.sweet,groups.walk,groups.steer_left,groups.steer_right,groups.feed,groups.antenna,...motorChannels.map(c=>c.indices)];
  // Read each cell once even when overlapping motor channels summarize it.
  readIds=Uint32Array.from(new Set([...readGroups.flat(),...model.io.motor_neurons.map(c=>c.index)]));
  const readLookup=new Map(Array.from(readIds,(id,k)=>[id,k]));
  readGroups=readGroups.map(group=>group.map(id=>readLookup.get(id)));
  allIds=Uint32Array.from({length:n},(_,i)=>i);
  let first,core;
  for(const fly of args.flies){
    let brain;
    if(first?.backend==='webgpu')brain=await WebGPUBrain.create(model,{shared:first});
    else if(!first){
      try{brain=await WebGPUBrain.create(model);}catch(error){
        self.postMessage({type:'progress',message:'WebGPU unavailable: '+error.message+'; using WASM physiology'});
        core=await createWasmCore();brain=new WasmBrain(core,model);
      }
    }else brain=new WasmBrain(core,model,{shared:first});
    first??=brain;
    const item={brain,fly:{...fly,brain:{time_ms:0,spikes:0,active_ever:0,motor:{}},senses:[0,0,0]},
      encoder:new SensoryEncoder(args.sensoryInputs,groups,environment,{visualMapping:args.visualMapping,colorMapping:args.colorMapping,tasteMapper}),
      graded:new GradedVision(visionModel,compiled,args.visualMapping,projection),color:new ColorVision(colorMapper,args.colorMapping),
      input:new Float32Array(n),lastInspection:-Infinity,lastSpikeMs:null,previousCounts:null,spikes:[],trace:[],
      readLookup,internal:{hunger:.65,insulin:0,akh:.65}};
    brains.push(item);byId.set(fly.id,item);
    self.postMessage({type:'progress',message:`Preparing BANC ${brains.length}/${args.flies.length} brains`});
  }
  core??=await createWasmCore();
  sensoryCurrent=await createBancSensoryCurrentMapper(core,model,{indices:brains[0].encoder.indices,
    onProfile:(_,i,n)=>self.postMessage({type:'progress',message:`Calibrating sensory physiology ${i+1}/${n}`})});
  applyPoses();running=true;
  self.postMessage({type:'ready',count:brains.length,backend:first.backend,heapBytes:(core?.HEAPU8.byteLength||0)+brains.reduce((s,b)=>s+(b.brain.allocatedBytes||0),0)+(first.shared?.buffers.reduce((s,b)=>s+b.size,0)||0)});
  await run();
}

function applyPoses(){
  for(const pose of poses){const item=byId.get(pose.id);if(item){Object.assign(item.fly,pose);if(pose.internal)item.internal=pose.internal;}}
}

function currents(item,input){
  // Match the isolated configured cell's release rate using the actual WASM
  // dynamics. Network input and internal-state modulation remain additional.
  // This calibrates the interface, not biological stimulus tuning.
  for(let k=0;k<input.indices.length;k++){
    const i=input.indices[k];item.input[i]=sensoryCurrent.current(i,input.ratesHz[k]);
  }
}

async function inspect(item,full){
  const {brain}=item;
  const ids=full?allIds:readIds,state=await brain.readState(ids,{includeSpikeTime:full}),lookup=full?null:item.readLookup,stride=full?9:8;
  const at=(i,k)=>state[(lookup?lookup.get(i):i)*stride+k];
  const stats={time_ms:brain.timeMs,spikes:state.totalSpikes,active_ever:state.activeEver,motor:{},backend:brain.backend,
    motorNeuronRates:model.io.motor_neurons.map(c=>at(c.index,4))};
  for(let k=0;k<readGroups.length;k++){
    const group=readGroups[k],rate=group.reduce((sum,j)=>sum+at(readIds[j],4),0)/group.length;
    if(k<8)stats[names[k]]=rate;else stats.motor[motorChannels[k-8].key]=rate;
  }
  item.fly.brain=stats;
  if(!full)return null;
  const n=model.manifest.neuron_count,activation=new Float64Array(n),counts=new Float32Array(n);
  item.lastSpikeMs??=new Float64Array(n).fill(-1e30);
  let total=0,active=0;
  for(let i=0;i<n;i++){
    activation[i]=state[i*9];counts[i]=state[i*9+3];total+=counts[i];active+=counts[i]>0;
    item.lastSpikeMs[i]=state[i*9+8];
  }
  stats.spikes=total;stats.active_ever=active;item.previousCounts=counts;
  item.spikes=state.spikes.slice(-256);
  if(diagnosticFly!==item.fly.id){diagnostics=new CircuitDiagnostics(probe,{restMv:-60,thresholdMv:-45});diagnosticFly=item.fly.id;if(brain.timeMs>2)diagnostics.reset(brain.timeMs,counts);}
  let circuit=null;
  if(diagnostics.ready(brain.timeMs))circuit=diagnostics.update(brain.timeMs,counts,activation,Float32Array.from(diagnostics.motorIds,i=>state[i*9+7]));
  const trace={neuronIndex:selectedNeuron,dtMs:dt,samples:item.trace.splice(0)};
  return {selectedId:item.fly.id,activation,lastSpikeMs:item.lastSpikeMs.slice(),spikes:item.spikes,circuit,trace};
}

async function run(){
  while(running){
    if(paused){
      if(requestedInspection){
        requestedInspection=false;const item=byId.get(selected);
        if(item){const sample=await inspect(item,true);self.postMessage({type:'update',inspectionOnly:true,flies:[{id:item.fly.id,brain:item.fly.brain,senses:item.fly.senses,sensory:item.fly.sensory,retina:item.inputFrame}],...sample},[sample.activation.buffer,sample.lastSpikeMs.buffer]);}
      }
      await sleep(20);continue;
    }
    // Pause can wake this wait for inspection, but never fabricates a body
    // acknowledgment. Resume rechecks the same completed neural time.
    const completedMs=Math.max(...brains.map(item=>item.brain.timeMs));
    if(bodySynchronized&&lastFeedbackMs+1e-6<completedMs){await new Promise(resolve=>{resolveFeedback=resolve;});continue;}
    // A worker may own several flies. Starting only the ones whose camera
    // arrived would strand them ahead of the population's minimum time.
    if(switches.vision&&brains.some(item=>!eyes.has(item.fly.id))){await sleep(1);continue;}
    const batchSwitches={...switches};
    const start=performance.now();let inspection=null;
    for(const item of brains){
      const {brain,fly,encoder,graded,color}=item,frame=eyes.get(fly.id);
      // Finish the in-flight 2 ms cohort before honoring pause/toggle changes.
      graded.update(frame,brain.timeMs,batchSwitches.vision);color.update(frame,batchSwitches.vision);
      const input=encoder.update(fly,frame,{...batchSwitches,graded,color});
      if(input)currents(item,input);
      // Retain per-timestep selected-neuron traces. Other cells are inspected
      // at bounded wall-clock frequency so readback cannot stall the population.
      if(fly.id===selected){
        const traceNeuron=selectedNeuron,startMs=brain.timeMs,steps=Math.round(2/dt);
        const trace=await brain.step(steps,item.input,item.internal,true,{traceIndex:traceNeuron});
        if(selectedNeuron===traceNeuron&&selected===fly.id)for(let k=0;k<steps;k++)item.trace.push([startMs+(k+1)*dt,trace[k*8],trace[k*8+3]]);
      }else await brain.step(Math.round(2/dt),item.input,item.internal);
      fly.senses=encoder.sample.food;fly.sensory=encoder.sample;item.inputFrame=frame;
      const full=fly.id===selected&&(requestedInspection||performance.now()-item.lastInspection>=100);
      const sample=await inspect(item,full);
      if(sample){inspection=sample;item.lastInspection=performance.now();requestedInspection=false;}
    }
    const message={type:'update',durationMs:2,wallMs:performance.now()-start,flies:brains.map(({fly,inputFrame})=>({id:fly.id,brain:fly.brain,senses:fly.senses,sensory:fly.sensory,...(fly.id===selected?{retina:inputFrame}:{})})),...inspection};
    self.postMessage(message,inspection?[inspection.activation.buffer,inspection.lastSpikeMs.buffer]:[]);
    await sleep(0);
  }
}

self.onmessage=({data})=>{
  if(data.type==='init')init(data).catch(error=>self.postMessage({type:'error',message:error.stack||error.message}));
  if(data.type==='poses'){
    poses=data.poses;applyPoses();
    lastFeedbackMs=Math.max(lastFeedbackMs,data.neuralTimeMs||0);
    if(resolveFeedback&&lastFeedbackMs+1e-6>=Math.max(...brains.map(item=>item.brain.timeMs))){resolveFeedback();resolveFeedback=null;}
  }
  if(data.type==='eyes')for(const frame of data.frames){const old=eyes.get(frame.id);if(Number.isInteger(frame.sequence)&&frame.sequence>(old?.sequence??-1)&&frame.pixels?.length===1024)eyes.set(frame.id,frame);}
  if(data.type==='control'){
    if('bodySynchronized'in data){bodySynchronized=data.bodySynchronized;if(!bodySynchronized&&resolveFeedback){resolveFeedback();resolveFeedback=null;}}
    if('paused'in data){paused=data.paused;if(paused&&resolveFeedback){resolveFeedback();resolveFeedback=null;}}
    if('selected'in data){selected=data.selected;diagnosticFly=null;requestedInspection=true;}
    if('selectedNeuron'in data){selectedNeuron=data.selectedNeuron;brains.forEach(item=>item.trace=[]);requestedInspection=true;}
    for(const key of Object.keys(switches))if(key in data)switches[key]=data[key];
  }
};
self.addEventListener('unhandledrejection',event=>self.postMessage({type:'error',message:String(event.reason)}));
