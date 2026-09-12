// Full FlyWire spiking brain plus a separate trained, graded visual front end.
import {createBrainModule,loadConnectome} from '/engine/index.js';
import {createVisionModule} from '/vision-engine/index.js';
import {createColorModule,loadFlyColorModel} from '/color-engine/index.js';
import {ColorVision,validateColorMapping} from './color-vision.js';
import {compileVisualModel,compileVisualProjection,GradedVision} from './graded-vision.js';
import {SIMULATION_MODES} from './simulation-modes.js';
import {createHabitat} from './body-world.js';
import {SensoryEncoder,validateSensoryManifest} from './sensory-encoder.js';
import {CircuitDiagnostics,validateCircuitProbe} from './circuit-diagnostics.js';
let brains=[],fruit=[],groups={},paused=false,selected=1,odorOn=true,tasteOn=true,selectedNeuron=0;
let visionOn=true,bodySenseOn=true,latestEyes=new Map();
let module,graph,indices,readIds,readGroups,alive=true,mode=SIMULATION_MODES.reference;
let environment,latestPoses=[],byId=new Map();
let motorChannels=[];
let circuitProbe,diagnostics=null,diagnosticFly=null;
let visualRuntime,visualModel,visualCompiled,visualProjection;
let colorMapper;
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
function applyPoses(){for(const pose of latestPoses){const item=byId.get(pose.id);if(item)Object.assign(item.fly,pose);}}
async function init(args){
  if(!Object.hasOwn(SIMULATION_MODES,args.mode))throw new Error('Unknown simulation mode');
  mode=SIMULATION_MODES[args.mode];
  paused=args.paused??false;selected=args.selected??1;odorOn=args.odor??true;tasteOn=args.taste??true;
  visionOn=args.vision??true;bodySenseOn=args.bodySense??true;
  selectedNeuron=args.selectedNeuron??args.groups.steer_left[0];
  self.postMessage({type:'progress',message:'Loading WASM and measured connectivity'});
  module=await createBrainModule({precision:mode.precision});
  const data=await loadConnectome(new URL('/connectome/',location.href));graph=module.createConnectome(data);
  if(args.visualMapping.neuron_count!==graph.neuronCount||args.visualMapping.ids_sha256!==data.metadata.prepared_sha256['ids.bin'])throw new Error('Visual projection does not match the connectome');
  visualRuntime=await createVisionModule();visualCompiled=compileVisualModel(args.visualModel);
  visualModel=visualRuntime.createModel(visualCompiled.csr);visualProjection=compileVisualProjection(visualCompiled,args.visualMapping);
  validateColorMapping(args.colorMapping,graph.neuronCount);
  const colorModel=await loadFlyColorModel();
  if(args.colorMapping.ids_sha256!==data.metadata.prepared_sha256['ids.bin']||args.colorMapping.lut_sha256!==colorModel.metadata.lut_sha256)throw new Error('Color inputs do not match the model/connectome');
  colorMapper=(await createColorModule()).createMapper(colorModel);
  groups=args.groups;fruit=args.fruit;environment=createHabitat(fruit);
  validateSensoryManifest(args.sensoryInputs,graph.neuronCount);
  circuitProbe=args.circuitProbe;validateCircuitProbe(circuitProbe,graph.neuronCount);
  indices=Uint32Array.from([...groups.odor_left,...groups.odor_right,...groups.sweet]);
  motorChannels=args.motorOutputs.channels;
  if(args.motorOutputs.neuron_count!==graph.neuronCount)throw new Error('Motor outputs do not match this connectome');
  if(motorChannels.some(c=>!c.indices.length||c.indices.some(i=>!Number.isInteger(i)||i<0||i>=graph.neuronCount)))throw new Error('Invalid motor output neuron indices');
  readGroups=[groups.odor_left,groups.odor_right,groups.sweet,groups.walk,groups.steer_left,groups.steer_right,groups.feed,groups.antenna,...motorChannels.map(c=>c.indices)];
  readIds=Uint32Array.from(readGroups.flat());
  const population=graph.createPopulation(args.flies.length,{...mode.parameters,seed:20260912+args.flies[0].id*100003});
  brains=population.map((brain,k)=>{
    brain.setRefractoryPeriod(indices,0);
    const encoder=new SensoryEncoder(args.sensoryInputs,groups,environment,{visualMapping:args.visualMapping,colorMapping:args.colorMapping});
    const graded=new GradedVision(visualModel,visualCompiled,args.visualMapping,visualProjection);
    const color=new ColorVision(colorMapper,args.colorMapping);
    // Preserve the source model's zero-refractory convention only for its food
    // inputs. New visual and body inputs retain the normal LIF refractory period.
    return {brain,encoder,graded,color,fly:{...args.flies[k],brain:{time_ms:0,spikes:0,active_ever:0,motor:{}},senses:[0,0,0]},previous:new Float64Array(readIds.length),rates:new Float64Array(readGroups.length)};
  });
  byId=new Map(brains.map(item=>[item.fly.id,item]));applyPoses();
  self.postMessage({type:'ready',count:brains.length,heapBytes:module.allocatedHeapBytes});
  run();
}
async function run(){
  const names=['odor_left_hz','odor_right_hz','sweet_hz','walk_hz','left_hz','right_hz','feed_hz','antenna_hz'];
  const durationMs=2,alpha=1-Math.exp(-durationMs/100);
  while(alive){
    if(paused){await sleep(20);continue;}
    const start=performance.now();let activation=null,spikes=null,selectedId=null,trace=null,lastSpikeMs=null,circuit=null;
    for(const item of brains){
      const {brain,fly:f,encoder,graded,color}=item,frame=latestEyes.get(f.id);
      if(visionOn&&!frame)continue; // Start with an actual rendered eye sample.
      graded.update(frame,brain.timeMs,visionOn);
      color.update(frame,visionOn);
      const input=encoder.update(f,frame,{odor:odorOn,taste:tasteOn,vision:visionOn,bodySense:bodySenseOn,graded,color});
      if(input){brain.setPoissonInputs(input);item.inputFrame=frame;}
      if(f.id===selected){
        const read=new Uint32Array([selectedNeuron]);trace={neuronIndex:selectedNeuron,dtMs:mode.dtMs,samples:[]};
        for(let tick=0;tick<Math.round(durationMs/mode.dtMs);tick++){
          brain.step(mode.dtMs);
          trace.samples.push([brain.timeMs,brain.readActivations({indices:read})[0],brain.readActivations({field:'spikeCount',indices:read})[0]]);
        }
      }else brain.step(durationMs);
      const counts=brain.readActivations({field:'spikeCount',indices:readIds});
      let offset=0;const stats={time_ms:brain.timeMs,spikes:brain.totalSpikes,active_ever:0,motor:{}};
      for(let k=0;k<readGroups.length;k++){
        let delta=0;for(let j=0;j<readGroups[k].length;j++,offset++)delta+=counts[offset]-item.previous[offset];
        item.rates[k]+=(delta*1000/(durationMs*readGroups[k].length)-item.rates[k])*alpha;
        if(k<8)stats[names[k]]=item.rates[k];else stats.motor[motorChannels[k-8].key]=item.rates[k];
      }
      item.previous=counts;
      Object.assign(f,{senses:encoder.sample.food,sensory:encoder.sample,brain:stats});
      if(f.id===selected){
        activation=brain.readActivations();const hist=brain.readSpikes();
        lastSpikeMs=new Float64Array(brain.neuronCount).fill(-1e30);
        for(let h=0;h<hist.timesMs.length;h++)lastSpikeMs[hist.neuronIndices[h]]=hist.timesMs[h];
        const activity=brain.readActivations({field:'spikeCount'});stats.active_ever=activity.reduce((sum,n)=>sum+(n>0),0);
        if(diagnosticFly!==f.id){
          diagnostics=new CircuitDiagnostics(circuitProbe,brain.parameters);diagnosticFly=f.id;
          if(brain.timeMs>durationMs)diagnostics.reset(brain.timeMs,activity);
        }
        if(diagnostics.ready(brain.timeMs))circuit=diagnostics.update(brain.timeMs,activity,activation,brain.readActivations({field:'synapticDrive',indices:diagnostics.motorIds}));
        spikes=Array.from(hist.timesMs.slice(-256),(t,i)=>[t,hist.neuronIndices[Math.max(0,hist.timesMs.length-256)+i]]);selectedId=f.id;
      }
    }
    self.postMessage({type:'update',flies:brains.map(({fly,inputFrame})=>({id:fly.id,brain:fly.brain,senses:fly.senses,sensory:fly.sensory,...(fly.id===selected?{retina:inputFrame}:{})})),durationMs,wallMs:performance.now()-start,selectedId,activation,spikes,lastSpikeMs,trace,circuit},activation?[activation.buffer,lastSpikeMs.buffer]:[]);
    await sleep(0);
  }
}
self.onmessage=({data})=>{
  if(data.type==='init')init(data).catch(error=>self.postMessage({type:'error',message:error.message}));
  if(data.type==='poses'){latestPoses=data.poses;applyPoses();}
  if(data.type==='eyes')for(const frame of data.frames){
    const previous=latestEyes.get(frame.id);
    if(Number.isInteger(frame.sequence)&&frame.sequence>(previous?.sequence??-1)&&frame.pixels?.length===1024&&Number.isFinite(frame.bodyTime))latestEyes.set(frame.id,frame);
  }
  if(data.type==='control'){
    if('paused'in data)paused=data.paused;if('selected'in data&&selected!==data.selected){selected=data.selected;diagnostics=null;diagnosticFly=null;}
    if('selectedNeuron'in data&&Number.isInteger(data.selectedNeuron)&&data.selectedNeuron>=0&&(!graph||data.selectedNeuron<graph.neuronCount))selectedNeuron=data.selectedNeuron;
    if('odor'in data)odorOn=data.odor;if('taste'in data)tasteOn=data.taste;
    if('vision'in data)visionOn=data.vision;if('bodySense'in data)bodySenseOn=data.bodySense;
    if(paused&&'selected'in data){
      const item=brains.find(item=>item.fly.id===selected);
      if(item){const activation=item.brain.readActivations(),hist=item.brain.readSpikes(),lastSpikeMs=new Float64Array(item.brain.neuronCount).fill(-1e30);
        for(let h=0;h<hist.timesMs.length;h++)lastSpikeMs[hist.neuronIndices[h]]=hist.timesMs[h];
        const start=Math.max(0,hist.timesMs.length-256),spikes=Array.from(hist.timesMs.slice(start),(t,i)=>[t,hist.neuronIndices[start+i]]);
        self.postMessage({type:'update',flies:[{id:selected,brain:item.fly.brain,senses:item.fly.senses,sensory:item.fly.sensory,retina:item.inputFrame}],selectedId:selected,activation,lastSpikeMs,spikes,trace:null},[activation.buffer,lastSpikeMs.buffer]);
      }
    }
  }
};
self.addEventListener('unhandledrejection',event=>self.postMessage({type:'error',message:String(event.reason)}));
