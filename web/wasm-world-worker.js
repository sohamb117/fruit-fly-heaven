// Example environment adapter. All neuroscience is in the generic WASM package.
import {createBrainModule,loadConnectome} from '/engine/index.js';
import {SIMULATION_MODES} from './simulation-modes.js';
let brains=[],fruit=[],groups={},paused=false,selected=1,odorOn=true,tasteOn=true,selectedNeuron=0;
let module,graph,indices,readIds,readGroups,alive=true,mode=SIMULATION_MODES.reference;
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
function distance(f,x,z){return f.kind==='apple'?Math.hypot(x-f.x,z-f.z):Math.min(...f.path.map(p=>Math.hypot(x-p[0],z-p[1])));}
function habitat(x,z){let y=1.5+.0037*(x*x+z*z),contact=false;for(const f of fruit){const d=distance(f,x,z);if(d<f.radius){const h=f.y+Math.sqrt(f.radius*f.radius-d*d);if(h>=y){y=h;contact=true;}}}return {y,contact};}
function odor(x,z){return Math.min(1,fruit.reduce((n,f)=>n+.3*Math.exp(-Math.max(0,distance(f,x,z)-f.radius)/18),0));}
function inputFor(f){const a=f.heading,x=f.x,z=f.z;return [odorOn?3+65*odor(x+.9*Math.cos(a)-.4*Math.sin(a),z+.9*Math.sin(a)+.4*Math.cos(a)):0,odorOn?3+65*odor(x+.9*Math.cos(a)+.4*Math.sin(a),z+.9*Math.sin(a)-.4*Math.cos(a)):0,tasteOn&&f.contact?150:0];}
async function init(args){
  if(!Object.hasOwn(SIMULATION_MODES,args.mode))throw new Error('Unknown simulation mode');
  mode=SIMULATION_MODES[args.mode];
  paused=args.paused??false;selected=args.selected??1;odorOn=args.odor??true;tasteOn=args.taste??true;
  selectedNeuron=args.selectedNeuron??args.groups.steer_left[0];
  self.postMessage({type:'progress',message:'Loading WASM and measured connectivity'});
  module=await createBrainModule({precision:mode.precision});
  const data=await loadConnectome(new URL('/connectome/',location.href));graph=module.createConnectome(data);
  groups=args.groups;fruit=args.fruit;
  indices=Uint32Array.from([...groups.odor_left,...groups.odor_right,...groups.sweet]);
  readGroups=[groups.odor_left,groups.odor_right,groups.sweet,groups.walk,groups.steer_left,groups.steer_right,groups.feed,groups.antenna];
  readIds=Uint32Array.from(readGroups.flat());
  const population=graph.createPopulation(args.flies.length,{...mode.parameters,seed:20260912+args.flies[0].id*100003});
  brains=population.map((brain,k)=>{
    brain.setRefractoryPeriod(indices,0);
    return {brain,fly:{...args.flies[k],brain:{},senses:[0,0,0]},previous:new Float64Array(readIds.length),rates:new Float64Array(8)};
  });
  self.postMessage({type:'ready',count:brains.length,heapBytes:module.allocatedHeapBytes});
  run();
}
async function run(){
  const names=['odor_left_hz','odor_right_hz','sweet_hz','walk_hz','left_hz','right_hz','feed_hz','antenna_hz'];
  const durationMs=2,alpha=1-Math.exp(-durationMs/100);
  while(alive){
    if(paused){await sleep(20);continue;}
    const start=performance.now();let activation=null,spikes=null,selectedId=null,trace=null,lastSpikeMs=null;
    for(const item of brains){
      const {brain,fly:f}=item,s=inputFor(f),ratesHz=new Float32Array(indices.length);
      ratesHz.fill(s[0],0,groups.odor_left.length);ratesHz.fill(s[1],groups.odor_left.length,groups.odor_left.length+groups.odor_right.length);ratesHz.fill(s[2],groups.odor_left.length+groups.odor_right.length);
      brain.setPoissonInputs({indices,ratesHz});
      if(f.id===selected){
        const read=new Uint32Array([selectedNeuron]);trace={neuronIndex:selectedNeuron,dtMs:mode.dtMs,samples:[]};
        for(let tick=0;tick<Math.round(durationMs/mode.dtMs);tick++){
          brain.step(mode.dtMs);
          trace.samples.push([brain.timeMs,brain.readActivations({indices:read})[0],brain.readActivations({field:'spikeCount',indices:read})[0]]);
        }
      }else brain.step(durationMs);
      const counts=brain.readActivations({field:'spikeCount',indices:readIds});
      let offset=0;const stats={time_ms:brain.timeMs,spikes:brain.totalSpikes,active_ever:0};
      for(let k=0;k<8;k++){
        let delta=0;for(let j=0;j<readGroups[k].length;j++,offset++)delta+=counts[offset]-item.previous[offset];
        item.rates[k]+=(delta*1000/(durationMs*readGroups[k].length)-item.rates[k])*alpha;stats[names[k]]=item.rates[k];
      }
      item.previous=counts;
      const speed=8*Math.tanh((stats.walk_hz+.15*(stats.left_hz+stats.right_hz))/30)/(1+stats.feed_hz/35);
      f.heading+=2.8*Math.tanh((stats.left_hz-stats.right_hz)/40)*durationMs/1000;
      f.x+=Math.cos(f.heading)*speed*durationMs/1000;f.z+=Math.sin(f.heading)*speed*durationMs/1000;
      const r=Math.hypot(f.x,f.z);if(r>62){f.x*=62/r;f.z*=62/r;}
      Object.assign(f,habitat(f.x,f.z),{velocity:speed,senses:s,brain:stats});
      if(f.id===selected){
        activation=brain.readActivations();const hist=brain.readSpikes();
        lastSpikeMs=new Float64Array(brain.neuronCount).fill(-1e30);
        for(let h=0;h<hist.timesMs.length;h++)lastSpikeMs[hist.neuronIndices[h]]=hist.timesMs[h];
        const activity=brain.readActivations({field:'spikeCount'});stats.active_ever=activity.reduce((sum,n)=>sum+(n>0),0);
        spikes=Array.from(hist.timesMs.slice(-256),(t,i)=>[t,hist.neuronIndices[Math.max(0,hist.timesMs.length-256)+i]]);selectedId=f.id;
      }
    }
    self.postMessage({type:'update',flies:brains.map(b=>b.fly),durationMs,wallMs:performance.now()-start,selectedId,activation,spikes,lastSpikeMs,trace},activation?[activation.buffer,lastSpikeMs.buffer]:[]);
    await sleep(0);
  }
}
self.onmessage=({data})=>{
  if(data.type==='init')init(data).catch(error=>self.postMessage({type:'error',message:error.message}));
  if(data.type==='control'){
    if('paused'in data)paused=data.paused;if('selected'in data)selected=data.selected;
    if('selectedNeuron'in data&&Number.isInteger(data.selectedNeuron)&&data.selectedNeuron>=0&&(!graph||data.selectedNeuron<graph.neuronCount))selectedNeuron=data.selectedNeuron;
    if('odor'in data)odorOn=data.odor;if('taste'in data)tasteOn=data.taste;
    if(paused&&'selected'in data){
      const item=brains.find(item=>item.fly.id===selected);
      if(item){const activation=item.brain.readActivations(),hist=item.brain.readSpikes(),lastSpikeMs=new Float64Array(item.brain.neuronCount).fill(-1e30);
        for(let h=0;h<hist.timesMs.length;h++)lastSpikeMs[hist.neuronIndices[h]]=hist.timesMs[h];
        const start=Math.max(0,hist.timesMs.length-256),spikes=Array.from(hist.timesMs.slice(start),(t,i)=>[t,hist.neuronIndices[start+i]]);
        self.postMessage({type:'update',flies:[],selectedId:selected,activation,lastSpikeMs,spikes,trace:null},[activation.buffer,lastSpikeMs.buffer]);
      }
    }
  }
};
self.addEventListener('unhandledrejection',event=>self.postMessage({type:'error',message:String(event.reason)}));
