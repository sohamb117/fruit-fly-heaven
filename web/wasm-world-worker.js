// Example environment adapter. All neuroscience is in the generic WASM package.
import {createBrainModule,loadConnectome} from '/engine/index.js';
let brains=[],fruit=[],groups={},paused=false,selected=1,odorOn=true,tasteOn=true;
let module,graph,indices,readIds,readGroups,alive=true;
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
function distance(f,x,z){return f.kind==='apple'?Math.hypot(x-f.x,z-f.z):Math.min(...f.path.map(p=>Math.hypot(x-p[0],z-p[1])));}
function habitat(x,z){let y=1.5+.0037*(x*x+z*z),contact=false;for(const f of fruit){const d=distance(f,x,z);if(d<f.radius){const h=f.y+Math.sqrt(f.radius*f.radius-d*d);if(h>=y){y=h;contact=true;}}}return {y,contact};}
function odor(x,z){return Math.min(1,fruit.reduce((n,f)=>n+.3*Math.exp(-Math.max(0,distance(f,x,z)-f.radius)/18),0));}
function inputFor(f){const a=f.heading,x=f.x,z=f.z;return [odorOn?3+65*odor(x+.9*Math.cos(a)-.4*Math.sin(a),z+.9*Math.sin(a)+.4*Math.cos(a)):0,odorOn?3+65*odor(x+.9*Math.cos(a)+.4*Math.sin(a),z+.9*Math.sin(a)-.4*Math.cos(a)):0,tasteOn&&f.contact?150:0];}
async function init(args){
  self.postMessage({type:'progress',message:'Loading WASM and measured connectivity'});
  module=await createBrainModule();
  const data=await loadConnectome(new URL('/connectome/',location.href));graph=module.createConnectome(data);
  groups=args.groups;fruit=args.fruit;
  indices=Uint32Array.from([...groups.odor_left,...groups.odor_right,...groups.sweet]);
  readGroups=[groups.odor_left,groups.odor_right,groups.sweet,groups.walk,groups.steer_left,groups.steer_right,groups.feed,groups.antenna];
  readIds=Uint32Array.from(readGroups.flat());
  const population=graph.createPopulation(args.flies.length,{seed:20260912+args.flies[0].id*100003});
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
    const start=performance.now();let activation=null,spikes=null,selectedId=null;
    for(const item of brains){
      const {brain,fly:f}=item,s=inputFor(f),ratesHz=new Float32Array(indices.length);
      ratesHz.fill(s[0],0,groups.odor_left.length);ratesHz.fill(s[1],groups.odor_left.length,groups.odor_left.length+groups.odor_right.length);ratesHz.fill(s[2],groups.odor_left.length+groups.odor_right.length);
      brain.setPoissonInputs({indices,ratesHz});brain.step(durationMs);
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
        const activity=brain.readActivations({field:'spikeCount'});stats.active_ever=activity.reduce((sum,n)=>sum+(n>0),0);
        spikes=Array.from(hist.timesMs.slice(-256),(t,i)=>[t,hist.neuronIndices[Math.max(0,hist.timesMs.length-256)+i]]);selectedId=f.id;
      }
    }
    self.postMessage({type:'update',flies:brains.map(b=>b.fly),durationMs,wallMs:performance.now()-start,selectedId,activation,spikes},activation?[activation.buffer]:[]);
    await sleep(0);
  }
}
self.onmessage=({data})=>{
  if(data.type==='init')init(data).catch(error=>self.postMessage({type:'error',message:error.message}));
  if(data.type==='control'){
    if('paused'in data)paused=data.paused;if('selected'in data)selected=data.selected;
    if('odor'in data)odorOn=data.odor;if('taste'in data)tasteOn=data.taste;
  }
};
self.addEventListener('unhandledrejection',event=>self.postMessage({type:'error',message:String(event.reason)}));
