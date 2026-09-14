struct Edge { source:u32, weight:f32, receptor:u32, delay:u32 }
struct Config { n:u32, tick:u32, slots:u32, gaps:u32, dt:f32, hunger:f32, insulin:f32, akh:f32 }
@group(0) @binding(0) var<storage,read> offsets:array<u32>;
@group(0) @binding(1) var<storage,read> edges:array<Edge>;
@group(0) @binding(2) var<storage,read> params:array<f32>;
@group(0) @binding(3) var<storage,read> old:array<f32>;
@group(0) @binding(4) var<storage,read_write> next:array<f32>;
@group(0) @binding(5) var<storage,read_write> history:array<f32>;
@group(0) @binding(6) var<storage,read_write> kinetics:array<f32>;
@group(0) @binding(7) var<uniform> cfg:Config;
// Current bits and an independent per-fly recent-spike ring share one binding.
@group(0) @binding(8) var<storage,read_write> inputs:array<atomic<u32>>;
var<workgroup> eventCount:atomic<u32>;
var<workgroup> eventStart:u32;
@compute @workgroup_size(128)
fn step(@builtin(global_invocation_id) invocation:vec3<u32>,@builtin(local_invocation_index) local:u32) {
  let i=invocation.x;
  if(local==0u){atomicStore(&eventCount,0u);}
  workgroupBarrier();
  var emitted=false;var eventOffset=0u;
  // All invocations, including padding in the final group, reach the barriers.
  if(i<cfg.n){
  let p=i*16u; let s=i*8u; let rbase=cfg.n*17u;
  var drive:array<f32,9>;
  for(var e=offsets[i];e<offsets[i+1u];e++) {
    let edge=edges[e];
    drive[edge.receptor]+=edge.weight*history[((cfg.tick+cfg.slots-edge.delay)%cfg.slots)*cfg.n+edge.source];
  }
  var conductance=params[p+1u]; var reversal=conductance*params[p+2u];
  var modulation:array<f32,4>;
  for(var r=0u;r<9u;r++) {
    let k=(i*9u+r)*2u; let rp=rbase+r*3u;
    kinetics[k]+=(drive[r]-kinetics[k])*(1.0-exp(-cfg.dt/params[rp]));
    kinetics[k+1u]+=(kinetics[k]-kinetics[k+1u])*(1.0-exp(-cfg.dt/params[rp+1u]));
    let g=kinetics[k+1u];
    if(r<5u){conductance+=g;reversal+=g*params[rp+2u];}
    else{modulation[r-5u]=g/(1.0+g);}
  }
  if(cfg.gaps!=0u) {
    for(var e=offsets[cfg.n+1u+i];e<offsets[cfg.n+2u+i];e++) {
      conductance+=edges[e].weight; reversal+=edges[e].weight*old[edges[e].source*8u];
    }
  }
  let gain=clamp(1.0+params[p+12u]*modulation[0]+params[p+13u]*(modulation[1]-.5*modulation[3])+params[p+14u]*modulation[2],.25,4.0);
  var adaptation=old[s+1u]*exp(-cfg.dt/params[p+6u]);
  let current=(bitcast<f32>(atomicLoad(&inputs[i]))+params[p+11u])*gain+params[p+15u]*(cfg.hunger+.3*cfg.akh-.3*cfg.insulin)-adaptation;
  var v=clamp((old[s]+cfg.dt*(reversal+current)/params[p])/(1.0+cfg.dt*conductance/params[p]),-100.0,60.0);
  var refractory=max(0.0,old[s+2u]-cfg.dt); var spike=0.0; var release=0.0;
  if(params[p+8u]>.5){release=.1/(1.0+exp(-(v-params[p+9u])/params[p+10u]));}
  else {
    if(old[s+2u]>0.0){v=params[p+4u];}
    else if(v>=params[p+3u]){spike=1.0;v=params[p+4u];refractory=params[p+5u];adaptation+=params[p+7u];}
    release=spike/cfg.dt;
  }
  next[s]=v; next[s+1u]=adaptation; next[s+2u]=refractory; next[s+3u]=old[s+3u]+spike;
  next[s+4u]=old[s+4u]+(release*1000.0-old[s+4u])*(1.0-exp(-cfg.dt/50.0));
  next[s+5u]=release; next[s+6u]=conductance; next[s+7u]=current;
  if(spike>0.0){
    let time=f32(cfg.tick+1u)*cfg.dt;kinetics[cfg.n*18u+i]=time;
    eventOffset=atomicAdd(&eventCount,1u);emitted=true;
  }
  history[(cfg.tick%cfg.slots)*cfg.n+i]=release;
  }
  workgroupBarrier();
  // Reserve one contiguous range per workgroup rather than contending on a
  // global counter for every spiking neuron. All events are still retained.
  if(local==0u){
    let count=atomicLoad(&eventCount);eventStart=0u;
    if(count>0u){eventStart=atomicAdd(&inputs[cfg.n],count);atomicMax(&inputs[cfg.n+1u],min(16384u,eventStart+count));}
  }
  workgroupBarrier();
  if(emitted){
    let slot=(eventStart+eventOffset)%16384u;let time=f32(cfg.tick+1u)*cfg.dt;
    atomicStore(&inputs[cfg.n+2u+slot*2u],bitcast<u32>(time));atomicStore(&inputs[cfg.n+3u+slot*2u],i);
  }
}
