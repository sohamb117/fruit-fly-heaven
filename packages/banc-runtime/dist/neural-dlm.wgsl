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
// Mathematical source: Hürkey et al., Nature (2023),
// doi:10.1038/s41586-023-06099-0. Newly expressed paper equations, not author code.
// x = voltage (mV), h and b; ms / nS / pA / pF. No reset or added tonic input.
fn dlmDerivative(x:vec3<f32>,current:f32,g:array<f32,5>)->vec3<f32>{
 let m=1.0/(1.0+exp(-.0392*3.0*(x.x+33.0)));
 let eh=exp(-.0392*5.2*(x.x+39.14));let eb=exp(-.0392*1.1056*(x.x+42.14));
 let hinf=1.0/(1.0+eh);let binf=1.0/(1.0+eb);
 let tauh=exp(-.0392*5.2*.38*(x.x+39.14))/(.2*(1.0+eh));
 let taub=exp(-.0392*1.1056*.38*(x.x+42.14))/(.2*(1.0+eb));
 var synaptic=0.0;
 for(var r=0u;r<5u;r++){synaptic+=g[r]*(params[cfg.n*17u+r*3u+2u]-x.x);}
 let sodium=431.2*m*m*m*(1.0-x.y)*(x.x-55.0);
 let potassium=137.68216*x.z*x.z*x.z*x.z*(x.x+72.0);
 return vec3<f32>((current+synaptic-8.624*(x.x+60.0)-sodium-potassium)/130.0,(hinf-x.y)/tauh,(binf-x.z)/taub);
}
fn dlmRK4(x:vec3<f32>,current:f32,g:array<f32,5>)->vec3<f32>{
 let a=dlmDerivative(x,current,g);let b=dlmDerivative(x+.05*a,current,g);
 let c=dlmDerivative(x+.05*b,current,g);let d=dlmDerivative(x+.1*c,current,g);
 return x+(.1/6.0)*(a+2.0*b+2.0*c+d);
}
fn dlmValid(x:vec3<f32>)->bool{
 return all(abs(x)<vec3<f32>(3.402823e38))&&x.y>=0.0&&x.y<=1.0&&x.z>=0.0&&x.z<=1.0;
}
fn dlmConductance(x:vec3<f32>,g:array<f32,5>)->f32{
 let m=1.0/(1.0+exp(-.0392*3.0*(x.x+33.0)));
 var result=8.624+431.2*m*m*m*(1.0-x.y)+137.68216*x.z*x.z*x.z*x.z;
 for(var r=0u;r<5u;r++){result+=g[r];}return result;
}

@compute @workgroup_size(128)
fn step(@builtin(global_invocation_id) invocation:vec3<u32>,@builtin(local_invocation_index) local:u32) {
  let i=invocation.x;
  if(local==0u){atomicStore(&eventCount,0u);}
  workgroupBarrier();
  var emitted=false;var eventOffset=0u;var eventTime=f32(cfg.tick+1u)*cfg.dt;
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
  var current=(bitcast<f32>(atomicLoad(&inputs[i]))+params[p+11u])*gain+params[p+15u]*(cfg.hunger+.3*cfg.akh-.3*cfg.insulin)-adaptation;
  var v=0.0;var refractory=0.0;var spike=0.0;var release=0.0;
  let slot=u32(params[cfg.n*17u+27u+i]);
  if(slot>0u){
    let base=cfg.n*19u+(slot-1u)*4u;
    var x=vec3<f32>(old[s],kinetics[base],kinetics[base+1u]);
    let storedGuard=kinetics[base+2u];
    var valid=kinetics[base+3u]==0.0&&dlmValid(x)&&storedGuard>=0.0&&storedGuard<=100.0&&floor(storedGuard)==storedGuard;
    var guard=u32(clamp(storedGuard,0.0,100.0));
    current=(bitcast<f32>(atomicLoad(&inputs[i]))+params[p+11u])*gain+params[p+15u]*(cfg.hunger+.3*cfg.akh-.3*cfg.insulin);
    var g:array<f32,5>;for(var r=0u;r<5u;r++){g[r]=kinetics[(i*9u+r)*2u+1u];}
    for(var sub=0u;sub<5u;sub++){
      if(valid){
        x=dlmRK4(x,current,g);valid=dlmValid(x);
        if(guard>0u){guard--;}
        if(valid&&x.x>-10.0&&guard==0u){spike=1.0;guard=100u;eventTime=f32(cfg.tick*5u+sub+1u)/10.0;}
      }
    }
    if(!valid){kinetics[base+3u]=1.0;spike=0.0;}
    kinetics[base]=x.y;kinetics[base+1u]=x.z;kinetics[base+2u]=f32(guard);
    v=x.x;adaptation=0.0;refractory=0.0;release=spike/cfg.dt;conductance=dlmConductance(x,g);
  }else{
  v=clamp((old[s]+cfg.dt*(reversal+current)/params[p])/(1.0+cfg.dt*conductance/params[p]),-100.0,60.0);
  refractory=max(0.0,old[s+2u]-cfg.dt);
  if(params[p+8u]>.5){release=.1/(1.0+exp(-(v-params[p+9u])/params[p+10u]));}
  else {
    if(old[s+2u]>0.0){v=params[p+4u];}
    else if(v>=params[p+3u]){spike=1.0;v=params[p+4u];refractory=params[p+5u];adaptation+=params[p+7u];}
    release=spike/cfg.dt;
  }
  }
  next[s]=v; next[s+1u]=adaptation; next[s+2u]=refractory; next[s+3u]=old[s+3u]+spike;
  next[s+4u]=old[s+4u]+(release*1000.0-old[s+4u])*(1.0-exp(-cfg.dt/50.0));
  next[s+5u]=release; next[s+6u]=conductance; next[s+7u]=current;
  if(spike>0.0){
    kinetics[cfg.n*18u+i]=eventTime;
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
    let slot=(eventStart+eventOffset)%16384u;let time=eventTime;
    atomicStore(&inputs[cfg.n+2u+slot*2u],bitcast<u32>(time));atomicStore(&inputs[cfg.n+3u+slot*2u],i);
  }
}
