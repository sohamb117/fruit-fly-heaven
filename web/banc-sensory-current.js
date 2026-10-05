// Convert requested sensory release rates to added current in the configured
// neuron model. Spiking profiles are calibrated with the production WASM
// neuron step, including its adaptation, refractory time, float32 and timestep.
// This is an isolated-cell interface calibration, not fitted fly physiology.
const cache=new Map(),clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const finite=(v,label)=>{if(!Number.isFinite(v))throw new Error(`Nonfinite ${label}`);return v;};
const ZERO_INTERNAL={hunger:0,insulin:0,akh:0};

// A small disconnected population: each cell tests a different input current.
// No production graph, neuron state, or external input buffer is modified.
function isolatedCells(core,p,dt,count){
 const pointers=[],alloc=array=>{
  const address=core._malloc(Math.max(8,array.byteLength));
  if(!address)throw new Error('Sensory calibration allocation failed');
  pointers.push(address);core.HEAPU8.set(new Uint8Array(array.buffer,array.byteOffset,array.byteLength),address);return address;
 };
 try{
  const parameters=new Float32Array(count*17+27);
  for(let i=0;i<count;i++)parameters.set(p,i*16);
  // No incoming edges; valid positive kinetics constants still avoid 0/0.
  for(let r=0;r<9;r++)parameters.set([1,1,0],count*17+r*3);
  const params=alloc(parameters),offsets=alloc(new Uint32Array(2*(count+1))),edges=alloc(new Uint32Array(4));
  const states=[alloc(new Float32Array(count*8)),alloc(new Float32Array(count*8))];
  const history=alloc(new Float32Array(count*2)),kinetics=alloc(new Float32Array(count*19)),events=alloc(new Uint32Array(2+32768));
  const reset=currents=>{
   core.HEAPF32.set(currents,params/4+count*16);
   for(const address of states){core.HEAPF32.fill(0,address/4,address/4+count*8);for(let i=0;i<count;i++)core.HEAPF32[address/4+i*8]=p[2];}
   core.HEAPF32.fill(0,history/4,history/4+count*2);core.HEAPF32.fill(0,kinetics/4,kinetics/4+count*19);
   core.HEAPF32.fill(-1e30,kinetics/4+count*18,kinetics/4+count*19);
   core.HEAPU8.fill(0,events,events+(2+32768)*4);
  };
  return {measure(currents,{settleMs=1000,measureMs=4000}={}){
   reset(currents);
   const settle=Math.ceil(settleMs/dt),steps=Math.ceil(measureMs/dt),before=new Float32Array(count);
   for(let tick=0;tick<settle+steps;tick++){
    if(tick===settle)for(let i=0;i<count;i++)before[i]=core.HEAPF32[states[tick%2]/4+i*8+3];
    core._br_step(count,tick,2,0,dt,0,0,0,offsets,edges,params,states[tick%2],states[1-tick%2],history,kinetics,events);
   }
   const address=states[(settle+steps)%2]/4;
   return Array.from({length:count},(_,i)=>p[8]>.5?core.HEAPF32[address+i*8+4]:(core.HEAPF32[address+i*8+3]-before[i])*1000/(steps*dt));
  },dispose(){pointers.forEach(address=>core._free(address));}};
 }catch(error){pointers.forEach(address=>core._free(address));throw error;}
}

export function measureIsolatedSensoryRates(core,parameters,dtMs,currents,options){
 const p=Array.from(parameters);validateProfile(p,dtMs);
 if(!currents.length||!Array.from(currents).every(Number.isFinite))throw new Error('Invalid calibration currents');
 const cells=isolatedCells(core,p,dtMs,currents.length);
 try{return cells.measure(currents,options);}finally{cells.dispose();}
}

function validateProfile(p,dt){
 if(p.length!==16||!p.every(Number.isFinite)||![p[0],p[1],p[6],p[10],dt].every(v=>v>0)||dt>1||p[5]<0||p[7]<0||p[3]<=p[4]||p[3]>60||p[4]<-100)throw new Error('Unsupported sensory physiology profile');
}

function gradedProfile(p,maxRateHz){
 const minimumHz=100/(1+Math.exp(-(-100-p[9])/p[10])),maximumHz=Math.min(maxRateHz,99,100/(1+Math.exp(-(60-p[9])/p[10])));
 const baselineVoltage=clamp(p[2]+p[11]/p[1],-100,60),baselineHz=100/(1+Math.exp(-(baselineVoltage-p[9])/p[10]));
 return {kind:'graded',parameters:p,minimumHz,maximumHz,baselineHz,
  note:'Sigmoid release in Hz-equivalent units; negative added current can reduce spontaneous graded release. Zero requested rate means no added current, not zero spontaneous release.',
  current(hz){if(hz<=0)return 0;const rate=clamp(hz,minimumHz,maximumHz),voltage=p[9]+p[10]*Math.log(rate/(100-rate));return p[1]*(voltage-p[2])-p[11];}};
}

function calibrateSpiking(core,p,dt,maxRateHz){
 // Match the float32 countdown rather than rounding a continuous refractory
 // duration: e.g. nonbinary timesteps can leave one extra positive remainder.
 let refractorySteps=0,remaining=Math.fround(p[5]);
 while(remaining>0&&refractorySteps<10000){remaining=Math.max(0,Math.fround(remaining-Math.fround(dt)));refractorySteps++;}
 const maximumHz=Math.min(maxRateHz,1000/((refractorySteps+1)*dt));
 if(maximumHz<2)throw new Error('Sensory refractory period precludes the calibration range');
 // A 2 Hz lower bound is explicit: very low tonic LIF rates approach rheobase
 // beyond float32 voltage precision. Zero remains an exact off input.
 const knots=[2,3,4,5,6,8,10,12,15,...Array.from({length:197},(_,i)=>20+i*5)].filter(x=>x<maximumHz);
 knots.push(maximumHz);
 const low=new Float64Array(knots.length),high=new Float64Array(knots.length),probe=new Float32Array(knots.length);
 const bestCurrent=new Float64Array(knots.length),bestRate=new Float64Array(knots.length),error=new Float64Array(knots.length).fill(Infinity);
 const rheobase=p[1]*(p[3]-p[2]),tau=p[0]/p[1];
 for(let i=0;i<knots.length;i++){
  const hz=knots[i],charge=Math.max(dt,1000/hz-p[5]),vReset=p[4]-p[2],vThreshold=p[3]-p[2],decay=Math.exp(-charge/tau);
  const estimate=p[1]*(vThreshold-vReset*decay)/(1-decay)+p[7]*p[6]*hz/1000-p[11];
  low[i]=-Math.abs(p[11])-rheobase;high[i]=Math.max(rheobase,estimate)*2+Math.abs(p[11])+p[7];
 }
 const cells=isolatedCells(core,p,dt,knots.length),settleMs=Math.max(1000,8*p[6]);
 try{
  for(let iteration=0;iteration<19;iteration++){
   for(let i=0;i<probe.length;i++)probe[i]=(low[i]+high[i])/2;
   const rates=cells.measure(probe,{settleMs,measureMs:4000});
   for(let i=0;i<probe.length;i++){
    const difference=Math.abs(rates[i]-knots[i]);
    if(difference<error[i]||(difference===error[i]&&probe[i]<bestCurrent[i])){error[i]=difference;bestCurrent[i]=probe[i];bestRate[i]=rates[i];}
    if(rates[i]<knots[i])low[i]=probe[i];else high[i]=probe[i];
   }
  }
 }finally{cells.dispose();}
 // Plateau ties and finite counting windows can produce tiny reversals.
 for(let i=1;i<bestCurrent.length;i++)bestCurrent[i]=Math.max(bestCurrent[i],bestCurrent[i-1]);
 const profile={kind:'spiking',parameters:p,minimumHz:knots[0],maximumHz,baselineHz:null,
  knotsHz:knots,currentsPa:Array.from(bestCurrent),measuredHz:Array.from(bestRate),maximumKnotErrorHz:Math.max(...error),
  calibration:{kernel:'production WASM br_step',dtMs:dt,settleMs,measureMs:4000,internal:ZERO_INTERNAL,network:'no incoming chemical or electrical synapses'},
  note:'Positive rates below minimumHz are clamped to minimumHz; rates above maximumHz are clamped. Interpolation remains approximate, including timestep-quantized firing plateaus. Synaptic input, hunger, and neuromodulation remain additional effects in the full brain.',
  current(hz){
   if(hz<=0)return 0;hz=clamp(hz,this.minimumHz,this.maximumHz);
   let lo=0,hi=this.knotsHz.length-1;
   while(hi-lo>1){const mid=(lo+hi)>>1;if(this.knotsHz[mid]<hz)lo=mid;else hi=mid;}
   const width=this.knotsHz[lo+1]-this.knotsHz[lo],t=width?(hz-this.knotsHz[lo])/width:0;
   return this.currentsPa[lo]+t*(this.currentsPa[lo+1]-this.currentsPa[lo]);
  }};
 if(knots.length===1)profile.current=hz=>hz<=0?0:profile.currentsPa[0];
 return profile;
}

export async function createBancSensoryCurrentMapper(core,model,{indices=null,maxRateHz=200,onProfile=null}={}){
 const params=model.params,dt=model.manifest.dt_ms,n=model.manifest.neuron_count;
 if(!(params instanceof Float32Array)||!Number.isInteger(n)||n<1||params.length!==n*16)throw new Error('Invalid sensory physiology model');
 if(!Number.isFinite(maxRateHz)||maxRateHz<2||maxRateHz>1000)throw new Error('Invalid sensory rate ceiling');
 const selected=indices?Array.from(new Set(indices)):Array.from({length:n},(_,i)=>i),profiles=[],byKey=new Map(),byIndex=new Int32Array(n).fill(-1);
 for(const index of selected){
  if(!Number.isInteger(index)||index<0||index>=n)throw new Error('Invalid sensory neuron index');
  const p=Array.from(params.subarray(index*16,index*16+16));validateProfile(p,dt);
  const key=JSON.stringify([dt,maxRateHz,p]);
  if(!byKey.has(key)){byKey.set(key,profiles.length);profiles.push({key,p});}
  byIndex[index]=byKey.get(key);
 }
 for(let i=0;i<profiles.length;i++){
  const {key,p}=profiles[i];
  if(!cache.has(key))cache.set(key,p[8]>.5?gradedProfile(p,maxRateHz):calibrateSpiking(core,p,dt,maxRateHz));
  profiles[i]=cache.get(key);onProfile?.(profiles[i],i,profiles.length);
  await Promise.resolve();
 }
 return {profiles,dtMs:dt,scope:'Isolated configured-cell sensory interface calibration, not biological fitting',
  current(index,hz){finite(hz,'sensory rate');const profile=profiles[byIndex[index]];if(!profile)throw new Error('Uncalibrated sensory neuron');return profile.current(hz);},
  limits(index){const profile=profiles[byIndex[index]];if(!profile)throw new Error('Uncalibrated sensory neuron');return {minimumHz:profile.minimumHz,maximumHz:profile.maximumHz,zeroMeans:'zero added current'};}};
}
