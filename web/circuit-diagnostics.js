// Read-only inspection. No neuron stimulation, gain changes, or body commands.
export function validateCircuitProbe(probe,neuronCount){
  if(probe.schema_version!==1||probe.neuron_count!==neuronCount)throw new Error('Circuit probes do not match this connectome');
  const valid=i=>Number.isInteger(i)&&i>=0&&i<neuronCount;
  for(const group of [...probe.groups,...probe.channels]){
    if(!group.indices.length||new Set(group.indices).size!==group.indices.length||!group.indices.every(valid))throw new Error('Invalid circuit probe indices');
  }
  for(const channel of probe.channels)for(const i of channel.indices){
    if(!probe.cells[i]||!Array.isArray(probe.incoming[i]))throw new Error('Missing motor circuit annotation');
    for(const [j,w]of probe.incoming[i])if(!valid(j)||!Number.isFinite(w)||!probe.cells[j])throw new Error('Invalid incoming circuit edge');
  }
}

export class CircuitDiagnostics{
  constructor(probe,{restMv=-52,thresholdMv=-45,windowMs=100}={}){
    validateCircuitProbe(probe,probe.neuron_count);
    this.probe=probe;this.restMv=restMv;this.thresholdMv=thresholdMv;this.windowMs=windowMs;
    this.motorIds=Uint32Array.from(probe.channels.flatMap(c=>c.indices));
    this.previous=new Float64Array(probe.neuron_count);this.previousTime=0;this.sample=null;
  }
  reset(timeMs,counts){this.previous=counts.slice();this.previousTime=timeMs;this.sample=null;}
  ready(timeMs){return timeMs-this.previousTime>=this.windowMs-1e-7;}
  update(timeMs,counts,voltage,synapticDrive){
    if(!this.ready(timeMs))return this.sample;
    const windowMs=timeMs-this.previousTime,scale=1000/windowMs;
    const rates=counts.map((n,i)=>(n-this.previous[i])*scale);
    const summarize=indices=>{
      let spikes=0,sum=0,active=0,minMv=Infinity,maxMv=-Infinity;
      for(const i of indices){spikes+=counts[i]-this.previous[i];sum+=voltage[i];active+=rates[i]>0;minMv=Math.min(minMv,voltage[i]);maxMv=Math.max(maxMv,voltage[i]);}
      return {n:indices.length,active,spikes,meanHz:spikes*scale/indices.length,meanMv:sum/indices.length,minMv,maxMv};
    };
    const groups=this.probe.groups.map(g=>({key:g.key,label:g.label,...summarize(g.indices)}));
    let offset=0;
    const channels=this.probe.channels.map(c=>{
      const incoming=new Map();
      const cells=c.indices.map(i=>{
        for(const [j,w]of this.probe.incoming[i]){
          const value=w*rates[j]/c.indices.length;
          if(!value)continue;
          const source=this.probe.cells[j],sign=w>0?'excitatory':'inhibitory',key=(source.type||'Unclassified')+'|'+sign;
          const entry=incoming.get(key)||{type:source.type||'Unclassified',sign,weightedHz:0};
          entry.weightedHz+=value;incoming.set(key,entry);
        }
        return {...this.probe.cells[i],rateHz:rates[i],voltageMv:voltage[i],synapticDriveMv:synapticDrive[offset++]};
      });
      const ranked=[...incoming.values()].sort((a,b)=>Math.abs(b.weightedHz)-Math.abs(a.weightedHz));
      return {key:c.key,label:c.label,...summarize(c.indices),cells,
        meanDriveMv:cells.reduce((s,n)=>s+n.synapticDriveMv,0)/cells.length,
        incomingPositive:ranked.filter(n=>n.sign==='excitatory').slice(0,3),incomingNegative:ranked.filter(n=>n.sign==='inhibitory').slice(0,3)};
    });
    this.sample={timeMs,windowMs,restMv:this.restMv,thresholdMv:this.thresholdMv,groups,channels};
    this.previous=counts.slice();this.previousTime=timeMs;return this.sample;
  }
}
