// Analyze completed recorded arms; no neural or mechanical execution.
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
const directory=path.resolve(process.argv[2]||'reports/flight-sensory-families/run');
const resultBytes=await fs.readFile(path.join(directory,'result.json')),planBytes=await fs.readFile(path.join(directory,'plan.json'));
const result=JSON.parse(resultBytes),plan=JSON.parse(planBytes),sha=b=>createHash('sha256').update(b).digest('hex');
assert(result.interpretationAllowed&&result.allInputReferenceGate.passed&&result.zeroTasteGate.passed&&!result.error);
assert.equal(result.planSha256,sha(planBytes));assert.equal(result.arms.length,7);
const offset=new Map(result.wingIndices.map((id,k)=>[id,k])),sources={'result.json':sha(resultBytes),'plan.json':sha(planBytes)},arms=[];
for(const entry of result.arms){
 const bytes=await fs.readFile(path.join(directory,entry.file)),arm=JSON.parse(bytes);sources[entry.file]=sha(bytes);
 assert(arm.completed&&!arm.error&&arm.trace.length===150);assert.equal(arm.trace.at(-1).timeMs,300);
 for(let i=0;i<arm.trace.length;i++){assert.equal(arm.trace[i].timeMs,(i+1)*2);assert.deepEqual(arm.trace[i].indices,result.wingIndices);}
 const events=arm.trace.flatMap(row=>row.events),groups=result.wingMappings.map(mapping=>{
  const indices=mapping.indices,positions=indices.map(id=>offset.get(id)),end=arm.trace.at(-1),start=arm.trace[49];
  const count=positions.reduce((sum,k)=>sum+end.counts[k]-start.counts[k],0);
  assert.equal(count,events.filter(event=>indices.includes(event.index)&&event.timeMs>100&&event.timeMs<=300).length,'Exact events and cumulative counts disagree');
  const mean=values=>values.reduce((a,b)=>a+b,0)/values.length;
  return {joint:mapping.joint,target:mapping.target,kind:mapping.kind,motorCount:indices.length,motorIndices:indices,
   totalSpikes:positions.reduce((sum,k)=>sum+end.counts[k],0),spikes100to300Ms:count,rawMeanRate100to300Hz:count/(.2*indices.length),
   filteredMeanRateAt300Hz:mean(positions.map(k=>end.ratesHz[k])),
   firstSpikeMs:events.find(event=>indices.includes(event.index))?.timeMs??null,
   perNeuron:indices.map((id,k)=>({index:id,totalSpikes:end.counts[positions[k]],spikes100to300Ms:end.counts[positions[k]]-start.counts[positions[k]],filteredRateAt300Hz:end.ratesHz[positions[k]]})),
   trace:arm.trace.map(row=>({timeMs:row.timeMs,count:positions.reduce((sum,k)=>sum+row.counts[k],0),meanFilteredRateHz:mean(positions.map(k=>row.ratesHz[k]))}))};
 });
 arms.push({name:entry.name,input:plan.arms.find(row=>row.name===entry.name),groups});
}
const summary={schemaVersion:1,kind:'frozen-sensory-family-analysis',sourceHashes:sources,scriptSha256:sha(await fs.readFile(fileURLToPath(import.meta.url))),
 gates:{allInput:result.allInputReferenceGate,zeroTaste:result.zeroTasteGate},
 scope:'Seven fresh300ms frozen-current neural trials; no body or muscle simulation. Only external current was masked; intrinsic/recurrent/internal-state terms stayed unchanged.',
 rawRateWindow:'Counts and independently checked exact events in(100,300]ms divided by0.2s and number of mapped MNs. This is distinct from the50ms filtered rate at300ms.',
 interpretation:'Zero added current and inactive taste do not recruit wing power neurons. Each active sensory family alone is sufficient for high DLM recruitment in this artificial context, so broad fallback is not the unique possible source. Single-family arms do not quantify additive causal contributions or prove family necessity under combined inputs. No physiological profile change follows from this one context.',arms};
await fs.writeFile(path.join(directory,'analysis.json'),JSON.stringify(summary)+'\n');
const find=(arm,side,target)=>arm.groups.find(group=>group.joint==='wing_power_'+side&&group.target===target);
const lines=['# Sensory-family recruitment results','',
 'All seven conditions completed. The all-input neural trajectory exactly matched the previous300ms prefix; taste-only and zero-external trajectories and full forward-state bytes matched each other.',
 '', 'Zero added sensory current produced no DLM or DVM spikes. Every nonzero sensory family alone produced substantial DLM firing. The broad fallback remains an anatomical mapping concern, but these results do not isolate it as the unique source of strong wing power drive.',
 '', '| External-current condition | DLM left raw Hz | DLM right raw Hz | DVM left raw Hz | DVM right raw Hz |',
 '|---|---:|---:|---:|---:|'];
for(const arm of arms)lines.push('| '+arm.name+' | '+[['left','dorsal_longitudinal_muscle'],['right','dorsal_longitudinal_muscle'],['left','dorsoventral_muscle'],['right','dorsoventral_muscle']].map(([side,target])=>find(arm,side,target).rawMeanRate100to300Hz.toFixed(2)).join(' | ')+' |');
lines.push('', 'Raw firing rates count events in(100,300]ms and average across the mapped neurons; they do not use the50ms rate filter. Cumulative counts were cross-checked against the exact timestamped events. Per-neuron counts, filtered rates, first spike times and full2ms group traces are in analysis.json.',
 '', 'The intrinsic current, recurrent graph, gap junctions and hunger/satiety modulation stayed fixed. No body, muscles, flight, optimizer or receptor-axis model was simulated. This is a sufficiency assay with constant frozen input, not a measurement of additive family contributions or a live closed-loop behavior.',
 '', 'The zero-external result argues against autonomous DLM firing under this specific unchanged model/state. Comparable recruitment from odor alone, native body transducers alone and fallback alone motivates examining stimulus-to-neural recruitment and the downstream rate-to-muscle interpretation before choosing a physiological change. It does not establish whether these input priors or motor firing rates are biologically appropriate.',
 '', 'Reproduce this analysis without simulation:', '', '```sh','node scripts/analyze-flight-sensory-families.mjs '+path.relative(process.cwd(),directory),'```');
await fs.writeFile(path.join(directory,'RESULTS.md'),lines.join('\n')+'\n');
console.log(JSON.stringify({arms:arms.length,output:path.join(directory,'RESULTS.md')}));
