// Postprocess recorded experiments only. This script never starts a brain/body.
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {execFileSync} from 'node:child_process';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const planPath=path.resolve(process.argv[2]||'reports/flight-afferent-identification/robustness/plan.json');
const folder=path.dirname(planPath),planBytes=await fs.readFile(planPath),plan=JSON.parse(planBytes);
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
assert.equal(plan.kind,'predeclared-frozen-context-afferent-robustness');
assert.equal(sha(await fs.readFile(path.join(root,plan.script))),plan.scriptSha256,'Declared assay script changed');
assert.equal(sha(await fs.readFile(path.join(folder,'identification-source.used.mjs'))),plan.scriptSha256);
const analyzer=path.join(root,'scripts/analyze-flight-afferent-response.mjs'),analyzerHash=sha(await fs.readFile(analyzer));
const sources={[path.relative(root,planPath)]:sha(planBytes)},loaded=[],pending=[];
const candidates=[{name:'historical-prefix100-pulse5',output:path.dirname(plan.historicalResult),historical:true,prefixMs:100,maxPulseHz:5},...plan.runs];
for(const entry of candidates){
 const directory=path.resolve(root,entry.output);assert(directory.startsWith(path.join(root,'reports')+path.sep));
 let bytes;try{bytes=await fs.readFile(path.join(directory,'result.json'));}catch(error){if(error.code==='ENOENT'){pending.push(entry.name);continue;}throw error;}
 const result=JSON.parse(bytes);sources[path.relative(root,path.join(directory,'result.json'))]=sha(bytes);
 assert(result.interpretationAllowed&&result.baselineGate.passed&&!result.error,entry.name+' failed experiment gates');
 assert(result.arms.every(arm=>arm.completed));
 assert.equal(result.neuralFingerprint,plan.neuralFingerprint);
 assert.deepEqual(result.sourceHashes,plan.sourceHashes);assert.deepEqual(result.artifactHashes,plan.artifactHashes);
 assert.equal(result.protocol.prefixMs,entry.prefixMs);
 if(entry.historical){assert.equal(sha(bytes),plan.historicalResultSha256);assert.equal(result.scriptSha256,plan.historicalScriptSha256);}
 else{
  assert.equal(result.scriptSha256,plan.scriptSha256);assert(result.referenceGate?.passed);assert.equal(result.referenceGate.sha256,plan.historicalResultSha256);
  assert.equal(result.protocol.maxPulseHz,entry.maxPulseHz);assert.equal(result.arms.length,8);assert.equal(result.protocol.groupSelection,'active3');
 }
 // Derived artifacts may be regenerated; recorded source files remain untouched.
 execFileSync(process.execPath,[analyzer,directory],{cwd:root,encoding:'utf8',maxBuffer:2**20});
 const analysisBytes=await fs.readFile(path.join(directory,'analysis.json')),analysis=JSON.parse(analysisBytes);
 assert.equal(analysis.scriptSha256,analyzerHash);assert.equal(analysis.sourceHashes['result.json'],sha(bytes));
 sources[path.relative(root,path.join(directory,'analysis.json'))]=sha(analysisBytes);
 const baselineBytes=await fs.readFile(path.join(directory,'baseline-a.json')),baseline=JSON.parse(baselineBytes);
 assert.equal(sha(baselineBytes),analysis.sourceHashes['baseline-a.json']);
 loaded.push({...entry,directory,result,analysis,baseline});
}
const historical=loaded.find(row=>row.historical);assert(historical,'Historical source must exist');
const groups=['haltere/right','wing_base/left','wing_base/right'];
for(const row of loaded){assert.deepEqual(row.result.context,historical.result.context);assert.deepEqual(row.analysis.muscleOrder,historical.analysis.muscleOrder);assert.deepEqual(row.analysis.wingNeuronOrder,historical.analysis.wingNeuronOrder);}
const dot=(a,b)=>a.reduce((sum,v,k)=>sum+v*b[k],0),norm=a=>Math.sqrt(dot(a,a));
const cosine=(a,b)=>norm(a)&&norm(b)?dot(a,b)/(norm(a)*norm(b)):null;
const difference=(a,b)=>a.map((v,k)=>v-b[k]);
const neuronLabel=k=>({index:historical.analysis.wingNeuronOrder[k],muscles:historical.result.wingMappings.filter(mapping=>mapping.indices.includes(historical.analysis.wingNeuronOrder[k])).map(mapping=>mapping.joint+':'+mapping.target)});
function response(positive,negative,delta){
 const contrast=difference(positive,negative),length=norm(contrast);
 return {positive,negative,contrast,centralGain:contrast.map(v=>v/(2*delta)),positiveGain:positive.map(v=>v/delta),negativeGain:negative.map(v=>-v/delta),
  oddSymmetryCosine:cosine(positive,negative.map(v=>-v)),evenOverOddNorm:length?norm(positive.map((v,k)=>v+negative[k]))/length:null,
  contrastNorm:length,changedPositive:positive.filter(v=>v!==0).length,changedNegative:negative.filter(v=>v!==0).length};
}
function temporalResponse(data){
 const contrasts=data.directSpikeTrace.map(row=>({elapsedMs:row.relativeMs-20,
  counts:difference(row.positiveCounts,row.negativeCounts),positiveCounts:difference(row.positiveCounts,row.baselineCounts),negativeCounts:difference(row.negativeCounts,row.baselineCounts),
  rates:difference(row.positiveRateHz,row.negativeRateHz),positiveRates:difference(row.positiveRateHz,row.baselineRateHz),negativeRates:difference(row.negativeRateHz,row.baselineRateHz)}));
 const first=(key,threshold)=>contrasts.find(row=>row[key].some(value=>Math.abs(value)>threshold))?.elapsedMs??null;
 const union=(rows,key,threshold)=>historical.analysis.wingNeuronOrder.flatMap((id,k)=>rows.some(row=>Math.abs(row[key][k])>threshold)?[id]:[]);
 return {firstCentralCountContrastMs:first('counts',0),firstPositiveCountContrastMs:first('positiveCounts',0),firstNegativeCountContrastMs:first('negativeCounts',0),
  firstCentralRawRateContrastMs:first('rates',1e-6),firstPositiveRawRateContrastMs:first('positiveRates',1e-6),firstNegativeRawRateContrastMs:first('negativeRates',1e-6),
  windows:Object.fromEntries([10,20,40].map(elapsed=>{const selected=contrasts.filter(row=>row.elapsedMs<=elapsed),last=selected.at(-1);
   return [`first${elapsed}ms`,{centralCountChangedNeurons:union(selected,'counts',0),positiveCountChangedNeurons:union(selected,'positiveCounts',0),negativeCountChangedNeurons:union(selected,'negativeCounts',0),
    centralRateChangedNeurons:union(selected,'rates',1e-6),
    transientCentralCountOnlyNeurons:historical.analysis.wingNeuronOrder.flatMap((id,k)=>last.counts[k]===0&&selected.some(row=>row.counts[k]!==0)?[id]:[])}];
  })),contrasts};
}
const conditions=loaded.map(row=>({name:row.name,prefixMs:row.prefixMs,maxPulseHz:row.maxPulseHz,historical:!!row.historical,
 sourceResult:path.relative(root,path.join(row.directory,'result.json')),baselineGate:row.result.baselineGate,referenceGate:row.result.referenceGate??null,
 groups:groups.map(group=>{
  const data=row.analysis.rows.find(item=>item.group===group);assert(data);
  if(!row.historical)assert.equal(data.deltaHz,row.maxPulseHz,'Declared active-group pulse was clipped');
  return {group,deltaHz:data.deltaHz,count:data.count,actualAfferentSpikesDuringPulse:data.afferentSamples.find(sample=>sample.relativeMs===40),temporal:temporalResponse(data),
   firstRateDifferenceMs:data.firstRateDifferenceAbove1HzMs===null?null:data.firstRateDifferenceAbove1HzMs-20,
   firstForceDifferenceMs:data.firstForceDifferenceAbove1eMinus4Ms===null?null:data.firstForceDifferenceAbove1eMinus4Ms-20,
   windows:Object.fromEntries([10,20,40].map(elapsed=>{
    const spikes=data.directSpikeWindows[`first${elapsed}ms`],trace=data.trace[(20+elapsed)/2-1];
    return [`first${elapsed}ms`,{
     rawSpikes48:response(spikes.raw48.positiveBaselineDifference,spikes.raw48.negativeBaselineDifference,data.deltaHz),
     pooledSpikes28:response(spikes.pooled28.positiveBaselineDifference,spikes.pooled28.negativeBaselineDifference,data.deltaHz),
     filteredRate28:response(trace.plusRateDifference,trace.minusRateDifference,data.deltaHz),
     muscleForce28:response(trace.plusForceDifference,trace.minusForceDifference,data.deltaHz)}];
   })),wholeResponseForceOddSymmetry:data.windows.wholeResponse.forceOddSymmetryCosine};
 })}));
function compareVectors(a,b){
 const na=norm(a),nb=norm(b),both=a.flatMap((v,k)=>Math.abs(v)>1e-12&&Math.abs(b[k])>1e-12?[k]:[]);
 return {cosine:cosine(a,b),firstNorm:na,secondNorm:nb,normRatio:na?nb/na:null,
  relativeDifferenceNorm:Math.max(na,nb)?norm(difference(b,a))/Math.max(na,nb):null,
  jointlyNonzeroChannels:both.length,oppositeSignChannels:both.filter(k=>Math.sign(a[k])!==Math.sign(b[k])).length};
}
const pairs=[];
function compare(a,b,kind){
 if(!a||!b)return;
 const rawA=loaded.find(row=>row.name===a.name),rawB=loaded.find(row=>row.name===b.name);
 let gate;
 if(a.prefixMs===b.prefixMs){
  assert.deepEqual(rawA.result.prefixForwardStateHashes,rawB.result.prefixForwardStateHashes,'Cross-amplitude neural snapshot differs');
  assert.equal(rawA.result.prefixMuscleState,rawB.result.prefixMuscleState,'Cross-amplitude muscle snapshot differs');
  assert.deepEqual(rawA.result.prefixTrace,rawB.result.prefixTrace,'Cross-amplitude prefix trace differs');
  assert.deepEqual(rawA.baseline.trace,rawB.baseline.trace,'Cross-amplitude baseline trace differs');
  gate={passed:true,kind:'same-prefix-exact-snapshot-and-trace',checkedBaselineBlocks:rawA.baseline.trace.length,checkedPrefixBlocks:rawA.result.prefixTrace.length};
 }else gate={passed:true,kind:'same-frozen-context-different-conditioned-neural-and-muscle-state',note:'Snapshot equality is intentionally not expected across100ms and300ms conditioning.'};
 pairs.push({kind,first:a.name,second:b.name,gate,groups:groups.map(group=>{
  const x=a.groups.find(row=>row.group===group),y=b.groups.find(row=>row.group===group);
  return {group,temporalWindows:Object.fromEntries([10,20,40].map(elapsed=>{const key=`first${elapsed}ms`,xw=x.temporal.windows[key],yw=y.temporal.windows[key];
   const shared=xw.centralCountChangedNeurons.filter(id=>yw.centralCountChangedNeurons.includes(id));
   return [key,{sharedAnySampleCountNeurons:xw.centralCountChangedNeurons.filter(id=>yw.centralCountChangedNeurons.includes(id)),
    sharedAnySampleRateNeurons:xw.centralRateChangedNeurons.filter(id=>yw.centralRateChangedNeurons.includes(id)),
    sharedCountDetails:shared.map(id=>{const k=historical.analysis.wingNeuronOrder.indexOf(id);
     const samples=condition=>condition.temporal.contrasts.filter(row=>row.elapsedMs<=elapsed&&row.counts[k]!==0).map(row=>({elapsedMs:row.elapsedMs,contrast:row.counts[k],positive:row.positiveCounts[k],negative:row.negativeCounts[k]}));
     const first=samples(x),second=samples(y),as=[...new Set(first.map(row=>Math.sign(row.contrast)))],bs=[...new Set(second.map(row=>Math.sign(row.contrast)))];
     return {...neuronLabel(k),firstSamples:first,secondSamples:second,matchingSingleContrastSign:as.length===1&&bs.length===1&&as[0]===bs[0]};
    }),
    rawCountTimeCourse:compareVectors(x.temporal.contrasts.filter(row=>row.elapsedMs<=elapsed).flatMap(row=>row.counts.map(v=>v/(2*x.deltaHz))),y.temporal.contrasts.filter(row=>row.elapsedMs<=elapsed).flatMap(row=>row.counts.map(v=>v/(2*y.deltaHz))))}];
  })),windows:Object.fromEntries([10,20,40].map(elapsed=>{const key=`first${elapsed}ms`;
   return [key,Object.fromEntries(['rawSpikes48','pooledSpikes28','filteredRate28','muscleForce28'].map(metric=>{
    const va=x.windows[key][metric],vb=y.windows[key][metric];
    return [metric,{central:compareVectors(va.centralGain,vb.centralGain),positive:compareVectors(va.positiveGain,vb.positiveGain),negative:compareVectors(va.negativeGain,vb.negativeGain)}];
   }))];
  }))};
 })});
}
const get=(prefix,amplitude)=>conditions.find(row=>row.prefixMs===prefix&&row.maxPulseHz===amplitude);
compare(get(100,1.25),get(100,2.5),'amplitude');compare(get(100,2.5),get(100,5),'amplitude-historical');compare(get(100,1.25),get(100,5),'amplitude-historical');
compare(get(300,1.25),get(300,2.5),'amplitude');
compare(get(100,1.25),get(300,1.25),'conditioned-state');compare(get(100,2.5),get(300,2.5),'conditioned-state');
const complete=plan.runs.every(run=>loaded.some(row=>row.name===run.name));
const directNeuronConsistency=pairs.filter(pair=>pair.kind==='conditioned-state').map(pair=>{
 const a=conditions.find(row=>row.name===pair.first),b=conditions.find(row=>row.name===pair.second);
 return {first:pair.first,second:pair.second,groups:groups.map(group=>({group,windows:Object.fromEntries([10,20,40].map(elapsed=>{
  const key=`first${elapsed}ms`,x=a.groups.find(row=>row.group===group).windows[key].rawSpikes48,y=b.groups.find(row=>row.group===group).windows[key].rawSpikes48;
  const common=x.contrast.flatMap((v,k)=>v!==0&&y.contrast[k]!==0?[k]:[]);
  return [key,{jointlyChangedNeurons:common.map(k=>({...neuronLabel(k),firstContrast:x.contrast[k],secondContrast:y.contrast[k],
   firstPositive:x.positive[k],firstNegative:x.negative[k],secondPositive:y.positive[k],secondNegative:y.negative[k],
   sameContrastSign:Math.sign(x.contrast[k])===Math.sign(y.contrast[k]),bothStatesBidirectional:x.positive[k]*x.negative[k]<0&&y.positive[k]*y.negative[k]<0})),
   sameSignCount:common.filter(k=>Math.sign(x.contrast[k])===Math.sign(y.contrast[k])).length,
   oppositeSignCount:common.filter(k=>Math.sign(x.contrast[k])!==Math.sign(y.contrast[k])).length,
   consistentBidirectionalCount:common.filter(k=>x.positive[k]*x.negative[k]<0&&y.positive[k]*y.negative[k]<0&&Math.sign(x.contrast[k])===Math.sign(y.contrast[k])).length}];
 }))}))};
});
const summary={schemaVersion:1,kind:'frozen-context-afferent-robustness-aggregate',complete,pending,createdAt:new Date().toISOString(),sourceHashes:sources,
 scriptSha256:sha(await fs.readFile(fileURLToPath(import.meta.url))),perRunAnalyzerSha256:analyzerHash,neuralFingerprint:plan.neuralFingerprint,
 scope:'Analysis of recorded neural identification only. No body integration or new neural execution. Anatomical population stimulation has no assigned roll/pitch/yaw meaning.100/300ms prefixes change both neural and muscle conditioning; this is not a live-flight state sweep.',
 conventions:{pulseOnsetMs:20,pulseDurationMs:20,windows:'10/20/40ms since pulse onset.40ms includes20ms recovery. Spike counts integrate each full window; rates and forces are sampled at its endpoint.',
  directSpikes:'Slot3 cumulative wing-MN spikes, prior to the50ms rate filter. Raw48 MN vectors and28 mean-per-mapped-neuron vectors are separate.',
  filteredRates:'Production slot4 rate state with50ms EMA; muscle input uses this rate and native muscle dynamics.',
  symmetry:'Cosine between positive-baseline and negated negative-baseline.1 means opposite, aligned responses;0 orthogonal;negative means the two pulse signs tend to shift outputs in the same direction. Null means one or both responses are zero.',
  alignment:'Cosines compare response vectors across conditions. Central, positive and negative gains divide by requested afferent pulse size before comparing norms. Locally linear amplitude scaling would have cosine1 and gain-norm ratio1.',
  limits:'Cumulative-window counts do not resolve within-window spike timing or wingbeat-phase shifts; zero net contrast does not prove no early timed response. Every saved2ms sample is inspected separately, but sub2ms timing remains unresolved. The current decoder consumes50ms filtered rates rather than spike timestamps/wingbeat phase. Discrete low counts and only two conditioning times preclude a stable local-Jacobian claim. No body is integrated; no flight, landing, or torque result is measured.'},
 wingNeuronOrder:historical.analysis.wingNeuronOrder,muscleOrder:historical.analysis.muscleOrder,conditions,pairs,directNeuronConsistency};
await fs.writeFile(path.join(folder,'aggregate.json'),JSON.stringify(summary)+'\n');
const fmt=value=>value===null?'—':Number(value).toFixed(3),lines=[`# Recorded afferent-response robustness`,
 '',complete?'All four declared runs are complete.':'Partial analysis; remaining runs: '+pending.join(', ')+'.',
 '',`Endpoint counts alone do not establish a stable early response across the two conditioned states. Inspecting intermediate2ms samples nevertheless finds sparse timing-sensitive responses; the temporal section below preserves these positive findings rather than treating zero net count as absent information. This experiment does not establish a robust multi-axis receptor projection.`,
 '',`Only saved neural/muscle traces were analyzed. No body integration or new neural simulation occurred.`,
 '',`Each run passed its duplicate-baseline gate. All available same-prefix amplitude comparisons also matched the complete conditioned snapshot, prefix muscle state, prefix trace, and baseline trace exactly.`,
 '',`Odd-symmetry cosine compares the positive response with the negated negative response: +1 is locally odd, negative values indicate that opposite pulse signs tend to move output in the same direction. A dash means a zero response prevents defining the cosine.`,
 '',`| Prefix / pulse | Population | Raw pooled spikes10ms | Spikes20ms | Spikes40ms | Filtered rate20ms | Muscle force20ms |`,
 `|---|---|---:|---:|---:|---:|---:|`];
for(const condition of conditions)for(const group of condition.groups)lines.push(`| ${condition.prefixMs}ms / ±${group.deltaHz}Hz${condition.historical?' (historical)':''} | ${group.group} | ${fmt(group.windows.first10ms.pooledSpikes28.oddSymmetryCosine)} | ${fmt(group.windows.first20ms.pooledSpikes28.oddSymmetryCosine)} | ${fmt(group.windows.first40ms.pooledSpikes28.oddSymmetryCosine)} | ${fmt(group.windows.first20ms.filteredRate28.oddSymmetryCosine)} | ${fmt(group.windows.first20ms.muscleForce28.oddSymmetryCosine)} |`);
lines.push('',`Across-condition alignment below uses central difference per requested Hz.1 means the same output direction; a negative value is a reversal.`,
 '',`| First → second | Population | Spikes20ms | Spikes40ms | Rate20ms | Force20ms | Force20ms gain norm ratio |`,
 `|---|---|---:|---:|---:|---:|---:|`);
for(const pair of pairs)for(const group of pair.groups)lines.push(`| ${pair.first} → ${pair.second} | ${group.group} | ${fmt(group.windows.first20ms.pooledSpikes28.central.cosine)} | ${fmt(group.windows.first40ms.pooledSpikes28.central.cosine)} | ${fmt(group.windows.first20ms.filteredRate28.central.cosine)} | ${fmt(group.windows.first20ms.muscleForce28.central.cosine)} | ${fmt(group.windows.first20ms.muscleForce28.central.normRatio)} |`);
lines.push('',`The direct per-neuron state check below requires a nonzero spike-count contrast in the same MN under both100ms and300ms prefixes. "Bidirectional" additionally requires opposite effects of + and − stimulation relative to baseline in both states, with matching polarity.`,
 '',`| Matched amplitude | Population | Jointly changed MNs20ms | Same / opposite sign40ms | Consistently bidirectional MNs40ms |`,
 `|---|---|---:|---:|---:|`);
for(const pair of directNeuronConsistency)for(const group of pair.groups)lines.push(`| ${conditions.find(row=>row.name===pair.first).maxPulseHz}Hz | ${group.group} | ${group.windows.first20ms.jointlyChangedNeurons.length} | ${group.windows.first40ms.sameSignCount} / ${group.windows.first40ms.oppositeSignCount} | ${group.windows.first40ms.consistentBidirectionalCount} |`);
lines.push('',`Net spike counts can hide within-window timing shifts. The following check inspects every saved2ms cumulative-count sample, including contrasts that have vanished by the window endpoint. It still cannot resolve sub2ms timing or establish wingbeat-phase coding.`,
 '',`| Prefix / pulse | Population | First count contrast | First raw-MN rate contrast | Transient-only count MNs20ms |`,
 `|---|---|---:|---:|---:|`);
for(const condition of conditions)for(const group of condition.groups)lines.push(`| ${condition.prefixMs}ms / ±${group.deltaHz}Hz | ${group.group} | ${group.temporal.firstCentralCountContrastMs??'—'}ms | ${group.temporal.firstCentralRawRateContrastMs??'—'}ms | ${group.temporal.windows.first20ms.transientCentralCountOnlyNeurons.length} |`);
lines.push('',`Across prefixes, shared MNs with a cumulative-count contrast at *any* sampled time in the first20ms: `+pairs.filter(pair=>pair.kind==='conditioned-state').map(pair=>`${conditions.find(row=>row.name===pair.first).maxPulseHz}Hz: ${pair.groups.map(group=>group.group+'='+group.temporalWindows.first20ms.sharedAnySampleCountNeurons.length).join(', ')}`).join('; ')+'.');
for(const pair of pairs.filter(pair=>pair.kind==='conditioned-state'))for(const group of pair.groups){
 const consistent=group.temporalWindows.first20ms.sharedCountDetails.filter(row=>row.matchingSingleContrastSign);
 if(consistent.length)lines.push('',`${group.group} at±${conditions.find(row=>row.name===pair.first).maxPulseHz}Hz has matching transient contrast signs in both states: `+consistent.map(row=>`MN${row.index} (${row.muscles.join(', ')}), first prefix at${row.firstSamples.map(sample=>sample.elapsedMs).join('/')}ms and second at${row.secondSamples.map(sample=>sample.elapsedMs).join('/')}ms`).join('; ')+'. These are timing-sensitive contrasts, not evidence of a validated wingbeat-phase code or anatomical axis tuning.');
}
lines.push('',`Spike counts integrate10/20/40ms after pulse onset. The40ms window includes20ms of recovery. Filtered rate and force comparisons use the window endpoint. Normalized gain ratios include unequal realized spike recruitment; requested Hz is not guaranteed firing rate inside the graph.`,
 '',`These are artificial frozen-context identifications. The100/300ms prefixes change neural and muscle state together. They do not represent restored live-flight states or validate body torque, closed-loop flight, or anatomical rotation tuning. Full vectors, one-sided alignments, source hashes, and exact comparison gates are in aggregate.json.`,
 '',`Reproduce without a simulation:`, '',`\`\`\`sh`, `node scripts/aggregate-flight-afferent-robustness.mjs ${path.relative(root,planPath)}`,`\`\`\``);
await fs.writeFile(path.join(folder,'AGGREGATE.md'),lines.join('\n')+'\n');
console.log(JSON.stringify({complete,pending,conditions:conditions.length,pairs:pairs.length,output:path.relative(root,path.join(folder,'aggregate.json'))}));
