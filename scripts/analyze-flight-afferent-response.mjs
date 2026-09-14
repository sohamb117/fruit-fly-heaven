// Read-only analysis of a completed frozen-context neural identification.
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';

const directory=path.resolve(process.argv[2]||'reports/flight-afferent-identification/dawn-frozen-context');
const sourceBytes=await fs.readFile(path.join(directory,'result.json')),source=JSON.parse(sourceBytes);
assert(source.baselineGate.passed&&source.interpretationAllowed&&source.arms.length===2+2*source.inputGroups.length&&source.arms.every(arm=>arm.completed));
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const unpack=text=>{const bytes=Buffer.from(text,'base64');return Float32Array.from({length:bytes.length/4},(_,i)=>bytes.readFloatLE(i*4));};
const sources={'result.json':sha(sourceBytes)};
const arms={};
for(const entry of source.arms){const bytes=await fs.readFile(path.join(directory,entry.file));sources[entry.file]=sha(bytes);
 arms[entry.name]=JSON.parse(bytes).trace.map(row=>({...row,state:unpack(row.neuralState),rates:unpack(row.wingGroupRatesHz),forces:Float32Array.from(unpack(row.wingMuscleStateAt2ms).filter((_,i)=>i%3===2)),input:unpack(row.wingMuscleInput)}));}
const base=arms['baseline-a'],ids=new Map(source.readIds.map((id,k)=>[id,k]));
const wingNeuronOffsets=source.wingIds.map(id=>ids.get(id));
const names=source.wingMappings.map(mapping=>mapping.joint.split('_').at(-1)+':'+mapping.target);
const steering=source.wingMappings.flatMap((mapping,k)=>mapping.kind==='wing_steering_assumption'?[k]:[]);
const minus=(a,b)=>Array.from(a,(v,k)=>v-b[k]);
const dot=(a,b)=>a.reduce((sum,v,k)=>sum+v*b[k],0),norm=a=>Math.sqrt(dot(a,a));
const mean=v=>v.reduce((a,b)=>a+b,0)/v.length;
const cosine=(a,b)=>norm(a)&&norm(b)?dot(a,b)/(norm(a)*norm(b)):null;
const columnsMean=(rows)=>rows[0].map((_,k)=>mean(rows.map(row=>row[k])));
function singularValues(columns){
 const n=columns.length,m=columns.map(a=>columns.map(b=>dot(a,b)));
 // Jacobi rotations on the small symmetric Gram matrix, with a relative stop.
 for(let sweep=0;sweep<100;sweep++){
  let p=0,q=1,max=0;for(let i=0;i<n;i++)for(let j=i+1;j<n;j++)if(Math.abs(m[i][j])>max){p=i;q=j;max=Math.abs(m[i][j]);}
  if(max<=1e-14*Math.max(...m.map((row,k)=>Math.abs(row[k])),Number.MIN_VALUE))break;
  const angle=.5*Math.atan2(2*m[p][q],m[q][q]-m[p][p]),c=Math.cos(angle),s=Math.sin(angle),pp=m[p][p],qq=m[q][q],pq=m[p][q];
  m[p][p]=c*c*pp-2*s*c*pq+s*s*qq;m[q][q]=s*s*pp+2*s*c*pq+c*c*qq;m[p][q]=m[q][p]=0;
  for(let k=0;k<n;k++)if(k!==p&&k!==q){const kp=m[k][p],kq=m[k][q];m[k][p]=m[p][k]=c*kp-s*kq;m[k][q]=m[q][k]=s*kp+c*kq;}
 }
 return m.map((row,k)=>Math.sqrt(Math.max(0,row[k]))).sort((a,b)=>b-a).slice(0,Math.min(n,columns[0].length));
}
function diversity(columns){
 const sv=singularValues(columns),unit=columns.map(column=>{const length=norm(column);return column.map(v=>length?v/length:0);});
 const unitSv=singularValues(unit);
 return {singularValues:sv,columnNorms:columns.map(norm),conditionNumber:sv.at(-1)?sv[0]/sv.at(-1):null,
  relativeSingularValues:sv.map(v=>sv[0]?v/sv[0]:0),unitColumnSingularValues:unitSv,unitColumnConditionNumber:unitSv.at(-1)?unitSv[0]/unitSv.at(-1):null,cosines:unit.map(a=>unit.map(b=>dot(a,b)))};
}
assert.deepEqual(singularValues([[1,0],[0,2]]),[2,1]);
assert(singularValues([[1,2],[2,4]])[1]<1e-7);
const windowRows={pulse:[10,20],earlyRecovery:[20,30],lateRecovery:[30,50],wholeResponse:[10,50]};
function oddSummary(positive,negative){
 const difference=positive.map((v,k)=>v-negative[k]),length=norm(difference);
 return {positiveBaselineDifference:positive,negativeBaselineDifference:negative,positiveNegativeDifference:difference,
  oddSymmetryCosine:cosine(positive,negative.map(v=>-v)),evenOverOddNorm:length?norm(positive.map((v,k)=>v+negative[k]))/length:null,
  changedPositive:positive.filter(v=>v!==0).length,changedNegative:negative.filter(v=>v!==0).length,
  increasedInContrast:difference.filter(v=>v>0).length,decreasedInContrast:difference.filter(v=>v<0).length};
}
function spikeCounts(arm,index){
 return wingNeuronOffsets.map(k=>{const end=arm[index].state[k*8+3],start=arm[9].state[k*8+3];
  assert(Number.isInteger(start)&&Number.isInteger(end)&&end>=start,'Invalid cumulative wing-MN spike count');return end-start;});
}
const wingNeuronIndex=new Map(source.wingIds.map((id,k)=>[id,k]));
const pooledCounts=counts=>source.wingMappings.map(mapping=>mean(mapping.indices.map(id=>counts[wingNeuronIndex.get(id)])));
const rows=[];
for(const group of source.inputGroups){
 const key=group.key.replace('/','-'),plus=arms[key+'-plus'],negative=arms[key+'-minus'];
 const groupOffsets=group.indices.map(id=>ids.get(id));
 const trace=base.map((baseline,t)=>({relativeMs:baseline.relativeMs,rateDifference:minus(plus[t].rates,negative[t].rates),forceDifference:minus(plus[t].forces,negative[t].forces),
  plusRateDifference:minus(plus[t].rates,baseline.rates),minusRateDifference:minus(negative[t].rates,baseline.rates),
  plusForceDifference:minus(plus[t].forces,baseline.forces),minusForceDifference:minus(negative[t].forces,baseline.forces)}));
 const first=(field,threshold)=>trace.find(row=>row.relativeMs>20&&row[field].some(v=>Math.abs(v)>threshold))?.relativeMs??null;
 const windows={};
 for(const [name,[start,end]]of Object.entries(windowRows)){
  const selected=trace.slice(start,end),rateDifference=columnsMean(selected.map(row=>row.rateDifference)),forceDifference=columnsMean(selected.map(row=>row.forceDifference));
  const pr=columnsMean(selected.map(row=>row.plusRateDifference)),mr=columnsMean(selected.map(row=>row.minusRateDifference));
  const pf=columnsMean(selected.map(row=>row.plusForceDifference)),mf=columnsMean(selected.map(row=>row.minusForceDifference));
  windows[name]={rateDifference,forceDifference,rateGain:rateDifference.map(v=>v/(2*group.deltaHz)),forceGain:forceDifference.map(v=>v/(2*group.deltaHz)),
   rateOddSymmetryCosine:cosine(pr,mr.map(v=>-v)),forceOddSymmetryCosine:cosine(pf,mf.map(v=>-v)),
   rateEvenOverOddNorm:norm(rateDifference)?norm(pr.map((v,k)=>v+mr[k]))/norm(rateDifference):null,
   forceEvenOverOddNorm:norm(forceDifference)?norm(pf.map((v,k)=>v+mf[k]))/norm(forceDifference):null};
 }
 const afferentSamples=trace.filter(row=>[20,22,40,60,100].includes(row.relativeMs)).map(row=>{
  const t=row.relativeMs/2-1;
  const branchStats=arm=>({meanRateHz:mean(groupOffsets.map(k=>arm[t].state[k*8+4])),
   spikeCountSincePulseOnset:groupOffsets.reduce((sum,k)=>sum+arm[t].state[k*8+3]-base[9].state[k*8+3],0)});
  return {relativeMs:row.relativeMs,baseline:branchStats(base),plus:branchStats(plus),minus:branchStats(negative)};
 });
 const top=field=>windows.wholeResponse[field].map((value,k)=>({muscle:names[k],value})).sort((a,b)=>Math.abs(b.value)-Math.abs(a.value)).slice(0,8);
 const direction=field=>({positive:windows.wholeResponse[field].filter(v=>v>1e-6).length,negative:windows.wholeResponse[field].filter(v=>v< -1e-6).length,
  bilateralDifferenceNorm:norm([...new Set(source.wingMappings.map(mapping=>mapping.target))].map(type=>{
   const l=names.indexOf('left:'+type),r=names.indexOf('right:'+type);assert(l>=0&&r>=0);return windows.wholeResponse[field][l]-windows.wholeResponse[field][r];}))});
 const directSpikeWindows={};
 for(const elapsedMs of [10,20,40]){
  const index=(20+elapsedMs)/2-1,baseline=spikeCounts(base,index),positive=spikeCounts(plus,index),neg=spikeCounts(negative,index);
  const bp=pooledCounts(baseline),pp=pooledCounts(positive),mp=pooledCounts(neg);
  directSpikeWindows[`first${elapsedMs}ms`]={elapsedMs,endsAtRelativeMs:20+elapsedMs,includesRecoveryMs:Math.max(0,elapsedMs-20),
   raw48:{baselineCounts:baseline,positiveCounts:positive,negativeCounts:neg,...oddSummary(minus(positive,baseline),minus(neg,baseline))},
   pooled28:{units:'Mean spikes per mapped motor neuron in this window; no50ms neural-rate filter and no muscle dynamics.',baselineCounts:bp,positiveCounts:pp,negativeCounts:mp,
    ...oddSummary(minus(pp,bp),minus(mp,bp))}};
 }
 const directSpikeTrace=trace.slice(10).map(row=>{const index=row.relativeMs/2-1,b=spikeCounts(base,index),p=spikeCounts(plus,index),m=spikeCounts(negative,index);
  return {relativeMs:row.relativeMs,baselineCounts:b,positiveCounts:p,negativeCounts:m,
   baselineRateHz:wingNeuronOffsets.map(k=>base[index].state[k*8+4]),positiveRateHz:wingNeuronOffsets.map(k=>plus[index].state[k*8+4]),negativeRateHz:wingNeuronOffsets.map(k=>negative[index].state[k*8+4])};});
 rows.push({group:group.key,count:group.count,deltaHz:group.deltaHz,meanCurrentPlusMinusSpan:mean(group.pulses[1].cells.map((cell,k)=>cell.current-group.pulses[0].cells[k].current)),
  baselineBeforePulseIdentical:trace.slice(0,10).every(row=>row.rateDifference.every(v=>v===0)&&row.forceDifference.every(v=>v===0)),
  firstRateDifferenceAboveMicroHzMs:first('rateDifference',1e-6),firstRateDifferenceAbove1HzMs:first('rateDifference',1),firstForceDifferenceAbove1eMinus4Ms:first('forceDifference',1e-4),
  maximumAbsoluteRateDifference:Math.max(...trace.flatMap(row=>row.rateDifference.map(Math.abs))),maximumAbsoluteForceDifference:Math.max(...trace.flatMap(row=>row.forceDifference.map(Math.abs))),
  afferentSamples,directSpikeWindows,directSpikeTrace,topMeanRateChanges:top('rateDifference'),topMeanForceChanges:top('forceDifference'),rateDirection:direction('rateDifference'),forceDirection:direction('forceDifference'),windows,trace});
}
const diversityByWindow=Object.fromEntries(Object.keys(windowRows).map(window=>[window,{
 rate28:diversity(rows.map(row=>row.windows[window].rateGain)),force28:diversity(rows.map(row=>row.windows[window].forceGain)),
 steeringForce24:diversity(rows.map(row=>steering.map(k=>row.windows[window].forceGain[k]))),
 rateTimeCourse:diversity(rows.map(row=>row.trace.slice(...windowRows[window]).flatMap(t=>t.rateDifference.map(v=>v/(2*row.deltaHz))))),
 forceTimeCourse:diversity(rows.map(row=>row.trace.slice(...windowRows[window]).flatMap(t=>t.forceDifference.map(v=>v/(2*row.deltaHz)))))
}]));
const accessibleGroupRows=rows.filter(row=>row.group!=='haltere/left');
const accessibleThreeByWindow=Object.fromEntries(Object.keys(windowRows).map(window=>[window,{
 rate28:diversity(accessibleGroupRows.map(row=>row.windows[window].rateGain)),
 steeringForce24:diversity(accessibleGroupRows.map(row=>steering.map(k=>row.windows[window].forceGain[k])))
}]));
const physicalPath='reports/flight-classical-control/result.json',physicalBytes=await fs.readFile(physicalPath),physical=JSON.parse(physicalBytes);
const physicalIndices=physical.controlNames.slice(0,24).map(name=>{const index=names.indexOf(name);assert(index>=0&&names.lastIndexOf(name)===index);assert(steering.includes(index));return index;});
assert.equal(new Set(physicalIndices).size,24);assert(physical.completed&&physical.responseJacobian.length===4);
const transferredPhysicalScreen={source:physicalPath,sourceSha256:sha(physicalBytes),scope:'Transferred quasi-steady screening only; not measured torque from these neural trials, not a current-runtime plant Jacobian, and not proof of controllability.',
 assumptions:'Historical plant used all24 force inputs=.35 and common power=.85, gains1, no force reference, fixed nonwing joints, restrained level body and16-cycle averaging. This screen maps neural force differences into that historical operating point and omits all four power-muscle changes. Reference offsets change the actual mean trajectory and can therefore change its Jacobian.',
 outputUnits:physical.conventions.calibrationResponse,physicalColumnToNeuralIndex:physicalIndices,windows:{}};
for(const window of Object.keys(windowRows)){
 const projected=rows.map(row=>physical.responseJacobian.map(matrixRow=>physicalIndices.reduce((sum,k,j)=>sum+matrixRow[j]*row.windows[window].forceGain[k],0)));
 transferredPhysicalScreen.windows[window]={groupColumns:projected,allTestedTorqueDiversity:diversity(projected.map(column=>column.slice(1))),accessibleThreeTorqueDiversity:diversity(projected.filter((_,k)=>rows[k].group!=='haltere/left').map(column=>column.slice(1)))};
}
const baselineMuscles=names.map((name,k)=>({name,meanRateHz:mean(base.map(t=>t.rates[k])),minimumRateHz:Math.min(...base.map(t=>t.rates[k])),maximumRateHz:Math.max(...base.map(t=>t.rates[k])),
 meanForce:mean(base.map(t=>t.forces[k])),fractionFullExcitation:mean(base.map(t=>Number(t.input[k*5]>=1)))}));
const report={schemaVersion:1,kind:'frozen-context-response-analysis',sourceHashes:sources,scriptSha256:sha(await fs.readFile(fileURLToPath(import.meta.url))),neuralFingerprint:source.neuralFingerprint,
 scope:'Deterministic finite-amplitude input-response measurements at one artificial frozen sensory operating point. Signed response means increased vs decreased stimulation, not assigned roll/pitch/yaw. Singular values quantify measured response-column diversity, not closed-loop controllability or a validated local derivative.',
 interpretation:{pulseOnsetMs:20,delays:'Reported first readout time minus20ms; readout resolution2ms. First1e-6Hz/1Hz/1e-4 force thresholds are descriptive, not statistical significance.',
 rate:'Production50ms exponentially filtered firing-rate state; not an instantaneous firing rate.',
 oddSymmetry:'Cosine between +baseline difference and negated -baseline difference;1 is odd-symmetric. Even/odd norm0 is locally linear; large values warn against interpreting central differences as an infinitesimal Jacobian.',
 singularValues:'Unweighted Euclidean channel units per uniform requestedHz per afferent. Unequal group sizes, baseline firing and realized spike recruitment make raw gains across groups noncommensurate. Unit-column singular values isolate response directions.',
 limitations:['One neural state and one amplitude per group; no amplitude scaling or state robustness test.','The muscles have frozen mechanical/internal inputs and no body feedback.','The four unsigned organ/side populations have no assigned axis receptive fields.','Mean response includes20ms stimulation and60ms post-stimulus evolution.']},
 accessibleThreeContext:'Haltere-left had zero asynchronous gate in the recorded native context; the other three groups had nonzero angular-norm sensitivity. Current live rotation input still has at most one unsigned scalar dimension because all use the same norm(angularVelocity). Independent anatomical-group stimulation here tests potential after an explicitly modeled encoder change.',
 groupOrder:rows.map(row=>row.group),accessibleGroupOrder:accessibleGroupRows.map(row=>row.group),muscleOrder:names,wingNeuronOrder:source.wingIds,
 directSpikeAnalysis:'Raw cumulative spike-count state at slot3; counts since20ms pulse onset sampled after10/20/40ms. first20ms is the pulse, first40ms includes20ms recovery. Raw48 vectors and28 mean-per-mapped-neuron vectors are separate; neither includes the50ms rate smoother.',
 rows,baselineMuscles,diversityByWindow,accessibleThreeByWindow,transferredPhysicalScreen};
await fs.writeFile(path.join(directory,'analysis.json'),JSON.stringify(report)+'\n');
console.log(JSON.stringify({output:path.join(directory,'analysis.json'),groups:rows.map(row=>({group:row.group,rateDelayMs:row.firstRateDifferenceAbove1HzMs===null?null:row.firstRateDifferenceAbove1HzMs-20,forceDelayMs:row.firstForceDifferenceAbove1eMinus4Ms===null?null:row.firstForceDifferenceAbove1eMinus4Ms-20,
 peakRate:row.maximumAbsoluteRateDifference,peakForce:row.maximumAbsoluteForceDifference,forceOddCosine:row.windows.wholeResponse.forceOddSymmetryCosine,forceEvenOverOdd:row.windows.wholeResponse.forceEvenOverOddNorm,
 directSpikes:Object.fromEntries(Object.entries(row.directSpikeWindows).map(([key,value])=>[key,{cosine:value.pooled28.oddSymmetryCosine,evenOverOdd:value.pooled28.evenOverOddNorm,changedPlus:value.raw48.changedPositive,changedMinus:value.raw48.changedNegative}]))})),
 accessibleThreeForceSummary:Object.fromEntries(Object.entries(accessibleThreeByWindow).map(([key,value])=>[key,{conditionNumber:value.steeringForce24.conditionNumber,unitSingularValues:value.steeringForce24.unitColumnSingularValues}])),
 transferredThreeTorqueSummary:Object.fromEntries(Object.entries(transferredPhysicalScreen.windows).map(([key,value])=>[key,{conditionNumber:value.accessibleThreeTorqueDiversity.conditionNumber,unitConditionNumber:value.accessibleThreeTorqueDiversity.unitColumnConditionNumber}]))},null,2));
