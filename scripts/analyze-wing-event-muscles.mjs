// Read completed characterization records only; no simulation or parameter fit.
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const directory=path.resolve(process.argv[2]||'reports/flight-motor-event-observer/characterization');
const sha=bytes=>createHash('sha256').update(bytes).digest('hex'),sources={};
async function read(file){const bytes=await fs.readFile(file);sources[path.relative(root,file)]=sha(bytes);return JSON.parse(bytes);}
const plan=await read(path.join(directory,'plan.json')),result=await read(path.join(directory,'result.json'));
assert(result.completed&&!result.error&&result.cases.length===10);assert.equal(result.planSha256,sources[path.relative(root,path.join(directory,'plan.json'))]);
const cases=[];
for(const row of result.cases){const value=await read(path.join(directory,row.file));assert(row.completed&&!row.error&&value.completed&&!value.error&&value.sourcePinsVerified);cases.push(value);}
const synthetic=cases.filter(row=>row.condition.kind==='periodic-synthetic'),recorded=cases.find(row=>row.condition.kind==='recorded-events');
const familyOf=m=>m.kind==='wing_steering_assumption'?'steering':m.target==='dorsal_longitudinal_muscle'?'dlm':'dvm';
const curves=synthetic.map(row=>{
 assert(row.settle.passed&&row.settle.checkedIntervals===2000&&row.analysisSamples===2000);
 const positions=row.mappings.map((m,k)=>familyOf(m)===row.condition.family?k:-1).filter(k=>k>=0),first=row.mappingStatistics[positions[0]];
 for(const k of positions)for(const key of ['nativeActivation','nativeFatigue','nativeForce'])assert.deepEqual(row.mappingStatistics[k][key],first[key],'Identical synthetic trains should have identical native output');
 return {family:row.condition.family,rateHz:row.condition.rateHz,name:row.name,
  meanExcitation:first.intervalMeanExcitation.mean,meanActivation:first.nativeActivation.mean,meanFatigue:first.nativeFatigue.mean,meanForce:first.nativeForce.mean,
  forceMinimum:first.nativeForce.min,forceMaximum:first.nativeForce.max,fatigueSlopePerSecond:row.fatigue.meanSlopePerSecond[positions[0]],
  maximumExcitation:first.intervalMeanExcitation.max,highExcitationFraction:first.intervalMeanExcitation.fractionAtLeast095,settle:row.settle};
});
const monotonicity=Object.fromEntries(['steering','dlm','dvm'].map(family=>{
 const rows=curves.filter(row=>row.family===family).sort((a,b)=>a.rateHz-b.rateHz);
 return [family,Object.fromEntries(['meanExcitation','meanActivation','meanForce'].map(key=>[key,rows.every((row,k)=>k===0||row[key]>rows[k-1][key])]))];
}));
assert(Object.values(monotonicity).every(row=>Object.values(row).every(Boolean)));
const captureFile=Object.keys(plan.artifacts).find(file=>file.includes('001-with-motor-events-'));
const capture=await read(path.join(root,captureFile));assert.equal(sources[captureFile],plan.artifacts[captureFile]);
const packetPositions=new Map(capture.motorEvents[0].indices.map((id,k)=>[id,k]));
const packetWindow=capture.motorEvents.filter(packet=>packet.timeMs>100&&packet.timeMs<=280);assert.equal(packetWindow.length,90);
const average=values=>values.reduce((a,b)=>a+b,0)/values.length;
const sd=stats=>Math.sqrt(Math.max(0,stats.rms*stats.rms-stats.mean*stats.mean));
const recordedMappings=recorded.mappings.map((mapping,k)=>{
 const native=recorded.mappingStatistics[k],legacy=packetWindow.map(packet=>Math.max(0,Math.min(1,average(mapping.indices.map(id=>packet.ratesHz[packetPositions.get(id)]))/80)));
 return {...mapping,family:familyOf(mapping),meanExcitation:native.intervalMeanExcitation.mean,meanActivation:native.nativeActivation.mean,
  meanFatigue:native.nativeFatigue.mean,meanForce:native.nativeForce.mean,forceStandardDeviation:sd(native.nativeForce),
  minimumExcitation:native.intervalMeanExcitation.min,maximumExcitation:native.intervalMeanExcitation.max,highExcitationFraction:native.intervalMeanExcitation.fractionAtLeast095,
  forceMinimum:native.nativeForce.min,forceMaximum:native.nativeForce.max,
  legacyCommand:{mean:average(legacy),minimum:Math.min(...legacy),maximum:Math.max(...legacy),fractionAtCeiling:legacy.filter(value=>value===1).length/legacy.length,
   definition:'clamp(saved50msEMA/80,0,1) at90 block-end samples in(100,280]ms. No legacy native force replay was performed.'}};
});
const dlmUnits=recorded.units.map((unit,k)=>({...unit,statistics:recorded.unitStatistics[k]})).filter(unit=>unit.family==='dlm');
const steeringContrasts=[...new Set(recordedMappings.filter(row=>row.family==='steering').map(row=>row.target))].map(target=>{
 const left=recordedMappings.find(row=>row.target===target&&row.joint.endsWith('_left')),right=recordedMappings.find(row=>row.target===target&&row.joint.endsWith('_right'));
 return {target,legacyCommandMeanLeftMinusRight:left.legacyCommand.mean-right.legacyCommand.mean,
  eventMeanExcitationLeftMinusRight:left.meanExcitation-right.meanExcitation,eventNativeMeanForceLeftMinusRight:left.meanForce-right.meanForce,
  legacyBothAlwaysAtCeiling:left.legacyCommand.fractionAtCeiling===1&&right.legacyCommand.fractionAtCeiling===1,
  eventNativeForceVariability:{leftStandardDeviation:left.forceStandardDeviation,rightStandardDeviation:right.forceStandardDeviation}};
});
const report={schemaVersion:1,kind:'offline-wing-event-muscle-analysis',sourceHashes:sources,
 scriptSha256:sha(await fs.readFile(fileURLToPath(import.meta.url))),nativeExecution:false,
 windows:{synthetic:'(4000,6000]ms after zero-state initialization',recorded:'(100,280]ms of unchanged0–388ms train'},
 curves,monotonicity,recordedMappings,dlmUnits,steeringContrasts,
 checks:{completedCases:10,allPeriodicChecksPassed:true,allEventSequenceChecksPassed:true,
  maximumQuadratureDiscrepancy:Math.max(...cases.map(row=>row.quadratureDiscrepancyMax)),
  maximumFloat32ExcitationRounding:Math.max(...cases.map(row=>row.excitationFloat32RoundingMax))},
 interpretation:{synthetic:'Monotonic excitation, activation and force across the tested finite grid and window; no extrapolation to all rates or steady fatigue. DVM equality follows from explicitly borrowed DLM priors.',
  recordedDLM:'All10 units stay above0.95 intervalmean excitation throughout the analysiswindow; slow calcium-like kernels saturate under the high-rate recorded train. This does not calibrate or conceal its neural recruitment.',
  steering:'Bilateral and temporal force contrasts remain in the event-native assay, including types whose two legacyclamp command samples are identical at1. The original moving-body/native schedule is not a matched legacy force control, so no force-effect improvement or closed-loop benefit is established.'}};
await fs.writeFile(path.join(directory,'analysis.json'),JSON.stringify(report,null,2)+'\n');
const n=(x,d=3)=>x.toFixed(d),lines=['# Event-to-native-muscle results','',
 'All ten fixed conditions completed with source and event-sequence checks intact. Synthetic kernel, excitation and activation passed the final 2 s periodicity test; no brain, body, controller or optimizer was simulated.',
 '', '| Family | MN rate (Hz) | Mean excitation | Native activation | Native fatigue | Native force |',
 '|---|---:|---:|---:|---:|---:|'];
for(const row of curves)lines.push(`| ${row.family} | ${row.rateHz} | ${n(row.meanExcitation)} | ${n(row.meanActivation)} | ${n(row.meanFatigue)} | ${n(row.meanForce)} |`);
lines.push('', 'Means use (4000, 6000] ms, after 4 s from zero state, with length 1, shortening velocity 0, Fmax 1 and energy 1. The tested excitation, activation and force means increase with rate. Force is normalized model output. Native fatigue remains active and can drift; periodic excitation and activation do not imply stationary force. The native 15 ms rise / 40 ms fall gate also makes mean activation differ from mean excitation.',
 '', 'DVM matches DLM because this experiment explicitly borrows the same kernel constants and excites every unit in a mapping with the same synthetic phase. That equality is an assumption check, not independent physiology evidence.',
 '', 'The original 1,460 events and all 48 identities were replayed without extension. During (100, 280] ms, all ten DLM units stayed above 0.95 mean excitation in every interval; their mean excitations span '+n(Math.min(...dlmUnits.map(row=>row.statistics.meanExcitation.mean)),4)+'–'+n(Math.max(...dlmUnits.map(row=>row.statistics.meanExcitation.mean)),4)+'. The left/right DLM groups produce mean excitation '+recordedMappings.filter(row=>row.family==='dlm').map(row=>n(row.meanExcitation,4)).join('/')+' and normalized native force '+recordedMappings.filter(row=>row.family==='dlm').map(row=>n(row.meanForce,4)).join('/')+'. Thus this event prior remains nearly saturated under the recorded high DLM rates; it does not turn that neural output into a calibrated low-rate motor regime.',
 '', '| Steering type | Legacy command L/R | Event mean excitation L/R | Event native force L/R |',
 '|---|---:|---:|---:|');
for(const target of ['b2_muscle','i1_muscle','i2_muscle','iv1_muscle']){
 const rows=['left','right'].map(side=>recordedMappings.find(row=>row.target===target&&row.joint.endsWith('_'+side)));
 lines.push(`| ${target.replace('_muscle','')} | ${rows.map(row=>n(row.legacyCommand.mean)).join('/')} | ${rows.map(row=>n(row.meanExcitation)).join('/')} | ${rows.map(row=>n(row.meanForce)).join('/')} |`);
}
lines.push('', 'The legacy values above are calculated directly from the saved 50 ms rate field using `clamp(rate/80,0,1)`, at its original 2 ms sample times. Both b2, both i2 and both iv1 commands equal 1 at every sample in this window. Their event-driven native outputs still have bilateral differences and temporal variability. This establishes retained contrast in the new assay and lost contrast at the old command clamp. It is **not a matched comparison of native force**: the old mechanical and energy history and update schedule differ, and no legacy native replay was run here. No flight or stabilization benefit is claimed.',
 '', 'The largest 4/8-point quadrature discrepancy was '+report.checks.maximumQuadratureDiscrepancy.toExponential(3)+' (limit 1e-7). Per-unit and per-mapping statistics, waveform samples, fatigue slopes and exact source/input hashes remain in the case files and `analysis.json`. No mean-force normalization or parameter fitting was applied.',
 '', 'Reproduce the offline analysis:', '', '```sh', 'node scripts/analyze-wing-event-muscles.mjs '+path.relative(root,directory), '```',
 '', 'The pinned characterization script can run in a new output directory:', '', '```sh','node scripts/characterize-wing-event-muscles.mjs reports/new-output --prepare-only','node scripts/characterize-wing-event-muscles.mjs reports/new-output','```',
 '', 'The current event kernels, unit recruitment scale, equal pooling, DVM borrowing and native activation stage remain explicit model priors. The next proposed check is a restrained mechanical phase response with a matched no-event baseline, not a new physiological fit.');
await fs.writeFile(path.join(directory,'README.md'),lines.join('\n')+'\n');
console.log(JSON.stringify({output:directory,completedCases:10,monotonicity,recordedDLMMeanExcitation:dlmUnits.map(row=>row.statistics.meanExcitation.mean),nativeExecution:false}));
