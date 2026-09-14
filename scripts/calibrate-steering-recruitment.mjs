// Mean-force calibration on a frozen, byte-verified native muscle trajectory.
// No neural execution, new body simulation, reward, pose fitting or feedback.
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {createWasmCore,WasmMuscles} from '../packages/banc-runtime/src/wasm.js';
import {createMotorExcitation} from '../web/flybody-motor-excitation.js';
import {STEERING_MUSCLE_TYPES,FLIGHT_PARAMETER_NAMES,flightParametersToInterpreter} from '../web/training/flight-parameters.js';

const args=Object.fromEntries(process.argv.slice(2).map(value=>value.replace(/^--/,'').split('=')));
const inputFile=args.profiles||'reports/steering-recruitment-calibration/legacy-replay/muscle-profiles.json';
const output=args.output||'reports/steering-recruitment-calibration/hill80-n1';
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const inputBytes=await fs.readFile(inputFile),capture=JSON.parse(inputBytes);
assert.equal(capture.kind,'exact-native-muscle-input-replay');
assert.equal(capture.baselineGate.passed,true);assert.equal(capture.observationGate.passed,true);
const sourcePaths={
 '/banc-engine/src/wasm.js':'packages/banc-runtime/src/wasm.js',
 '/banc-engine/src/model.js':'packages/banc-runtime/src/model.js',
 '/banc-engine/dist/core.js':'packages/banc-runtime/dist/core.js',
 '/banc-engine/dist/core.wasm':'packages/banc-runtime/dist/core.wasm',
 '/training/flight-parameters.js':'web/training/flight-parameters.js'};
const sourceHashes={};
for(const [url,file]of Object.entries(sourcePaths)){
 sourceHashes[url]=hash(await fs.readFile(file));
 assert.equal(sourceHashes[url],capture.sourceHashes[url],'Muscle solver/parameter contract changed since capture: '+url);
}
const helperPath='web/flybody-motor-excitation.js',helperSha256=hash(await fs.readFile(helperPath));
const configBytes=await fs.readFile('web/training/config.json'),config=JSON.parse(configBytes);
assert.deepEqual(config.parameters.map(p=>p.name),FLIGHT_PARAMETER_NAMES);
const decode=text=>{
 const bytes=Buffer.from(text,'base64');assert.equal(bytes.length%4,0);
 const values=new Float32Array(bytes.length/4);
 for(let i=0;i<values.length;i++)values[i]=bytes.readFloatLE(i*4);
 assert(values.every(Number.isFinite));return values;
};
const encode=values=>{const bytes=Buffer.alloc(values.length*4);for(let i=0;i<values.length;i++)bytes.writeFloatLE(values[i],i*4);return bytes.toString('base64');};
const n=capture.dimensions.muscleCount,mappings=capture.mappings;
assert.equal(n,135);assert.equal(mappings.length,n);assert.equal(capture.motorNeuronIndices.length,805);
const steering=mappings.flatMap((mapping,index)=>mapping.kind==='wing_steering_assumption'?[{index,...mapping,side:mapping.joint.endsWith('_left')?'left':'right'}]:[]);
const power=mappings.flatMap((mapping,index)=>mapping.kind==='asynchronous_wing'?[{index,...mapping}]:[]);
assert.equal(steering.length,24);assert.equal(power.length,4);
for(const type of STEERING_MUSCLE_TYPES)assert.deepEqual(steering.filter(row=>row.target===type).map(row=>row.side).sort(),['left','right']);
const steeringIndices=new Set(steering.map(row=>row.index));
const recruitment={steering:{kind:'hill',halfActivationHz:80,exponent:1}};
const oldDecoder=createMotorExcitation(),newDecoder=createMotorExcitation(recruitment);
const frameRates=capture.frames.map(frame=>{
 const rates=decode(frame.ratesHz);assert.equal(rates.length,capture.motorNeuronIndices.length);
 const byIndex=new Map(capture.motorNeuronIndices.map((index,k)=>[index,rates[k]]));
 return mappings.map(mapping=>mapping.indices.reduce((sum,index)=>sum+(byIndex.get(index)||0),0)/mapping.indices.length);
});
const core=await createWasmCore(),legacy=new WasmMuscles(core,n),hill=new WasmMuscles(core,n);
const initial=decode(capture.initial.packed.muscleWasmState.data);
assert.equal(initial.length,n*3);
core.HEAPF32.set(initial,legacy.state/4);core.HEAPF32.set(initial,hill.state/4);
const profiles=[],window=[];
const gate={legacyStatesByteExact:true,checkedMuscleSteps:0,nonSteeringStatesByteExact:true,
 unchangedNonExcitationInputs:true,unchangedPowerMappings:true,changedExcitationMappings:[],nativeReplay:capture.baselineGate};
const changed=new Set();let elapsed=0;
try{
 for(const [index,row]of capture.profiles.entries()){
  assert.equal(row.index,index);assert(Math.abs(row.timeBefore-elapsed)<1e-8);assert.equal(row.dt,.001);
  assert.equal(row.frameIndex,Math.floor(index/2));
  const input=decode(row.input),nextInput=input.slice(),rates=frameRates[row.frameIndex];
  assert.equal(input.length,n*5);
  for(let k=0;k<n;k++){
   const expected=capture.frames[row.frameIndex].bodyOptions.coupling?Math.fround(oldDecoder.fromRate(mappings[k].kind,rates[k])):0;
   assert(Object.is(input[k*5],expected),'Legacy excitation differs from recorded motor rates');
   if(steeringIndices.has(k))nextInput[k*5]=capture.frames[row.frameIndex].bodyOptions.coupling?newDecoder.fromRate(mappings[k].kind,rates[k]):0;
   if(!Object.is(nextInput[k*5],input[k*5]))changed.add(k);
   for(let column=1;column<5;column++)assert(Object.is(nextInput[k*5+column],input[k*5+column]));
  }
  const oldState=legacy.step(input,row.dt),newState=hill.step(nextInput,row.dt);
  assert.equal(encode(oldState),row.state,'Native legacy muscle state differs at step '+index);
  for(let k=0;k<n;k++)if(!steeringIndices.has(k))for(let field=0;field<3;field++)assert(Object.is(oldState[k*3+field],newState[k*3+field]),'Nonsteering muscle changed');
  gate.checkedMuscleSteps++;
  const weight=Math.max(0,Math.min(elapsed+row.dt,.28)-Math.max(elapsed,.1));
  if(weight>1e-12)window.push({index,weight,rates,oldInput:input,newInput:nextInput,oldState,newState});
  profiles.push({index,frameIndex:row.frameIndex,timeBefore:row.timeBefore,dt:row.dt,input:encode(nextInput),state:encode(newState)});
  elapsed+=row.dt;
 }
}finally{legacy.dispose();hill.dispose();}
assert(elapsed>=.28);assert.equal(gate.checkedMuscleSteps,capture.profiles.length);
gate.changedExcitationMappings=[...changed].sort((a,b)=>a-b);
assert(gate.changedExcitationMappings.every(index=>steeringIndices.has(index)));
const windowSeconds=window.reduce((sum,row)=>sum+row.weight,0);assert(Math.abs(windowSeconds-.18)<1e-9);
function stats(values){
 const mean=values.reduce((sum,[value,weight])=>sum+value*weight,0)/windowSeconds;
 const variance=values.reduce((sum,[value,weight])=>sum+(value-mean)**2*weight,0)/windowSeconds;
 return {mean,standardDeviation:Math.sqrt(variance),minimum:Math.min(...values.map(([v])=>v)),maximum:Math.max(...values.map(([v])=>v))};
}
const metric=fn=>stats(window.map(row=>[fn(row),row.weight]));
const perMapping=steering.map(mapping=>({index:mapping.index,target:mapping.target,side:mapping.side,indices:mapping.indices,
 ratesHz:metric(row=>row.rates[mapping.index]),oldExcitation:metric(row=>row.oldInput[mapping.index*5]),newExcitation:metric(row=>row.newInput[mapping.index*5]),
 oldForce:metric(row=>row.oldState[mapping.index*3+2]),newForce:metric(row=>row.newState[mapping.index*3+2])}));
const baselineParameters=capture.captureProvenance.job.parameters;
const baseline=flightParametersToInterpreter(baselineParameters);
assert.equal(baseline.powerGain,1.5);assert.equal(baseline.deploymentTauScale,1);assert.equal(baseline.frequencyScale,1);
const parameters=[...baselineParameters],perType=[];
for(const [typeIndex,type]of STEERING_MUSCLE_TYPES.entries()){
 const pair=perMapping.filter(row=>row.target===type),left=pair.find(row=>row.side==='left'),right=pair.find(row=>row.side==='right');
 const oldMean=(left.oldForce.mean+right.oldForce.mean)/2,newMean=(left.newForce.mean+right.newForce.mean)/2;
 const evidence=oldMean>1e-12&&newMean>1e-12,multiplier=evidence?oldMean/newMean:1;
 assert(Number.isFinite(multiplier)&&multiplier>0);
 const gains={};
 for(const [offset,key]of [[0,'biasGain'],[1,'amplitudeGain']]){
  const index=3+typeIndex*2+offset;
  parameters[index]=baselineParameters[index]+Math.log(multiplier);
  assert(parameters[index]>=config.parameters[index].min&&parameters[index]<=config.parameters[index].max,'Mean-force calibration exceeds configured bound: '+FLIGHT_PARAMETER_NAMES[index]);
  gains[key]={old:baseline.steering[type][key],calibrated:Math.exp(parameters[index])};
 }
 const sides=pair.map(row=>({side:row.side,oldMeanForce:row.oldForce.mean,newMeanForce:row.newForce.mean,
  compensatedNewMeanForce:multiplier*row.newForce.mean,residualMeanForce:multiplier*row.newForce.mean-row.oldForce.mean,
  residualMeanFraction:row.oldForce.mean>1e-12?(multiplier*row.newForce.mean-row.oldForce.mean)/row.oldForce.mean:null,
  oldForceStandardDeviation:row.oldForce.standardDeviation,compensatedNewForceStandardDeviation:multiplier*row.newForce.standardDeviation}));
 const leftIndex=left.index,rightIndex=right.index;
 perType.push({type,evidence:evidence?'mean-force calibration':'silent or unresolved; baseline gain retained',
  oldMeanForce:oldMean,newMeanForce:newMean,multiplier,compensatedMeanForce:newMean*multiplier,
  aggregateResidualMeanForce:newMean*multiplier-oldMean,gains,sides,
  oldBilateralForceDifference:metric(row=>row.oldState[leftIndex*3+2]-row.oldState[rightIndex*3+2]),
  compensatedNewBilateralForceDifference:metric(row=>multiplier*(row.newState[leftIndex*3+2]-row.newState[rightIndex*3+2])),
  clippedBilateralDistinctionsRestoredFraction:window.reduce((sum,row)=>sum+(row.rates[leftIndex]!==row.rates[rightIndex]&&row.oldInput[leftIndex*5]===row.oldInput[rightIndex*5]&&row.newInput[leftIndex*5]!==row.newInput[rightIndex*5]?row.weight:0),0)/windowSeconds});
}
assert.equal(parameters.length,27);flightParametersToInterpreter(parameters);
assert.equal(hash(await fs.readFile(helperPath)),helperSha256,'Recruitment helper changed during calibration');
const report={schemaVersion:1,kind:'frozen-muscle-steering-recruitment-calibration',createdAt:new Date().toISOString(),
 sourceFile:inputFile,sourceSha256:hash(inputBytes),captureFile:capture.captureFile,captureSha256:capture.captureSha256,
 sourceConfigHash:capture.configHash,captureProvenance:capture.captureProvenance,sourceHashes,helperPath,helperSha256,
 parameterBoundsReferenceConfigHash:hash(configBytes),scriptSha256:hash(await fs.readFile('scripts/calibrate-steering-recruitment.mjs')),
 scope:'Native WASM muscle replay on frozen legacy input profiles. Only 24 steering excitations change. All 135 initial states and length/velocity/Fmax/energy inputs are preserved; no new neural or body feedback, reward, pose or outcome fitting.',
 recruitment,window:{fromSeconds:.1,toSeconds:.28,durationSeconds:windowSeconds,samples:window.length,
  weighting:'Time integral of the force held during each native 1 ms control interval; both sides receive equal weight.'},
 gate,counts:{muscles:n,steering:steering.length,power:power.length,totalWing:steering.length+power.length},
 perMapping,perType,baselineParameters,parameterNames:FLIGHT_PARAMETER_NAMES,parameters,
 candidateInterpreter:flightParametersToInterpreter(parameters),
 caveats:['The Hill curve is a model prior, not measured steering-muscle physiology.',
  'Matching aggregate pre-basis mean force preserves average linear contribution for each type, not each side or the subsequently clipped wing command.',
  'A silent muscle supplies no calibration evidence; its baseline gain is retained.',
  'Temporal and bilateral statistics use this frozen recorded motor sequence. They do not demonstrate closed-loop flight or generalization.']};
await fs.mkdir(output,{recursive:true});
await fs.writeFile(path.join(output,'result.json'),JSON.stringify(report)+'\n');
await fs.writeFile(path.join(output,'candidate.json'),JSON.stringify({schemaVersion:1,kind:'mean-force-calibrated-steering-candidate',
 sourceCalibrationSha256:hash(JSON.stringify(report)+'\n'),sourceConfigHash:capture.configHash,recruitment,
 parameterNames:FLIGHT_PARAMETER_NAMES,parameters,requiresOptInMetadata:true,validatedFlight:false},null,2)+'\n');
await fs.writeFile(path.join(output,'hill-muscle-profiles.json'),JSON.stringify({schemaVersion:1,kind:'frozen-input-hill-muscle-replay',
 sourceFile:inputFile,sourceSha256:hash(inputBytes),helperSha256,recruitment,gate,initial:capture.initial.packed.muscleWasmState,
 dimensions:capture.dimensions,mappings,encoding:capture.encoding,profiles})+'\n');
const rows=perType.map(row=>`| ${row.type} | ${row.oldMeanForce.toFixed(6)} | ${row.newMeanForce.toFixed(6)} | ${row.multiplier.toFixed(4)} | ${row.gains.biasGain.calibrated.toFixed(5)} | ${row.sides.map(side=>side.residualMeanFraction===null?'n/a':(side.residualMeanFraction*100).toFixed(2)+'%').join(' / ')} |`);
await fs.writeFile(path.join(output,'README.md'),`# Steering recruitment calibration\n\nAll ${gate.checkedMuscleSteps} legacy native muscle states are byte-identical to the captured profiles. Nonsteering states are byte-identical under the Hill variant. The 100–280 ms force integral sets a single bilateral multiplier per type, applied to its bias and amplitude gains. No reward or body outcome was optimized.\n\n| Type | Legacy mean force | Hill mean force | Multiplier | New gain | Per-side mean residual |\n|---|---:|---:|---:|---:|---|\n${rows.join('\n')}\n\nForce is the native muscle output with the captured Fmax input. Aggregate mean is preserved before the fixed steering basis and command clipping. The candidate keeps power 1.5, deployment scale 1 and frequency scale 1. The Hill curve requires explicit model metadata; this is not a flight validation. Exact profiles, per-side variability and source identities are in the adjacent JSON artifacts.\n`);
console.log(JSON.stringify({output,gate,window:report.window,perType:perType.map(row=>({type:row.type,multiplier:row.multiplier,newGain:row.gains.biasGain.calibrated,restoredFraction:row.clippedBilateralDistinctionsRestoredFraction})),parameters}));
