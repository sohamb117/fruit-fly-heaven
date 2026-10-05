#!/usr/bin/env node
// Offline bounded linear calibration. No brain/body/optimizer allocation.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
import {buildMotorDecoderContract, createMotorDecoder, validateMotorDecoderVector} from '../web/motor-decoder.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const finite = x => typeof x === 'number' && Number.isFinite(x);
const requireThat = (condition, message) => { if (!condition) throw new Error(message); };
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);
const copy = x => JSON.parse(JSON.stringify(x));
function vector(value, n, predicate, label) {
  requireThat(Array.isArray(value) && value.length === n && value.every(predicate), `${label}: invalid values or length`);
  return value;
}

/** Minimize sum_i weight_i*(X_i*w-y_i)^2 + ridge*||w-initial||^2.
 * Cyclic exact coordinate minimization with explicit box constraints. The
 * prior is the starting checkpoint; unexcited columns retain their values.
 * Validation rows never enter this fit or its convergence criterion.
 */
export function fitBoundedRidge(rows, targets, weights, initial, bounds,
  {ridge = 1e-5, maxSweeps = 1000, tolerance = 1e-9} = {}) {
  const n = rows.length, p = initial.length;
  requireThat(n > 0 && p > 0 && targets.length === n && weights.length === n && bounds.length === p,
    'Invalid regression dimensions');
  requireThat(finite(ridge) && ridge > 0 && Number.isInteger(maxSweeps) && maxSweeps > 0 && maxSweeps <= 10000 &&
    finite(tolerance) && tolerance > 0, 'Invalid regression options');
  requireThat(rows.every(row => row.length === p && Array.from(row).every(finite)) && targets.every(finite) &&
    weights.every(x => finite(x) && x > 0) && Math.abs(weights.reduce((a,b) => a+b,0)-1) < 1e-8,
    'Invalid regression rows, targets or normalized weights');
  bounds.forEach(([lo, hi], j) => requireThat(finite(lo) && finite(hi) && lo < hi && finite(initial[j]) &&
    initial[j] >= lo && initial[j] <= hi, 'Invalid coefficient bounds or initial value'));
  const columns = Array.from({length:p}, (_, j) => Float64Array.from(rows, row => row[j]));
  const norm = columns.map(column => column.reduce((sum, x, i) => sum + weights[i]*x*x, 0));
  const w = Float64Array.from(initial), residual = Float64Array.from(targets, (y, i) => y-dot(rows[i],w));
  const objective = () => residual.reduce((sum, r, i) => sum + weights[i]*r*r, 0) +
    w.reduce((sum, x, j) => sum + ridge*(x-initial[j])**2, 0);
  const before = objective(); let sweeps = 0, maxDelta = Infinity, projectedGradient = Infinity;
  for (; sweeps < maxSweeps; sweeps++) {
    maxDelta = 0;
    for (let j = 0; j < p; j++) {
      let correlation = 0;
      for (let i = 0; i < n; i++) correlation += weights[i]*columns[j][i]*residual[i];
      const next = clamp(w[j] + (correlation-ridge*(w[j]-initial[j]))/(norm[j]+ridge), ...bounds[j]);
      const delta = next-w[j];
      if (delta !== 0) for (let i = 0; i < n; i++) residual[i] -= columns[j][i]*delta;
      w[j] = next; maxDelta = Math.max(maxDelta, Math.abs(delta));
    }
    if (maxDelta <= tolerance) { sweeps++; break; }
  }
  projectedGradient = 0;
  for (let j = 0; j < p; j++) {
    let gradient = ridge*(w[j]-initial[j]);
    for (let i = 0; i < n; i++) gradient -= weights[i]*columns[j][i]*residual[i];
    const projected = w[j] - clamp(w[j]-gradient, ...bounds[j]);
    projectedGradient = Math.max(projectedGradient, Math.abs(projected));
  }
  const after = objective();
  requireThat(finite(after) && after <= before + 1e-10*Math.max(1,before), 'Regression did not decrease its training objective');
  return {values:Array.from(w), sweeps, converged:maxDelta <= tolerance,
    maxCoordinateChange:maxDelta, projectedGradient, objectiveBefore:before, objectiveAfter:after,
    unexcitedColumns:norm.flatMap((value,j) => value === 0 ? [j] : []),
    columnsAtBounds:Array.from(w).flatMap((value,j) => value === bounds[j][0] || value === bounds[j][1] ? [j] : [])};
}

export function validateDataset(dataset, contract, ioSha256) {
  requireThat(dataset?.schemaVersion === 1 && dataset.kind === 'paired-motor-decoder-calibration' &&
    dataset.intervalMs === 1 && dataset.targetSpace === 'pre-clamp-decoder', 'Unsupported dataset contract');
  requireThat(dataset.ioSha256 === ioSha256, 'Dataset IO hash does not match the supplied prepared IO');
  assert.deepEqual(dataset.indices, contract.indices, 'Dataset motor indices/order differ from decoder contract');
  requireThat(['synthetic-teacher','native-control-calibration','measured-control'].includes(dataset.provenance?.kind) &&
    typeof dataset.provenance.description === 'string' && dataset.provenance.description.length > 0 &&
    dataset.provenance.sources && typeof dataset.provenance.sources === 'object', 'Explicit target provenance is required');
  requireThat(Array.isArray(dataset.trials) && dataset.trials.length >= 2, 'At least two independent trials are required');
  const ids = new Set(), groupSplits = new Map(); let power = null;
  for (const trial of dataset.trials) {
    requireThat(typeof trial.id === 'string' && trial.id.length > 0 && !ids.has(trial.id), 'Trial IDs must be unique'); ids.add(trial.id);
    requireThat(typeof trial.groupId === 'string' && trial.groupId.length > 0 && ['train','validation'].includes(trial.split),
      'Each trial needs groupId and an explicit train/validation split');
    requireThat(!groupSplits.has(trial.groupId) || groupSplits.get(trial.groupId) === trial.split,
      'Related trials cannot cross train/validation groups'); groupSplits.set(trial.groupId,trial.split);
    requireThat(Array.isArray(trial.history) && trial.history.length <= 5, 'Declare 0–5 preceding 1 ms excitation rows, oldest first');
    trial.history.forEach(row => vector(row,48,x=>finite(x)&&x>=0&&x<=1,'Initial history'));
    requireThat(Array.isArray(trial.rows) && trial.rows.length > 0, 'Empty trial');
    trial.rows.forEach((row,i) => {
      requireThat(row.step === i+1 && finite(row.wingPhaseRadians), 'Rows require sequential 1 ms steps and finite phase');
      vector(row.unitExcitation,48,x=>finite(x)&&x>=0&&x<=1,'Unit excitation');
      vector(row.targetSteering,2,x=>Array.isArray(x),'Steering sides').forEach(v=>vector(v,3,finite,'Steering target'));
      const hasPower = Object.hasOwn(row,'targetPower');
      if (power === null) power = hasPower;
      requireThat(power === hasPower, 'Power targets must be present on every row or none');
      if (hasPower) vector(row.targetPower,2,x=>finite(x)&&x>=0&&x<=4,'Unclipped power target');
    });
  }
  requireThat([...groupSplits.values()].includes('train') && [...groupSplits.values()].includes('validation'),
    'A separate validation group is required');
  return {fitPower:power};
}

export function validateInitialCheckpoint(checkpoint,contract,ioSha256) {
  requireThat(checkpoint?.schemaVersion===1&&checkpoint.kind==='motor-decoder-calibration-checkpoint'&&
    checkpoint.profile===contract.version&&checkpoint.contractSha256===hash(JSON.stringify(contract))&&
    checkpoint.ioSha256===ioSha256,'Initial checkpoint schema/profile/contract mismatch');
  return Array.from(validateMotorDecoderVector(contract,checkpoint.parameters));
}

// No duplicate FIR implementation: use the production decoder's feature clock.
function layout(contract) {
  requireThat(contract.parameters?.length === 672 && contract.indices?.length === 48, 'Unexpected decoder dimensions');
  const groups = ['steering','power'].flatMap(kind => [0,1].flatMap(side =>
    (kind === 'power' ? [null] : [0,1,2]).map(axis => ({kind, side, axis,
      name:kind+'_'+['left','right'][side]+(axis===null?'':'_'+['yaw','roll','pitch'][axis]),
      indices:contract.parameters.flatMap((p,i) => p.kind === kind && p.sideIndex === side &&
        (kind === 'power' || p.axisIndex === axis) ? [i] : [])}))));
  groups.forEach(g=>requireThat(g.indices.length === (g.kind === 'power'?12:108), 'Unexpected per-output feature count'));
  requireThat(new Set(groups.flatMap(g=>g.indices)).size === 672, 'Output groups do not cover each coefficient exactly once');
  return groups;
}

function compileTrials(io, contract, dataset) {
  const initial = contract.parameters.map(p=>p.initial);
  return dataset.trials.map(trial => {
    const decoder = createMotorDecoder(io,initial);
    trial.history.forEach(excitation=>decoder.advance(excitation));
    return {...trial, rows:trial.rows.map(row=>{
      decoder.advance(row.unitExcitation);
      const features = decoder.features(row.wingPhaseRadians);
      requireThat(features instanceof Float64Array && features.length === 672 && features.every(finite), 'Invalid decoder features');
      return {...row,features:features.slice()};
    })};
  });
}

const target = (row,g) => g.kind === 'power' ? row.targetPower[g.side] : row.targetSteering[g.side][g.axis];
const prediction = (row,g,parameters) => g.indices.reduce((sum,j)=>sum+row.features[j]*parameters[j],0);
const executed = (value,g) => clamp(value,g.kind==='power'?0:-.25,g.kind==='power'?1:.25);

function evaluate(trials,groups,parameters) {
  function summarize(selected) {
    if (!selected.length) return null;
    const outputs = groups.map(g=>{
      let square=0, clippedSquare=0, maximum=0, predictionsClipped=0, targetsClipped=0, samples=0;
      for (const trial of selected) for (const row of trial.rows) {
        const weight = 1/(selected.length*trial.rows.length), y=target(row,g), p=prediction(row,g,parameters);
        square+=weight*(p-y)**2; clippedSquare+=weight*(executed(p,g)-executed(y,g))**2;
        maximum=Math.max(maximum,Math.abs(p-y)); predictionsClipped+=Number(executed(p,g)!==p);
        targetsClipped+=Number(executed(y,g)!==y); samples++;
      }
      return {name:g.name,rawRmse:Math.sqrt(square),executedRmse:Math.sqrt(clippedSquare),
        maximumAbsoluteRawError:maximum,predictionsClipped,targetsClipped,samples};
    });
    return {trialCount:selected.length,groupCount:new Set(selected.map(t=>t.groupId)).size,
      rawRmse:Math.sqrt(outputs.reduce((sum,o)=>sum+o.rawRmse**2,0)/outputs.length),outputs};
  }
  return {train:summarize(trials.filter(t=>t.split==='train')),validation:summarize(trials.filter(t=>t.split==='validation')),
    trials:trials.map(t=>({id:t.id,groupId:t.groupId,split:t.split,...summarize([t])}))};
}

// Pivoted Gram-Schmidt on weighted observed columns: a numerical span check,
// not a claim that correlated physiological parameters have been identified.
function featureCoverage(rows, weights, indices) {
  const columns = indices.map(j=>Float64Array.from(rows,(r,i)=>r.features[j]*Math.sqrt(weights[i])));
  const squaredNorm = column=>column.reduce((sum,x)=>sum+x*x,0);
  const originalNorms=columns.map(squaredNorm), maximum=Math.sqrt(Math.max(...originalNorms));
  const threshold=maximum*1e-8, pivots=[], remaining=columns.map((column,k)=>({column,k}));
  while (remaining.length) {
    remaining.sort((a,b)=>squaredNorm(b.column)-squaredNorm(a.column));
    const next=remaining.shift(), norm=Math.sqrt(squaredNorm(next.column));
    if (norm <= threshold || norm === 0) break;
    pivots.push(norm); const q=next.column.map(x=>x/norm);
    // Reorthogonalize to limit false rank caused by cancellation.
    for (const item of remaining) for(let pass=0;pass<2;pass++) {
      const projection=dot(item.column,q);
      for(let i=0;i<q.length;i++)item.column[i]-=projection*q[i];
    }
  }
  return {columns:indices.length,numericalRank:pivots.length,rankRelativeTolerance:1e-8,
    pivotNorms:pivots,unexcitedParameters:indices.filter((_,i)=>originalNorms[i]===0),
    note:'Weighted pivoted-QR span estimate; not singular values, condition number, or biological identifiability.'};
}

export function calibrateDataset(io,contract,dataset,initial,options={}) {
  const groups=layout(contract).filter(g=>g.kind!=='power'||dataset.trials[0].rows[0].targetPower!==undefined);
  requireThat(initial.length===672,'Initial vector length must be 672');
  createMotorDecoder(io,initial); // Validate bounds using the actual implementation.
  const trials=compileTrials(io,contract,dataset), train=trials.filter(t=>t.split==='train');
  const rows=train.flatMap(t=>t.rows), weights=train.flatMap(t=>t.rows.map(()=>1/(train.length*t.rows.length)));
  const parameters=initial.slice(), fits=[],coverage={};
  for(const group of groups) {
    const x=rows.map(row=>group.indices.map(j=>row.features[j])), y=rows.map(row=>target(row,group));
    const fit=fitBoundedRidge(x,y,weights,group.indices.map(j=>initial[j]),
      group.indices.map(j=>[contract.parameters[j].min,contract.parameters[j].max]),options);
    group.indices.forEach((j,k)=>{parameters[j]=fit.values[k];});
    fits.push({output:group.name,...fit,values:undefined,
      unexcitedParameters:fit.unexcitedColumns.map(k=>group.indices[k]),
      parametersAtBounds:fit.columnsAtBounds.map(k=>group.indices[k])});
    const key=group.kind+'_'+group.side;
    if(!coverage[key])coverage[key]=featureCoverage(rows,weights,group.indices);
  }
  createMotorDecoder(io,parameters);
  return {parameters,fits,coverage,before:evaluate(trials,groups,initial),after:evaluate(trials,groups,parameters),
    changedParameterCount:parameters.filter((x,i)=>x!==initial[i]).length,
    updateL2:Math.sqrt(parameters.reduce((sum,x,i)=>sum+(x-initial[i])**2,0)),
    unchangedPowerWithoutTargets:groups.some(g=>g.kind==='power')?null:parameters.slice(0,24).every((x,i)=>x===initial[i])};
}

function syntheticTeacher(io,contract,ioSha256) {
  let seed=0x51a7f00d;
  const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/2**32;};
  const groups=layout(contract),teacher=contract.parameters.map(p=>p.kind==='power'?.5+random()*1.2:(random()-.5)*.012);
  const dataset={schemaVersion:1,kind:'paired-motor-decoder-calibration',intervalMs:1,targetSpace:'pre-clamp-decoder',
    ioSha256,indices:contract.indices.slice(),provenance:{kind:'synthetic-teacher',
      description:'Deterministic random known decoder coefficients and independent excitation sequences. Fitting plumbing only; no native or biological targets.',sources:{}},
    trials:Array.from({length:6},(_,trialIndex)=>{
      const decoder=createMotorDecoder(io,teacher),rows=[];
      for(let i=0;i<256;i++) {
        const unitExcitation=Array.from({length:48},random),wingPhaseRadians=(trialIndex*.7+(i+1)*2*Math.PI*(.180+.013*trialIndex))%(2*Math.PI);
        decoder.advance(unitExcitation);const features=decoder.features(wingPhaseRadians);
        const row={step:i+1,unitExcitation,wingPhaseRadians,targetSteering:[[0,0,0],[0,0,0]],targetPower:[0,0]};
        for(const group of groups) {
          const value=prediction({features},group,teacher);
          if(group.kind==='power')row.targetPower[group.side]=value;else row.targetSteering[group.side][group.axis]=value;
        }
        const actual=decoder.sample(wingPhaseRadians);
        for(const group of groups) assert(Math.abs((group.kind==='power'?actual.power[group.side]:actual.steering[group.side][group.axis])-
          executed(target(row,group),group))<1e-12,'Feature prediction differs from executed decoder');
        rows.push(row);
      }
      return {id:'synthetic-'+trialIndex,groupId:'independent-sequence-'+trialIndex,split:trialIndex<4?'train':'validation',history:[],rows};
    })};
  return {dataset,teacher};
}

function solverSelfTests() {
  const bounded=fitBoundedRidge([[1,0],[1,0]],[3,3],[.5,.5],[0,.7],[[0,1],[0,1]]);
  assert.deepEqual(bounded.values,[1,.7]);assert.deepEqual(bounded.unexcitedColumns,[1]);
  const signed=fitBoundedRidge([[1],[-1]],[-.1,.1],[.5,.5],[0],[[-.25,.25]],{ridge:1e-10});
  assert(Math.abs(signed.values[0]+.1)<1e-8,'Signed fit did not recover negative coefficient');
  assert.throws(()=>fitBoundedRidge([[NaN]],[0],[1],[0],[[-1,1]]),/Invalid regression/);
}

async function main() {
  const args = {};
  for (const argument of process.argv.slice(2)) {
    requireThat(argument.startsWith('--'), 'Use --name=value options');
    const at = argument.indexOf('='), name = argument.slice(2,at < 0 ? undefined : at);
    requireThat(!Object.hasOwn(args,name), 'Repeated option: '+name);
    args[name] = at < 0 ? true : argument.slice(at+1);
  }
  requireThat(Object.keys(args).every(k=>['help','self-test','dataset','io','output','initial','ridge','sweeps','tolerance'].includes(k)),
    'Unknown option');
  for(const flag of ['help','self-test'])requireThat(args[flag]===undefined||args[flag]===true,'Use --'+flag+' without a value');
  if (args.help) { console.log('node scripts/calibrate-motor-decoder.mjs --dataset=paired.json --output=NEW_DIRECTORY [--io=data/prepared/banc888/io.json]\nExplicit synthetic plumbing check: --self-test --output=NEW_DIRECTORY'); return; }
  requireThat(Boolean(args['self-test']) !== Boolean(args.dataset), 'Choose --dataset=FILE or explicit --self-test');
  requireThat(typeof args.output === 'string' && args.output.length > 0, 'An explicit new output directory is required');
  const ioPath = path.resolve(ROOT,args.io || 'data/prepared/banc888/io.json');
  const ioBytes = await fs.readFile(ioPath), io = JSON.parse(ioBytes), contract = buildMotorDecoderContract(io);
  layout(contract);const ioSha256=hash(ioBytes),contractSha256=hash(JSON.stringify(contract));
  const sourcePaths=['scripts/calibrate-motor-decoder.mjs','web/motor-decoder.js','web/training/flight-parameters.js'];
  const sourceHashes=Object.fromEntries(await Promise.all(sourcePaths.map(async p=>[p,hash(await fs.readFile(path.join(ROOT,p)))])));
  const output=path.resolve(ROOT,args.output);
  try {await fs.access(output);throw new Error('Output directory already exists; preserve previous calibration artifacts');}
  catch(error){if(error.code!=='ENOENT')throw error;}
  const options={ridge:args.ridge===undefined?1e-5:Number(args.ridge),maxSweeps:args.sweeps===undefined?1000:Number(args.sweeps),
    tolerance:args.tolerance===undefined?1e-9:Number(args.tolerance)};
  let dataset,datasetBytes,teacher=null;
  if(args['self-test']) {
    solverSelfTests();({dataset,teacher}=syntheticTeacher(io,contract,ioSha256));datasetBytes=JSON.stringify(dataset)+'\n';
  } else {datasetBytes=await fs.readFile(path.resolve(ROOT,args.dataset));dataset=JSON.parse(datasetBytes);}
  validateDataset(dataset,contract,ioSha256);
  // Source descriptors are local files, never fetched from external locations.
  // Real calibration must pin its independently generated target evidence.
  requireThat(dataset.provenance.kind==='synthetic-teacher'||Object.keys(dataset.provenance.sources).length>0,
    'Native/measured targets require at least one pinned local evidence source');
  for(const [file,sha] of Object.entries(dataset.provenance.sources)) {
    requireThat(typeof sha==='string'&&/^[a-f0-9]{64}$/.test(sha),'Invalid evidence digest');
    requireThat(hash(await fs.readFile(path.resolve(ROOT,file)))===sha,'Calibration evidence source hash mismatch: '+file);
  }
  let initial=contract.parameters.map(p=>p.initial),initialCheckpointSha256=null;
  if(args.initial) {
    const bytes=await fs.readFile(path.resolve(ROOT,args.initial)),checkpoint=JSON.parse(bytes);initialCheckpointSha256=hash(bytes);
    initial=validateInitialCheckpoint(checkpoint,contract,ioSha256);
  }
  const started=performance.now(), result=calibrateDataset(io,contract,dataset,initial,options);
  if(args['self-test']) {
    const compatible={schemaVersion:1,kind:'motor-decoder-calibration-checkpoint',profile:contract.version,
      contractSha256,ioSha256,parameters:initial};
    assert.deepEqual(validateInitialCheckpoint(compatible,contract,ioSha256),initial);
    for(const patch of [{schemaVersion:2},{profile:'unknown'},{ioSha256:'0'.repeat(64)},{contractSha256:'0'.repeat(64)}])
      assert.throws(()=>validateInitialCheckpoint({...compatible,...patch},contract,ioSha256),/mismatch/);
    assert.throws(()=>validateInitialCheckpoint({...compatible,parameters:[1]},contract,ioSha256),/exactly 672/);
    const leaked=copy(dataset);leaked.trials.at(-1).groupId=leaked.trials[0].groupId;
    assert.throws(()=>validateDataset(leaked,contract,ioSha256),/cannot cross/);
    const invalid=copy(dataset);invalid.trials[0].rows[1].step=7;
    assert.throws(()=>validateDataset(invalid,contract,ioSha256),/sequential/);
    requireThat(result.after.validation.rawRmse < .05*result.before.validation.rawRmse,
      'Synthetic fitting did not reduce held-out raw RMSE by at least 95%');
    requireThat(result.changedParameterCount===672,'Synthetic fixture failed to update every coefficient');
    const steeringOnly=copy(dataset);
    steeringOnly.trials.forEach(trial=>{trial.rows=trial.rows.slice(0,12);trial.rows.forEach(row=>{delete row.targetPower;});});
    validateDataset(steeringOnly,contract,ioSha256);
    const partial=calibrateDataset(io,contract,steeringOnly,initial,{...options,maxSweeps:10});
    requireThat(partial.unchangedPowerWithoutTargets===true,'Unlabeled power weights changed');
  }
  for(const [file,sha]of Object.entries(sourceHashes))requireThat(hash(await fs.readFile(path.join(ROOT,file)))===sha,'Source changed during fit: '+file);
  requireThat(hash(await fs.readFile(ioPath))===ioSha256,'Prepared IO changed during fit');
  for(const [file,sha]of Object.entries(dataset.provenance.sources))
    requireThat(hash(await fs.readFile(path.resolve(ROOT,file)))===sha,'Evidence source changed during fit: '+file);
  const provenance={sourceHashes,ioPath,ioSha256,contractSha256,datasetSha256:hash(datasetBytes),initialCheckpointSha256,
    targetProvenance:dataset.provenance,options,executionSeconds:(performance.now()-started)/1000,
    split:dataset.trials.map(t=>({id:t.id,groupId:t.groupId,split:t.split,rows:t.rows.length,historyRows:t.history.length})),
    objective:'Each training trial has equal total weight; bounded ridge about the initial vector. No validation-target fitting.',
    limitations:['Control prediction is not force/torque calibration or flight success.',
      '672 available coefficients do not establish 672 independently identifiable biological parameters.',
      'Pre-clamp targets are mandatory; clipped execution metrics are reported separately.',
      'No BANC, native physics, reinforcement learning, coordinator, or deployment was run.']};
  const checkpoint={schemaVersion:1,kind:'motor-decoder-calibration-checkpoint',profile:contract.version,
    status:'unverified',parameters:result.parameters,ioSha256,contractSha256,datasetSha256:hash(datasetBytes),sourceHashes,
    targetKind:dataset.provenance.kind,converged:result.fits.every(f=>f.converged),
    trainingRmse:result.after.train.rawRmse,validationRmse:result.after.validation.rawRmse};
  const report={schemaVersion:1,kind:'motor-decoder-calibration-report',completed:true,
    syntheticSelfTest:!!args['self-test'],...provenance,...result,parameters:undefined,
    teacherMaximumCoefficientError:teacher?Math.max(...result.parameters.map((x,i)=>Math.abs(x-teacher[i]))):null};
  await fs.mkdir(path.dirname(output),{recursive:true});await fs.mkdir(output);
  await fs.writeFile(path.join(output,'checkpoint.json'),JSON.stringify(checkpoint,null,2)+'\n',{flag:'wx'});
  await fs.writeFile(path.join(output,'report.json'),JSON.stringify(report,null,2)+'\n',{flag:'wx'});
  await fs.writeFile(path.join(output,'dataset.json'),datasetBytes,{flag:'wx'});
  await fs.writeFile(path.join(output,'contract.json'),JSON.stringify(contract,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({output,synthetic:report.syntheticSelfTest,changedParameterCount:result.changedParameterCount,
    trainingRmse:[result.before.train.rawRmse,result.after.train.rawRmse],
    validationRmse:[result.before.validation.rawRmse,result.after.validation.rawRmse],
    converged:result.fits.every(f=>f.converged),executionSeconds:provenance.executionSeconds}));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url))
  main().catch(error => { console.error(error.stack || error); process.exitCode = 1; });
