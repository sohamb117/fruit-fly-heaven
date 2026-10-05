import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {interfaceParameterValues,evaluatePromotionGate,createOperatorPromotionGate} from '../training/interface-parameters.js';
import {parameterValues,PARAMETER_NAMES} from '../training/episode.js';
import {safeUrl,manifestFingerprint} from '../../scripts/package-training-client.mjs';
const config={parameterContract:'banc-motor-interface-v2',environmentVersion:'banc-flybody-interface-v2',behaviorCriteriaVersion:2,freezeNeuralParameters:true,parameters:[{name:'hindGripScale',min:.25,max:1,initial:1}]};
test('interface candidates freeze neural parameters and retain unselected reference mechanics',()=>{
 const v=interfaceParameterValues(config,[.5]);assert.equal(v.profile.hindGripScale,.5);assert.equal(v.profile.dlmRateHz,80);assert.equal(v.neuralParametersFrozen,true);
 assert.throws(()=>interfaceParameterValues({...config,freezeNeuralParameters:false},[.5]));
 assert.throws(()=>interfaceParameterValues({...config,parameterContract:'v1'},[.5]));
 assert.throws(()=>interfaceParameterValues({...config,environmentVersion:'banc-flybody-rl-v1'},[.5]));
 assert.throws(()=>interfaceParameterValues({...config,behaviorCriteriaVersion:1},[.5]));
 assert.throws(()=>interfaceParameterValues({...config,behaviorCriteriaVersion:undefined},[.5]));
 assert.throws(()=>interfaceParameterValues(config,[2]));
 assert.throws(()=>interfaceParameterValues({...config,parameters:[{name:'muscle_wing_log_gain',min:0,max:1,initial:0}]},[.5]));
});
const identity={configHash:'a'.repeat(64),modelFingerprint:'b'.repeat(64),interfaceHash:'c'.repeat(64)};
const run=(executionId,stage='flight',success=true,score=2)=>({...identity,executionId,stage,split:'validation',results:[11,12].map(seed=>({seed,success,return:score}))});
const input=()=>({candidate:run('candidate'),incumbent:{...run('incumbent','flight',false,1),interfaceHash:'d'.repeat(64)},
 requiredSeeds:[11,12],trainingSeeds:[1,2],requiredPriorStages:['posture'],stage:'flight',minimumImprovement:.1,
 mechanicalProtocolHash:'e'.repeat(64),mechanicalGate:{...identity,executionId:'mechanics',passed:true,protocolHash:'e'.repeat(64)},priorStages:[run('posture','posture')]});
function attested(input){
 const gate=createOperatorPromotionGate(),record=value=>value&&typeof value==='object'?gate.recordLocalEvidence(value):value;
 return {gate,contract:{...input,candidate:record(input.candidate),incumbent:record(input.incumbent),mechanicalGate:record(input.mechanicalGate),priorStages:input.priorStages?.map(record)}};
}
const evaluate=input=>{const {gate,contract}=attested(input);return gate.evaluate(contract);};

test('operator promotion requires matched independent execution and an identified mechanical pass',()=>{
 const result=evaluate(input());assert.equal(result.passed,true);assert.equal(result.improvement,1);assert.equal(result.biologicalSuccessValidated,false);
 const x=input();x.stage=x.candidate.stage=x.incumbent.stage='posture';x.requiredPriorStages=[];x.priorStages=[];
 assert.equal(evaluate(x).passed,true,'An explicitly initial stage may have no prior requirements');
});

test('anonymous JSON flags and copied trusted reports cannot authorize promotion',()=>{
 const raw=input();for(const run of [raw.candidate,raw.incumbent,raw.mechanicalGate,...raw.priorStages])run.trustedExecution=true;
 assert.equal(evaluatePromotionGate(raw).reason,'untrusted_mechanical_evidence');
 const {gate,contract}=attested(raw);
 assert.equal(gate.evaluate({...contract,candidate:JSON.parse(JSON.stringify(contract.candidate))}).reason,'untrusted_execution_evidence');
 const otherGate=createOperatorPromotionGate();assert.equal(otherGate.evaluate(contract).reason,'untrusted_mechanical_evidence');
});

test('operator evidence snapshots cannot change after trust registration',()=>{
 const raw=input(),{gate,contract}=attested(raw);
 raw.candidate.results[0].return=-9;raw.mechanicalGate.passed=false;
 assert(Object.isFrozen(contract.candidate));assert(Object.isFrozen(contract.candidate.results));assert(Object.isFrozen(contract.candidate.results[0]));
 assert.throws(()=>{contract.candidate.results[0].return=-9;},TypeError);
 assert.equal(gate.evaluate(contract).passed,true);
});

test('failed or missing mechanical evidence reports a calibration block without asserting trust',()=>{
 for(const mechanicalGate of [undefined,{passed:false},{passed:'true'}]){
  assert.equal(evaluatePromotionGate({mechanicalGate}).reason,'mechanical_calibration_required');
  assert.equal(createOperatorPromotionGate().evaluate({mechanicalGate}).passed,false);
 }
});

test('all identities and independent execution identifiers are required',()=>{
 for(const mutate of [
  x=>delete x.candidate.configHash,x=>x.candidate.interfaceHash='',x=>x.incumbent.modelFingerprint='f'.repeat(64),
  x=>x.incumbent.configHash='f'.repeat(64),x=>x.candidate.stage='posture',x=>x.incumbent.executionId=x.candidate.executionId,
  x=>x.mechanicalGate.interfaceHash='d'.repeat(64),x=>x.mechanicalGate.modelFingerprint='f'.repeat(64),
  x=>x.mechanicalGate.configHash='f'.repeat(64),x=>x.mechanicalGate.protocolHash='f'.repeat(64),
  x=>delete x.mechanicalGate.executionId,x=>x.priorStages[0].executionId=x.candidate.executionId,
 ]){const x=input();mutate(x);assert.equal(evaluate(x).passed,false);}
});

test('matched validation seeds must be unique uint32 values and disjoint from training',()=>{
 for(const mutate of [
  x=>x.requiredSeeds=[],x=>x.requiredSeeds=[11,11],x=>x.requiredSeeds=[11,-1],x=>x.requiredSeeds=[11,2**32],
  x=>x.trainingSeeds=[12],x=>x.trainingSeeds=[],x=>delete x.trainingSeeds,x=>x.trainingSeeds=[1,1],
  x=>x.candidate.split='train',x=>x.incumbent.results.pop(),x=>x.candidate.results[0].seed=12,
  x=>x.candidate.results[0].seed=1,x=>x.incumbent.results[0].seed='11',
 ]){const x=input();mutate(x);assert.equal(evaluate(x).passed,false);}
});

test('every declared prior stage must pass with the same candidate and model on matched seeds',()=>{
 for(const mutate of [
  x=>delete x.requiredPriorStages,x=>x.requiredPriorStages=['posture','posture'],x=>x.requiredPriorStages=['flight'],
  x=>x.requiredPriorStages=['posture','feeding'],x=>x.priorStages=[],x=>x.priorStages[0].stage='feeding',
  x=>x.priorStages[0].interfaceHash='d'.repeat(64),x=>x.priorStages[0].configHash='f'.repeat(64),
  x=>x.priorStages[0].modelFingerprint='f'.repeat(64),x=>x.priorStages[0].results[0].success=false,
  x=>x.priorStages[0].results[0].seed=1,
 ]){const x=input();mutate(x);assert.equal(evaluate(x).passed,false);}
});

test('promotion rejects invalid or failed trials and finite-score tricks',()=>{
 for(const mutate of [
  x=>x.candidate.results[0].success=false,x=>x.candidate.results[0].cancelled=true,x=>x.incumbent.results[0].error='crashed',
  x=>x.candidate.results[0].return=NaN,x=>x.candidate.results[0].return=Infinity,x=>x.candidate.results[0].return=11,
  x=>x.incumbent.results[0].return=-11,x=>delete x.incumbent.results[0].success,
 ]){const x=input();mutate(x);assert.equal(evaluate(x).passed,false);}
});

test('minimum improvement is finite and nonnegative, and even the default requires improvement',()=>{
 for(const minimumImprovement of [NaN,Infinity,-1,'0',null]){const x=input();x.minimumImprovement=minimumImprovement;assert.equal(evaluate(x).reason,'invalid_validation_contract');}
 const tie=input();delete tie.minimumImprovement;tie.candidate.results.forEach(r=>r.return=1);assert.equal(evaluate(tie).reason,'candidate_did_not_improve');
 const threshold=input();threshold.minimumImprovement=1.01;assert.equal(evaluate(threshold).reason,'candidate_did_not_improve');
 threshold.minimumImprovement=1;assert.equal(evaluate(threshold).passed,true);
});

const flightConfig=()=>({schemaVersion:1,dtMs:.5,bodyBlockMs:2,vision:false,
 parameters:PARAMETER_NAMES.map(name=>({name,min:-1,max:1,initial:0}))});
test('the episode decoder requires the ordered 27-parameter physical interpreter contract',()=>{
 const current=flightConfig(),vector=current.parameters.map(p=>p.initial),values=parameterValues(current,vector);
 assert.equal(PARAMETER_NAMES.length,27);assert.equal(new Set(PARAMETER_NAMES).size,27);
 assert.deepEqual(values.vector,vector);assert.deepEqual(Object.keys(values.gains),PARAMETER_NAMES);
 assert(Object.values(values.gains).every(v=>v===1));
 for(const count of [1,14,26,28])assert.throws(()=>parameterValues(current,Array(count).fill(0)),/parameter count/);
 const reordered=structuredClone(current);[reordered.parameters[0],reordered.parameters[1]]=[reordered.parameters[1],reordered.parameters[0]];
 assert.throws(()=>parameterValues(reordered,vector),/Unknown training parameter contract/);
 assert.throws(()=>parameterValues({...current,...config},[.5]),/Unknown training parameter contract/);
 const oldNames=structuredClone(current);oldNames.parameters[0].name='synapse_exc_log_gain';
 assert.throws(()=>parameterValues(oldNames,vector),/Unknown training parameter contract/);
});

test('packaging permits explicit asset URLs and rejects traversal or ambiguous paths',()=>{
 for(const url of ['/training/episode.js','/banc-data/manifest.json','/body-model/flybody-mujoco.json'])assert.equal(safeUrl(url),url);
 for(const url of ['training/episode.js','https://example.com/file.js','/../secret','/training/./episode.js','/training//episode.js','/.env','/training/file.js?x=1','/training/file.js#x','/training/%2e%2e/file.js'])assert.throws(()=>safeUrl(url),/Unsafe bundle URL/);
});

test('packaging fingerprint binds exact interpreter source, asset names and manifest order',()=>{
 const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
 const episode=fs.readFileSync(new URL('../training/episode.js',import.meta.url));
 const parameters=fs.readFileSync(new URL('../training/flight-parameters.js',import.meta.url));
 const assets={'/training/episode.js':sha(episode),'/training/flight-parameters.js':sha(parameters)};
 const fingerprint=manifestFingerprint(assets);
 assert.equal(fingerprint,sha(`/training/episode.js:${sha(episode)}\n/training/flight-parameters.js:${sha(parameters)}\n`));
 assert.notEqual(manifestFingerprint({...assets,'/training/episode.js':sha(Buffer.concat([episode,Buffer.from('\n')]))}),fingerprint);
 assert.notEqual(manifestFingerprint(Object.fromEntries(Object.entries(assets).reverse())),fingerprint);
 assert.notEqual(manifestFingerprint({'/training/other.js':assets['/training/episode.js'],'/training/flight-parameters.js':assets['/training/flight-parameters.js']}),fingerprint);
});

test('packager CLI checks manifest and asset bytes before requiring the pinned BANC graph',()=>{
 // These tiny isolated repositories stop before attribution downloads, server
 // startup or archive construction. The packager is exercised without exports
 // of its private Python server template or a retired operator-policy gate.
 const temporary=fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(),'fly-package-gate-')));
 const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
 try{
  fs.mkdirSync(path.join(temporary,'scripts'));fs.mkdirSync(path.join(temporary,'web/training'),{recursive:true});
  fs.mkdirSync(path.join(temporary,'data/prepared/banc888'),{recursive:true});
  fs.copyFileSync(new URL('../../scripts/package-training-client.mjs',import.meta.url),path.join(temporary,'scripts/package-training-client.mjs'));
  const episode=fs.readFileSync(new URL('../training/episode.js',import.meta.url)),graph=Buffer.from(JSON.stringify({dataset:'BANC',materialization:887,files:{}}));
  const episodePath=path.join(temporary,'web/training/episode.js');fs.writeFileSync(episodePath,episode);
  fs.writeFileSync(path.join(temporary,'data/prepared/banc888/manifest.json'),graph);
  const assets={'/training/episode.js':sha(episode),'/banc-data/manifest.json':sha(graph)};
  const fixture={...flightConfig(),assets,modelFingerprint:manifestFingerprint(assets)};
  const run=value=>{
   fs.writeFileSync(path.join(temporary,'web/training/config.json'),JSON.stringify(value));
   const result=spawnSync(process.execPath,[path.join(temporary,'scripts/package-training-client.mjs')],{encoding:'utf8',timeout:5000});
   assert.ifError(result.error);assert.equal(result.signal,null);assert.equal(result.status,1,result.stderr);return result.stderr;
  };
  assert.match(run({...fixture,modelFingerprint:'0'.repeat(64)}),/Finalize the canonical training asset manifest first/);
  assert(!fs.existsSync(path.join(temporary,'dist')),'Bad manifest must fail before output creation');
  fs.appendFileSync(episodePath,'\n');assert.match(run(fixture),/Source checksum mismatch: web\/training\/episode\.js/);
  fs.writeFileSync(episodePath,episode);assert.match(run(fixture),/BANC v888 graph manifest required/);
  assert.deepEqual(fs.readdirSync(path.join(temporary,'dist/training-client')),[],'Failed attempts leave no staged or published bundle');
 }finally{fs.rmSync(temporary,{recursive:true,force:true});}
});
