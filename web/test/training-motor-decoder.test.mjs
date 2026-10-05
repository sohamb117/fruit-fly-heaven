import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {MOTOR_DECODER_VERSION,buildMotorDecoderContract} from '../motor-decoder.js';
import {parameterValues,validateMotorDecoderTrainingConfig} from '../training/episode.js';
import {DEFAULT_FLIGHT_INTERPRETER,flightParametersToInterpreter} from '../training/flight-parameters.js';
import {validateConfig,checkpoint,readCheckpoint,makeGeneration} from '../training/optimizer.js';

const configUrl=new URL('../training/config.json',import.meta.url);
const legacy=JSON.parse(await fs.readFile(configUrl,'utf8'));
const io=JSON.parse(await fs.readFile(new URL('../../data/prepared/banc888/io.json',import.meta.url),'utf8'));
const contract=buildMotorDecoderContract(io);
function configuration(){
 return {...structuredClone(legacy),parameterContract:MOTOR_DECODER_VERSION,
  motorDecoderContract:structuredClone(contract),freezeNeuralParameters:true,
  wingEventExcitation:{schemaVersion:1},
  parameters:contract.parameters.map(({name,min,max,initial})=>({name,min,max,initial}))};
}

test('explicit decoder config binds 672 raw coordinates to actual prepared identities',()=>{
 const config=configuration(),vector=config.parameters.map(p=>p.initial),steering=contract.powerParameterCount;
 vector[steering]=config.parameters[steering].min/2;
 const values=parameterValues(config,new Float64Array(vector),io);
 assert.equal(vector.length,672);assert.equal(contract.powerParameterCount,24);assert.equal(contract.steeringParameterCount,648);
 assert.deepEqual(values.vector,vector);assert.deepEqual(values.motorDecoder,vector);
 assert.deepEqual(values.interpreter,DEFAULT_FLIGHT_INTERPRETER);assert(!('gains' in values));
 assert(values.motorDecoder[steering]<0);assert.notEqual(values.vector,vector);
 vector[steering]=0;assert(values.vector[steering]<0);
 assert.equal(validateConfig(config),config);
});

test('calibrated bounded initials and search scales are separate from the immutable contract',()=>{
 const config=configuration(),index=contract.powerParameterCount;
 config.parameters[index].initial=.25*config.parameters[index].min;config.parameters[index].searchScale=.2;
 const declared=structuredClone(config.motorDecoderContract);
 assert.equal(parameterValues(config,config.parameters.map(p=>p.initial)).vector[index],config.parameters[index].initial);
 assert.deepEqual(config.motorDecoderContract,declared);assert.equal(validateConfig(config),config);
});

test('decoder contract, topology, bounds, event path and frozen-neural declarations fail closed',()=>{
 const changes=[
  c=>{c.parameterContract='unknown-decoder';},c=>{delete c.motorDecoderContract;},
  c=>{c.freezeNeuralParameters=false;},c=>{delete c.wingEventExcitation;},c=>{c.wingEventExcitation=null;},
  c=>{c.parameters[0].name+='-changed';},c=>{c.parameters[0].max+=1;},
  c=>{[c.parameters[0],c.parameters[1]]=[c.parameters[1],c.parameters[0]];},
  c=>{c.parameters[0].initial=Infinity;},c=>{c.parameters[0].searchScale=0;},
  c=>{c.motorDecoderContract.indices[0]+=1;},c=>{c.motorDecoderContract.extra=true;},
 ];
 for(const change of changes){
  const config=configuration();change(config);
  assert.throws(()=>parameterValues(config,config.parameters.map(p=>p.initial)));
  assert.throws(()=>validateConfig(config));
 }
 const other=structuredClone(io),mapping=other.muscles.find(m=>m.kind==='wing_steering_assumption'),old=mapping.indices[0];
 // Change a complete identity consistently, so the serialized contract is
 // internally valid but is still rejected against the loaded prepared io.
 const rootId=(BigInt(mapping.root_ids[0])+1n).toString();
 for(const muscle of other.muscles)muscle.indices.forEach((index,k)=>{if(index===old)muscle.root_ids[k]=rootId;});
 other.motor_neurons.find(m=>m.index===old).root_id=rootId;
 const config=configuration();config.motorDecoderContract=buildMotorDecoderContract(other);
 config.parameters=config.motorDecoderContract.parameters.map(({name,min,max,initial})=>({name,min,max,initial}));
 assert(validateMotorDecoderTrainingConfig(config));
 assert.throws(()=>parameterValues(config,config.parameters.map(p=>p.initial),io),/prepared io/);
});

test('decoder vectors and checkpoints cannot silently use old 27-coordinate assignments',()=>{
 const config=configuration(),vector=config.parameters.map(p=>p.initial);
 for(const bad of [vector.slice(1),Array(27).fill(0),[NaN,...vector.slice(1)],[config.parameters[0].max+1,...vector.slice(1)]])
  assert.throws(()=>parameterValues(config,bad));
 const saved=checkpoint(config,'decoder-config-pin',vector,0,config.stage);
 assert.deepEqual(readCheckpoint(saved,config,'decoder-config-pin').parameters,vector);
 assert.throws(()=>readCheckpoint({...saved,parameterNames:legacy.parameters.map(p=>p.name)},config,'decoder-config-pin'));
 assert.throws(()=>readCheckpoint({...saved,parameters:Array(27).fill(0)},config,'decoder-config-pin'));
 const unversioned={...config};delete unversioned.parameterContract;delete unversioned.motorDecoderContract;
 assert.throws(()=>validateConfig(unversioned),/parameter schema/);
});

test('legacy mapping and optimizer retain their previous parameter semantics',()=>{
 const vector=legacy.parameters.map(p=>p.initial),values=parameterValues(legacy,vector);
 assert.deepEqual(values,{vector,gains:Object.fromEntries(legacy.parameters.map((p,i)=>[p.name,Math.exp(vector[i])])),interpreter:flightParametersToInterpreter(vector)});
 assert.equal(validateConfig(legacy),legacy);assert.equal(values.motorDecoder,undefined);
 if(legacy.schemaVersion===1){
  const generation=makeGeneration(vector,0,legacy);
  assert.equal(generation.pairs.length,legacy.optimizer.populationPairs);
  assert(generation.pairs.every(pair=>pair.jobs.every(job=>job.parameters.length===27)));
 }
 const generic={...structuredClone(legacy),parameters:Array.from({length:256},(_,i)=>({name:'p'+i,min:-1,max:1,initial:0}))};
 assert.equal(validateConfig(generic),generic);
 generic.parameters.push({name:'p256',min:-1,max:1,initial:0});assert.throws(()=>validateConfig(generic),/parameter schema/);
});

test('manifest module import and builder are read-only and pin the decoder dependency',async()=>{
 const before=await fs.readFile(configUrl);
 const {buildTrainingAssetManifest}=await import('../../scripts/prepare-training-manifest.mjs');
 assert.deepEqual(await fs.readFile(configUrl),before);
 const result=await buildTrainingAssetManifest();
 const digest=bytes=>createHash('sha256').update(bytes).digest('hex');
 assert.equal(result.assets['/motor-decoder.js'],digest(await fs.readFile(new URL('../motor-decoder.js',import.meta.url))));
 assert(result.assets['/training/episode.js']);assert(result.assets['/training/flight-parameters.js']);
 assert.equal(result.modelFingerprint,digest(Object.entries(result.assets).map(([url,sha])=>`${url}:${sha}\n`).join('')));
 assert.deepEqual(await fs.readFile(configUrl),before);
});
