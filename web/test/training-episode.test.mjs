import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {parameterValues,PARAMETER_NAMES,seededRandom} from '../training/episode.js';
import {STEERING_MUSCLE_TYPES} from '../training/flight-parameters.js';
const config=JSON.parse(readFileSync(new URL('../training/config.json',import.meta.url)));
const vector=config.parameters.map(p=>p.initial);
test('canonical contract exposes only 27 interpreter gains with neutral defaults',()=>{
 assert.equal(PARAMETER_NAMES.length,27);
 assert.deepEqual(config.parameters.map(p=>p.name),PARAMETER_NAMES);
 assert(PARAMETER_NAMES.every(name=>name.startsWith('flight_')));
 const values=parameterValues(config,vector);
 assert(Object.values(values.gains).every(v=>v===1));
 assert.equal(values.interpreter.powerGain,1);assert.equal(values.interpreter.deploymentTauScale,1);assert.equal(values.interpreter.frequencyScale,1);
 for(const type of STEERING_MUSCLE_TYPES)assert.deepEqual(values.interpreter.steering[type],{biasGain:1,amplitudeGain:1});
});
test('parameter bounds include their endpoints and reject incompatible checkpoints',()=>{
 for(const key of ['min','max'])assert.equal(parameterValues(config,config.parameters.map(p=>p[key])).vector.length,27);
 assert.throws(()=>parameterValues(config,[...vector.slice(0,-1),Infinity]),/bounds/);
 const bad=[...vector];bad[0]=config.parameters[0].max+.001;assert.throws(()=>parameterValues(config,bad),/bounds/);
 assert.throws(()=>parameterValues(config,Array(18).fill(0)),/count/);
 const reordered=structuredClone(config);[reordered.parameters[3],reordered.parameters[4]]=[reordered.parameters[4],reordered.parameters[3]];
 assert.throws(()=>parameterValues(reordered,vector),/contract/);
 const old=structuredClone(config);old.parameters[0].name='synapse_exc_log_gain';assert.throws(()=>parameterValues(old,vector),/contract/);
});
test('a muscle coefficient changes only its named interpreter setting',()=>{
 const changed=[...vector],type='iii3_muscle';changed[PARAMETER_NAMES.indexOf(`flight_${type}_amplitude_log_gain`)]=Math.log(1.5);
 const {interpreter}=parameterValues(config,changed);
 assert.equal(interpreter.steering[type].amplitudeGain,1.5);
 for(const name of STEERING_MUSCLE_TYPES){assert.equal(interpreter.steering[name].biasGain,1);if(name!==type)assert.equal(interpreter.steering[name].amplitudeGain,1);}
 assert.equal(interpreter.powerGain,1);assert.equal(interpreter.frequencyScale,1);assert.equal(interpreter.deploymentTauScale,1);
});
test('seeded disturbances repeat exactly and differ across seeds',()=>{const a=seededRandom(888),b=seededRandom(888),c=seededRandom(889);const values=Array.from({length:20},()=>a());assert.deepEqual(values,Array.from({length:20},()=>b()));assert.notDeepEqual(values,Array.from({length:20},()=>c()));});
