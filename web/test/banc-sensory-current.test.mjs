import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createWasmCore} from '../../packages/banc-runtime/src/wasm.js';
import {createBancSensoryCurrentMapper} from '../banc-sensory-current.js';
import {runSensoryCurrentAssay,measurePublicRuntime} from '../../scripts/verify-banc-sensory-current.mjs';

test('configured sensory currents reproduce held-out rates through the production public runtime',async()=>{
 const report=await runSensoryCurrentAssay();
 assert.equal(report.preparedProfileCounts.unrecognized,undefined,'prepared cells must match the assayed configuration');
 for(const result of report.results){
  assert.ok(result.maximumErrorHz<2.5,`${result.name}: ${result.maximumErrorHz} Hz maximum held-out error`);
  assert.ok(result.meanAbsoluteErrorHz<result.legacyMeanAbsoluteErrorHz*.15,`${result.name}: substantially reduced interface error`);
  assert.equal(result.samples[0].addedCurrentPa,0,'zero request adds no current');
  assert.equal(result.samples.at(-1).addedCurrentPa,result.samples.find(s=>s.requestedHz===200).addedCurrentPa,'ceiling is bounded');
 }
});

test('timestep, adaptation, refractory time and graded release are profile-specific',async()=>{
 const config=JSON.parse(await readFile(new URL('../../configs/banc-physiology.json',import.meta.url))),core=await createWasmCore();
 const p=Float32Array.from(config.profiles.default),adapted=p.slice(),slow=p.slice();adapted[7]=1;slow[5]=12;
 const model={params:Float32Array.from([...p,...adapted,...slow,...config.profiles.lamina]),manifest:{dt_ms:.25,neuron_count:4}};
 const copy=model.params.slice(),mapper=await createBancSensoryCurrentMapper(core,model,{indices:[0,1,2,3,0],maxRateHz:200});
 assert.deepEqual(model.params,copy,'calibration does not mutate model parameters');
 assert.equal(mapper.profiles.length,4);assert.ok(mapper.current(1,50)>mapper.current(0,50),'adaptation needs extra current');
 assert.ok(mapper.limits(2).maximumHz<82,'refractory period reduces rate ceiling');
 const rates=measurePublicRuntime(core,[p,adapted,slow,config.profiles.lamina],[mapper.current(0,17),mapper.current(1,50),mapper.current(2,200),mapper.current(3,7)],.25);
 assert.ok(Math.abs(rates[0]-17)<.5);assert.ok(Math.abs(rates[1]-50)<.5);
 assert.ok(Math.abs(rates[2]-mapper.limits(2).maximumHz)<.5);assert.ok(Math.abs(rates[3]-7)<.001);
 assert.equal(mapper.current(0,0),0);assert.equal(mapper.current(3,0),0);assert.equal(mapper.current(0,-1),0);
 assert.equal(mapper.current(0,.01),mapper.current(0,2),'explicit positive rate floor');
 assert.throws(()=>mapper.current(4,10),/Uncalibrated/);assert.throws(()=>mapper.current(0,NaN),/Nonfinite/);
 const restricted=await createBancSensoryCurrentMapper(core,model,{indices:[3],maxRateHz:40});assert.equal(restricted.limits(3).maximumHz,40);
 assert.equal(restricted.current(3,40),restricted.current(3,100));
});
