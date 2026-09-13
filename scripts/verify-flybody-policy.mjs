import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {FlyBodyPolicy} from '../web/flybody-policy.js';
const file='models/flybody-flight-policy.json',bytes=await fs.readFile(file),model=JSON.parse(bytes),policy=new FlyBodyPolicy(model);
const fixtures=model.validation.fixtures;assert(fixtures?.length>=3,'Require original TensorFlow inference fixtures');
let maximumError=0;
for(const fixture of fixtures){
 const actual=policy.predict(fixture.input);assert(actual.every(Number.isFinite));assert.equal(actual.length,fixture.output.length);
 maximumError=Math.max(maximumError,...actual.map((x,i)=>Math.abs(x-fixture.output[i])));
}
assert(maximumError<1e-4,`Published policy inference mismatch: ${maximumError}`);
const runs=1000,start=performance.now();for(let i=0;i<runs;i++)policy.predict(fixtures[i%fixtures.length].input);
const report={date:new Date().toISOString(),scope:'Exact exported trained-policy forward pass against original TensorFlow outputs; not BANC behavior or a physical flight test.',sourceSha256:createHash('sha256').update(bytes).digest('hex'),inputSize:policy.inputSize,outputSize:model.output.size,fixtures:fixtures.length,maximumError,meanInferenceMs:(performance.now()-start)/runs,passed:true};
console.log(JSON.stringify(report,null,2));await fs.writeFile('reports/flybody-policy-validation.json',JSON.stringify(report,null,2)+'\n');
