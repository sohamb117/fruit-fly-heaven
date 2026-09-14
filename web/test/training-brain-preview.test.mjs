import test from 'node:test';
import assert from 'node:assert/strict';
import {validateBrainSample,TrainingBrainPreview} from '../training/brain-preview.js';
const config={modelFingerprint:'a'.repeat(64)};
const sample={schemaVersion:1,dataset:'BANC v888',modelFingerprint:config.modelFingerprint,preparedIdsSha256:'b'.repeat(64),neuronCount:175401,
  sampleCount:2,indices:[3,7],positions:[1,2,3,4,5,6],ids:['1001','1002'],labels:['First','Second']};
test('brain positions require matching identity and aligned finite unique neuron data',()=>{
  assert.equal(validateBrainSample(sample,config),sample);
  for(const patch of [{modelFingerprint:'c'.repeat(64)},{indices:[3,3]},{indices:[3,175401]},{positions:[1,2,3]},{ids:['1001']},{labels:['First',4]},
    {preparedIdsSha256:'bad'},{positions:[1,2,3,4,5,NaN]}])assert.throws(()=>validateBrainSample({...sample,...patch},config));
});
test('brain samples stay tied to their evaluation and restart correctly for a re-leased job',()=>{
  const preview=new TrainingBrainPreview({addEventListener(){},removeEventListener(){},replaceChildren(){}});preview.setSample(sample);preview.setJob('same-job',1);
  const snapshot={jobId:'same-job',neuralTimeMs:100,indices:[3,7],voltage:[-60,-50],rates:[1,2],lastSpikeMs:[-1,99]};
  assert.equal(preview.setSnapshot(snapshot),true);assert.equal(preview.setSnapshot({...snapshot,neuralTimeMs:80}),false);
  assert.equal(preview.setSnapshot({...snapshot,jobId:'other'}),false);assert.equal(preview.setSnapshot({...snapshot,indices:[7,3]}),false);
  preview.setJob('same-job',1);assert.equal(preview.snapshot,snapshot);
  preview.setJob('same-job',2);assert.equal(preview.snapshot,null);assert.equal(preview.setSnapshot({...snapshot,neuralTimeMs:0}),true);
  assert.equal(preview.renderer,undefined,'Receiving data while closed creates no renderer');preview.dispose();
});
