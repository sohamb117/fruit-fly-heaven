import test from 'node:test';
import assert from 'node:assert/strict';
import {SensoryEncoder} from '../sensory-encoder.js';

const manifest={schema_version:1,neuron_count:8,
 vision:{width:32,height:16,receptors:[{index:3,side:'left',u:0,v:0}]},
 channels:[{key:'antenna_left',indices:[4,5]},{key:'antenna_right',indices:[6,7]}],
 body_transducer_exclusions:[{index:4,function:'auditory_high_frequency',organ:'antenna'},
  {index:6,function:'auditory_low_frequency',organ:'antenna'}]};
const groups={odor_left:[0],odor_right:[1],sweet:[2]};
const environment={odor:()=>0,surface:()=>({y:0,contact:false})};
const pose={x:0,y:0,z:0,heading:0,bodyTime:0,contact:false,
 feedback:{legs:Array.from({length:6},()=>({loadBodyWeights:0})),
  antennae:[{angle:.4,speed:0},{angle:-.3,speed:0}],speed:0,tilt:.7}};

test('explicit auditory exclusions cannot borrow static-antenna aggregate rates from old channels',()=>{
 const encoder=new SensoryEncoder(manifest,groups,environment);
 const out=encoder.update(pose,null,{odor:false,taste:false,vision:false});
 const rates=new Map(Array.from(out.indices,(index,k)=>[index,out.ratesHz[k]]));
 assert.equal(rates.get(4),0);assert.equal(rates.get(6),0);
 assert.equal(rates.get(5),21);assert.equal(rates.get(7),17.5);
 assert.equal(out.sample.body.rates.antenna_left,10.5);
 assert.equal(out.sample.body.rates.antenna_right,8.75);
});

test('body switch clears remaining input and removing exclusions preserves existing aggregate behavior',()=>{
 const encoder=new SensoryEncoder(manifest,groups,environment);
 const out=encoder.update(pose,null,{odor:false,taste:false,vision:false,bodySense:false});
 assert(out.ratesHz.every(rate=>rate===0));
 const previous=new SensoryEncoder({...manifest,body_transducer_exclusions:[]},groups,environment);
 const legacy=previous.update(pose,null,{odor:false,taste:false,vision:false});
 assert.deepEqual(Array.from(legacy.ratesHz.slice(4)),[21,21,17.5,17.5]);
});
