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

test('explicit exclusions also apply before native leg feedback is available',()=>{
 for(const legs of [undefined,Array.from({length:6},()=>({angle:.2,speed:1,support:1}))]){
  const encoder=new SensoryEncoder(manifest,groups,environment);
  const out=encoder.update({...pose,feedback:{...pose.feedback,legs}},null,{odor:false,taste:false,vision:false});
  assert.deepEqual(Array.from(out.ratesHz.slice(4)),[0,21,0,17.5]);
  assert.equal(out.sample.body.rates.antenna_left,10.5);
  assert.equal(out.sample.body.rates.antenna_right,8.75);
 }
});

test('unsupported proximal sensors stay silent while neighboring native and aggregate sensors retain their routes',()=>{
 const proximal={...manifest,channels:[{key:'self_motion_left',indices:[4,5,6,7]}],
  body_transducers:[{index:5,kind:'load',leg:0,side:'left'}],
  body_transducer_exclusions:[{index:4,function:'joint_angle',organ:'coxa,front_leg'},
   {index:6,function:'joint_angle',organ:'front_leg,trochanter'}]};
 const nativePose={...pose,feedback:{...pose.feedback,speed:10,
  legs:Array.from({length:6},()=>({loadBodyWeights:.2,angle:.2,speed:1,support:1}))}};
 const native=new SensoryEncoder(proximal,groups,environment).update(nativePose,null,{odor:false,taste:false,vision:false});
 assert.equal(native.ratesHz[4],0);assert.equal(native.ratesHz[6],0);
 assert(native.ratesHz[5]>0);assert(native.ratesHz[7]>0);
 assert.notEqual(native.ratesHz[5],native.ratesHz[7]);
 const aggregatePose={...nativePose,feedback:{...nativePose.feedback,
  legs:nativePose.feedback.legs.map(({loadBodyWeights,...leg})=>leg)}};
 const aggregate=new SensoryEncoder(proximal,groups,environment).update(aggregatePose,null,{odor:false,taste:false,vision:false});
 assert.equal(aggregate.ratesHz[4],0);assert.equal(aggregate.ratesHz[6],0);
 assert.equal(aggregate.ratesHz[5],aggregate.ratesHz[7]);
 assert(aggregate.ratesHz[5]>0);
});
