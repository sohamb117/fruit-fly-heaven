import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {createWasmCore,WasmBrain,WasmMuscles} from '../src/wasm.js';
import {CalibratedWasmMuscles} from '../src/calibrated-muscles.js';
import {fixture} from './fixture.mjs';

const core=await createWasmCore();
const defaults=count=>Float32Array.from({length:count*4},(_,i)=>[.015,.04,.08,.03][i%4]);
const bits=array=>new Uint32Array(array.buffer,array.byteOffset,array.length);
const state=muscles=>muscles.core.HEAPF32.slice(muscles.state/4,muscles.state/4+muscles.count*3);
const excitation=(count,value)=>Float32Array.from({length:count*5},(_,i)=>[value,1,0,1,1][i%5]);

test('configured defaults reproduce pre-extension legacy activation, fatigue and force bitexact',()=>{
 const legacy=new WasmMuscles(core,5),configured=new CalibratedWasmMuscles(core,defaults(5)),hash=createHash('sha256');
 try{
  for(let step=0;step<2400;step++){
   const input=new Float32Array(25);
   for(let i=0;i<5;i++)input.set([step<800?1:step<1600?((step+3*i)%17)/16:0,
    .6+((step+i)%7)*.12,((step+2*i)%10-5)/3,.2+i*.3,((step+7*i)%11)/10],i*5);
   const dt=[.0002,.001,.002,.007,.02,.05][step%6],a=legacy.step(input,dt),b=configured.step(input,dt);
   assert.deepEqual(bits(b),bits(a),`Full state differs at muscle step ${step}`);
   hash.update(new Uint8Array(a.buffer,a.byteOffset,a.byteLength));
  }
  // Captured from the unchanged, production WASM before adding this ABI.
  assert.equal(hash.digest('hex'),'028a318ae34a8f15e0be6550cab534554b8e612d0701b1fd7d54cc0fead2c9e2');
 }finally{legacy.dispose();configured.dispose();}
});

test('slower power-muscle kinetics affect only their selected profile and preserve other muscles',()=>{
 const profiles=defaults(3);profiles.set([.075,.12,.08,.03],4);
 const ordinary=new WasmMuscles(core,3),configured=new CalibratedWasmMuscles(core,profiles);
 try{
  let a,b;
  for(let i=0;i<30;i++){a=ordinary.step(excitation(3,1),.001);b=configured.step(excitation(3,1),.001);}
  assert(b[3]<a[3]*.5,'Slower power activation must be observably slower');
  assert(b[5]<a[5]*.5,'The force output must follow the slower activation');
  assert.deepEqual(bits(b.slice(0,3)),bits(a.slice(0,3)));assert.deepEqual(bits(b.slice(6)),bits(a.slice(6)));
  // Match the initial activation/fatigue, isolating deactivation from prior recruitment.
  const matched=new Float32Array([.8,.1,.72,.8,.1,.72,.8,.1,.72]);
  core.HEAPF32.set(matched,ordinary.state/4);core.HEAPF32.set(matched,configured.state/4);
  a=ordinary.step(excitation(3,0),.02);b=configured.step(excitation(3,0),.02);
  assert(b[3]>a[3]+.15,'A longer decay time must retain activation during release');
  assert.deepEqual(bits(b.slice(0,3)),bits(a.slice(0,3)));assert.deepEqual(bits(b.slice(6)),bits(a.slice(6)));
 }finally{ordinary.dispose();configured.dispose();}
});

test('fatigue and recovery profiles alter sustained force without changing activation kinetics',()=>{
 const profiles=defaults(3);profiles[2]=0;profiles[6]=.16;profiles[11]=.09;
 const configured=new CalibratedWasmMuscles(core,profiles);
 try{
  let s;for(let i=0;i<100;i++)s=configured.step(excitation(3,1),.01);
  assert.equal(s[1],0);assert(s[4]>s[7]*1.9);assert(s[2]>s[8]&&s[8]>s[5]);
  assert.deepEqual([s[0],s[3],s[6]],[s[0],s[0],s[0]]);
  core.HEAPF32.set(new Float32Array([0,.3,0,0,.3,0,0,.3,0]),configured.state/4);
  for(let i=0;i<10;i++)s=configured.step(excitation(3,0),.01);
  assert(s[7]<s[1]-.005);assert.equal(s[1],s[4]);
 }finally{configured.dispose();}
});

test('profile copies and returned states remain independent through allocation and disposal',()=>{
 const active=new Set(),freed=[],facade=Object.create(core);
 facade._malloc=bytes=>{const p=core._malloc(bytes);assert(p);assert(!active.has(p));active.add(p);return p;};
 facade._free=p=>{assert(active.delete(p),'Free must correspond to one live allocation');freed.push(p);core._free(p);};
 const profiles=defaults(2),a=new CalibratedWasmMuscles(facade,profiles);
 const b=new CalibratedWasmMuscles(facade,defaults(2));
 try{
  assert.equal(active.size,6);profiles.fill(.9);
  assert.deepEqual(core.HEAPF32.slice(a.profiles/4,a.profiles/4+8),defaults(2));
  const saved=a.step(excitation(2,1),.02),savedCopy=saved.slice();
  a.step(excitation(2,0),.01);assert.deepEqual(saved,savedCopy,'Returned snapshots must not alias native state');
  const aBefore=state(a);b.step(excitation(2,.3),.01);b.dispose();
  assert.deepEqual(state(a),aBefore,'Disposing another episode must preserve this episode state');
  b.dispose();assert.equal(active.size,3);assert.equal(freed.length,3);
  assert.throws(()=>b.step(excitation(2,1),.001),/disposed/);
  a.dispose();a.dispose();assert.equal(active.size,0);assert.equal(freed.length,6);
 }finally{a.dispose();b.dispose();}
});

test('partial native allocation failure releases owned buffers and leaves other episodes intact',()=>{
 const existing=new CalibratedWasmMuscles(core,defaults(1));existing.step(excitation(1,1),.01);
 const before=state(existing),allocated=[],freed=[],facade=Object.create(core);let calls=0;
 facade._malloc=bytes=>{if(++calls===3)return 0;const p=core._malloc(bytes);allocated.push(p);return p;};
 facade._free=p=>{freed.push(p);core._free(p);};
 try{
  assert.throws(()=>new CalibratedWasmMuscles(facade,defaults(2)),/allocation failed/);
  assert.equal(allocated.length,2);assert.deepEqual(freed,allocated);assert.deepEqual(state(existing),before);
 }finally{existing.dispose();}
});

test('the existing neural ABI matches its pre-extension mixed-physiology trace bitexact',()=>{
 const model=fixture({n:12,graded:[0,2,4,6,8,10],
  edges:Array.from({length:9},(_,r)=>({source:r,post:r+1,weight:12+r*2,receptor:r,delay:1+r%5})),
  gaps:[{a:0,b:9,weight:.6},{a:4,b:11,weight:1.2}],
  overrides:Object.fromEntries(Array.from({length:12},(_,i)=>[i,{12:.5,13:.7,14:.8,15:.4}]))});
 const brain=new WasmBrain(core,model),hash=createHash('sha256');
 try{
  for(let step=0;step<960;step++){
   const input=Float32Array.from({length:12},(_,i)=>((step+13*i)%97<37?45:4)+i);
   brain.step(1,input,{hunger:(step%100)/100,insulin:.2,akh:.4},step%17!==0);
   const s=brain.readState();hash.update(new Uint8Array(s.buffer,s.byteOffset,s.byteLength));
  }
  assert.equal(brain.readState().totalSpikes,194);
  assert.equal(hash.digest('hex'),'0e43c6c1ca15c629ad240836656e0d34724fcbe4e0391a31172c9f3f03bcb1f5');
 }finally{brain.dispose();}
});
