import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createColorModule,loadFlyColorModel} from '../../packages/fly-color-wasm/dist/index.js';
import {ColorVision,validateColorMapping} from '../color-vision.js';
import {SensoryEncoder} from '../sensory-encoder.js';
import {createHabitat} from '../body-world.js';
const json=async path=>JSON.parse(await readFile(new URL(path,import.meta.url)));
const mapping=await json('../color-inputs.json'),sense=await json('../sensory-inputs.json'),motors=await json('../motor-outputs.json'),visual=await json('../visual-projections.json');
const model=await loadFlyColorModel(),mapper=(await createColorModule()).createMapper(model);
const frame=(left,right,sequence=0)=>({sequence,bodyTime:sequence*.05,pixels:new Uint8Array(1024).fill(100),rgb:Uint8Array.from({length:3072},(_,i)=>(i<1536?left:right)[i%3])});

test('color changes excite distinct existing receptors without changing achromatic input; off and missing RGB clear rates',()=>{
  const a=new ColorVision(mapper,mapping),b=new ColorVision(mapper,mapping);
  const blue=frame([0,0,255],[0,255,0]),green=frame([0,255,0],[0,255,0]);
  a.update(blue);b.update(green);
  assert.deepEqual(blue.pixels,green.pixels);
  assert.notDeepEqual(a.summary.left,b.summary.left);assert.deepEqual(a.summary.right,b.summary.right);
  assert.ok(a.ratesHz.every(v=>Number.isFinite(v)&&v>=2&&v<=80));
  const copy=a.ratesHz.slice();b.update(frame([255,0,0],[255,0,0],1));assert.deepEqual(a.ratesHz,copy);
  const serial=a.serial;a.update(blue);assert.equal(a.serial,serial);
  a.update(blue,false);assert.ok(a.ratesHz.every(v=>v===0));assert.equal(a.summary.ready,false);
  a.update(blue,true);assert.deepEqual(a.ratesHz,copy);
  a.update({...blue,rgb:null});assert.ok(a.ratesHz.every(v=>v===0));
});

test('opsin types follow documented pairing and mapped input IDs never overlap motors or other sensory targets',()=>{
  assert.doesNotThrow(()=>validateColorMapping(mapping,138639));assert.equal(mapping.cells.length,2498);
  const others=new Set([...motors.channels.flatMap(c=>c.indices),...sense.vision.receptors.map(c=>c.index),...sense.channels.flatMap(c=>c.indices),...visual.cells.map(c=>c.index)]);
  assert.ok(mapping.cells.every(c=>!others.has(c.index)&&c.subtype_assigned));
  const mosaic=new Map();
  for(const c of mapping.cells){const key=[c.side,c.p,c.q].join(','),pale=['Rh3','Rh5'].includes(c.opsin);if(mosaic.has(key))assert.equal(mosaic.get(key),pale);mosaic.set(key,pale);}
  const bad=structuredClone(mapping);bad.cells[0].opsin='Rh1';assert.throws(()=>validateColorMapping(bad,138639),/Invalid/);
  const duplicates=structuredClone(mapping);duplicates.cells[1].index=duplicates.cells[0].index;assert.throws(()=>validateColorMapping(duplicates,138639),/Invalid/);
});

test('encoder appends only color inputs and disconnection preserves food/body/achromatic inputs',()=>{
  const groups={odor_left:[],odor_right:[],sweet:[]},pose={x:0,y:0,z:0,heading:0,bodyTime:0},environment=createHabitat([]);
  const base=new SensoryEncoder(sense,groups,environment),encoder=new SensoryEncoder(sense,groups,environment,{colorMapping:mapping});
  const f=frame([0,0,255],[0,255,0]),color=new ColorVision(mapper,mapping).update(f),old=base.update(pose,f);
  const out=encoder.update(pose,f,{color}),prefix=old.ratesHz.length;
  assert.deepEqual(out.ratesHz.slice(0,prefix),old.ratesHz);assert.deepEqual(out.ratesHz.slice(prefix),color.ratesHz);
  const disconnected=encoder.update(pose,f,{color:null});assert.ok(disconnected.ratesHz.slice(prefix).every(v=>v===0));assert.deepEqual(disconnected.ratesHz.slice(0,prefix),old.ratesHz);
  const blind=encoder.update(pose,f,{color,vision:false});assert.ok(blind.ratesHz.slice(prefix).every(v=>v===0));
});
