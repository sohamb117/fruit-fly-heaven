import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createColorModule,loadFlyColorModel} from '../dist/index.js';
const model=await loadFlyColorModel(),runtime=await createColorModule();
const reference=JSON.parse(await readFile(new URL('../model/reference.json',import.meta.url)));

test('WASM LUT agrees with the independent full spectral integral including off-grid colors',()=>{
  const mapper=runtime.createMapper(model),values=mapper.map(Uint8Array.from(reference.rgb.flat()));
  const expected=reference.captures.flat();let max=0;
  for(let i=0;i<values.length;i++)max=Math.max(max,Math.abs(values[i]-expected[i]));
  assert.ok(max<.003,`absolute relative-capture error ${max}`);
  assert.ok(values.every(v=>Number.isFinite(v)&&v>=0));mapper.dispose();
});

test('RGB primaries produce distinct receptor signals, darkness zero, and no fabricated UV illumination',()=>{
  const mapper=runtime.createMapper(model);
  const [black,white,red,green,blue]=reference.rgb.slice(0,5).map(rgb=>mapper.map(Uint8Array.from(rgb)));
  assert.deepEqual([...black],[0,0,0,0]);assert.ok(blue[2]>blue[3]);assert.ok(green[3]>green[2]*2);
  assert.ok(red[3]<green[3]);assert.ok(white[0]<.003&&white[1]<.14);
  assert.equal(model.metadata.illumination_cutoff_nm,400);mapper.dispose();
});

test('batched buffers, independent mappers, subarrays, validation and disposal',()=>{
  const a=runtime.createMapper(model),b=runtime.createMapper(model),rgb=new Uint8Array([9,0,0,255,9]).subarray(1,4);
  const first=a.map(rgb),copy=first.slice(),out=new Float32Array(4);
  assert.equal(b.map(rgb,out),out);assert.deepEqual(out,copy);
  a.map(new Uint8Array([255,0,0]));assert.deepEqual(first,copy);
  assert.throws(()=>a.map(new Uint8Array(4)),/RGB/);assert.throws(()=>a.map(new Uint8Array(3075)),/capacity/);
  assert.throws(()=>a.map(rgb,new Float32Array(3)),/Output/);
  a.dispose();a.dispose();assert.throws(()=>a.map(rgb),/disposed/);assert.deepEqual(b.map(rgb),copy);b.dispose();
  assert.throws(()=>runtime.createMapper({...model,size:1}),/size/);
  const invalid=model.lut.slice();invalid[0]=NaN;assert.throws(()=>runtime.createMapper({...model,lut:invalid}),/LUT/);
});
