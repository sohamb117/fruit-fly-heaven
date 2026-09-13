import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createBancProboscisDecoder} from '../banc-proboscis.js';
const io=JSON.parse(await readFile(new URL('../../data/prepared/banc888/io.json',import.meta.url)));
const channels=['rostrumExtend','rostrumRetract','haustellumExtend','haustellumRetract','labellarExtend','labellarAbduct'];
const expected={m1:'rostrumRetract',m2da:'rostrumRetract',m2db:'rostrumRetract',m2v:'rostrumRetract',m3l:'haustellumRetract',m3m:'haustellumRetract',m4a:'haustellumExtend',m4b:'haustellumExtend',m6:'labellarExtend',m7:'labellarAbduct',m9:'rostrumExtend'};

test('each exact BANC muscle target activates only its supported positioning channel',()=>{
 const decoder=createBancProboscisDecoder(io.muscles);
 for(const [muscle,channel]of Object.entries(expected)){
  const target=`proboscis_${muscle}_muscle`,index=io.muscles.findIndex(m=>m.target===target),state=new Float32Array(io.muscles.length*3);
  assert.ok(index>=0,`${target} exists in the prepared BANC mapping`);state[index*3+2]=.8;
  const result=decoder.read(state);
  assert.ok(result[channel]>.39,`${target} -> ${channel}`);
  for(const other of channels.filter(x=>x!==channel))assert.equal(result[other],0,`${target} must not leak into ${other}`);
 }
 assert.deepEqual(decoder.unsupported.map(m=>m.target),['proboscis_m8_muscle']);
 assert.ok(decoder.unsupported[0].rootIds.every(id=>typeof id==='string'));
});

test('antagonists remain separate, unrelated labellar activity does not dilute rostrum extension',()=>{
 const decoder=createBancProboscisDecoder(io.muscles),state=new Float32Array(io.muscles.length*3);
 for(const target of ['proboscis_m9_muscle','proboscis_m1_muscle','proboscis_m6_muscle','proboscis_m7_muscle','proboscis_m8_muscle'])state[io.muscles.findIndex(m=>m.target===target)*3+2]=.75;
 const result=decoder.read(state);
 assert.equal(result.rostrumExtend,.75);assert.equal(result.rostrumRetract,.75);
 assert.equal(result.haustellumExtend,0);assert.equal(result.haustellumRetract,0);
 assert.equal(result.labellarExtend,.75);assert.equal(result.labellarAbduct,.75);
 state.fill(0);assert.ok(channels.every(c=>decoder.read(state)[c]===0),'no stored tonic positioning drive');
});

test('distinct retractors add while subdivisions average only within their own muscle class',()=>{
 const mappings=['m1','m2da','m2db','m2v'].map(name=>({joint:'proboscis',target:`proboscis_${name}_muscle`})),decoder=createBancProboscisDecoder(mappings),state=new Float32Array(12);
 state[2]=.2;state[5]=.4;state[8]=.8;state[11]=.1;
 const result=decoder.read(state);assert.ok(Math.abs(result.byClass.m2D-.6)<1e-7);assert.ok(Math.abs(result.rostrumRetract-.9)<1e-7);
 assert.equal(result.rostrumExtend,0);state[2]=1;assert.equal(decoder.read(state).rostrumRetract,1);
 assert.throws(()=>decoder.read(new Float32Array(0)),/Invalid/);
});
