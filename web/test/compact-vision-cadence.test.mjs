import test from 'node:test';
import assert from 'node:assert/strict';
import {createCompactVision} from '../training/compact-vision.js';

const width=32,height=16;
const cells=[];
for(const side of ['left','right'])for(const type of ['T4a','T4b','T4c','T4d','T5a','T5b','T5c','T5d'])
  cells.push({index:cells.length,type,side,u:.5,v:.5});
const create=()=>createCompactVision({mapping:{neuron_count:cells.length,cells},width,height,maxGapSeconds:.1});
function frame(sequence,bodyTime){
  const pixels=new Uint8Array(2*width*height);
  for(let side=0;side<2;side++)for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    const u=x-sequence*.25;
    pixels[side*width*height+y*width+x]=Math.round(128+36*Math.sin(.23*u+.15*y)+31*Math.cos(.10*u-.33*y)+23*Math.sin(.51*u+.49*y));
  }
  return {width,height,pixels,sequence,bodyTime};
}

test('100 ms retinal cadence tolerates timestamp subtraction roundoff',()=>{
  const vision=create(),times=[.2,.3,.4,.5,.6,.7,.8];
  assert(times.some((time,i)=>i>0&&time-times[i-1]>.1),'The fixture must exercise subtraction above the exact limit');
  vision.update(frame(0,times[0]));assert.equal(vision.summary.ready,false);
  for(let sequence=1;sequence<times.length;sequence++){
    vision.update(frame(sequence,times[sequence]));
    assert.equal(vision.summary.ready,true,'Unexpected sampling reset at '+times[sequence]);
    assert.equal(vision.summary.dt,times[sequence]-times[sequence-1],'The actual dt must not be rounded');
    assert(vision.ratesHz.some(rate=>rate>0));
  }
});

test('a genuine over-limit retinal gap still clears motion until the next valid frame',()=>{
  const vision=create();vision.update(frame(0,.2));vision.update(frame(1,.3));
  vision.update(frame(2,.400001));
  assert.equal(vision.summary.ready,false);assert.equal(vision.summary.reason,'sampling_gap');
  assert.equal(vision.fields,null);assert(vision.ratesHz.every(rate=>rate===0));
  vision.update(frame(3,.420001));
  assert.equal(vision.summary.ready,true);assert(vision.ratesHz.some(rate=>rate>0));
});
