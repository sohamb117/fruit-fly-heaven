import test from 'node:test';
import assert from 'node:assert/strict';
import {TrainingEyePreview,validateEyeSnapshot} from '../training/eye-preview.js';

const snapshot=(patch={})=>({jobId:'job-1',sampleSequence:1,trialSequence:0,sequence:1,width:2,height:1,sourceWidth:2,sourceHeight:1,
  frameTimeSeconds:.02,nativeTimeSeconds:.022,neuralTimeMs:22,format:'rgba8',source:'sensory-retina',sensoryInput:'luminance-derived-from-rgb',
  left:Uint8Array.from([1,2,3,255,4,5,6,255]),right:Uint8Array.from([7,8,9,255,10,11,12,255]),...patch});
const canvas=()=>({width:300,height:150,contexts:0,images:[],cleared:0,getContext(type){
  assert.equal(type,'2d');this.contexts++;return {createImageData:(w,h)=>({data:new Uint8ClampedArray(w*h*4)}),
    putImageData:frame=>this.images.push(Array.from(frame.data)),clearRect:()=>this.cleared++};}});

test('eye renderer validates source, dimensions, format, clocks and exact byte lengths',()=>{
  assert.equal(validateEyeSnapshot(snapshot()).source,'sensory-retina');
  for(const patch of [{source:'observer-camera'},{format:'rgb8'},{sensoryInput:'made-up'},{width:257},{height:129},
    {sourceWidth:1},{left:new Uint8Array(1)},{right:[7,8,9,255,10,11,12,255]},{frameTimeSeconds:NaN},{trialSequence:-1}])
    assert.throws(()=>validateEyeSnapshot(snapshot(patch)),/Invalid eye image/);
});

test('received pixels stay separate and draw only on demand without a scene renderer',()=>{
  const left=canvas(),right=canvas(),status=[],preview=new TrainingEyePreview(left,right,{onStatus:value=>status.push(value)});
  preview.setJob('job-1',1);const value=snapshot();assert.equal(preview.setSnapshot(value),true);
  assert.equal(left.contexts,0);assert.equal(right.contexts,0);
  preview.setActive(true);
  assert.deepEqual(left.images,[Array.from(value.left)]);assert.deepEqual(right.images,[Array.from(value.right)]);
  assert.equal(left.width,2);assert.equal(left.height,1);assert.equal(status.at(-1).timeSeconds,.02);
  preview.setActive(true);assert.equal(left.images.length,1);
  preview.setActive(false);preview.setSnapshot(snapshot({sampleSequence:2,sequence:2,frameTimeSeconds:.04}));
  assert.equal(left.images.length,1);preview.setActive(true);assert.equal(left.images.length,2);
  preview.dispose();assert.equal(preview.snapshot,null);assert.equal(left.width,0);
});

test('late or duplicate images cannot replace a newer image, while a new demonstration can reset its clock',()=>{
  const preview=new TrainingEyePreview(canvas(),canvas());preview.setJob('job-1',1);
  assert.equal(preview.setSnapshot(snapshot()),true);
  assert.equal(preview.setSnapshot(snapshot({jobId:'other'})),false);
  assert.equal(preview.setSnapshot(snapshot()),false);
  assert.equal(preview.setSnapshot(snapshot({sampleSequence:2,sequence:2,frameTimeSeconds:.01})),false);
  assert.equal(preview.setSnapshot(snapshot({sampleSequence:2,trialSequence:1,sequence:0,frameTimeSeconds:0})),true);
  preview.setJob('job-1',2);assert.equal(preview.snapshot,null);
  assert.equal(preview.setSnapshot(snapshot()),true);
});
