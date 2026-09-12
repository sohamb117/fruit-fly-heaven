import test from 'node:test';
import assert from 'node:assert/strict';
import {createBrainViewModule} from '../dist/index.js';
const runtime=await createBrainViewModule();
const fixture=()=>runtime.createGeometry({positions:new Float32Array([-1,0,0,0,0,0,1,0,0]),neuronIndices:new Uint32Array([2,0,1]),neuronCount:3});

test('activity colors follow neuron IDs and snapshots are owned',()=>{
  const geometry=fixture();
  const colors=geometry.updateActivity({voltage:new Float64Array([-52,-45,-60])});
  assert(colors[2]>colors[0]);assert(colors[6]>.99);assert(colors[3]<.2);
  geometry.updateActivity({voltage:new Float64Array([-45,-52,-52])});
  assert(colors[6]>.99);geometry.dispose();
});
test('cutaway and slab use anatomical distances and arbitrary plane normals',()=>{
  const geometry=fixture();
  assert.deepEqual([...geometry.filter({normal:[2,0,0],offset:0,mode:'cutaway'})],[0,1]);
  assert.deepEqual([...geometry.filter({normal:[1,0,0],offset:0,halfThickness:.2,mode:'slab'})],[1]);
  geometry.dispose();
});
test('picking respects projection, pixel radius, and clipping',()=>{
  const geometry=fixture(),matrix=new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);
  assert.equal(geometry.pick({matrix,x:0,y:0,width:800,height:600}),1);
  assert.equal(geometry.pick({matrix,x:1,y:0,width:800,height:600,normal:[1,0,0],offset:0,mode:'cutaway'}),-1);
  assert.equal(geometry.pick({matrix,x:0,y:.5,width:800,height:600}),-1);geometry.dispose();
});
test('surface intersection lies exactly on the cutting plane',()=>{
  const positions=new Float32Array([-1,-1,-1,1,-1,1,0,1,1]),triangles=new Uint32Array([0,1,2]);
  const result=runtime.sliceMesh({positions,triangles,normal:[0,0,1],offset:0});
  assert.equal(result.length,6);assert.deepEqual([...result],[0,-1,0,-.5,0,0]);
});
test('volume slices interpolate a known 3D ramp, including oblique sampling',()=>{
  const values=Uint8Array.from({length:27},(_,i)=>{const x=i%3,y=Math.floor(i/3)%3,z=Math.floor(i/9);return x*10+y*20+z*30;});
  const volume=runtime.createVolume({values,dimensions:[3,3,3]});
  assert.deepEqual([...volume.samplePlane({origin:[0,0,1],u:[1,0,0],v:[0,1,0],width:3,height:3})],[30,40,50,50,60,70,70,80,90]);
  assert.deepEqual([...volume.samplePlane({origin:[.5,.5,.5],u:[.5,0,.5],v:[0,.5,0],width:2,height:2})],[30,50,40,60]);
  assert.deepEqual([...volume.samplePlane({origin:[-1,0,0],u:[1,0,0],v:[0,1,0],width:2,height:1})],[0,0]);volume.dispose();
});
test('invalid input and disposed handles fail before native memory access',()=>{
  assert.throws(()=>runtime.createGeometry({positions:new Float32Array([0,NaN,0]),neuronIndices:new Uint32Array([0]),neuronCount:1}),/finite/);
  const geometry=fixture();assert.throws(()=>geometry.filter({normal:[0,0,0]}),/nonzero/);
  assert.throws(()=>geometry.updateActivity({voltage:new Float64Array([1])}),/one Float64/);
  geometry.dispose();assert.throws(()=>geometry.filter(),/disposed/);
  const volume=runtime.createVolume({values:new Uint8Array(8),dimensions:[2,2,2]});volume.dispose();
  assert.throws(()=>volume.samplePlane({origin:[0,0,0],u:[1,0,0],v:[0,1,0],width:2,height:2}),/disposed/);
});
