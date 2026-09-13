import {test} from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../vendor/three.module.js';
import {createFruitCameraOcclusion} from '../fruit-camera-occlusion.js';

const input=()=>({target:new THREE.Vector3(0,0,0),azimuth:0,elevation:.3,distance:10,following:true});
const fruit=(x=0,y=2,z=4,r=1.5)=>{const m=new THREE.Mesh(new THREE.SphereGeometry(r,24,16),new THREE.MeshBasicMaterial());m.position.set(x,y,z);return m;};

test('foreground fruit raises only effective elevation until the target sightline is clear',()=>{
  const guard=createFruitCameraOcclusion(),solid=fruit();guard.addSolid(solid);const request=input(),out=guard.resolve(request);
  assert(out.occluded&&out.adjusted&&out.resolved);assert(out.elevation>request.elevation&&out.elevation<=1.47);assert(out.checks<=29);
  assert.equal(guard.resolve({...request,elevation:out.elevation}).occluded,false);
  assert.equal(request.elevation,.3);assert.equal(request.azimuth,0);assert.equal(request.distance,10);assert.deepEqual(request.target.toArray(),[0,0,0]);
});

test('clear sightline and non-following camera preserve the original angle',()=>{
  const guard=createFruitCameraOcclusion();guard.addSolid(fruit());
  const clear=guard.resolve({...input(),elevation:1.2});assert.equal(clear.elevation,1.2);assert.equal(clear.adjusted,false);
  const manual=guard.resolve({...input(),following:false});assert.equal(manual.elevation,.3);assert.equal(manual.checks,0);
});

test('original hidden apple mesh still occludes after the renderer batches it',()=>{
  const guard=createFruitCameraOcclusion(),solid=fruit();solid.visible=false;guard.addSolid(solid);
  assert(guard.resolve(input()).adjusted);assert.equal(solid.visible,false);assert.equal(solid.material.side,THREE.FrontSide);
});

test('decorative children and an unregistered fly mesh do not affect the result',()=>{
  const guard=createFruitCameraOcclusion(),solid=fruit(20,2,4),decoration=fruit(-20,0,0),fly=fruit();
  solid.add(decoration);guard.addSolid(solid);
  assert.equal(guard.resolve(input()).adjusted,false);assert.equal(guard.solidCount,1);
  assert(fly.isMesh);assert.throws(()=>guard.addSolid(new THREE.Group()),/original fruit solid/);
});

test('fruit behind the target or camera is excluded by the finite ray segment',()=>{
  for(const solid of [fruit(0,-2,-4),fruit(0,4,15)]){
    const guard=createFruitCameraOcclusion();guard.addSolid(solid);assert.equal(guard.resolve(input()).adjusted,false);
  }
});

test('transformed fruit uses its current world geometry without scene or material mutation',()=>{
  const guard=createFruitCameraOcclusion(),group=new THREE.Group(),solid=fruit(0,0,0,1);
  group.position.set(0,2,4);solid.scale.set(1.5,1.5,1.5);group.add(solid);guard.addSolid(solid);
  const before=solid.position.toArray(),scale=solid.scale.toArray(),opacity=solid.material.opacity;
  assert(guard.resolve(input()).adjusted);assert.deepEqual(solid.position.toArray(),before);assert.deepEqual(solid.scale.toArray(),scale);assert.equal(solid.material.opacity,opacity);
});

test('a target inside fruit remains an unresolved obstruction rather than moving the target or hiding fruit',()=>{
  const guard=createFruitCameraOcclusion(),solid=fruit(0,0,0,2);guard.addSolid(solid);const out=guard.resolve(input());
  assert(out.occluded);assert.equal(out.resolved,false);assert.equal(out.adjusted,false);assert.equal(out.elevation,.3);assert(out.checks<=25);
  assert.equal(solid.visible,true);assert.equal(solid.material.opacity,1);
});

test('invalid positions and duplicate registration cannot cause unbounded camera work',()=>{
  const guard=createFruitCameraOcclusion(),solid=fruit();guard.addSolid(solid);guard.addSolid(solid);assert.equal(guard.solidCount,1);
  const out=guard.resolve({...input(),target:new THREE.Vector3(NaN,0,0)});assert(out.invalid);assert.equal(out.checks,0);
  assert.throws(()=>createFruitCameraOcclusion({searchStep:0}),/Invalid/);
});

test('triangle-block acceleration agrees with original Three.js raycasting on varied sightlines',()=>{
  const guard=createFruitCameraOcclusion(),solids=[fruit(),fruit(-2,4,6,2),fruit(3,1,2,.7)];
  solids[1].scale.set(.7,1.1,1.4);solids[1].visible=false;solids.forEach(s=>guard.addSolid(s));
  const ray=new THREE.Raycaster(),direction=new THREE.Vector3(),camera=new THREE.Vector3();
  for(let k=0;k<80;k++){
    const request={...input(),target:new THREE.Vector3(Math.sin(k)*2,Math.cos(k*.7),Math.sin(k*.3)),azimuth:k*.19,elevation:.25+(k%12)*.1};
    const {target,azimuth,elevation,distance}=request;
    solids.forEach(s=>s.updateWorldMatrix(true,false));
    camera.set(target.x+distance*Math.cos(elevation)*Math.sin(azimuth),target.y+distance*Math.sin(elevation),target.z+distance*Math.cos(elevation)*Math.cos(azimuth));
    direction.subVectors(camera,target).normalize();ray.set(target,direction);ray.near=.1;ray.far=distance-1e-5;
    let expected=ray.intersectObjects(solids,false).length>0;
    direction.negate();ray.set(camera,direction);ray.near=1e-5;ray.far=distance-.1;
    expected ||= ray.intersectObjects(solids,false).length>0;
    assert.equal(guard.resolve(request).occluded,expected,`sightline ${k}`);
  }
});
