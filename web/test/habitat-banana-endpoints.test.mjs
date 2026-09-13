import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as THREE from '../vendor/three.module.js';
import {createHabitat} from '../body-world.js';
import {originalFruitGeometry,originalBowlGeometry} from '../../scripts/original-fruit-geometry.mjs';

const fruit=JSON.parse(fs.readFileSync(new URL('../habitat.json',import.meta.url))).fruit;
const bowlY=(x,z)=>1.5+.0037*(x*x+z*z);
const straight={kind:'banana',x:0,z:0,y:11,radius:6,path:[[-10,0],[0,0],[10,0]]};

test('open banana ends retain only the original small tip spheres',()=>{
  const h=createHabitat([straight]);
  for(const direction of [-1,1]){
    const beyond=direction*14;
    assert.equal(h.surface(beyond,0).y,bowlY(beyond,0));
    assert.equal(h.surface(beyond,0).contact,false);
    const tip=h.surface(direction*12,0);
    assert.equal(tip.y,12.5);assert.equal(tip.fruitIndex,0);
    assert.ok(tip.normal.every(Number.isFinite));
    assert.equal(h.surface(direction*9,0).y,17);
  }
});

test('an endpoint plane does not cut a different nearby part of a curved tube',()=>{
  const f={...straight,path:[[-10,0],[0,0],[0,10],[-20,10]]};
  // x < -10 is beyond the first endpoint plane, but this point lies on the
  // interior of the final segment and must retain the tube's full radius.
  assert.equal(createHabitat([f]).surface(-15,10).y,17);
});

test('actual banana ends no longer create a full-radius invisible cap',()=>{
  const ray=new THREE.Raycaster(),bowl=originalBowlGeometry({reverse:true});
  for(const f of fruit.filter(f=>f.kind==='banana')){
    const h=createHabitat([f]),visual=originalFruitGeometry([f],{decorations:true});
    for(const end of [0,f.path.length-1]){
      const p=f.path[end],near=f.path[end===0?1:end-1],length=Math.hypot(p[0]-near[0],p[1]-near[1]);
      const dx=(p[0]-near[0])/length,dz=(p[1]-near[1])/length;
      for(const lateral of [-2.75,2.75]){
        const rim=h.surface(p[0]-dz*lateral,p[1]+dx*lateral);
        assert.equal(rim.fruitIndex,0,'roundoff must preserve the endpoint plane rim');
        assert.ok(rim.y>f.y+3);
      }
      for(const distance of [3.5,4,5]){
        const x=p[0]+dx*distance,z=p[1]+dz*distance;
        ray.set(new THREE.Vector3(x,100,z),new THREE.Vector3(0,-1,0));
        const hit=ray.intersectObjects([...visual.all,bowl],false)[0];
        assert.equal(h.surface(x,z).fruitIndex,-1);
        assert.equal(hit.object,bowl,'actual original mesh has no fruit at the removed cap');
        assert.ok(Math.abs(h.surface(x,z).y-hit.point.y)<.012);
      }
      const x=p[0]+dx*1.5,z=p[1]+dz*1.5;
      ray.set(new THREE.Vector3(x,100,z),new THREE.Vector3(0,-1,0));
      const hit=ray.intersectObjects([...visual.all,bowl],false)[0],height=h.surface(x,z).y;
      assert.equal(h.surface(x,z).fruitIndex,0);
      assert.ok(Math.abs(height-hit.point.y)<.2,'tip sphere differs only by existing low-poly tessellation');
    }
    visual.dispose();
  }
  bowl.geometry.dispose();bowl.material.dispose();
});

test('spawn, apples and bare floor retain their original surfaces',()=>{
  const h=createHabitat(fruit);
  assert.ok(Math.abs(h.surface(-32.85577942512126,-12.281421463553833).y-16.869576398523403)<1e-12);
  for(const f of fruit.filter(f=>f.kind==='apple')){
    assert.equal(h.surface(f.x,f.z).y,f.y+.94*f.radius);
  }
  const frame150=h.surface(-1.1632325616512753,12.974724140286762);
  assert.equal(frame150.fruitIndex,-1);assert.ok(Math.abs(frame150.y-2.1278773330833958)<1e-12);
});
