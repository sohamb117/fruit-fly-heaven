import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHabitat} from '../body-world.js';
import {createFlybodyHabitatCollision} from '../flybody-habitat-collision.js';

const fruit=JSON.parse(fs.readFileSync(new URL('../habitat.json',import.meta.url))).fruit;
test('fruit solids stay in worldbody and the heightfield contains only bowl',()=>{
  const result=createFlybodyHabitatCollision(createHabitat(fruit));
  assert.equal(result.stats.fruitGeoms,189);assert.equal(Object.keys(result.fruitGeomNames).length,189);
  assert.equal(result.stats.bananaSlices,180);assert.equal(result.stats.bananaTips,6);assert.equal(result.stats.apples,3);
  assert.equal(result.groundName,'ground');assert.ok(result.geoms.startsWith('<geom name="ground"'));
  assert.equal(result.geoms.includes('<body'),false);
  assert.equal((result.assets.match(/scale="1 1 1"/g)||[]).length,189,'native vertices override inherited FlyBody mesh scale');
  const empty=createFlybodyHabitatCollision({...createHabitat(fruit),fruit:[]});
  assert.deepEqual(result.heights,empty.heights,'fruit never fills the bowl heightfield');
  assert.equal(result.fruitGeomNames.habitat_fruit_1_banana_30,1);assert.equal(result.fruitGeomNames.habitat_fruit_4_apple,4);
});
test('legacy synthetic surface fixtures preserve their supplied floor',()=>{
  const result=createFlybodyHabitatCollision({surface:()=>({y:.1}),ceiling:50},{resolution:9});
  assert.equal(result.stats.fruitGeoms,0);assert.equal(result.heights.length,81);
  assert.ok(result.heights.every(x=>x===1));assert.match(result.assets,/size="6.6 6.6 0.01 .1"/);
});
test('invalid collision inputs fail before producing invalid native geometry',()=>{
  assert.throws(()=>createFlybodyHabitatCollision({surface:()=>({y:NaN}),ceiling:50}));
  assert.throws(()=>createFlybodyHabitatCollision({surface:()=>({y:-1}),ceiling:50}));
  assert.throws(()=>createFlybodyHabitatCollision(createHabitat(fruit),{resolution:1}));
});
