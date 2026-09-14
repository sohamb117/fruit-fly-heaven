import test from 'node:test';
import assert from 'node:assert/strict';
import {createHabitat} from '../body-world.js';
import {createFlybodyHabitatCollision} from '../flybody-habitat-collision.js';
import {SPACIOUS_MAINTAINED_SCENE as profile, validateMaintainedScene, matchMaintainedScene,
  maintainedFloorHeightScene, assertMaintainedObservationScene} from '../flight-scene-profile.js';

const fruit = [{kind: 'apple', x: 0, z: 0, y: 11, radius: 2, remaining: 1}];
const near = (actual, expected) => assert(Math.abs(actual - expected) < 1e-12, `${actual} != ${expected}`);
const geoms = xml => new Map([...xml.matchAll(/<geom\s+([^>]+)\/>/g)].map(match => {
  const attributes = Object.fromEntries([...match[1].matchAll(/([\w]+)="([^"]*)"/g)].map(m => [m[1], m[2]]));
  return [attributes.name, attributes];
}));
const vec = text => text.split(/\s+/).map(Number);

test('scene profiles have exact finite bounds, own keys and immutable copied values', () => {
  assert.equal(validateMaintainedScene(undefined), null);
  assert.deepEqual(profile, {schemaVersion: 1, profile: 'spacious-maintained-flight-v1', radiusCm: 50,
    ceilingCm: 50, floorCapRadiusCm: 6.5});
  for (const value of [null, [], {}, Object.create(profile), {...profile, schemaVersion: 2},
    {...profile, profile: 'unknown'}, {...profile, radiusCm: Infinity}, {...profile, radiusCm: 49},
    {...profile, ceilingCm: NaN}, {...profile, ceilingCm: 500}, {...profile, floorCapRadiusCm: 6.6},
    {...profile, extra: true}, {...profile, [Symbol('extra')]: true}])
    assert.throws(() => validateMaintainedScene(value), TypeError);
  const supplied = {...profile}, validated = validateMaintainedScene(supplied);
  supplied.radiusCm = 1;
  assert.deepEqual(validated, profile); assert(Object.isFrozen(validated)); assert.notEqual(validated, supplied);
});

test('scene selection requires matching config, body metadata and observation units', () => {
  assert.equal(matchMaintainedScene(undefined, undefined), null);
  assert.deepEqual(matchMaintainedScene({...profile}, {...profile}), profile);
  assert.throws(() => matchMaintainedScene(profile, undefined), /must agree/);
  assert.throws(() => matchMaintainedScene(undefined, profile), /must agree/);
  assert.throws(() => matchMaintainedScene(profile, {...profile, ceilingCm: 49}));
  assert.doesNotThrow(() => assertMaintainedObservationScene({ceiling: 5}, null));
  assert.throws(() => assertMaintainedObservationScene({curriculumScene: profile, ceiling: 50}, null));
  assert.throws(() => assertMaintainedObservationScene({ceiling: 50}, profile));
  assert.throws(() => assertMaintainedObservationScene({curriculumScene: profile, ceiling: 500}, profile));
  assert.doesNotThrow(() => assertMaintainedObservationScene({curriculumScene: {...profile}, ceiling: 50}, profile));
});

test('optional scene preserves central fruit, floor and odor while capping only the outer floor', () => {
  const legacy = createHabitat(fruit), explicitDefault = createHabitat(fruit, undefined), spacious = createHabitat(fruit, profile);
  assert.equal(legacy.ceiling, 43); assert(!Object.hasOwn(legacy, 'maintainedScene'));
  assert.deepEqual(Object.keys(legacy), Object.keys(explicitDefault)); assert.equal(spacious.ceiling, 500);
  for (const radius of [0, 2, 20, 64, 65]) for (const angle of [0, .7, 2.1, 4.3]) {
    const x = radius * Math.cos(angle), z = radius * Math.sin(angle);
    assert.deepEqual(spacious.surface(x, z), legacy.surface(x, z));
    assert.deepEqual(explicitDefault.surface(x, z), legacy.surface(x, z));
  }
  for (const [x, y, z] of [[0, 11, 0], [30, 20, -10], [400, 40, 300]])
    assert.equal(spacious.odor(x, y, z), legacy.odor(x, y, z));
  near(legacy.surface(100, 0).y, 38.5);
  for (const [x, z] of [[66, 0], [0, -499], [400, 300]]) {
    near(spacious.surface(x, z).y, 17.1325);
    assert.deepEqual(spacious.surface(x, z).normal, [0, 1, 0]);
    near(maintainedFloorHeightScene(x, z, profile), 17.1325);
  }
});

test('spacious collision geometry retains the central grid and fruit, with a finite closed apron', () => {
  const legacy = createFlybodyHabitatCollision(createHabitat(fruit));
  const spacious = createFlybodyHabitatCollision(createHabitat(fruit, profile));
  assert.equal(spacious.heights.length, 257 * 257);
  assert.equal(spacious.assets, legacy.assets, 'Keep native grid scale and fruit mesh bytes');
  assert.deepEqual(spacious.stats, legacy.stats); assert.deepEqual(spacious.fruitGeomNames, legacy.fruitGeomNames);
  for (let row = 0; row < 257; row += 16) for (let col = 0; col < 257; col += 16) {
    const x = (col / 256 * 2 - 1) * 66, z = (row / 256 * 2 - 1) * 66;
    if (x * x + z * z <= 65 * 65) assert.equal(spacious.heights[row * 257 + col], legacy.heights[row * 257 + col]);
  }
  const old = geoms(legacy.geoms), current = geoms(spacious.geoms);
  assert.deepEqual(current.get('ground'), old.get('ground'));
  assert.deepEqual(current.get('habitat_fruit_0_apple'), old.get('habitat_fruit_0_apple'));
  assert.equal(current.size, old.size + 4);
  assert.deepEqual(vec(old.get('ceiling').pos), [0, 0, 4.45]);
  assert.deepEqual(vec(current.get('ceiling').pos), [0, 0, 50.15]);
  const apron = [...current].filter(([name]) => name.startsWith('curriculum_apron'))
    .map(([, g]) => ({position: vec(g.pos), size: vec(g.size)}));
  assert.equal(apron.length, 4);
  for (const {position: p, size: s} of apron) {near(p[2] + s[2], 1.71325); near(p[2] - s[2], -1);}
  for (let k = 0; k < 64; k++) {
    const wall = current.get('wall' + k), p = vec(wall.pos), s = vec(wall.size);
    near(Math.hypot(p[0], p[1]), 49.9); near(p[2] + s[2], 50.15); near(p[2] - s[2], -1);
  }
  for (const radius of [7, 20, 49.8]) for (let k = 0; k < 16; k++) {
    const x = radius * Math.cos(k), y = radius * Math.sin(k);
    if (Math.abs(x) <= 6.6 && Math.abs(y) <= 6.6) continue;
    assert(apron.some(({position: p, size: s}) => Math.abs(x - p[0]) <= s[0] + 1e-10 && Math.abs(y - p[1]) <= s[1] + 1e-10));
  }
});

test('scene collision generation rejects mismatched bounds or altered central resolution', () => {
  const habitat = createHabitat(fruit, profile);
  assert.throws(() => createFlybodyHabitatCollision(habitat, {resolution: 129}), /original central grid/);
  assert.throws(() => createFlybodyHabitatCollision(habitat, {extent: 7}), /original central grid/);
  assert.throws(() => createFlybodyHabitatCollision({...habitat, ceiling: 50}), /declared ceiling/);
  assert.equal(createFlybodyHabitatCollision(createHabitat(fruit), {resolution: 9}).heights.length, 81,
    'The absent-profile legacy option remains available');
});
