import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {BodyWorld} from '../body-world.js';
const data=JSON.parse(await readFile(new URL('../habitat.json',import.meta.url)));
const make=(mode='behavior')=>new BodyWorld(data.fruit,data.flies.map(f=>({...f,brain:{time_ms:100,walk_hz:0,left_hz:35,right_hz:5,feed_hz:20,motor:{proboscis:20,landing:10}}})),{movementMode:mode});
const step=(w,seconds)=>{for(let i=0;i<seconds*60;i++)w.advance(1/60);};

test('behavior readouts produce takeoff, sustained flight and landing for 100 flies without wing spikes',()=>{
  const w=make(),brains=structuredClone(w.flies.map(f=>f.brain));let airborne=0;
  for(let i=0;i<8*60;i++){w.advance(1/60);airborne=Math.max(airborne,w.flies.filter(f=>f.airborne).length);}
  assert.equal(w.takeoffs,100);assert.equal(airborne,100);assert.ok(w.landings>=95);
  assert.ok(w.flies.every(f=>Number.isFinite(f.x+f.y+f.z)&&f.y>=w.habitat.surface(f.x,f.z).y-1e-8));
  assert.deepEqual(w.flies.map(f=>f.brain),brains);
});

test('flight disconnection lands bodies and stops preparation; direct mode has no automatic flight',()=>{
  const direct=make('direct');step(direct,12);assert.equal(direct.takeoffs,0);
  const w=make();step(w,4.5);assert.ok(w.flies.some(f=>f.airborne));
  w.setFlightEnabled(false);step(w,6);const takeoffs=w.takeoffs;
  assert.ok(w.flies.every(f=>!f.airborne&&f.actuators.wing===0&&f.flightProgram.preparation===0));
  step(w,4);assert.equal(w.takeoffs,takeoffs);
  w.setFlightEnabled(true);w.setMotorCoupling(false);step(w,8);assert.equal(w.takeoffs,takeoffs);
});

test('quiet brains never prepare flight; pause and mode switches cannot create a takeoff',()=>{
  const w=make();for(const f of w.flies)f.brain={time_ms:100,motor:{}};step(w,15);
  assert.equal(w.takeoffs,0);assert.ok(w.flies.every(f=>f.flightProgram.preparation===0));
  const poses=w.poses();w.advance(0);assert.deepEqual(w.poses(),poses);
  w.setMovementMode('direct');w.setMovementMode('behavior');step(w,5);assert.equal(w.takeoffs,0);
});
