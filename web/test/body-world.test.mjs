import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {BodyWorld,createHabitat,sensoryRates,isAirborne,applyNeuralOutput,decodeMotorOutput,decodeBehaviorOutput,MOTOR_DECODER} from '../body-world.js';

const data=JSON.parse(await readFile(new URL('../habitat.json',import.meta.url)));
const outputs=JSON.parse(await readFile(new URL('../motor-outputs.json',import.meta.url)));
const make=(n=1,motor={},options={})=>new BodyWorld(data.fruit,data.flies.slice(0,n).map(f=>({...f,brain:{time_ms:100,motor:{...motor}}})),options);
const advance=(world,seconds,hz=60)=>{for(let i=0;i<seconds*hz;i++)world.advance(1/hz);};
const setMotor=(world,motor)=>{for(const f of world.flies)f.brain={time_ms:f.brain.time_ms+2,motor:{...motor}};};
const flat=(motor={})=>new BodyWorld(data.fruit,[{id:1,x:50,z:0,heading:Math.PI,brain:{time_ms:100,motor}}]);
const pose=(f)=>[f.x,f.y,f.z,f.heading,f.gait,f.wingPhase,f.groomPhase,f.feedPhase,f.antennaPhase];

test('original behavior mapping walks from steering, turns by side, and slows for feeding',()=>{
  const brain={time_ms:100,walk_hz:0,left_hz:40,right_hz:0,feed_hz:0,motor:{}};
  const a=decodeBehaviorOutput(brain),fed=decodeBehaviorOutput({...brain,feed_hz:35});
  assert.ok(Math.abs(a.forward*MOTOR_DECODER.walkSpeed-1.579002561799232)<1e-12);
  assert.ok(Math.abs(a.turn*MOTOR_DECODER.yawRate-2.1324636366761416)<1e-12);
  assert.equal(fed.forward,a.forward/2);assert.equal(fed.turn,a.turn);
  const right=decodeBehaviorOutput({...brain,left_hz:0,right_hz:40});
  assert.equal(right.forward,a.forward);assert.equal(right.turn,-a.turn);
  const walking=decodeBehaviorOutput({...brain,walk_hz:30,left_hz:0});
  assert.ok(walking.forward>a.forward);assert.equal(walking.turn,0);
  assert.equal(decodeMotorOutput(brain).forward,0);
});

test('behavior mode moves 100 bodies with real-rate readout shape; switching never edits brain state',()=>{
  const world=make(100,{}, {movementMode:'behavior'});
  for(const f of world.flies)Object.assign(f.brain,{walk_hz:0,left_hz:35,right_hz:5,feed_hz:20});
  const before=world.poses(),brains=structuredClone(world.flies.map(f=>f.brain));advance(world,1);
  assert.ok(world.flies.every((f,i)=>Math.hypot(f.x-before[i].x,f.z-before[i].z)>.1));
  assert.ok(world.flies.every(f=>f.feedback.speed>0&&f.feedback.legs.some(l=>l.speed>0)));
  const time=world.time;world.setMovementMode('direct');assert.equal(world.time,time);
  advance(world,2);const stopped=world.flies.map(pose);advance(world,1);
  assert.deepEqual(world.flies.map(pose),stopped);
  world.setMovementMode('behavior');advance(world,1);
  assert.ok(world.flies.some((f,i)=>f.x!==stopped[i][0]||f.z!==stopped[i][2]));
  assert.deepEqual(world.flies.map(f=>f.brain),brains);
  assert.throws(()=>world.setMovementMode('unknown'),RangeError);assert.equal(world.movementMode,'behavior');
});

test('behavior mode preserves zero-input stillness, disconnection and identified flight outputs',()=>{
  const world=make(100,{}, {movementMode:'behavior'}),initial=world.flies.map(pose);advance(world,2);
  world.flies.forEach((f,i)=>pose(f).forEach((n,j)=>assert.ok(Math.abs(n-initial[i][j])<1e-10)));
  const brain={time_ms:100,walk_hz:40,left_hz:40,feed_hz:35,motor:{wing_left:42,wing_right:42,takeoff:42,groom:20,proboscis:30}};
  const direct=decodeMotorOutput(brain),behavior=decodeBehaviorOutput(brain);
  for(const key of Object.keys(direct).filter(k=>k!=='forward'&&k!=='turn'))assert.equal(behavior[key],direct[key]);
  assert.ok(Object.values(decodeBehaviorOutput(brain,{connected:false})).every(n=>n===0));
  assert.equal(decodeBehaviorOutput(brain,{flightEnabled:false}).wing,0);
  for(const invalid of [{time_ms:0,walk_hz:100},{time_ms:100,walk_hz:NaN,left_hz:Infinity,right_hz:-100,feed_hz:NaN}])
    assert.ok(Object.values(decodeBehaviorOutput(invalid)).every(n=>n===0));
  for(const f of world.flies)f.brain=structuredClone(brain);
  world.setFlightEnabled(false);advance(world,1);world.setMotorCoupling(false);advance(world,2);
  const stopped=world.flies.map(pose);advance(world,1);assert.deepEqual(world.flies.map(pose),stopped);
});

test('zero motor input leaves 100 bodies and every animated actuator still',()=>{
  const world=make(100);
  // Old aggregate signals must not act as a hidden fallback for identified outputs.
  for(const f of world.flies)Object.assign(f.brain,{spikes:90000,walk_hz:100,left_hz:100,feed_hz:100});
  const initial=world.flies.map(pose);advance(world,30);
  for(let i=0;i<100;i++)pose(world.flies[i]).forEach((n,j)=>assert.ok(Math.abs(n-initial[i][j])<1e-10,`fly ${i}, coordinate ${j}`));
  assert.equal(world.takeoffs,0);assert.equal(world.landings,0);
  assert.ok(world.flies.every(f=>f.motion==='resting'));
  assert.ok(Object.values(decodeMotorOutput({time_ms:100,motor:{forward:NaN,wing_left:Infinity}})).every(n=>n===0));
});

test('identified forward, reverse and left/right signals determine ground movement',()=>{
  const forward=flat({forward:42}),reverse=flat({reverse:42});
  advance(forward,.5);advance(reverse,.5);
  assert.ok(forward.flies[0].x<47);assert.ok(reverse.flies[0].x>52);
  const left=flat({turn_left:22}),right=flat({turn_right:22});
  advance(left,.5);advance(right,.5);
  assert.ok(left.flies[0].heading<0);assert.ok(right.flies[0].heading>0);
  assert.equal(left.flies[0].x,50);assert.equal(right.flies[0].x,50);
  assert.equal(left.flies[0].z,0);assert.equal(right.flies[0].z,0);
  assert.ok(left.flies[0].gait>0);
});

test('wing power produces lift; landing output and gravity return bodies to surfaces',()=>{
  const world=make(100,{wing_left:42,wing_right:42});advance(world,3);
  assert.equal(world.takeoffs,100);assert.ok(world.flies.every(isAirborne));
  assert.ok(world.flies.every(f=>f.altitude>1&&!f.contact));
  setMotor(world,{wing_left:42,wing_right:42,landing:42});advance(world,8);
  assert.equal(world.flies.some(isAirborne),false);assert.equal(world.landings,100);
  for(const f of world.flies){
    assert.ok(Number.isFinite(f.x+f.y+f.z));assert.ok(Math.hypot(f.x,f.z)<=61.000001);
    assert.ok(f.y>=world.habitat.surface(f.x,f.z).y-1e-8);assert.ok(f.y<=world.habitat.ceiling);
  }
});

test('a held takeoff command does not schedule repeated jumps',()=>{
  const world=flat({takeoff:42});advance(world,5);
  assert.equal(world.takeoffs,1);assert.equal(world.landings,1);assert.equal(world.flies[0].airborne,false);
  setMotor(world,{});advance(world,.1);setMotor(world,{takeoff:42});advance(world,.1);
  assert.equal(world.takeoffs,2);
});

test('grooming, proboscis and antenna motion require their own output',()=>{
  const world=flat({groom:22,proboscis:22,antenna_left:22});advance(world,1);
  const f=world.flies[0];assert.equal(f.x,50);assert.equal(f.z,0);assert.equal(f.wingPhase,0);
  assert.notEqual(f.groomPhase,0);assert.notEqual(f.feedPhase,0);assert.notEqual(f.antennaPhase,0);
  assert.ok(f.actuators.antennaLeft>0);assert.equal(f.actuators.antennaRight,0);
  const phases=[f.groomPhase,f.feedPhase,f.antennaPhase];setMotor(world,{});advance(world,1);
  assert.deepEqual([f.groomPhase,f.feedPhase,f.antennaPhase],phases);
});

test('disconnecting motors removes drive while preserving neural output and allowing passive settling',()=>{
  const world=make(8,{wing_left:42,wing_right:42,forward:42,turn_left:22});advance(world,2);
  const neural=structuredClone(world.flies.map(f=>f.brain));world.setMotorCoupling(false);advance(world,10);
  assert.deepEqual(world.flies.map(f=>f.brain),neural);
  assert.ok(world.flies.every(f=>Object.values(f.actuators).every(n=>n===0)));
  assert.equal(world.flies.some(isAirborne),false);
  const stopped=world.flies.map(pose);advance(world,5);assert.deepEqual(world.flies.map(pose),stopped);
  world.setMotorCoupling(true);advance(world,2);assert.ok(world.flies.some(isAirborne));
  world.setFlightEnabled(false);advance(world,10);assert.equal(world.flies.some(isAirborne),false);
});

test('waiting, pause and fixed stepping do not invent movement',()=>{
  const waiting=make(1,{forward:42});waiting.flies[0].brain.time_ms=0;
  const initial=waiting.poses();advance(waiting,2);assert.deepEqual(waiting.poses(),initial);
  const a=make(8,{forward:20,turn_left:10}),b=make(8,{forward:20,turn_left:10});
  advance(a,4,60);advance(b,4,30);assert.deepEqual(a.poses(),b.poses());
  const frozen=JSON.stringify(a.flies.map(pose)),time=a.time;a.advance(0);a.advance(NaN);a.advance(-1);
  assert.equal(JSON.stringify(a.flies.map(pose)),frozen);assert.equal(a.time,time);
});

test('actual 3D pose and surface contact feed the sensory adapter',()=>{
  const habitat=createHabitat(data.fruit),apple=data.fruit[3],top=habitat.surface(apple.x,apple.z);
  assert.equal(top.y,apple.y+apple.radius*.94);assert.deepEqual(top.normal,[0,1,0]);
  const landed={x:apple.x,y:top.y,z:apple.z,heading:0,contact:true},airborne={...landed,y:top.y+25,contact:false};
  assert.equal(sensoryRates(landed,habitat)[2],150);assert.equal(sensoryRates(airborne,habitat)[2],0);
  assert.ok(sensoryRates(airborne,habitat)[0]<sensoryRates(landed,habitat)[0]);
  assert.deepEqual(sensoryRates(landed,habitat,{odor:false,taste:false}),[0,0,0]);
});

test('late neural messages cannot roll body poses or neural time back',()=>{
  const world=flat({forward:30});advance(world,1);const f=world.flies[0],before=world.poses()[0];
  applyNeuralOutput(f,{x:999,y:-999,z:999,brain:{time_ms:102,motor:{forward:10}},senses:[30,40,150]});
  assert.deepEqual(world.poses()[0],before);assert.equal(f.brain.time_ms,102);
  applyNeuralOutput(f,{brain:{time_ms:101,motor:{}},senses:[0,0,0]});assert.equal(f.brain.time_ms,102);
  assert.deepEqual(f.senses,[30,40,150]);assert.doesNotThrow(()=>structuredClone(world.poses()));
});

test('all actuator readouts retain identifiable, bounded v783 neuron IDs',()=>{
  assert.equal(outputs.neuron_count,138639);assert.equal(outputs.channels.length,12);
  const all=new Set();
  for(const channel of outputs.channels){
    assert.ok(channel.indices.length>0);assert.equal(channel.indices.length,channel.cells.length);
    assert.match(channel.source,/^https:\/\//);
    channel.cells.forEach((cell,i)=>{assert.equal(cell.index,channel.indices[i]);assert.ok(cell.index>=0&&cell.index<outputs.neuron_count);assert.match(cell.root_id,/^\d+$/);all.add(cell.index);});
  }
  assert.equal(all.size,97);
  for(const side of ['left','right'])assert.ok(outputs.channels.find(c=>c.key==='wing_'+side).cells.every(c=>/^DNg02_[a-h]$/.test(c.type)&&c.side===side));
});
