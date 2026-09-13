import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {InternalState,ReducedBody,sensoryCurrents,TaskMonitor} from '../../../web/banc/embodiment.js';
import {createWasmCore,WasmMuscles,WasmJoints} from '../src/wasm.js';
const core=await createWasmCore();
const mechanics=JSON.parse(await fs.readFile(new URL('../../../models/flybody-reduced.json',import.meta.url)));
const io={muscles:[{joint:'coxa_T1_left',sign:1,kind:'leg',indices:[0]},{joint:'proboscis',sign:1,kind:'proboscis_assumption',indices:[1]},{joint:'pump',sign:1,kind:'pump',indices:[2]}]};
const runtime={createMuscles:n=>new WasmMuscles(core,n),createJoints:j=>new WasmJoints(core,j)};
test('quiet and disconnected motor neurons never generate active translation or flight',()=>{
  const a=new ReducedBody(mechanics,io,runtime),b=new ReducedBody(mechanics,io,runtime);
  for(let i=0;i<100;i++){a.step(new Map(),.01);b.step(new Map([[0,100],[1,100],[2,100]]),.01,{coupling:false});}
  assert.equal(a.x,-1);assert.equal(a.y,0);assert.equal(a.airborne,false);assert.equal(a.internal.ingested,0);assert.equal(b.x,a.x);assert.equal(b.internal.ingested,0);a.dispose();b.dispose();
});
test('feeding requires mouth contact and motor-driven probing plus pumping; food is conserved',()=>{
  const b=new ReducedBody(mechanics,io,runtime);b.x=b.food.x-.1;
  for(let i=0;i<20;i++)b.step(new Map([[1,100]]),.01);assert.equal(b.internal.ingested,0);
  for(let i=0;i<100;i++)b.step(new Map([[1,100],[2,100]]),.01);
  assert(b.internal.ingested>0);assert(Math.abs(1-b.food.remaining-b.internal.ingested)<1e-10);assert(b.monitor.events.some(e=>e.stage==='feeding'));b.dispose();
});
test('energy balance drives hunger, absorption and counter-regulatory hormones',()=>{
  const fed=new InternalState(.2),empty=new InternalState(.2);fed.step(.01,.5,0);
  for(let i=0;i<5000;i++){fed.step(.01,0,0);empty.step(.01,0,0);}
  assert(fed.energy>empty.energy);assert(fed.hunger<empty.hunger);assert(fed.insulin>empty.insulin);assert(fed.akh<empty.akh);assert(fed.crop<.5);
});
test('sensory switches remove currents without writing motor indices',()=>{
  const body=new ReducedBody(mechanics,io,runtime),model={manifest:{neuron_count:8},io:{sensory:[{index:4,kind:'odor',side:'left',cell_type:'ORN_DM1'},{index:5,kind:'vision',side:'right'},{index:6,kind:'taste'}]}};
  const input=sensoryCurrents(model,body);assert(input[4]>0&&input[5]>0);assert(input.subarray(0,4).every(v=>v===0));
  assert(sensoryCurrents(model,body,{odor:false,vision:false,taste:false,proprioception:false}).every(v=>v===0));body.dispose();
});
test('stage monitor cannot synthesize a feeding-to-flight cycle',()=>{
  const monitor=new TaskMonitor(),body={x:0,y:0,food:{x:1,y:0},vx:0,vy:0,odor:[0,0],time:0,airborne:false};
  for(let i=0;i<100;i++){body.time+=.01;monitor.step(body,.01,0);}assert.equal(monitor.events.length,0);
});
test('initial contact settling and tiny shuffling on food do not count as landing or approach',()=>{
  const monitor=new TaskMonitor(),body={x:0,y:0,food:{x:1,y:0,radius:2},vx:.02,vy:0,odor:[0,0],time:0,airborne:true,onFood:false};
  monitor.step(body,.001,0);body.airborne=false;body.onFood=true;
  for(let i=0;i<100;i++){body.x+=.0002;body.time+=.01;monitor.step(body,.01,0);}
  assert(!monitor.events.some(e=>['landing','approach'].includes(e.stage)));
});
test('falling off food after feeding is not takeoff',()=>{
  const monitor=new TaskMonitor(),body={x:0,y:0,food:{x:0,y:0,radius:1},vx:0,vy:0,vz:0,odor:[0,0],time:0,airborne:false,onFood:true,wingPower:.8};
  monitor.step(body,.01,.01);body.airborne=true;body.onFood=false;body.vz=-1;
  monitor.step(body,.01,0);assert(!monitor.events.some(e=>e.stage==='takeoff'));
  body.airborne=false;monitor.step(body,.01,0);body.airborne=true;body.vz=1;
  monitor.step(body,.01,0);assert(monitor.events.some(e=>e.stage==='takeoff'));
});
test('surface initialization supports a grounded body on curved terrain without a false landing',()=>{
  const body=new ReducedBody(mechanics,io,runtime,{environment:{surface:(x,y)=>.1*x*x+.1*y*y,foodAt:()=>null,odor:()=>0}});
  body.settleOnSurface();body.step(new Map(),.01);
  assert.equal(body.airborne,false);assert(!body.monitor.events.some(e=>e.stage==='landing'));body.dispose();
});
test('bilateral odor respects body coordinates and excludes unrelated ORNs',()=>{
  const b=new ReducedBody(mechanics,io,runtime);b.heading=0;b.food.y=.5;
  const model={manifest:{neuron_count:3},io:{sensory:[{index:0,kind:'odor',side:'left',cell_type:'ORN_DM1'},{index:1,kind:'odor',side:'right',cell_type:'ORN_DM1'},{index:2,kind:'odor',side:'left',cell_type:'ORN_DA1'}]}};
  const currents=sensoryCurrents(model,b);assert(currents[0]>currents[1]);assert.equal(currents[2],0);b.dispose();
});
test('bilateral flight muscles can lift the body and wing disconnection returns it to support',()=>{
  const ioFlight={muscles:[{joint:'wing_power_left',sign:1,kind:'asynchronous_wing',indices:[0]},{joint:'wing_power_right',sign:1,kind:'asynchronous_wing',indices:[1]}]};
  // Exercise the actual hungry default, which previously could never lift even
  // with fully saturated wing excitation. Fed-only tests hid this failure.
  const body=new ReducedBody(mechanics,ioFlight,runtime),rates=new Map([[0,100],[1,100]]);
  for(let i=0;i<40;i++)body.step(rates,.01);
  assert(body.airborne&&body.z>.3);
  // This reduced model has no roll/pitch state. Lift alone cannot establish
  // upright, supported flight, so the event observer must abstain.
  assert(!body.monitor.events.some(e=>e.stage==='flight'));
  assert.equal(body.monitor.flightEvidence.reason,'missing kinematic evidence');
  for(let i=0;i<300;i++)body.step(rates,.01,{flight:false});
  assert.equal(body.airborne,false);assert.equal(body.wingPower,0);body.dispose();
});
test('depleted reserves cannot supply active wing force',()=>{
  const flight={muscles:[{joint:'wing_power_left',sign:1,kind:'asynchronous_wing',indices:[0]},{joint:'wing_power_right',sign:1,kind:'asynchronous_wing',indices:[1]}]};
  const body=new ReducedBody(mechanics,flight,runtime,{energy:0});
  for(let i=0;i<50;i++)body.step(new Map([[0,100],[1,100]]),.01);
  assert.equal(body.airborne,false);assert.equal(body.wingPower,0);body.dispose();
});
