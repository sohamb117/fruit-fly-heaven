import {test} from 'node:test';
import assert from 'node:assert/strict';
import {TaskMonitor} from '../../../web/banc/embodiment.js';

const baseBody=()=>({time:0,x:0,y:0,z:1,vx:0,vy:0,vz:0,heading:0,
  quaternion:[1,0,0,0],angularVelocity:[0,0,0],wingPower:.8,
  airborne:true,onFood:false,mouthContact:false,proboscis:0,environmentContactCount:0,
  food:{x:10,y:0,radius:.1},odor:[0,0],
  data:{ncon:0,xfrc_applied:[0,0,0,0,0,0],qfrc_applied:[0,0,0,0,0,0]}});
function observe(monitor,body,seconds,update=()=>{},dt=.001){
  const start=body.time;
  for(let i=0;i<Math.round(seconds/dt);i++){
    body.time=start+(i+1)*dt;update(body,body.time);monitor.step(body,dt,0);
  }
}
const has=(monitor,stage)=>monitor.events.some(e=>e.stage===stage);
const hover=(monitor,body,seconds=.4)=>observe(monitor,body,seconds);
const contact=body=>{body.airborne=false;body.onFood=true;body.data.ncon=1;body.environmentContactCount=1;body.vx=body.vy=body.vz=0;};

test('active wings during upright ballistic falling do not establish flight',()=>{
  const m=new TaskMonitor(),b=baseBody();
  observe(m,b,.5,(b,t)=>{b.z=10-981*t*t/2;b.vz=-981*t;});
  assert.equal(has(m,'flight'),false);assert.equal(m.flightEvidence.reason,'ballistic or insufficient support');
});

test('rapid descent and upright spinning each fail sustained flight evidence',()=>{
  for(const update of [
    (b,t)=>{b.z=20-30*t;b.vz=-30;},
    (b,t)=>{b.quaternion=[Math.cos(40*t/2),0,0,Math.sin(40*t/2)];b.angularVelocity=[0,0,40];},
  ]){
    const m=new TaskMonitor(),b=baseBody();observe(m,b,.5,update);assert.equal(has(m,'flight'),false);
  }
});

test('tumbling with wings active does not establish flight',()=>{
  const m=new TaskMonitor(),b=baseBody();
  observe(m,b,1,(b,t)=>{b.quaternion=[Math.cos(20*t/2),Math.sin(20*t/2),0,0];b.angularVelocity=[20,0,0];});
  assert.equal(has(m,'flight'),false);
});

test('missing attitude, missing angular motion, environment collision or applied force cannot count as flight',()=>{
  for(const alter of [b=>delete b.quaternion,b=>delete b.angularVelocity,b=>delete b.environmentContactCount,b=>b.environmentContactCount=1,
    b=>b.data.xfrc_applied[2]=1,b=>b.data.qfrc_applied[0]=1,b=>b.data.qfrc_applied[0]=NaN]){
    const m=new TaskMonitor(),b=baseBody();alter(b);hover(m,b);assert.equal(has(m,'flight'),false);
  }
});

test('body self contact does not falsely reject supported flight with no environment contact',()=>{
  const m=new TaskMonitor(),b=baseBody();b.data.ncon=8;hover(m,b);
  assert(has(m,'flight'));assert.equal(m.flightEvidence.qualified,true);
});

test('sustained supported hover tolerates real wingbeat-scale oscillatory motion',()=>{
  const m=new TaskMonitor(),b=baseBody();
  observe(m,b,.24);assert.equal(has(m,'flight'),false);
  observe(m,b,.4,(b,t)=>{
    const phase=2*Math.PI*200*t,pitch=47.5*Math.PI/180+.01*Math.sin(phase);
    b.z=1+.0002*Math.sin(phase);b.vz=.0002*2*Math.PI*200*Math.cos(phase);
    b.quaternion=[Math.cos(pitch/2),0,Math.sin(pitch/2),0];
    b.angularVelocity=[4*Math.sin(phase),.01*2*Math.PI*200*Math.cos(phase),2*Math.cos(phase)];
  });
  assert(has(m,'flight'));assert.equal(m.flightEvidence.qualified,true);assert.equal(m.phase,'flight');
  assert(m.flightEvidence.inferredSupportFraction>.9);assert(m.flightEvidence.meanAngularSpeed<1);
});

test('controlled slow descent can satisfy powered flight evidence',()=>{
  const m=new TaskMonitor(),b=baseBody();
  observe(m,b,.4,(b,t)=>{b.z=3-2*t;b.vz=-2;});
  assert(has(m,'flight'));assert.equal(m.flightEvidence.meanVerticalSpeed,-2);
});

test('sparse observations cannot turn two endpoint samples into sustained flight',()=>{
  const m=new TaskMonitor(),b=baseBody();observe(m,b,1,()=>{},.1);
  assert.equal(has(m,'flight'),false);
});

test('historical flight remains in history while current falling state loses the flight label',()=>{
  const m=new TaskMonitor(),b=baseBody();hover(m,b);assert(has(m,'flight'));
  b.wingPower=0;observe(m,b,.01);
  assert(has(m,'flight'));assert.equal(m.phase,'airborne unverified');assert.equal(m.flightEvidence.qualified,false);
});

test('an upright food contact bounce after qualified flight does not count as landing',()=>{
  const m=new TaskMonitor(),b=baseBody();hover(m,b);contact(b);observe(m,b,.05);
  b.airborne=true;b.onFood=false;b.data.ncon=0;b.environmentContactCount=0;observe(m,b,.2);
  assert.equal(has(m,'landing'),false);
  assert.equal(m.landingCount,0);
});

test('stable upright food contact after qualified flight counts as landing only after 100 ms',()=>{
  const m=new TaskMonitor(),b=baseBody();hover(m,b);contact(b);observe(m,b,.1);
  assert.equal(has(m,'landing'),false);observe(m,b,.002);assert(has(m,'landing'));
  assert.match(m.events.find(e=>e.stage==='landing').evidence,/modeled observation criteria/);
  assert.equal(m.landingCount,1);observe(m,b,.3);assert.equal(m.landingCount,1);
  for(let i=0;i<10;i++){observe(m,b,.001);assert.equal(m.phase,'landing','current qualified landing keeps a stable label');}
  b.airborne=true;b.onFood=false;b.environmentContactCount=0;hover(m,b);
  contact(b);observe(m,b,.12);assert.equal(m.landingCount,2);
});

test('a crash settling on food is not a landing, including a crash after earlier qualified flight',()=>{
  for(const initiallyQualified of [false,true]){
    const m=new TaskMonitor(),b=baseBody();if(initiallyQualified)hover(m,b);
    observe(m,b,.06,(b,t)=>{b.vz=-40;b.z=10-40*t;});
    contact(b);hover(m,b,.3);assert.equal(has(m,'landing'),false);
  }
});

test('explicit placement cannot join earlier flight evidence into a landing',()=>{
  const m=new TaskMonitor(),b=baseBody();hover(m,b);contact(b);observe(m,b,.12);
  assert.equal(m.landingCount,1);
  b.airborne=true;b.onFood=false;b.environmentContactCount=0;hover(m,b);assert(m.flightEvidence.qualified);
  const history=m.events.slice();m.resetContinuity();b.x=4;b.y=5;contact(b);observe(m,b,.15);
  assert.equal(m.landingCount,1);assert.deepEqual(m.events,history);assert.notEqual(m.phase,'landing');
  assert.equal(m.flightEvidence.qualified,false);
});

test('upside-down, moving or externally supported food contact cannot count as stable landing',()=>{
  for(const update of [
    b=>{b.quaternion=[0,1,0,0];},
    (b,t)=>{b.x=2*t;},
    b=>{b.data.xfrc_applied[2]=1;},
  ]){
    const m=new TaskMonitor(),b=baseBody();hover(m,b);contact(b);observe(m,b,.3,update);
    assert.equal(has(m,'landing'),false);
  }
});

test('TaskMonitor only reads physical observations and never writes the body or accesses neural commands',()=>{
  const m=new TaskMonitor(),b=baseBody();
  Object.defineProperty(b,'brain',{get(){throw new Error('observer accessed controller');}});
  Object.defineProperty(b,'motor',{get(){throw new Error('observer accessed commands');}});
  Object.freeze(b.quaternion);Object.freeze(b.angularVelocity);Object.freeze(b.data);Object.freeze(b);
  assert.doesNotThrow(()=>m.step(b,.001,0));assert.equal(b.time,0);assert.equal(b.wingPower,.8);
});
