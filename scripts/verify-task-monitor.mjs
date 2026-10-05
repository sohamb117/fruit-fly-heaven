import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import loadMujoco from '../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {TaskMonitor} from '../web/banc/embodiment.js';
import {FlyBodyFlightReference} from '../web/flybody-flight-reference.js';

const report={date:new Date().toISOString(),scope:'Reporting-only TaskMonitor validation: archived failed BANC flight and a fresh published-policy flight reference. Reference flight is not BANC-controlled flight; synthetic stable landing tests do not demonstrate physical BANC landing.',
  observationCriteria:TaskMonitor.observationCriteria,
  embodimentSha256:createHash('sha256').update(await fs.readFile('web/banc/embodiment.js')).digest('hex'),archivedTraces:[]};
const body=()=>({time:0,x:0,y:0,z:1,vx:0,vy:0,vz:0,heading:0,wingPower:0,
  airborne:false,onFood:false,mouthContact:false,proboscis:0,food:{x:100,y:0,radius:.1},odor:[0,0]});
for(const file of ['reports/flybody-flight-live-audit.json','reports/flybody-flight-repair-final-live.json']){
  const source=await fs.readFile(file),rows=JSON.parse(source).flightTrace,m=new TaskMonitor(),b=body();
  let oldAirTime=0,oldFlightTicks=0,previous=0;
  const reasons={};
  for(const r of rows){
    const dt=r.t-previous;previous=r.t;
    [b.x,b.y,b.z]=r.position;[b.vx,b.vy,b.vz]=r.velocity;
    Object.assign(b,{time:r.t,tilt:r.tilt,angularSpeed:r.angularSpeed,
      wingPower:(r.wings[0]+r.wings[1])/2,airborne:r.airborne,data:{ncon:r.contacts.length},
      environmentContactCount:r.contacts.filter(c=>c.geoms.includes('ground')&&c.distance<=.002).length});
    m.step(b,dt,0);oldAirTime=r.airborne?oldAirTime+dt:0;
    if(oldAirTime>.15&&b.wingPower>.1)oldFlightTicks++;
    reasons[m.flightEvidence.reason]=(reasons[m.flightEvidence.reason]??0)+1;
  }
  const observed=m.events.filter(e=>['landing','flight'].includes(e.stage));
  assert.equal(observed.length,0,`${file}: ${JSON.stringify(observed)}`);
  report.archivedTraces.push({file,sha256:createHash('sha256').update(source).digest('hex'),samples:rows.length,
    simulationSeconds:rows.at(-1).t,legacyFlightEligibleTicks:oldFlightTicks,newFlightOrLandingEvents:observed,reasons,
    limitation:'Archive records scalar angular speed, so angular oscillation cannot be averaged by axis; conservative fallback is used. Environment contact is identified by the archived ground geometry name. No food-contact or external-force evidence was logged.'});
}
const [mj,xml,metadata,policy]=await Promise.all([loadMujoco(),fs.readFile('models/flybody-flight-reference.xml','utf8'),
  ...['models/flybody-flight-reference.json','models/flybody-flight-policy.json'].map(p=>fs.readFile(p,'utf8').then(JSON.parse))]);
const reference=new FlyBodyFlightReference(mj,xml,metadata,policy),monitor=new TaskMonitor(),b=body(),start=performance.now();
let maximumAppliedForce=0,maximumRootActuatorForce=0,maximumSelfContacts=0;
for(let i=0;i<2000;i++){
  assert(reference.step());const d=reference.data,root=reference.rootBody;
  [b.x,b.y,b.z]=d.xpos.subarray(root*3,root*3+3);[b.vx,b.vy,b.vz]=d.qvel.subarray(0,3);
  let environmentContactCount=0;
  const contacts=d.ncon?d.contact:null;
  try{for(let k=0;k<d.ncon;k++){
    const contact=contacts.get(k);
    try{if(contact.dist<=.002&&(reference.model.geom_bodyid[contact.geom[0]]===0||reference.model.geom_bodyid[contact.geom[1]]===0))environmentContactCount++;}
    finally{contact.delete();}
  }}finally{contacts?.delete();}
  maximumSelfContacts=Math.max(maximumSelfContacts,d.ncon-environmentContactCount);
  Object.assign(b,{time:d.time,quaternion:d.xquat.subarray(root*4,root*4+4),data:d,environmentContactCount,
    wingPower:1,airborne:environmentContactCount===0});
  monitor.step(b,metadata.control_timestep,0);
  maximumAppliedForce=Math.max(maximumAppliedForce,...Array.from(d.xfrc_applied,Math.abs),...Array.from(d.qfrc_applied,Math.abs));
  maximumRootActuatorForce=Math.max(maximumRootActuatorForce,...Array.from(d.qfrc_actuator.subarray(0,6),Math.abs));
}
report.reference={seconds:b.time,wallSeconds:(performance.now()-start)/1000,events:monitor.events,
  evidence:monitor.flightEvidence,maximumAppliedForce,maximumRootActuatorForce,maximumSelfContacts,
  activeWingEvidence:'The test runs the released learned controller plus wingbeat generator on every control step. wingPower=1 marks this known active reference actuator path only.'};
assert(monitor.events.some(e=>e.stage==='flight'),JSON.stringify(report.reference));
assert(monitor.flightEvidence.qualified,JSON.stringify(monitor.flightEvidence));
assert.equal(maximumAppliedForce,0);assert.equal(maximumRootActuatorForce,0);
reference.dispose();report.passed=true;
await fs.writeFile('reports/task-monitor-validation.json',JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));
