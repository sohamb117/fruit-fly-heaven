import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
import loadMujoco from '../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {FlyBodyFlightReference} from '../web/flybody-flight-reference.js';
import {createBoundedFlightTrajectory} from '../web/flybody-reference-trajectory.js';

const args=Object.fromEntries(process.argv.slice(2).map(v=>v.replace(/^--/,'').split('=')));
const options={kind:args.kind||'brake-hover',durationSeconds:Number(args.seconds||10),
  brakingSeconds:Number(args.braking||.3),radiusCm:Number(args.radius||2.5),bankScale:Number(args.bank||1)};
const output=args.output||`reports/flybody-bounded-${options.kind}-${options.brakingSeconds}-${options.radiusCm}-${options.bankScale}.json`;
const [mj,xml,original,policy]=await Promise.all([loadMujoco(),fs.readFile('models/flybody-flight-reference.xml','utf8'),
  fs.readFile('models/flybody-flight-reference.json','utf8').then(JSON.parse),fs.readFile('models/flybody-flight-policy.json','utf8').then(JSON.parse)]);
const metadata=createBoundedFlightTrajectory(original,options),body=new FlyBodyFlightReference(mj,xml,metadata,policy);
const initial=body.sample().position,trace=[],start=performance.now();
let maxError=0,maxAttitude=0,maxDistance=0,minHeight=Infinity,maxHeight=-Infinity,maxExternal=0,maxRootActuator=0;
while(!body.done){
  body.step();
  if(body.controlStep%25!==0&&!body.done)continue;
  const sample=body.sample(),distance=Math.hypot(sample.position[0]-initial[0],sample.position[1]-initial[1]);
  maxError=Math.max(maxError,sample.referenceError);maxAttitude=Math.max(maxAttitude,sample.attitudeError);
  maxDistance=Math.max(maxDistance,distance);minHeight=Math.min(minHeight,sample.position[2]);maxHeight=Math.max(maxHeight,sample.position[2]);
  maxExternal=Math.max(maxExternal,sample.maximumAppliedForce);maxRootActuator=Math.max(maxRootActuator,sample.maximumRootActuatorForce);
  trace.push({...sample,distanceFromStartCm:distance});
  if(body.controlStep%5000===0)console.log(JSON.stringify({time:sample.time,error:sample.referenceError,distance,height:sample.position[2]}));
}
const report={date:new Date().toISOString(),scope:'Prescribed bounded trajectory tracked by released joint policy; one fly, original airborne dynamics, no BANC control or takeoff/landing claim.',
  options,source:{xml:metadata.source.xml_sha256,trajectory:createHash('sha256').update(await fs.readFile('web/flybody-reference-trajectory.js')).digest('hex')},
  seconds:body.data.time,wallSeconds:(performance.now()-start)/1000,termination:body.terminationReason,
  maximumReferenceErrorCm:maxError,maximumAttitudeErrorDegrees:maxAttitude,maximumDistanceFromStartCm:maxDistance,
  minimumHeightCm:minHeight,maximumHeightCm:maxHeight,maximumAppliedForce:maxExternal,maximumRootActuatorForce:maxRootActuator,
  passed:body.terminationReason==='trajectory complete'&&maxDistance<7&&minHeight>.5&&maxExternal===0&&maxRootActuator===0,
  trace};
await fs.writeFile(output,JSON.stringify(report,null,2)+'\n');body.dispose();
console.log(JSON.stringify({...report,trace:undefined,output},null,2));
