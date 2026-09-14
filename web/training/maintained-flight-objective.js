import {validateMaintainedScene,assertMaintainedObservationScene} from '../flight-scene-profile.js';
// Pure proposed objective. It never supplies forces, motor commands or resets.
import {FLIGHT_CRITERIA as base} from './flight-objective.js';

export const MAINTAINED_FLIGHT_STAGE='maintained_flight';
export const MAINTAINED_FLIGHT_CRITERIA=Object.freeze({
 version:2,stage:MAINTAINED_FLIGHT_STAGE,durationSeconds:5,sourceFlightCriteriaVersion:base.version,
 interpretation:'Powered controlled airtime following an explicitly airborne reset; no takeoff or landing credit. Thresholds are modeled criteria.',
 verticalSpeedMeasurement:'COM height difference over a fresh contact-free 50 ms window',
 verticalSpeedWindowSeconds:base.flight.supportWindowSeconds,
 flight:base.flight,maximumAngularSpeed:base.maximumAngularSpeed,overturnedSeconds:base.overturnedSeconds,
 maximumRadiusCm:base.maximumRadiusCm,minimumHeightCm:base.minimumHeightCm,ceilingAllowanceCm:base.ceilingAllowanceCm,
 gravityCmPerSecondSquared:base.gravityCmPerSecondSquared,maximumStepSeconds:base.maximumStepSeconds,
});
export function maintainedFlightCriteria(sceneValue){
 const scene=validateMaintainedScene(sceneValue);
 return scene?Object.freeze({...MAINTAINED_FLIGHT_CRITERIA,maximumRadiusCm:scene.radiusCm,curriculumScene:scene}):MAINTAINED_FLIGHT_CRITERIA;
}
const EPS=1e-9,clip=(v,lo=0,hi=1)=>Math.max(lo,Math.min(hi,v));
function finite(value,name,lo=-Infinity,hi=Infinity){
 if(typeof value!=='number'||!Number.isFinite(value)||value<lo||value>hi)throw new TypeError('Invalid maintained-flight observation: '+name);
 return value;
}
function count(value,name,max=Number.MAX_SAFE_INTEGER){const v=finite(value,name,0,max);if(!Number.isInteger(v))throw new TypeError('Invalid count: '+name);return v;}
function observation(value){
 if(!value||value.finite!==true||typeof value.externalForce!=='boolean')throw new TypeError('Invalid maintained-flight observation flags');
 const o={up:clip(finite(value.up,'up',-1-1e-6,1+1e-6),-1,1),angularSpeed:finite(value.angularSpeed,'angularSpeed',0),
  speedCmPerSecond:finite(value.speedCmPerSecond,'speedCmPerSecond',0),verticalSpeed:finite(value.verticalSpeed,'verticalSpeed'),
  height:finite(value.height,'height'),radius:finite(value.radius,'radius',0),ceiling:finite(value.ceiling,'ceiling'),
  wingPower:finite(value.wingPower,'wingPower',0,1),environmentContacts:count(value.environmentContacts,'environmentContacts'),
  nonFootEnvironmentContacts:count(value.nonFootEnvironmentContacts,'nonFootEnvironmentContacts'),
  footSupportCount:count(value.footSupportCount,'footSupportCount',6),footSupportFraction:finite(value.footSupportFraction,'footSupportFraction',0),externalForce:value.externalForce};
 if(o.nonFootEnvironmentContacts>o.environmentContacts||o.footSupportCount>o.environmentContacts||
  ((o.footSupportCount===0)!==(o.footSupportFraction===0))||o.speedCmPerSecond+1e-6<Math.abs(o.verticalSpeed))
  throw new TypeError('Contradictory maintained-flight observation');
 return o;
}
function angularWindow(){
 const rows=[];let seconds=0,integral=0;
 return {clear(){rows.length=0;seconds=0;integral=0;},append(speed,dt){
  const square=speed*speed,limit=base.flight.angularSpeedWindowSeconds;rows.push({dt,square});seconds+=dt;integral+=dt*square;
  while(rows.length&&seconds-rows[0].dt>=limit-EPS){const r=rows.shift();seconds-=r.dt;integral-=r.dt*r.square;}
  if(rows.length&&seconds>limit){const removed=seconds-limit;rows[0].dt-=removed;seconds-=removed;integral-=removed*rows[0].square;}
  return {seconds,rms:Math.sqrt(Math.max(0,integral/seconds))};
 }};
}
function supportWindow(){
 const rows=[];let seconds=0;
 return {clear(){rows.length=0;seconds=0;},append(from,to,fromHeight,toHeight,dt){
  const limit=base.flight.supportWindowSeconds;rows.push({from,to,fromHeight,toHeight,dt});seconds+=dt;
  while(rows.length&&seconds-rows[0].dt>=limit-EPS)seconds-=rows.shift().dt;
  if(rows.length&&seconds>limit){const r=rows[0],removed=seconds-limit;r.from+=(r.to-r.from)*removed/r.dt;r.fromHeight+=(r.toHeight-r.fromHeight)*removed/r.dt;r.dt-=removed;seconds-=removed;}
  return {seconds,fraction:1+(rows.at(-1).to-rows[0].from)/seconds/base.gravityCmPerSecondSquared,verticalSpeed:(rows.at(-1).toHeight-rows[0].fromHeight)/seconds};
 }};
}

/** Five scored seconds; absolute neural/body time belongs to the reset clock.
 * Contact/failed flight gates reset current airtime. Success is assessed only
 * at the full horizon and requires the current continuous bout >= existing1s
 * criterion. This does not claim five entirely qualified seconds of flight. */
export function createMaintainedFlightScore(duration,initial,sceneValue){
 const scene=validateMaintainedScene(sceneValue);
 const readObservation=value=>{assertMaintainedObservationScene(value,scene);return observation(value);};
 if(duration!==MAINTAINED_FLIGHT_CRITERIA.durationSeconds)throw new RangeError('Maintained flight requires a full 5 second horizon');
 let previous=readObservation(initial),finished=null;
 if(previous.environmentContacts!==0)throw new RangeError('Maintained flight requires an explicitly contact-free airborne reset');
 const angular=angularWindow(),support=supportWindow();
 const state={stage:MAINTAINED_FLIGHT_STAGE,duration,elapsed:0,success:false,reason:null,return:0,
  startedAirborne:true,hasTakenOff:false,takeoffTime:null,flightSeconds:0,bestFlightSeconds:0,totalQualifiedFlightSeconds:0,
  landingSeconds:0,landingTime:null,phase:'unqualified_airborne',
  diagnostics:{confirmedTakeoffs:0,resetSuppliesTakeoffCredit:false,maximumAngularSpeed:previous.angularSpeed,overturnedSeconds:0,
   supportWindowSeconds:0,inferredSupportFraction:null,meanVerticalSpeed:null,angularSpeedWindowSeconds:0,angularSpeedRms:null,
   contactSeconds:0,contactSamples:0,bestEarnedProgress:0,failurePenalty:0,invalidObservation:null}};
 function finish(reason,success,terminated){state.reason=reason;state.success=success;if(terminated&&!success)state.phase='failed';
  Object.freeze(state.diagnostics);Object.freeze(state);finished=Object.freeze({reason,success,terminated});return finished;}
 function fail(reason){
  state.diagnostics.failurePenalty=reason==='invalid_observation'||reason==='unexpected_external_force'?10:3;
  state.return=state.diagnostics.failurePenalty===10?-10:clip(state.diagnostics.bestEarnedProgress-3,-8,3);
  return finish(reason,false,true);
 }
 const physicalFailure=o=>o.externalForce?'unexpected_external_force':
  o.radius>(scene?scene.radiusCm:base.maximumRadiusCm)||o.height<base.minimumHeightCm||o.height>o.ceiling+base.ceilingAllowanceCm?'outside_habitat':
  o.angularSpeed>base.maximumAngularSpeed?'excessive_rotation':state.diagnostics.overturnedSeconds+EPS>=base.overturnedSeconds?'overturned':null;
 const initialFailure=physicalFailure(previous);if(initialFailure)fail(initialFailure);
 return {state,step(value,dt){
  if(finished)return finished;
  if(!Number.isFinite(dt)||dt<=0||dt>base.maximumStepSeconds||state.elapsed+dt>duration+1e-7)throw new RangeError('Invalid maintained-flight timestep');
  let o;try{o=readObservation(value);}catch(error){state.diagnostics.invalidObservation=error.message;return fail('invalid_observation');}
  const h=Math.min(dt,duration-state.elapsed);state.elapsed+=h;if(duration-state.elapsed<EPS)state.elapsed=duration;
  const d=state.diagnostics;d.maximumAngularSpeed=Math.max(d.maximumAngularSpeed,o.angularSpeed);
  d.overturnedSeconds=o.up<0?d.overturnedSeconds+h:0;
  const failure=physicalFailure(o);if(failure)return fail(failure);
  const inAir=o.environmentContacts===0,wasAir=previous.environmentContacts===0;
  if(inAir){const w=angular.append(o.angularSpeed,h);d.angularSpeedWindowSeconds=w.seconds;d.angularSpeedRms=w.rms;}
  else{angular.clear();d.angularSpeedWindowSeconds=0;d.angularSpeedRms=null;d.contactSeconds+=h;d.contactSamples++;}
  // Empty at release. No warm-up restraint or contact impulse enters the
  // airborne COM support window, even when a later contact is followed by air.
  if(inAir&&wasAir){const w=support.append(previous.verticalSpeed,o.verticalSpeed,previous.height,o.height,h);d.supportWindowSeconds=w.seconds;d.inferredSupportFraction=w.fraction;d.meanVerticalSpeed=w.verticalSpeed;}
  else{support.clear();d.supportWindowSeconds=0;d.inferredSupportFraction=null;d.meanVerticalSpeed=null;}
  const qualified=inAir&&o.wingPower>base.flight.minimumWingPower&&o.up>=base.flight.minimumUp&&
   d.angularSpeedWindowSeconds+EPS>=base.flight.angularSpeedWindowSeconds&&d.angularSpeedRms<=base.flight.maximumAngularSpeed&&
   d.meanVerticalSpeed+EPS>=base.flight.minimumVerticalSpeed&&d.supportWindowSeconds+EPS>=base.flight.supportWindowSeconds&&
   d.inferredSupportFraction>=base.flight.minimumSupportFraction;
  state.flightSeconds=qualified?state.flightSeconds+h:0;
  if(qualified)state.totalQualifiedFlightSeconds+=h;
  state.bestFlightSeconds=Math.max(state.bestFlightSeconds,state.flightSeconds);
  state.phase=qualified?'flight':inAir?'unqualified_airborne':'contact';
  const earned=7*clip(state.flightSeconds/duration);d.bestEarnedProgress=Math.max(d.bestEarnedProgress,earned);state.return=earned;
  previous=o;
  if(state.elapsed===duration){
   const success=qualified&&state.flightSeconds+EPS>=base.flight.minimumContinuousSeconds;
   if(success){state.return=clip(state.return+3,-10,10);return finish('stage_success',true,true);}
   return finish('time_limit',false,false);
  }
  return {terminated:false,success:false,reason:null};
 }};
}
