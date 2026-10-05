// Simulation engineering task, not measured receptor calibration. A seeded
// release velocity is applied once by the audited airborne initializer; this
// module only specifies initial conditions and observes subsequent recovery.
import {createMaintainedFlightScore,maintainedFlightCriteria} from './maintained-flight-objective.js';
const freeze=x=>{if(x&&typeof x==='object'){Object.values(x).forEach(freeze);Object.freeze(x);}return x;};
const record=x=>x!==null&&typeof x==='object'&&!Array.isArray(x);
const finite=x=>typeof x==='number'&&Number.isFinite(x);
const check=(ok,message)=>{if(!ok)throw new Error('Flight recovery: '+message);};
const clip=(x,lo=0,hi=1)=>Math.max(lo,Math.min(hi,x));
export const RECOVERY_STAGE='recovery';
export const DEFAULT_RECOVERY_DISTURBANCE=freeze({schema:1,profile:'release-velocity-disturbance-v1',
 angularSpeedRangeRadS:[24,36],linearSpeedRangeCmS:[2,6]});
export const RECOVERY_CRITERIA=freeze({version:1,stage:RECOVERY_STAGE,durationSeconds:5,
 maximumRecoverySeconds:2,settlementSeconds:.25,minimumFinalRecoveredSeconds:1,
 minimumUp:Math.cos(Math.PI/6),maximumAngularSpeedRms:10,requireContactFree:true,
 interpretation:'Recover controlled powered flight after one seeded initial velocity disturbance. No takeoff or landing credit; no sustaining external forces. Thresholds are engineering assumptions.'});
export function validateRecoveryDisturbance(value){
 const expected=Object.keys(DEFAULT_RECOVERY_DISTURBANCE);
 check(record(value)&&Object.keys(value).length===expected.length&&expected.every(k=>Object.hasOwn(value,k)), 'explicit disturbance declaration required');
 check(value.schema===1&&value.profile===DEFAULT_RECOVERY_DISTURBANCE.profile,'unsupported disturbance profile');
 for(const [key,limit]of [['angularSpeedRangeRadS',80],['linearSpeedRangeCmS',30]]){
  const range=value[key];check(Array.isArray(range)&&range.length===2&&range.every(finite)&&range[0]>=0&&range[0]<=range[1]&&range[1]<=limit,'invalid '+key);
 }
 check(value.angularSpeedRangeRadS[0]>RECOVERY_CRITERIA.maximumAngularSpeedRms,'disturbance must exceed the recovered angular-speed threshold');
 return freeze(structuredClone(value));
}
export function createRecoveryDisturbance(value,seed){
 const config=validateRecoveryDisturbance(value);check(Number.isInteger(seed)&&seed>=0&&seed<=0xffffffff,'invalid seed');
 // Independent deterministic stream; varying a held-out seed changes both
 // direction and magnitude without changing the declared task distribution.
 let state=(seed^0x9e3779b9)>>>0;
 const random=()=>{state+=0x6D2B79F5;let x=state;x=Math.imul(x^(x>>>15),x|1);x^=x+Math.imul(x^(x>>>7),x|61);return ((x^(x>>>14))>>>0)/4294967296;};
 const vector=range=>{const z=2*random()-1,phi=2*Math.PI*random(),r=Math.sqrt(1-z*z),length=range[0]+(range[1]-range[0])*random();return [r*Math.cos(phi)*length,r*Math.sin(phi)*length,z*length];};
 const angularVelocityRootRadS=vector(config.angularSpeedRangeRadS),linearVelocityWorldCmS=vector(config.linearSpeedRangeCmS);
 return freeze({schema:1,profile:config.profile,seed,config,angularVelocityRootRadS,linearVelocityWorldCmS,
  releaseVelocity:[...linearVelocityWorldCmS,...angularVelocityRootRadS],
  application:'One native free-joint velocity assignment inside the audited release boundary; absolute neural/event/native clocks retained.'});
}
export const recoveryFlightCriteria=scene=>freeze({...RECOVERY_CRITERIA,maintainedFlight:maintainedFlightCriteria(scene)});

/** Uses fresh maintained-flight support/angular windows after release. Its
 * stricter settling gates must be reached within2s and held for the final1s;
 * contacts cannot count as recovery. Rejected observations and catastrophic
 * failures retain the maintained objective's penalties. */
export function createRecoveryScore(duration,initial,scene){
 check(duration===RECOVERY_CRITERIA.durationSeconds,'full five-second horizon required');
 const flight=createMaintainedFlightScore(duration,initial,scene),c=RECOVERY_CRITERIA;
 let recoveredSeconds=0,bestRecoveredSeconds=0,recoveredAt=null,finished=null;
 const state={...flight.state,stage:RECOVERY_STAGE,recoveredSeconds,bestRecoveredSeconds,recoveredAt,recoveryDeadlineSeconds:c.maximumRecoverySeconds,
  diagnostics:{...flight.state.diagnostics,recoveryCriteria:c}};
 const sync=()=>{Object.assign(state,flight.state,{stage:RECOVERY_STAGE,recoveredSeconds,bestRecoveredSeconds,recoveredAt,recoveryDeadlineSeconds:c.maximumRecoverySeconds});state.diagnostics={...flight.state.diagnostics,recoveryCriteria:c};};
 return {state,step(value,dt){
  if(finished)return finished;
  const result=flight.step(value,dt);sync();
  if(flight.state.phase==='failed'){finished=result;return result;}
  const settled=flight.state.phase==='flight'&&value.up>=c.minimumUp&&
   flight.state.diagnostics.angularSpeedRms!==null&&flight.state.diagnostics.angularSpeedRms<=c.maximumAngularSpeedRms;
  recoveredSeconds=settled?recoveredSeconds+dt:0;bestRecoveredSeconds=Math.max(bestRecoveredSeconds,recoveredSeconds);
  if(recoveredAt===null&&recoveredSeconds+1e-9>=c.settlementSeconds)recoveredAt=flight.state.elapsed;
  sync();state.phase=settled?'recovered_flight':flight.state.phase;
  // Reward earned controlled airtime plus current settled recovery. An early
  // posture correction without subsequent powered flight earns no success.
  state.return=6*clip(flight.state.flightSeconds/duration)+3*clip(recoveredSeconds/c.minimumFinalRecoveredSeconds);
  if(flight.state.elapsed===duration){
   const success=result.success&&recoveredAt!==null&&recoveredAt<=c.maximumRecoverySeconds+1e-9&&
    recoveredSeconds+1e-9>=c.minimumFinalRecoveredSeconds&&flight.state.diagnostics.contactSamples===0;
   state.success=success;state.reason=success?'recovery_success':'time_limit';if(success)state.return=Math.min(10,state.return+1);
   finished=Object.freeze({terminated:success,success,reason:state.reason});return finished;
  }
  return {terminated:false,success:false,reason:null};
 }};
}
