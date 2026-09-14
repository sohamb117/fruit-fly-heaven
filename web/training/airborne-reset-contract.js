// Pure proposed setup contract/clock assertions. No state is set by this file.
import {MAINTAINED_FLIGHT_STAGE} from './maintained-flight-objective.js';
const LEGACY=['takeoff','flight','landing'],DT=.002,TOL=1e-8;
const record=v=>v!==null&&typeof v==='object'&&!Array.isArray(v);
const exactKeys=(v,keys)=>record(v)&&Reflect.ownKeys(v).length===keys.length&&keys.every(k=>Object.hasOwn(v,k));
function requireThat(ok,message){if(!ok)throw new TypeError(message);}

/** The absent old-stage selector returns null, so a later integration can
 * execute its original grounded branch with identical arithmetic/state. */
export function selectFlightInitialCondition(stage,value,context={}){
 if(LEGACY.includes(stage)){requireThat(value===undefined,'Airborne initialization is exclusive to maintained_flight');return null;}
 requireThat(stage===MAINTAINED_FLIGHT_STAGE,'Unknown initial-condition stage');
 requireThat(exactKeys(value,['schemaVersion','profile','warmupSeconds','bodyVariant','bodyVariantHash','rootQpos']),'Explicit airborne reset declaration required');
 requireThat(value.schemaVersion===1&&value.profile==='airborne-live-warmup-v1','Unsupported airborne reset profile');
 requireThat(Number.isFinite(value.warmupSeconds)&&value.warmupSeconds>=.1&&value.warmupSeconds<=2&&
  Math.abs(value.warmupSeconds/DT-Math.round(value.warmupSeconds/DT))<1e-9,'Warm-up must be explicit, within0.1–2s and a whole2ms block');
 requireThat(value.bodyVariant==='full-native','Only the currently supported full-native body variant is allowed');
 requireThat(typeof value.bodyVariantHash==='string'&&/^[a-f0-9]{64}$/.test(value.bodyVariantHash),'Pin exact body metadata hash');
 requireThat(context.bodyMetadataSha256===value.bodyVariantHash,'Body variant differs from verified metadata asset');
 requireThat(Array.isArray(value.rootQpos)&&value.rootQpos.length===7&&value.rootQpos.every(Number.isFinite),'Explicit finite seven-component root pose required');
 requireThat(Math.abs(Math.hypot(...value.rootQpos.slice(3))-1)<1e-10,'Root quaternion must already be normalized');
 return Object.freeze({...value,rootQpos:Object.freeze([...value.rootQpos])});
}

/** Snapshot fields refer to the SAME continuously advanced BANC/body pair.
 * rootWriteCount is a cumulative count supplied by an independent native write
 * audit. A false caller declaration is not made trustworthy by this helper. */
export function createMaintainedFlightClock(initialization,context){
 const profile=selectFlightInitialCondition(MAINTAINED_FLIGHT_STAGE,initialization,context);
 const warmupBlocks=Math.round(profile.warmupSeconds/DT),warmupMs=warmupBlocks*2;
 let origin=null,steps=0;
 function check(s){
  requireThat(record(s),'Clock snapshot required');
  for(const k of ['nativeTimeSeconds','neuralTimeMs','bodyEventElapsedMs','bodyEventObservedMs','lastPacketTimeMs','remainderSeconds'])
   requireThat(Number.isFinite(s[k])&&s[k]>=0,'Invalid '+k);
  requireThat(Number.isInteger(s.neuralTimeMs/2),'Neural time must end on a2ms body block');
  requireThat(Math.abs(s.nativeTimeSeconds-s.neuralTimeMs/1000)<=TOL,'Neural/native clock mismatch');
  requireThat(s.bodyEventElapsedMs===s.neuralTimeMs&&s.bodyEventObservedMs===s.neuralTimeMs&&s.lastPacketTimeMs===s.neuralTimeMs,'Unconsumed or unobserved motor interval');
  requireThat(s.remainderSeconds===0&&s.pendingEvents===false,'Incomplete native/event block');
  requireThat(s.rootRestraintActive===false&&s.externalForceApplied===false,'Scored release requires an unrestrained root with no applied external force');
  requireThat(Number.isSafeInteger(s.rootWriteCount)&&s.rootWriteCount>=0,'Native root-write audit count required');
 }
 return Object.freeze({profile,warmupBlocks,bodyBlockSeconds:DT,
  release(snapshot){
   requireThat(origin===null,'Scored release already established');check(snapshot);
   requireThat(snapshot.neuralTimeMs===warmupMs,'Release differs from predeclared warm-up duration');
   origin=Object.freeze({nativeTimeSeconds:snapshot.nativeTimeSeconds,neuralTimeMs:snapshot.neuralTimeMs,rootWriteCount:snapshot.rootWriteCount});
   return Object.freeze({releaseNativeTimeSeconds:origin.nativeTimeSeconds,releaseNeuralTimeMs:origin.neuralTimeMs,scoredElapsedSeconds:0,scoredSteps:0});
  },
  advance(snapshot){
   requireThat(origin!==null,'Scored release must precede scored steps');check(snapshot);
   requireThat(steps<2500,'Maintained-flight horizon already complete');
   requireThat(snapshot.neuralTimeMs===origin.neuralTimeMs+(steps+1)*2,'Missing or repeated scored block');
   requireThat(Math.abs(snapshot.nativeTimeSeconds-origin.nativeTimeSeconds-(steps+1)*DT)<=TOL,'Scored native clock drift');
   requireThat(snapshot.rootWriteCount===origin.rootWriteCount,'Direct root writes after scored release are forbidden');
   steps++;return Object.freeze({scoredSteps:steps,scoredElapsedSeconds:steps*DT,nativeTimeSeconds:snapshot.nativeTimeSeconds,neuralTimeMs:snapshot.neuralTimeMs});
  },
 });
}
