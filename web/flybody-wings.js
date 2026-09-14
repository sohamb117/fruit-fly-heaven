import {STEERING_MUSCLE_TYPES,validateFlightInterpreter} from './training/flight-parameters.js';
const clamp=(x,a=0,b=1)=>Math.max(a,Math.min(b,x));
const TAU=2*Math.PI;

function powerTransfer(config){
 if(!Object.hasOwn(config,'power_transfer'))return null;
 const value=config.power_transfer,keys=['schemaVersion','profile','activationGain'];
 if(value===null||typeof value!=='object'||
  (Object.getPrototypeOf(value)!==Object.prototype&&Object.getPrototypeOf(value)!==null))
  throw new TypeError('Wing power transfer must be a plain object');
 if(Reflect.ownKeys(value).length!==keys.length||Reflect.ownKeys(value).some(key=>!keys.includes(key))||
  keys.some(key=>!Object.hasOwn(value,key)))
  throw new TypeError('Wing power transfer requires exactly schemaVersion, profile and activationGain');
 if(value.schemaVersion!==1||value.profile!=='activation-amplitude-v1')
  throw new TypeError('Unsupported wing power transfer profile');
 if(typeof value.activationGain!=='number'||!Number.isFinite(value.activationGain)||value.activationGain<=0)
  throw new TypeError('Wing power transfer activationGain must be finite and positive');
 // Fixed numerical normalization of the existing native muscle force, not a
 // measured biological recruitment gain or an additional learned coordinate.
 return Object.freeze({schemaVersion:1,profile:'activation-amplitude-v1',activationGain:value.activationGain});
}

function forceReference(config){
 if(!Object.hasOwn(config,'steering_force_reference'))return null;
 const values=config.steering_force_reference;
 if(values===null||typeof values!=='object'||
  (Object.getPrototypeOf(values)!==Object.prototype&&Object.getPrototypeOf(values)!==null))
  throw new TypeError('Steering force reference must be a twelve-muscle object');
 for(const name of Reflect.ownKeys(values))if(!STEERING_MUSCLE_TYPES.includes(name))
  throw new TypeError('Unknown steering force reference: '+String(name));
 for(const name of STEERING_MUSCLE_TYPES)if(!Object.hasOwn(values,name)||
  typeof values[name]!=='number'||!Number.isFinite(values[name])||values[name]<0||values[name]>1)
  throw new TypeError('Steering force reference must be finite within [0,1]: '+name);
 // A local operating-point prior in normalized WASM force units, not a
 // measured physiological baseline or another set of learned parameters.
 return Object.freeze(Object.fromEntries(STEERING_MUSCLE_TYPES.map(name=>[name,values[name]])));
}

// Thin BANC -> FlyBody flight interpreter. The calibrated FlyBody wingbeat
// tables remain fixed. Three common scales and two coefficients per steering
// muscle type are learnable: 27 total. Coefficients are tied across homologues,
// while left/right forces remain separate. No root/task/reward input is added.
export class FlyBodyWings{
 constructor(metadata){
  this.config={...metadata.wing_actuation};this.baseFrequencyHz=this.config.frequency_hz;
  if(!this.config?.steering)throw new Error('Missing calibrated wing actuator model');
  if(Object.keys(this.config.steering).length!==STEERING_MUSCLE_TYPES.length||STEERING_MUSCLE_TYPES.some(name=>
   !Object.hasOwn(this.config.steering,name)||!Array.isArray(this.config.steering[name])||this.config.steering[name].length!==6||!this.config.steering[name].every(Number.isFinite)))
   throw new Error('Calibrated wing steering basis does not match the twelve-muscle contract');
  this.config.steering=Object.freeze(Object.fromEntries(STEERING_MUSCLE_TYPES.map(name=>[name,Object.freeze([...this.config.steering[name]])])));
  this.powerTransfer=powerTransfer(this.config);
  if(this.powerTransfer)this.config.power_transfer=this.powerTransfer;
  this.steeringForceReference=forceReference(this.config);
  if(this.steeringForceReference)this.config.steering_force_reference=this.steeringForceReference;
  this.names=['left','right'].flatMap(side=>['yaw','roll','pitch'].map(axis=>`wing_${axis}_${side}`));
  this.joints=this.names.map(name=>metadata.joints.find(j=>j.name===name));
  this.actuators=this.names.map(name=>metadata.actuators.find(a=>a.name===name));
  if(this.joints.some(x=>!x)||this.actuators.some(x=>!x))throw new Error('Missing FlyBody wing joints or actuators');
  this.tables=this.config.targets.map(rows=>Float64Array.from(rows.flat()));
  this.centers=this.config.targets.map(rows=>Array.from({length:6},(_,k)=>rows.reduce((sum,r)=>sum+r[k],0)/rows.length));
  this.count=this.config.targets[0].length;
  this.deployment=new Float64Array(2);this.power=new Float64Array(2);this.phase=0;this.target=new Float64Array(6);
  this.residuals=[new Float64Array(6),new Float64Array(6)];
  this.interpreter=validateFlightInterpreter();this.frequencyHz=this.config.frequency_hz;
 }
 setInterpreterParameters(values={}){
  const next=validateFlightInterpreter(values,this.interpreter);
  if(this.powerTransfer&&next.powerGain>1)
   throw new RangeError('activation-amplitude-v1 requires powerGain <= 1');
  this.interpreter=next;this.frequencyHz=this.baseFrequencyHz*next.frequencyScale;this.config.frequency_hz=this.frequencyHz;
 }
 controlState(){return {power:Array.from(this.power),deployment:Array.from(this.deployment),frequencyHz:this.frequencyHz,parameters:{...this.interpreter,
  steering:Object.fromEntries(STEERING_MUSCLE_TYPES.map(name=>[name,{...this.interpreter.steering[name]}]))}};}
 // The learned bridge supplies normalized power plus bounded corrections in
 // native actuator-control units. Its only mechanical phase is the existing
 // wing oscillator; it has no access to root state or a target trajectory.
 stepDecoded(q,ctrl,decoded,dt){
  if(!decoded||decoded.power?.length!==2||decoded.steering?.length!==2||
   !Array.from(decoded.power).every(v=>Number.isFinite(v)&&v>=0&&v<=1)||
   !decoded.steering.every(side=>side?.length===3&&Array.from(side).every(v=>Number.isFinite(v)&&Math.abs(v)<=.25)))
   throw new Error('Invalid decoded wing actuator controls');
  this.step(q,ctrl,decoded.power[0],decoded.power[1],{},dt,true);
  const residual=[new Float64Array(3),new Float64Array(3)];
  for(let side=0;side<2;side++)for(let axis=0;axis<3;axis++){
   const a=this.actuators[side*3+axis];
   // Steering cannot generate powered flight without ipsilateral power MNs.
   residual[side][axis]=decoded.steering[side][axis]*this.power[side];
   ctrl[a.id]=clamp(ctrl[a.id]+residual[side][axis],...a.range);
  }
  this.decodedControls={power:Array.from(decoded.power),steering:decoded.steering.map(side=>Array.from(side)),
   appliedResidual:residual.map(side=>Array.from(side)),unit:'native actuator control'};
 }
 step(q,ctrl,left,right,steering,dt,normalizedPower=false){
  if(!Number.isFinite(dt)||dt<=0)throw new Error('Invalid wing timestep');
  const c=this.config,p=this.interpreter;
  const requested=normalizedPower?[clamp(left),clamp(right)]:this.powerTransfer?
   [p.powerGain*clamp(left*this.powerTransfer.activationGain),p.powerGain*clamp(right*this.powerTransfer.activationGain)]:
   [clamp(left*p.powerGain),clamp(right*p.powerGain)];
  this.frequencyHz=this.baseFrequencyHz*p.frequencyScale;c.frequency_hz=this.frequencyHz;
  for(let side=0;side<2;side++){
   const muscles=steering?.[side===0?'left':'right']||{},residual=this.residuals[side];residual.fill(0);
   // Decoded controls own steering; do not add the legacy muscle basis or
   // subtract its optional force reference from an absent muscle-force input.
   if(normalizedPower)continue;
   if(this.steeringForceReference){
    // Center before applying either gain, so the declared operating point
    // always maps exactly to the calibrated zero-residual wing table.
    for(const name of STEERING_MUSCLE_TYPES){
     const force=muscles[name]??0,delta=force-this.steeringForceReference[name];
     const response=c.steering[name],gains=p.steering[name];
     for(let axis=0;axis<3;axis++){
      residual[axis]+=delta*response[axis]*gains.biasGain;
      residual[axis+3]+=delta*response[axis+3]*gains.amplitudeGain;
     }
    }
   }else{
   // Keep the legacy iteration order and arithmetic unchanged when absent.
   for(const [name,force] of Object.entries(muscles)){
    if(!Object.hasOwn(c.steering,name))continue;const response=c.steering[name],gains=p.steering[name];
    for(let axis=0;axis<3;axis++){
     residual[axis]+=force*response[axis]*gains.biasGain;
     residual[axis+3]+=force*response[axis+3]*gains.amplitudeGain;
    }
   }
   }
  }
  const index=this.phase/TAU*this.count,i=Math.floor(index),t=index-i,next=(i+1)%this.count;
  const tau=Math.max(1e-6,c.deployment_tau_s*p.deploymentTauScale),alpha=-Math.expm1(-dt/tau);
  for(let side=0;side<2;side++){
   // Deployment is a mechanical release gate only. Steering muscles no longer
   // decide whether a wing exists/open; their effect is confined to the fixed
   // FlyBody steering basis below. Left/right async power stays independent.
   this.deployment[side]+=((requested[side]>.01?1:0)-this.deployment[side])*alpha;
   const ramp=clamp((this.deployment[side]-c.deployment_before_beating)/(1-c.deployment_before_beating));
   this.power[side]=Math.min(requested[side],ramp);
   const powers=c.powers;let lo=0;
   while(lo<powers.length-2&&powers[lo+1]<this.power[side])lo++;
   const u=clamp((this.power[side]-powers[lo])/(powers[lo+1]-powers[lo]));
   const a=this.tables[lo],b=this.tables[lo+1],residuals=this.residuals[side];
   for(let axis=0;axis<3;axis++){
    const k=side*3+axis,j=this.joints[k],act=this.actuators[k];
    const low=a[i*6+k]*(1-t)+a[next*6+k]*t,high=b[i*6+k]*(1-t)+b[next*6+k]*t;
    const cycle=low+(high-low)*u,center=this.centers[lo][k]*(1-u)+this.centers[lo+1][k]*u;
    const residual=clamp(residuals[axis]+(cycle-center)*residuals[axis+3],-.15,.15);
    this.target[k]=clamp(j.neutral+this.deployment[side]*(cycle-j.neutral)+residual*this.power[side],...j.range);
    ctrl[act.id]=clamp(this.target[k]-q[j.qpos],...act.range);
   }
  }
  this.phase=(this.phase+TAU*this.frequencyHz*dt)%TAU;
 }
}
