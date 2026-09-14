const clamp=(x,a=0,b=1)=>Math.max(a,Math.min(b,x));
const TAU=2*Math.PI;
const DEFAULT_INTERPRETER=Object.freeze({powerGain:1,deploymentTauScale:1,frequencyScale:1,steeringBiasGain:1,steeringAmplitudeGain:1});

// Thin BANC -> FlyBody flight interpreter. The calibrated FlyBody wingbeat
// tables remain the mechanical primitive; only five positive transfer scales
// are learnable. No root pose, task state or reward enters this module.
export class FlyBodyWings{
 constructor(metadata){
  this.config={...metadata.wing_actuation};this.baseFrequencyHz=this.config.frequency_hz;
  if(!this.config?.steering)throw new Error('Missing calibrated wing actuator model');
  this.names=['left','right'].flatMap(side=>['yaw','roll','pitch'].map(axis=>`wing_${axis}_${side}`));
  this.joints=this.names.map(name=>metadata.joints.find(j=>j.name===name));
  this.actuators=this.names.map(name=>metadata.actuators.find(a=>a.name===name));
  if(this.joints.some(x=>!x)||this.actuators.some(x=>!x))throw new Error('Missing FlyBody wing joints or actuators');
  this.tables=this.config.targets.map(rows=>Float64Array.from(rows.flat()));
  this.centers=this.config.targets.map(rows=>Array.from({length:6},(_,k)=>rows.reduce((sum,r)=>sum+r[k],0)/rows.length));
  this.count=this.config.targets[0].length;
  this.deployment=new Float64Array(2);this.power=new Float64Array(2);this.phase=0;this.target=new Float64Array(6);
  this.residuals=[new Float64Array(6),new Float64Array(6)];
  this.interpreter={...DEFAULT_INTERPRETER};this.frequencyHz=this.config.frequency_hz;
 }
 setInterpreterParameters(values={}){
  const next={...this.interpreter,...values};
  for(const name of Object.keys(DEFAULT_INTERPRETER))if(!Number.isFinite(next[name])||next[name]<=0)throw new Error('Invalid flight interpreter parameter: '+name);
  for(const name of Object.keys(values))if(!(name in DEFAULT_INTERPRETER))throw new Error('Unknown flight interpreter parameter: '+name);
  this.interpreter=next;this.frequencyHz=this.baseFrequencyHz*next.frequencyScale;this.config.frequency_hz=this.frequencyHz;
 }
 controlState(){return {power:Array.from(this.power),deployment:Array.from(this.deployment),frequencyHz:this.frequencyHz,parameters:{...this.interpreter}};}
 step(q,ctrl,left,right,steering,dt){
  if(!Number.isFinite(dt)||dt<=0)throw new Error('Invalid wing timestep');
  const c=this.config,p=this.interpreter,requested=[clamp(left*p.powerGain),clamp(right*p.powerGain)];
  this.frequencyHz=this.baseFrequencyHz*p.frequencyScale;c.frequency_hz=this.frequencyHz;
  for(let side=0;side<2;side++){
   const muscles=steering?.[side===0?'left':'right']||{},residual=this.residuals[side];residual.fill(0);
   for(const [name,force] of Object.entries(muscles)){
    const response=c.steering[name];if(!response)continue;
    for(let axis=0;axis<3;axis++){
     residual[axis]+=force*response[axis]*p.steeringBiasGain;
     residual[axis+3]+=force*response[axis+3]*p.steeringAmplitudeGain;
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
