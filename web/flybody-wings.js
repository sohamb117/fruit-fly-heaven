const clamp=(x,a=0,b=1)=>Math.max(a,Math.min(b,x));
const TAU=2*Math.PI;

// A distilled wing hinge, not a flight controller. Its only inputs are muscle
// forces and elapsed time. Forces on the body still come from native fluid and
// contact mechanics; this module never reads or changes the root pose.
export class FlyBodyWings{
 constructor(metadata){
  this.config=metadata.wing_actuation;
  if(!this.config?.steering)throw new Error('Missing calibrated wing actuator model');
  this.names=['left','right'].flatMap(side=>['yaw','roll','pitch'].map(axis=>`wing_${axis}_${side}`));
  this.joints=this.names.map(name=>metadata.joints.find(j=>j.name===name));
  this.actuators=this.names.map(name=>metadata.actuators.find(a=>a.name===name));
  this.tables=this.config.targets.map(rows=>Float64Array.from(rows.flat()));
  this.centers=this.config.targets.map(rows=>Array.from({length:6},(_,k)=>rows.reduce((sum,r)=>sum+r[k],0)/rows.length));
  this.count=this.config.targets[0].length;
  this.deployment=new Float64Array(2);this.power=new Float64Array(2);this.phase=0;this.target=new Float64Array(6);
  this.residuals=[new Float64Array(6),new Float64Array(6)];this.opening=new Float64Array(2);
 }
 step(q,ctrl,left,right,steering,dt){
  const power=clamp((left+right)/2),c=this.config;
  if(steering!==this.lastSteering){
   this.lastSteering=steering;
   for(let side=0;side<2;side++){
    const muscles=steering[side===0?'left':'right'],residual=this.residuals[side];residual.fill(0);
    this.opening[side]=1-clamp((muscles.iii1_muscle||0)-(muscles.b1_muscle||0));
    for(const [name,force] of Object.entries(muscles)){
     const response=c.steering[name];
     if(response)for(let k=0;k<6;k++)residual[k]+=force*response[k];
    }
   }
  }
  const index=this.phase/TAU*this.count,i=Math.floor(index),t=index-i,next=(i+1)%this.count;
  for(let side=0;side<2;side++){
   // In Drosophila the reported retraction/termination muscle is III1, not I1
   // (Heide & Gotz 1996). I1 retains its distinct stroke-reducing steering map.
   // Numerical competition against basalar tension remains a hinge prior.
   const opening=this.opening[side],residuals=this.residuals[side];
   this.deployment[side]+=((power>.01?opening:0)-this.deployment[side])*(-Math.expm1(-dt/c.deployment_tau_s));
   this.power[side]=Math.min(power*opening*opening,clamp((this.deployment[side]-c.deployment_before_beating)/(1-c.deployment_before_beating)));
   const powers=c.powers;let lo=0;
   while(lo<powers.length-2&&powers[lo+1]<this.power[side])lo++;
   const u=clamp((this.power[side]-powers[lo])/(powers[lo+1]-powers[lo]));
   const a=this.tables[lo],b=this.tables[lo+1];
   for(let axis=0;axis<3;axis++){
   const k=side*3+axis;
   const j=this.joints[k],act=this.actuators[k];
   const low=a[i*6+k]*(1-t)+a[next*6+k]*t,high=b[i*6+k]*(1-t)+b[next*6+k]*t;
   const cycle=low+(high-low)*u,center=this.centers[lo][k]*(1-u)+this.centers[lo+1][k]*u;
   const residual=clamp(residuals[axis]+(cycle-center)*residuals[axis+3],-.15,.15);
   this.target[k]=clamp(j.neutral+this.deployment[side]*(cycle-j.neutral)+residual*this.power[side],...j.range);
   ctrl[act.id]=clamp(this.target[k]-q[j.qpos],...act.range);
   }
  }
  this.phase=(this.phase+TAU*c.frequency_hz*dt)%TAU;
 }
}
