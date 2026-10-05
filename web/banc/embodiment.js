const clamp=(x,a=0,b=1)=>Math.max(a,Math.min(b,x));
export const TASK_STAGES=['localization','approach','landing','probing','feeding','takeoff','flight'];
// Energy is a reserve, not a linear multiplier on instantaneous strength.
// Force becomes fuel-limited near depletion. The 0.1 reserve threshold is an
// explicit modeling prior, not a measured BANC or FlyBody parameter.
export const MUSCLE_FUEL_RESERVE=.1;
export const muscleFuel=energy=>clamp(energy/MUSCLE_FUEL_RESERVE);

export class InternalState {
  constructor(energy=.35){this.energy=clamp(energy);this.crop=0;this.insulin=0;this.akh=1-this.energy;this.ingested=0;this.hunger=1-this.energy;}
  step(dt,intake,effort){
    if(!(dt>0&&dt<=.05)||!Number.isFinite(intake)||!Number.isFinite(effort))throw new Error('Invalid metabolism step');
    const accepted=Math.max(0,Math.min(intake,1-this.crop));this.crop+=accepted;this.ingested+=accepted;
    const absorbed=this.crop*(1-Math.exp(-dt/25));this.crop-=absorbed;
    this.energy=clamp(this.energy+absorbed-dt*(.0003+.0015*clamp(effort)));
    this.hunger=clamp(1-this.energy-.5*this.crop);
    this.insulin+=(clamp(this.energy+.5*this.crop)-this.insulin)*(1-Math.exp(-dt/15));
    this.akh+=(clamp(1-this.energy)-this.akh)*(1-Math.exp(-dt/20));
  }
}

/** Records evidence; never selects a target or supplies actuator commands. */
export class TaskMonitor {
  // Conservative event-observation criteria, not measured biological limits.
  // They affect reporting only and never feed the motor/physics computation.
  static observationCriteria=Object.freeze({flightWindowSeconds:.25,velocityAverageSeconds:.04,maximumSampleGapSeconds:.02,
    minimumWingPower:.1,maximumTiltRadians:75*Math.PI/180,maximumMeanAngularSpeed:20,
    gravityCmPerSecondSquared:981,minimumSupportFraction:.5,maximumDescentCmPerSecond:10,
    landingContactSeconds:.1,maximumLandingTiltRadians:60*Math.PI/180,
    maximumLandingSpeedCmPerSecond:1,maximumLandingAngularSpeed:10,maximumLandingApproachSpeed:10});
  constructor(){
    this.events=[];this.lastDistance=null;this.initialBearingError=null;this.approachTime=0;this.approachDistance=0;
    this.previousAir=false;this.wasFed=false;this.airTime=0;this.phase='unobserved';
    this.flightSamples=[];this.flightBoutQualified=false;this.landingCandidate=null;this.landingCount=0;
    this.flightEvidence={qualified:false,reason:'insufficient observations'};
  }
  resetContinuity(reason='physical pose explicitly repositioned'){
    // An explicit placement/mode switch is not a physical trajectory. Retain
    // historical observations, but do not join evidence across the jump.
    this.lastDistance=null;this.initialBearingError=null;this.approachTime=0;this.approachDistance=0;
    this.previousAir=false;this.airTime=0;this.phase='unobserved';this.flightSamples=[];
    this.flightBoutQualified=false;this.landingCandidate=null;this.flightEvidence={qualified:false,reason};
  }
  sample(body){
    const q=body.quaternion;
    const norm=q?.length===4?q.reduce((sum,v)=>sum+v*v,0):0;
    const up=norm>0&&Number.isFinite(norm)?1-2*(q[1]*q[1]+q[2]*q[2])/norm:
      Number.isFinite(body.tilt)?Math.cos(body.tilt):null;
    const raw=body.angularVelocity??body.data?.qvel?.slice(3,6);
    const omega=raw?.length===3&&Array.from(raw).every(Number.isFinite)?Array.from(raw):
      Number.isFinite(body.angularSpeed)?[body.angularSpeed,0,0]:null;
    // Native ncon includes self contact. The body wrapper separately counts
    // collisions with environment geometry, including walls and ceilings.
    const nativeContact=body.environmentContactCount;
    const external=[body.data?.xfrc_applied,body.data?.qfrc_applied].some(a=>a&&Array.from(a).some(v=>!Number.isFinite(v)||Math.abs(v)>1e-12));
    return {time:body.time,x:body.x,y:body.y,z:body.z,vz:body.vz,up,omega,
      airborne:body.airborne===true,wing:body.wingPower,
      contactKnown:Number.isFinite(nativeContact),contactFree:nativeContact===0,external};
  }
  assessFlight(sample){
    const c=TaskMonitor.observationCriteria;
    const usable=[sample.time,sample.x,sample.y,sample.z,sample.vz,sample.wing].every(Number.isFinite)&&
      sample.up!==null&&sample.omega!==null&&sample.contactKnown;
    const supported=usable&&sample.airborne&&sample.contactFree&&!sample.external&&
      sample.wing>c.minimumWingPower&&sample.up>=Math.cos(c.maximumTiltRadians);
    if(!supported){
      this.flightSamples=[];
      return {qualified:false,reason:!usable?'missing kinematic evidence':!sample.airborne?'surface contact':
        !sample.contactFree?'environment collision':sample.external?'external force':sample.wing<=c.minimumWingPower?'wings inactive':'not upright'};
    }
    let rows=this.flightSamples;
    const previous=rows.at(-1);
    if(previous&&(sample.time<=previous.time||sample.time-previous.time>c.maximumSampleGapSeconds))rows=this.flightSamples=[];
    rows.push(sample);
    while(rows.length>1&&rows[1].time<=sample.time-c.flightWindowSeconds)rows.shift();
    const span=sample.time-rows[0].time;
    if(span+1e-9<c.flightWindowSeconds)return {qualified:false,reason:'insufficient sustained observations',windowSeconds:span};
    const early=rows.filter(s=>s.time<=rows[0].time+c.velocityAverageSeconds),late=rows.filter(s=>s.time>=sample.time-c.velocityAverageSeconds);
    const mean=(a,key)=>a.reduce((sum,s)=>sum+s[key],0)/a.length;
    const acceleration=(mean(late,'vz')-mean(early,'vz'))/(mean(late,'time')-mean(early,'time'));
    const verticalSpeed=(sample.z-rows[0].z)/span;
    const meanOmega=[0,1,2].map(axis=>rows.reduce((sum,s)=>sum+s.omega[axis],0)/rows.length);
    const angularSpeed=Math.hypot(...meanOmega),supportFraction=1+acceleration/c.gravityCmPerSecondSquared;
    const qualified=supportFraction>=c.minimumSupportFraction&&verticalSpeed>=-c.maximumDescentCmPerSecond&&angularSpeed<=c.maximumMeanAngularSpeed;
    return {qualified,reason:qualified?'sustained powered upright flight':supportFraction<c.minimumSupportFraction?'ballistic or insufficient support':
      verticalSpeed < -c.maximumDescentCmPerSecond?'rapid descent':'sustained rapid rotation',windowSeconds:span,
      meanVerticalSpeed:verticalSpeed,meanVerticalAcceleration:acceleration,inferredSupportFraction:supportFraction,meanAngularSpeed:angularSpeed};
  }
  step(body,dt,intake){
    if(!(dt>0)||!Number.isFinite(dt))return;
    const c=TaskMonitor.observationCriteria,priorPhase=this.phase,sample=this.sample(body);
    const previousRows=this.flightSamples;
    if(!this.previousAir&&body.airborne){this.flightBoutQualified=false;this.landingCandidate=null;}
    if(this.previousAir&&!body.airborne){
      // A crash or a bounce is not a landing. Require a previously qualified
      // flight bout, a bounded approach speed, then sustained upright contact.
      const recent=previousRows.filter(s=>s.time>body.time-c.velocityAverageSeconds);
      const approachSpeed=recent.length?Math.abs(recent.reduce((sum,s)=>sum+s.vz,0)/recent.length):Infinity;
      this.landingCandidate=this.flightBoutQualified&&this.airTime>=c.flightWindowSeconds&&body.onFood&&
        approachSpeed<=c.maximumLandingApproachSpeed?{rows:[],approachSpeed}:null;
    }
    this.flightEvidence=this.assessFlight(sample);
    if(this.flightEvidence.qualified)this.flightBoutQualified=true;
    const distance=Math.hypot(body.x-body.food.x,body.y-body.food.y),speed=Math.hypot(body.vx,body.vy);
    let recordedCurrentStage=false;
    const record=(stage,evidence)=>{if(!this.events.some(e=>e.stage===stage))this.events.push({stage,time:body.time,evidence});this.phase=stage;recordedCurrentStage=true;};
    const bearing=Math.atan2(body.food.y-body.y,body.food.x-body.x)-(body.heading||0),error=Math.abs(Math.atan2(Math.sin(bearing),Math.cos(bearing)));
    if(body.odor[0]+body.odor[1]>.02){
      this.initialBearingError??=error;
      if(this.initialBearingError>.3&&error<.15)record('localization','Body turned from an initial bearing error above 0.3 rad to within 0.15 rad of food; orientation proxy only.');
    }
    if(!body.onFood&&distance>body.food.radius&&this.lastDistance!==null&&distance<this.lastDistance-1e-6&&speed>.01){
      this.approachTime+=dt;this.approachDistance+=this.lastDistance-distance;
    }else{this.approachTime=0;this.approachDistance=0;}
    if(this.approachTime>.2&&this.approachDistance>.23)record('approach','Distance to food decreased by a body length (0.23 cm) over at least 200 ms outside food contact.');
    if(this.landingCandidate){
      const stable=!body.airborne&&body.onFood&&sample.contactKnown&&!sample.contactFree&&!sample.external&&sample.up!==null&&sample.up>=Math.cos(c.maximumLandingTiltRadians)&&
        sample.omega!==null&&[sample.time,sample.x,sample.y,sample.z].every(Number.isFinite);
      if(!stable)this.landingCandidate=null;
      else{
        const rows=this.landingCandidate.rows,previous=rows.at(-1);
        if(previous&&(sample.time<=previous.time||sample.time-previous.time>c.maximumSampleGapSeconds))rows.length=0;
        rows.push(sample);
        while(rows.length>1&&rows[1].time<=sample.time-c.landingContactSeconds)rows.shift();
        const span=sample.time-rows[0].time;
        if(span+1e-9>=c.landingContactSeconds){
          const displacement=Math.max(...rows.map(s=>Math.hypot(s.x-rows[0].x,s.y-rows[0].y,s.z-rows[0].z)));
          const rotation=Math.hypot(...[0,1,2].map(axis=>rows.reduce((sum,s)=>sum+s.omega[axis],0)/rows.length));
          if(displacement/span<=c.maximumLandingSpeedCmPerSecond&&rotation<=c.maximumLandingAngularSpeed){
            if(!this.landingCandidate.recorded){this.landingCount++;this.landingCandidate.recorded=true;}
            record('landing','Upright food contact persisted for 100 ms after qualified flight, with bounded approach speed and low mean contact motion; modeled observation criteria.');
          }
        }
      }
    }
    if(body.mouthContact&&body.proboscis>.05)record('probing','Extended proboscis contacts food.');
    if(intake>0){record('feeding','Finite food transfer into crop.');this.wasFed=true;}
    if(!this.previousAir&&body.airborne&&this.wasFed&&body.wingPower>.1&&body.vz>.5)record('takeoff','Upward departure from ground contact with active wings after feeding.');
    this.airTime=body.airborne?this.airTime+dt:0;
    if(this.flightEvidence.qualified)record('flight','At least 250 ms airborne with active wings, upright pose, no environment contacts or external forces, bounded mean descent/rotation and kinematic support against gravity; modeled observation criteria, not measured biological limits.');
    else if(!recordedCurrentStage&&['flight','landing','takeoff'].includes(priorPhase)&&this.phase===priorPhase)
      this.phase=body.airborne?'airborne unverified':'surface contact unverified';
    this.previousAir=body.airborne;this.lastDistance=distance;
  }
}

function footFeatures(q){return [1,...q,...q.map(x=>x*x),q[0]*q[1],q[0]*q[2],q[1]*q[2]];}
export function footPositions(model,state){
  return model.leg_kinematics.map(leg=>{
    const q=leg.joint_indices.map(i=>state[i*2]-model.active_joints[i].neutral),f=footFeatures(q);
    return [0,1,2].map(axis=>f.reduce((sum,v,k)=>sum+v*leg.coefficients[k][axis],0));
  });
}

export class ReducedBody {
  constructor(model,io,runtime,{energy=.35,initialCondition='grounded',environment=null}={}){
    if(!['grounded','airborne'].includes(initialCondition))throw new Error('Invalid initial condition');
    this.model=model;this.environment=environment;this.mappings=io.muscles;this.muscles=runtime.createMuscles(io.muscles.length);this.joints=runtime.createJoints(model.active_joints);
    this.internal=new InternalState(energy);this.monitor=new TaskMonitor();this.time=0;
    this.x=-1;this.y=0;this.heading=.5;this.vx=0;this.vy=0;this.vz=0;this.yaw=0;
    this.food={x:.6,y:0,radius:.35,height:.12,remaining:1};this.jointState=Float32Array.from(model.active_joints.flatMap(j=>[j.neutral,0]));
    this.feet=footPositions(model,this.jointState);this.z=-Math.min(...this.feet.map(p=>p[2]));
    this.airborne=false;this.onFood=false;this.mouthContact=false;this.proboscis=0;this.pump=0;this.wingPower=0;this.wingPhase=0;this.odor=[0,0];
    this.input=new Float32Array(io.muscles.length*5);this.muscleState=new Float32Array(io.muscles.length*3);
    this.jointIndex=new Map(model.active_joints.map((j,i)=>[j.name,i]));
    if(initialCondition==='airborne'){this.z+=.8;this.vx=3*Math.cos(this.heading);this.vy=3*Math.sin(this.heading);this.airborne=true;this.monitor.previousAir=true;}
  }
  surface(x,y){return this.environment?this.environment.surface(x,y):Math.hypot(x-this.food.x,y-this.food.y)<this.food.radius?this.food.height:0;}
  settleOnSurface(){
    const c=Math.cos(this.heading),s=Math.sin(this.heading);
    this.z=Math.max(...this.feet.map(([x,y,z])=>this.surface(this.x+c*x-s*y,this.y+s*x+c*y)-z));
    this.vz=0;this.airborne=false;
  }
  smell(x,y,z){return this.environment?this.environment.odor(x,y,z):this.food.remaining>0?Math.exp(-Math.hypot(x-this.food.x,y-this.food.y,Math.max(0,z-this.food.height))/.8):0;}
  step(rates,duration,{coupling=true,flight=true}={}){
    if(!(duration>0&&duration<=.1))throw new Error('Invalid body duration');
    const steps=Math.ceil(duration/.001),dt=duration/steps;
    for(let sub=0;sub<steps;sub++){
      for(let i=0;i<this.mappings.length;i++){
        const m=this.mappings[i],j=this.jointIndex.get(m.joint);
        const hz=m.indices.reduce((sum,id)=>sum+(rates.get(id)||0),0)/m.indices.length;
        const activation=coupling?clamp(hz/80):0;
        const length=j===undefined?1:1+.2*m.sign*(this.jointState[j*2]-this.model.active_joints[j].neutral);
        const velocity=j===undefined?0:.2*m.sign*this.jointState[j*2+1];
        this.input.set([activation,length,velocity,1,muscleFuel(this.internal.energy)],i*5);
      }
      this.muscleState=this.muscles.step(this.input,dt);
      const torque=new Float32Array(this.model.active_joints.length),power={left:0,right:0},steer={left:0,right:0};
      const counts={left:0,right:0};let proboscis=0,pump=0,probN=0,pumpN=0,effort=0;
      for(let i=0;i<this.mappings.length;i++){
        const m=this.mappings[i],force=this.muscleState[i*3+2];effort+=this.muscleState[i*3];
        const j=this.jointIndex.get(m.joint),side=m.joint.endsWith('left')?'left':'right';
        if(j!==undefined)torque[j]+=m.sign*force*.003;
        if(m.kind==='asynchronous_wing'){power[side]+=force;counts[side]++;}
        if(m.kind==='wing_steering_assumption')steer[side]+=force*.05;
        if(m.joint==='proboscis'){proboscis+=force;probN++;}
        if(m.joint==='pump'){pump+=force;pumpN++;}
      }
      power.left=flight?power.left/Math.max(1,counts.left):0;power.right=flight?power.right/Math.max(1,counts.right):0;
      this.wingPowerLeft=power.left;this.wingPowerRight=power.right;
      this.proboscis=proboscis/Math.max(1,probN);this.pump=pump/Math.max(1,pumpN);
      for(const name of ['rostrum','haustellum'])torque[this.jointIndex.get(name)]=-.006*this.proboscis;
      this.jointState=this.joints.step(torque,dt);
      const previousFeet=this.feet;this.feet=footPositions(this.model,this.jointState);
      const c=Math.cos(this.heading),s=Math.sin(this.heading),support=[];
      let minClearance=Infinity;
      for(let leg=0;leg<6;leg++){
        const [fx,fy,fz]=this.feet[leg],wx=this.x+c*fx-s*fy,wy=this.y+s*fx+c*fy;
        const clearance=this.z+fz-this.surface(wx,wy);minClearance=Math.min(minClearance,clearance);
        if(clearance<.012)support.push({leg,x:fx,y:fy});
      }
      this.wingPower=(power.left+power.right)/2;
      // Averaged lift from asynchronous flight-muscle activation, not spike phase.
      const lift=2400*this.wingPower*this.wingPower;
      this.vz+=(lift-981-4*this.vz)*dt;
      if(support.length){
        let desiredX=0,desiredY=0,desiredYaw=0;
        for(const p of support){const leg=p.leg,dx=(this.feet[leg][0]-previousFeet[leg][0])/dt,dy=(this.feet[leg][1]-previousFeet[leg][1])/dt;
          desiredX-=dx/support.length;desiredY-=dy/support.length;desiredYaw+=(p.y*dx-p.x*dy)/(p.x*p.x+p.y*p.y+.001)/support.length;}
        const targetX=c*desiredX-s*desiredY,targetY=s*desiredX+c*desiredY;
        this.vx+=clamp((targetX-this.vx)*40,-.6*981,.6*981)*dt;
        this.vy+=clamp((targetY-this.vy)*40,-.6*981,.6*981)*dt;
        this.yaw+=(desiredYaw-this.yaw)*(1-Math.exp(-dt*30));
      }else{
        this.vx+=(c*lift*.1-3*this.vx)*dt;this.vy+=(s*lift*.1-3*this.vy)*dt;
        this.yaw+=(20*(power.right-power.left+steer.right-steer.left)-4*this.yaw)*dt;
      }
      this.x+=this.vx*dt;this.y+=this.vy*dt;this.z+=this.vz*dt;this.heading+=this.yaw*dt;
      const clearance=Math.min(...this.feet.map(([fx,fy,fz])=>this.z+fz-this.surface(this.x+c*fx-s*fy,this.y+s*fx+c*fy)));
      if(clearance<0){this.z-=clearance;this.vz=Math.max(0,this.vz);}
      this.airborne=clearance>.012;this.onFood=(this.environment?!!this.environment.foodAt(this.x,this.y):this.surface(this.x,this.y)>0)&&!this.airborne;
      const mouthX=this.x+c*.12,mouthY=this.y+s*.12;
      const contactFood=this.environment?this.environment.foodAt(mouthX,mouthY):this.food;
      this.mouthContact=!!contactFood&&contactFood.remaining>0&&(this.environment||Math.hypot(mouthX-this.food.x,mouthY-this.food.y)<this.food.radius)&&this.z-this.surface(mouthX,mouthY)<.23&&!this.airborne;
      const intake=this.mouthContact&&this.proboscis>.05&&this.pump>.05?Math.min(contactFood.remaining,dt*.03*this.pump,1-this.internal.crop):0;
      if(contactFood)contactFood.remaining-=intake;this.internal.step(dt,intake,effort/this.mappings.length);
      this.odor=[1,-1].map(side=>this.smell(this.x+c*.1-side*s*.04,this.y+s*.1+side*c*.04,this.z));
      this.wingPhase=(this.wingPhase+2*Math.PI*200*dt)%(2*Math.PI);this.time+=dt;
      this.monitor.step(this,dt,intake);
      if(![this.x,this.y,this.z,this.vx,this.vy,this.vz].every(Number.isFinite))throw new Error('Body dynamics became nonfinite');
    }
  }
  snapshot(){return {time:this.time,x:this.x,y:this.y,z:this.z,heading:this.heading,vx:this.vx,vy:this.vy,vz:this.vz,airborne:this.airborne,onFood:this.onFood,
    mouthContact:this.mouthContact,proboscis:this.proboscis,pump:this.pump,wingPower:this.wingPower,wingPhase:this.wingPhase,feet:this.feet,
    internal:{...this.internal},food:{...this.food},events:this.monitor.events,phase:this.monitor.phase,muscleState:this.muscleState,jointState:this.jointState,odor:this.odor};}
  dispose(){this.muscles.dispose();this.joints.dispose();}
}

export function sensoryCurrents(model,body,{odor=true,vision=true,taste=true,proprioception=true}={}){
  const input=new Float32Array(model.manifest.neuron_count),c=Math.cos(body.heading),s=Math.sin(body.heading);
  const dx=body.food.x-body.x,dy=body.food.y-body.y,bearing=Math.atan2(dy,dx)-body.heading;
  for(const sensor of model.io.sensory){
    const side=sensor.side==='left'?1:-1;
    if(sensor.kind==='odor'&&odor&&['ORN_DM1','ORN_VA2'].includes(sensor.cell_type))input[sensor.index]=5+70*body.smell(body.x+c*.1-side*s*.04,body.y+s*.1+side*c*.04,body.z);
    if(sensor.kind==='taste'&&taste&&body.mouthContact)input[sensor.index]=45;
    // Coarse bilateral retinal irradiance surrogate; no FlyWire IDs or mapped T4/T5 injection.
    if(sensor.kind==='vision'&&vision)input[sensor.index]=10+8*Math.max(0,Math.cos(bearing-side*Math.PI/3))*Math.exp(-Math.hypot(dx,dy)/2);
    if((sensor.kind==='proprioception'||sensor.kind==='load')&&proprioception){
      const segment=String(sensor.body_part||'').includes('front')?0:String(sensor.body_part||'').includes('middle')?1:String(sensor.body_part||'').includes('hind')?2:-1;
      if(segment>=0){const leg=segment*2+(side===1?0:1),j=leg*3;input[sensor.index]=sensor.kind==='load'?(!body.airborne?12:0):Math.min(35,Math.abs(body.jointState[j*2+1])*2+Math.abs(body.jointState[j*2]-body.model.active_joints[j].neutral)*10);}
    }
  }
  return input;
}
