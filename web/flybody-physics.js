import {InternalState,TaskMonitor,muscleFuel} from './banc/embodiment.js';
import {initializeStance} from './flybody-stance.js';
import {FlyBodyWings} from './flybody-wings.js';
import {createBancProboscisDecoder} from './banc-proboscis.js';
import {nativeLegDirection} from './flybody-leg-actuation.js';
import {createFlybodyHabitatCollision} from './flybody-habitat-collision.js';
import {createMotorExcitation} from './flybody-motor-excitation.js';
import {createWingEventExcitation} from './flybody-wing-event-excitation.js';
import {createWingLoadSampler} from './flybody-wing-load.js';
import {createMotorDecoder} from './motor-decoder.js';

const clamp=(v,a=0,b=1)=>Math.max(a,Math.min(b,v));
// Only the bowl is a heightfield; separate fruit solids preserve overhangs.
export function flybodyScene(xml,habitat,{resolution=257,extent=6.6}={}){
  const scene=createFlybodyHabitatCollision(habitat,{resolution,extent});
  return {...scene,xml:xml.replace('<asset />',`<asset>${scene.assets}</asset>`).replace('<worldbody>',`<worldbody>${scene.geoms}`)};
}

/** A BANC motor-neuron decoder, muscle activation filter, and real MuJoCo body.
 * No desired root velocity, height, heading, food bearing or task phase enters
 * the actuator decoder. The published WPG is the asynchronous wing model gap.
 */
export class FlyBodyPhysics{
  constructor(mj,model,metadata,io,createMuscles,environment){
    // Validate and freeze optional recruitment priors before native allocation.
    this.motorExcitation=createMotorExcitation(metadata.motor_excitation);
    Object.assign(this,{mj,model,metadata,environment});
    this.data=new mj.MjData(model);this.mappings=io.muscles;this.muscles=createMuscles(io.muscles.length);
    this._createMuscles=createMuscles;this._wingMotorEvents=null;this._wingLoadFeedback=null;
    // Model topology is immutable. Own this small copy so native heap growth
    // cannot detach a retained view, and avoid a new Embind wrapper per contact.
    this.geomBodyIds=Int32Array.from(model.geom_bodyid);
    this.contactForce=new mj.DoubleBuffer(6);this.legLoads=new Float64Array(6);this.legCollisions=new Float64Array(6);
    this.jointIndex=new Map(metadata.joints.map((j,i)=>[j.name,i]));
    this.muscleDirections=Float64Array.from(io.muscles,m=>m.kind==='leg'?nativeLegDirection(m):m.sign);
    this.byJoint=new Map(metadata.joints.map(j=>[j.name,j]));
    this.actuators=new Map(metadata.actuators.map(a=>[a.name,a]));
    this.legFoodContact=new Uint8Array(6);this.wingFoodContact=new Uint8Array(2);this.mouthFoodContact=new Uint8Array(2);
    this.tasteBodyToLeg=new Int8Array(model.nbody).fill(-1);this.tasteBodyToWing=new Int8Array(model.nbody).fill(-1);
    const labelDescendants=(joint,label,lookup)=>{
      const ancestor=model.jnt_bodyid[this.byJoint.get(joint).id];
      for(let body=1;body<model.nbody;body++)for(let parent=body;parent;parent=model.body_parentid[parent])if(parent===ancestor){lookup[body]=label;break;}
    };
    for(const [sideIndex,side]of ['left','right'].entries()){
      for(let segment=1;segment<=3;segment++)labelDescendants(`tarsus_T${segment}_${side}`,sideIndex*3+segment-1,this.tasteBodyToLeg);
      labelDescendants(`wing_yaw_${side}`,sideIndex,this.tasteBodyToWing);
    }
    this.input=new Float32Array(io.muscles.length*5);this.muscleState=new Float32Array(io.muscles.length*3);
    this.proboscisDecoder=createBancProboscisDecoder(io.muscles);
    this.jointState=new Float32Array(metadata.joints.length*2);this.internal=new InternalState();this.monitor=new TaskMonitor();
    this.activation=new Float32Array(io.muscles.length);this.wingPhase=0;this.wingPower=0;this.wingPowerLeft=0;this.wingPowerRight=0;
    this.halterePower=[0,0];this.haltereSteering={left:{},right:{}};
    this.wings=new FlyBodyWings(metadata);
    this.proboscis=0;this.pump=0;this.time=0;this.remainder=0;this.airborne=false;this.onFood=false;this.mouthContact=false;this.odor=[0,0];
    this.food={x:0,y:0,radius:.1,height:0,remaining:0};this.contactCount=0;
    for(const j of metadata.joints)this.data.qpos[j.qpos]=j.neutral;
    // Evaluate the reference pose above the terrain; forwarding it at z=0
    // would initialize every body segment inside the bowl's collision field.
    this.data.qpos[2]=environment.surface(0,0)+1;
    for(const a of metadata.actuators)if(a.joint!==null&&!a.name.startsWith('wing_'))this.data.ctrl[a.id]=this.byJoint.get(a.name).neutral;
    mj.mj_forward(model,this.data);
    this.restHeight=this.data.qpos[2]-Math.min(...metadata.feet.map(i=>this.data.site_xpos[i*3+2]));
    this.restPose=Float64Array.from(this.data.qpos);
    this.refresh();
  }
  surface(x,y){return this.environment.surface(x,y);}
  smell(x,y,z){return this.environment.odor(x,y,z);}
  /** Call after the initial mj_forward and final placement, before copying
   * feedback. No native operation or force write is performed by this observer. */
  enableWingLoadFeedback(){
    if(this._disposed||this._wingLoadFeedback)throw new Error('Wing load feedback already enabled or body disposed');
    if(this.time!==0||this.data.time!==0||this.remainder!==0)throw new Error('Wing load feedback requires a fresh zero-time body');
    if(this.metadata.timestep!==.00005||this.model.opt.timestep!==this.metadata.timestep)
      throw new Error('Wing load feedback requires the50us native/1ms observation contract');
    const sampler=createWingLoadSampler({mj:this.mj,model:this.model});
    sampler.capture(this.data,this.data.time); // matching just-forwarded cache
    this._wingLoadFeedback=sampler;
    return sampler.read();
  }
  readWingLoadFeedback(){return this._wingLoadFeedback?.read()??null;}
  /** Experimental learned actuator bridge. It receives individual motor-unit
   * excitation histories, never root state, sensory targets or reward. */
  enableMotorDecoder(parameters){
    if(this._disposed||this.motorDecoder||!this._wingMotorEvents)
      throw new Error('Motor decoder requires wing events and an unconfigured active body');
    if(this.time!==0||this.data.time!==0||this.remainder!==0||this._wingMotorEvents.elapsedMs!==0)
      throw new Error('Motor decoder requires a fresh zero-time body');
    this.motorDecoder=createMotorDecoder({muscles:this.mappings},parameters);
    this._silentMotorUnits=new Float64Array(this.motorDecoder.contract.indices.length);
    return this.motorDecoder.contract;
  }
  /** Opt-in only, before the first body interval. Priors are owned by the
   * adapter; the existing full-count native kernel retains silent wing rows.
   * This changes the wing effector hypothesis, not neural physiology. */
  enableWingMotorEvents(config,eventContract){
    if(this._disposed||this._wingMotorEvents)throw new Error('Wing motor events already enabled or body disposed');
    if(this.time!==0||this.data.time!==0||this.remainder!==0||
      !this.muscleState.every(value=>value===0)||!this.activation.every(value=>value===0))
      throw new Error('Wing motor events require a fresh zero-time muscle/body state');
    if(this.metadata.timestep!==.00005)throw new Error('Wing motor events require the50us native/1ms muscle timing contract');
    const adapter=createWingEventExcitation({io:{muscles:this.mappings},config,eventContract}),description=adapter.readState();
    const mappingIndices=Array.from(description.mappingIndices),mask=new Uint8Array(this.mappings.length);
    for(const index of mappingIndices){
      if(this.byJoint.has(this.mappings[index].joint))throw new Error('Wing event muscles require virtual unit-length, zero-velocity mappings');
      mask[index]=1;
    }
    const muscles=this._createMuscles(28);
    this._wingMotorEvents={adapter,muscles,mappingIndices,mask,input:new Float32Array(28*5),
      nativeState:new Float32Array(28*3),elapsedMs:0,observedMs:0,initialized:false,lastInterval:null};
    return this.readWingMotorEvents();
  }
  _checkWingEventClock(){
    const event=this._wingMotorEvents;
    if(!event||this._disposed)throw new Error('Wing motor events are not enabled on an active body');
    // Native time accumulates50us floating additions. Integer packet clocks
    // remain authoritative; tolerate only arithmetic drift, not a lost step.
    if(this.remainder!==0||![this.time,this.data.time].every(time=>Number.isFinite(time)&&
      Math.abs(time*1000-event.elapsedMs)<=1e-5))throw new Error('Wing event/native body clock mismatch');
    return event;
  }
  acceptWingMotorEvents(packet){
    const event=this._checkWingEventClock();
    // Adapter validation is transactional. No native state, rate conversion,
    // or body motion occurs when accepting the packet, including its baseline.
    event.adapter.accept(packet);event.observedMs=packet.timeMs;event.initialized=true;
  }
  readWingMotorEvents(){
    const event=this._wingMotorEvents;if(!event)return null;
    const interval=event.lastInterval;
    return {mode:'wing-motor-events-v1',...event.adapter.readState(),bodyConsumedThroughMs:event.elapsedMs,
      nativeInput:event.input.slice(),nativeState:event.nativeState.slice(),
      lastInterval:interval?{...interval,excitation:interval.excitation.slice(),unitExcitation:interval.unitExcitation.slice(),
        unitMeanKernel:interval.unitMeanKernel.slice()}:null};
  }
  _finishWingMotorInterval(coupling){
    const event=this._wingMotorEvents,interval=event.adapter.finishInterval(event.elapsedMs+1);
    // One completed 1 ms interval enters the decoder. Sampling a wingbeat
    // phase below does not advance this history or consume future events.
    if(this.motorDecoder)this.motorDecoder.advance(coupling?interval.unitExcitation:this._silentMotorUnits);
    const fuel=muscleFuel(this.internal.energy);
    // Each virtual wing muscle has the existing length1/velocity0/Fmax1
    // convention. Fuel is held from this completed interval's start, before
    // the common metabolism update below, matching the nonwing input context.
    for(let i=0;i<28;i++)event.input.set([coupling?interval.excitation[i]:0,1,0,1,fuel],i*5);
    const state=event.muscles.step(event.input,.001);
    if(!(state instanceof Float32Array)||state.length!==84||!state.every(Number.isFinite))
      throw new Error('Invalid native wing muscle output');
    event.nativeState=state;event.elapsedMs++;event.lastInterval=interval;
    // One merge of newly available wing state, in original mapping order.
    // Full-kernel wing state is never overwritten with this public readback.
    for(let i=0;i<28;i++){
      const target=event.mappingIndices[i];
      this.muscleState.set(state.subarray(i*3,i*3+3),target*3);
      this.input.set(event.input.subarray(i*5,i*5+5),target*5);
      this.activation[target]=event.input[i*5];
    }
  }
  place(x,y,heading){
    if(this._wingMotorEvents&&this._wingMotorEvents.elapsedMs!==0)throw new Error('Cannot re-place a running wing event body');
    if(this._wingLoadFeedback&&this.data.time!==0)throw new Error('Cannot re-place a running wing load body');
    this.monitor.resetContinuity();
    const eps=.002,nx=-(this.surface(x+eps,y)-this.surface(x-eps,y))/(2*eps),ny=-(this.surface(x,y+eps)-this.surface(x,y-eps))/(2*eps),length=Math.hypot(nx,ny,1);
    const tw=Math.sqrt((1+1/length)/2),tx=-ny/length/(2*tw),ty=nx/length/(2*tw),c=Math.cos(heading/2),s=Math.sin(heading/2);
    const q=this.data.qpos;q[0]=x;q[1]=y;q[2]=this.surface(x,y)+1;
    q.set([tw*c,tx*c+ty*s,ty*c-tx*s,tw*s],3);
    this.mj.mj_forward(this.model,this.data);
    const height=this.data.qpos[2]+Math.max(...this.metadata.feet.map(i=>{const p=this.data.site_xpos;return this.surface(p[i*3],p[i*3+1])-p[i*3+2];}))+.003;
    this.data.qpos[2]=height;
    this.restPose=this.metadata.initializeStance===false?Float64Array.from(this.data.qpos):initializeStance(this.mj,this.model,this.data,this.metadata,(a,b)=>this.surface(a,b));
    this.data.qvel.fill(0);this.mj.mj_forward(this.model,this.data);
    if(this._wingLoadFeedback)this._wingLoadFeedback.capture(this.data,this.data.time);
    this.refresh();
  }
  step(rates,duration,{coupling=true,flight=true}={}){
    if(!(duration>0&&duration<=.25))throw new Error('Invalid body duration');
    const wingEvents=this._wingMotorEvents;
    if(wingEvents){
      this._checkWingEventClock();
      if(duration!==.002||!wingEvents.initialized||wingEvents.observedMs!==wingEvents.elapsedMs+2)
        throw new Error('Wing motor events require one complete observed2ms packet per body step');
    }
    const h=this.metadata.timestep,controlSteps=20,controlDt=h*controlSteps;
    for(let i=0;i<this.mappings.length;i++){
      if(wingEvents?.mask[i])continue; // Wing rate values remain diagnostic only.
      const m=this.mappings[i],hz=m.indices.reduce((s,id)=>s+(rates.get(id)||0),0)/m.indices.length;
      // An explicit metadata prior can desaturate wing steering only. The
      // default and all other muscle kinds retain the original rate/80 clamp.
      this.activation[i]=coupling?this.motorExcitation.fromRate(m.kind,hz):0;
    }
    this.remainder+=duration;
    while(this.remainder+1e-12>=controlDt){
      const q=this.data.qpos,v=this.data.qvel;
      for(let i=0;i<this.mappings.length;i++){
        const m=this.mappings[i],j=this.byJoint.get(m.joint),direction=this.muscleDirections[i];
        // Positive actuator torque shortens its agonist: dl/dq has the
        // opposite sign. WASM velocity is positive for shortening.
        this.input.set([wingEvents?.mask[i]?0:this.activation[i],j?1-.2*direction*(q[j.qpos]-j.neutral):1,j?.2*direction*v[j.dof]:0,1,muscleFuel(this.internal.energy)],i*5);
      }
      if(wingEvents){
        const state=this.muscles.step(this.input,controlDt);
        for(let i=0;i<this.mappings.length;i++){
          if(wingEvents.mask[i]){
            if(state[i*3]!==0||state[i*3+1]!==0||state[i*3+2]!==0)throw new Error('Full muscle kernel has nonzero hidden wing state');
          }else this.muscleState.set(state.subarray(i*3,i*3+3),i*3);
        }
      }else this.muscleState=this.muscles.step(this.input,controlDt);
      const commands=new Map(),power={left:[],right:[]},haltere={left:[],right:[]},steer={left:{},right:{}},haltereSteering={left:{},right:{}};let effort=0,pump=0,pumpN=0;
      for(let i=0;i<this.mappings.length;i++){
        const m=this.mappings[i],force=this.muscleState[i*3+2];effort+=this.muscleState[i*3];
        if(m.kind==='leg'){
          const command=commands.get(m.joint)||{positive:[],negative:[]};command[this.muscleDirections[i]>0?'positive':'negative'].push(force);commands.set(m.joint,command);
        }
        if(m.kind==='asynchronous_wing')power[m.joint.endsWith('left')?'left':'right'].push(force);
        if(m.kind==='wing_steering_assumption')steer[m.joint.endsWith('left')?'left':'right'][m.target]=force;
        if(m.kind==='asynchronous_haltere')haltere[m.joint.endsWith('left')?'left':'right'].push(force);
        if(m.kind==='haltere_steering_assumption')haltereSteering[m.joint.endsWith('left')?'left':'right'][m.target]=force;
        if(m.kind==='claw_grip_assumption')this.data.ctrl[this.actuators.get(m.joint).id]=clamp(force);
        if(m.joint==='pump'){pump+=force;pumpN++;}
      }
      const mean=a=>a.reduce((s,x)=>s+x,0)/Math.max(1,a.length);
      this.proboscisChannels=this.proboscisDecoder.read(this.muscleState);this.pump=pump/Math.max(1,pumpN);
      for(const [name,command]of commands){
        const j=this.byJoint.get(name),a=this.actuators.get(name),drive=clamp(mean(command.positive)-mean(command.negative),-1,1);
        const rest=this.restPose[j.qpos],excursion=Math.min(this.metadata.maxJointExcursion??.35,drive>0?j.range[1]-rest:rest-j.range[0]);
        this.data.ctrl[a.id]=clamp(rest+drive*excursion,...a.range);
      }
      for(const name of ['rostrum','haustellum']){
        const j=this.byJoint.get(name),a=this.actuators.get(name),channels=this.proboscisChannels;
        const drive=channels[name+'Extend']-channels[name+'Retract'];
        // Native joint-sign assay: negative rotates into extension; positive
        // retracts. Separate anatomical antagonists act on their own joint.
        // Labellar channels remain explicit and unrepresented (frozen DoFs).
        this.data.ctrl[a.id]=j.neutral+Math.max(0,drive)*(a.range[0]-j.neutral)+Math.max(0,-drive)*(a.range[1]-j.neutral);
      }
      this.wingDriveLeft=flight?mean(power.left):0;this.wingDriveRight=flight?mean(power.right):0;
      this.halterePower=[mean(haltere.left),mean(haltere.right)];this.haltereSteering=haltereSteering;
      for(let sub=0;sub<controlSteps;sub+=4){
      if(this.motorDecoder){
        const decoded=this.motorDecoder.sample(this.wings.phase);
        // The old native wing-muscle states above remain diagnostic in this
        // mode. Individual event excitation drives the learned bridge directly.
        // Flight/coupling switches gate all active power and steering together.
        if(!flight||!coupling){decoded.power.fill(0);for(const side of decoded.steering)side.fill(0);}
        this.wingDriveLeft=decoded.power[0];this.wingDriveRight=decoded.power[1];
        this.wings.stepDecoded(this.data.qpos,this.data.ctrl,decoded,h*4);
      }else this.wings.step(this.data.qpos,this.data.ctrl,this.wingDriveLeft,this.wingDriveRight,steer,h*4);
      for(let k=0;k<4;k++)this.mj.mj_step(this.model,this.data);
      this.wingPhase=this.wings.phase;
      }
      // Euler leaves force and hinge frames from its final pre-integration
      // evaluation. Capture before any subsequent forward/refresh, including
      // contact work that may grow the native heap. This is aerodynamic load.
      if(this._wingLoadFeedback)this._wingLoadFeedback.capture(this.data,this.data.time-h);
      [this.wingPowerLeft,this.wingPowerRight]=this.wings.power;
      this.wingPower=(this.wingPowerLeft+this.wingPowerRight)/2;
      // Commands above used the previously available wing force throughout
      // this1ms interval. New event-derived force becomes usable only now.
      // Effort was counted once from held wing activation and updated nonwing
      // activation; the newly merged wing state belongs to the next interval.
      if(wingEvents)this._finishWingMotorInterval(coupling);
      this.remainder-=controlDt;this.refresh();
      const c=Math.cos(this.heading),s=Math.sin(this.heading),contactFood=this.contactFood;
      this.mouthContact=!!contactFood&&contactFood.remaining>0;
      const intake=this.mouthContact&&this.proboscis>.05&&this.pump>.05?Math.min(contactFood.remaining,controlDt*.03*this.pump,1-this.internal.crop):0;
      this.lastIntake=intake;
      if(contactFood)contactFood.remaining-=intake;
      this.internal.step(controlDt,intake,effort/this.mappings.length);
      this.odor=[1,-1].map(side=>this.smell(this.x+c*.1-side*s*.04,this.y+s*.1+side*c*.04,this.z));
      this.monitor.step(this,controlDt,intake);
    }
  }
  refresh(){
    const d=this.data,q=d.qpos.slice();this.x=q[0];this.y=q[1];this.z=q[2];this.vx=d.qvel[0];this.vy=d.qvel[1];this.vz=d.qvel[2];this.time=d.time;
    this.heading=Math.atan2(2*(q[3]*q[6]+q[4]*q[5]),1-2*(q[5]*q[5]+q[6]*q[6]));
    this.quaternion=Array.from(q.slice(3,7));
    this.contactCount=0;this.environmentContactCount=0;this.foodContactCount=0;
    this.legLoads.fill(0);this.legCollisions.fill(0);this.contactFood=null;
    this.legFoodContact.fill(0);this.wingFoodContact.fill(0);this.mouthFoodContact.fill(0);
    // Embind contact access copies a C++ vector; neither it nor its element
    // handles are JS-GC-owned. Copy once per refresh and release both levels.
    const contacts=d.ncon?d.contact:null;
    try{for(let i=0;i<d.ncon;i++){
      const contact=contacts.get(i);
      try{
      const geoms=contact.geom,g0=geoms[0],g1=geoms[1],distance=contact.dist;
      const static0=this.geomBodyIds[g0]===0,static1=this.geomBodyIds[g1]===0;
      // Every static habitat solid can support a limb. Self contacts and
      // static/static pairs cannot supply environmental touch or food taste.
      if(static0===static1)continue;
      const geom=static0?g1:g0,environmentGeom=static0?g0:g1;
      if(distance<=0)this.environmentContactCount++;
      if(distance>=.002)continue;
      this.contactCount++;const body=this.geomBodyIds[geom],leg=this.metadata.body_to_leg?.[body]??-1;
      if(leg>=0){
        this.mj.mj_contactForce(this.model,d,i,this.contactForce);
        this.legLoads[leg]+=Math.max(0,this.contactForce.GetView()[0])/(this.metadata.mass_g*981);
        if(!this.metadata.claw_bodies.includes(body))this.legCollisions[leg]=1;
      }
      if(distance<=0){
        const position=contact.pos;
        const food=this.environment.foodForContact?this.environment.foodForContact(environmentGeom,position):this.environment.foodAt(position[0],position[1]);
        const mouth=this.metadata.mouth_bodies?.indexOf(body)??-1;
        if(food)this.foodContactCount++;
        if(mouth>=0&&food)this.contactFood=food;
        if(food?.remaining>0){
          const foot=this.tasteBodyToLeg[body],wing=this.tasteBodyToWing[body];
          if(foot>=0)this.legFoodContact[foot]=1;if(wing>=0)this.wingFoodContact[wing]=1;if(mouth>=0)this.mouthFoodContact[mouth]=1;
        }
      }
      }finally{contact.delete();}
    }}finally{contacts?.delete();}
    // Acquire ephemeral views after native contact allocation/deletion. They
    // are consumed synchronously and never retained across native operations.
    const velocities=d.qvel,sites=d.site_xpos;
    this.metadata.joints.forEach((j,i)=>this.jointState.set([q[j.qpos],velocities[j.dof]],i*2));
    this.proboscis=['rostrum','haustellum'].reduce((sum,name)=>{const j=this.byJoint.get(name);return sum+clamp((j.neutral-q[j.qpos])/(j.neutral-j.range[0]));},0)/2;
    this.feet=this.metadata.feet.map(i=>Array.from(sites.subarray(i*3,i*3+3)));
    // A projected fruit top can lie above an unsupported fly. Native proximity
    // contacts determine support; TaskMonitor qualifies sustained flight.
    this.airborne=this.contactCount===0;
    this.mouthContact=!!this.contactFood&&this.contactFood.remaining>0;
    this.onFood=this.foodContactCount>0;
    if(!q.every(Number.isFinite)||Math.max(...velocities.map(Math.abs))>1e5)throw new Error('MuJoCo body became unstable');
  }
  dispose(){
    if(this._disposed)return;this._disposed=true;
    this._wingMotorEvents?.muscles.dispose();this.muscles.dispose();this.contactForce.delete();this.data.delete();
  }
}
