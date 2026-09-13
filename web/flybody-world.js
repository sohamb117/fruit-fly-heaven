import {BodyWorld,jointPose,senseBody,decodeMotorOutput} from './body-world.js';
import {FlyBodyPhysics,flybodyScene} from './flybody-physics.js';
import {createWasmCore,WasmMuscles} from '/banc-engine/src/index.js';
import loadMujoco from '/body-engine/mujoco.js';
import {createMouthLandmarks,sampleMouthPose} from './flybody-mouth-pose.js';
import {createWingLandmarks,sampleWingPose} from './flybody-wing-pose.js';
import {createContactFoodResolver} from './flybody-contact-environment.js';

const SCALE=10,clamp=(v,a=0,b=1)=>Math.max(a,Math.min(b,v));
export async function createBancBodyFactory(){
  const [mj,core,xml,metadata,io]=await Promise.all([loadMujoco(),createWasmCore(),
    fetch('/body-model/flybody-mujoco.xml').then(r=>r.text()),fetch('/body-model/flybody-mujoco.json').then(r=>r.json()),fetch('/banc-data/io.json').then(r=>r.json())]);
  const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(xml))),x=>x.toString(16).padStart(2,'0')).join('');
  if(hash!==metadata.xml_sha256)throw new Error('FlyBody model checksum mismatch');
  return (fruit,flies,options)=>new FlyBodyWorld(fruit,flies,options,{mj,core,xml,metadata,io});
}

export class FlyBodyWorld extends BodyWorld{
  constructor(fruit,flies,options,{mj,core,xml,metadata,io}){
    super(fruit.map(f=>({...f,remaining:10})),flies,options);
    Object.assign(this,{mj,metadata,io});this.backend='mujoco-wasm';this.bodies=new Map();
    const scene=flybodyScene(xml,this.habitat);this.model=mj.MjModel.from_xml_string(scene.xml);this.model.hfield_data.set(scene.heights);
    this.mouthLandmarks=createMouthLandmarks(this.model,metadata);
    this.wingLandmarks=createWingLandmarks(this.model,metadata);
    const environment={surface:(x,y)=>this.habitat.surface(x*SCALE,y*SCALE).y/SCALE,
      odor:(x,y,z)=>this.habitat.odor(x*SCALE,z*SCALE,y*SCALE),
      foodForContact:createContactFoodResolver(mj,this.model,this.habitat.fruit,scene.fruitGeomNames),
      foodAt:(x,y)=>{const i=this.habitat.surface(x*SCALE,y*SCALE).fruitIndex;return i>=0?this.habitat.fruit[i]:null;}};
    for(const f of flies){
      const b=new FlyBodyPhysics(mj,this.model,metadata,io,n=>new WasmMuscles(core,n),environment);
      b.place(f.x/SCALE,f.z/SCALE,f.heading);this.bodies.set(f.id,b);f.internal={...b.internal};f.task={phase:'unobserved',events:[]};
      this.copyPose(f,b,0);
    }
  }
  setMovementMode(mode){
    super.setMovementMode(mode);
    if(this.bodies)for(const f of this.flies){const b=this.bodies.get(f.id);b.place(f.x/SCALE,f.z/SCALE,f.heading);}
  }
  advance(seconds){
    if(!Number.isFinite(seconds)||seconds<=0)return;
    if(this.movementMode==='behavior'){super.advance(seconds);return;}
    this.tick(Math.min(seconds,.1));
  }
  tick(dt){
    if(this.movementMode==='behavior'){
      for(const f of this.flies){delete f.physicsQuaternion;delete f.physicsPosition;delete f.physicsLegs;delete f.physicsMouth;delete f.physicsWings;}
      super.tick(dt);
      for(const f of this.flies){
        const b=this.bodies.get(f.id),food=this.habitat.surface(f.x,f.z);
        const pumpIds=this.io.muscles.filter(m=>m.joint==='pump').flatMap(m=>m.indices),rates=this.rates(f);
        const pump=pumpIds.reduce((s,i)=>s+(rates.get(i)||0),0)/Math.max(1,pumpIds.length)/80;
        const source=food.fruitIndex>=0?this.habitat.fruit[food.fruitIndex]:null;
        const intake=source&&f.contact&&f.actuators.proboscis>.05?Math.min(source.remaining,dt*.03*clamp(pump),1-b.internal.crop):0;
        if(source)source.remaining-=intake;
        b.internal.step(dt,intake,f.actuators.forward+f.actuators.wing);
        f.internal={...b.internal};f.task={phase:'assisted behavior',events:[]};
      }
      return;
    }
    for(const f of this.flies){
      if(!(f.brain?.time_ms>0))continue;
      const b=this.bodies.get(f.id),food=this.habitat.fruit.reduce((best,p)=>Math.hypot(p.x-f.x,p.z-f.z)<Math.hypot(best.x-f.x,best.z-f.z)?p:best);
      b.food={x:food.x/SCALE,y:food.z/SCALE,radius:food.radius/SCALE,height:food.y/SCALE,remaining:food.remaining};
      const wasAir=f.airborne;b.step(this.rates(f),dt,{coupling:this.motorCoupling,flight:this.flightEnabled});
      this.copyPose(f,b,dt);
      if(!wasAir&&f.airborne)this.takeoffs++;
    }
    // Contact loss/reacquisition includes bounces and crashes. The displayed
    // landing counter must use the same evidence as the task monitor.
    this.landings=Array.from(this.bodies.values()).reduce((sum,b)=>sum+b.monitor.landingCount,0);
    this.time+=dt;
  }
  copyPose(f,b,dt){
    const previous=f.joints,ground=this.habitat.surface(b.x*SCALE,b.y*SCALE),[w,x,y,z]=b.quaternion;
    Object.assign(f,{x:b.x*SCALE,y:(b.z-b.restHeight)*SCALE,z:b.y*SCALE,heading:b.heading,
      vx:b.vx*SCALE,vy:b.vz*SCALE,vz:b.vy*SCALE,yawVelocity:b.data.qvel[5],airborne:b.airborne,contact:b.onFood,
      bodyTime:b.time,normal:ground.normal,altitude:Math.max(0,(b.z-b.restHeight)*SCALE-ground.y),velocity:Math.hypot(b.vx,b.vy)*SCALE,
      internal:{...b.internal},task:{phase:b.monitor.phase,events:b.monitor.events,flightEvidence:b.monitor.flightEvidence,landings:b.monitor.landingCount,environmentContacts:b.environmentContactCount,
        foodContacts:{legs:Array.from(b.legFoodContact),wings:Array.from(b.wingFoodContact),mouth:Array.from(b.mouthFoodContact)}},muscleState:b.muscleState,jointState:b.jointState,
      physicsQuaternion:[-x,-z,-y,w],physicsBackend:this.backend});
    f.pitch=Math.asin(clamp(2*(w*y-z*x),-1,1));f.bank=Math.atan2(2*(w*x+y*z),1-2*(x*x+y*y));
    // Skin the original two-segment leg meshes to physical limb landmarks.
    // Rotation transpose maps world coordinates into the physical root frame.
    const rot=[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)];
    // Register the original mesh's thorax center to the physical root, including
    // pitch and roll. A world-vertical offset detached the body from its feet.
    f.physicsPosition=[b.x*SCALE-rot[0]*.13-rot[2]*.91,b.z*SCALE-rot[6]*.13-rot[8]*.91,b.y*SCALE-rot[3]*.13-rot[5]*.91];
    f.physicsLegs=this.metadata.leg_bodies.map((ids,leg)=>ids.map((id,k)=>{
      const p=k===2?b.feet[leg]:b.data.xpos.subarray(id*3,id*3+3),dx=p[0]-b.x,dy=p[1]-b.y,dz=p[2]-b.z;
      return [(rot[0]*dx+rot[3]*dy+rot[6]*dz)*SCALE+.13,(rot[2]*dx+rot[5]*dy+rot[8]*dz)*SCALE+.91,(rot[1]*dx+rot[4]*dy+rot[7]*dz)*SCALE];
    }));
    f.physicsMouth=sampleMouthPose(b.data,this.mouthLandmarks,[b.x,b.y,b.z],rot);
    f.physicsWings=sampleWingPose(b.data,this.wingLandmarks,[b.x,b.y,b.z],rot);
    f.actuators={...decodeMotorOutput(f.brain,{connected:this.motorCoupling,flightEnabled:this.flightEnabled}),forward:clamp(f.velocity/10),wing:b.wingPower,wingLeft:b.wingPowerLeft,wingRight:b.wingPowerRight,proboscis:b.proboscis};
    f.wingPhase=b.wingPhase;
    f.antennaPhase=(f.antennaPhase+dt*9*Math.max(f.actuators.antennaLeft,f.actuators.antennaRight))%(Math.PI*2);
    f.joints=jointPose(f);
    // Frozen native antenna joints cannot send proprioception from the old
    // display oscillator. Report deviation from the native rest posture.
    f.joints.antennae=['left','right'].map(side=>{
      const j=b.byJoint.get(`antenna_${side}`);return j?b.data.qpos[j.qpos]-j.neutral:0;
    });
    f.joints.legs=[];
    for(const side of ['left','right'])for(const segment of ['T1','T2','T3'])f.joints.legs.push(['coxa','femur','tibia'].map(name=>{const j=b.byJoint.get(`${name}_${segment}_${side}`);return b.data.qpos[j.qpos]-j.neutral;}));
    f.joints.wings=['left','right'].map((side,k)=>['roll','yaw','pitch'].map(axis=>b.data.qpos[b.byJoint.get(`wing_${axis}_${side}`).qpos]*(k?1:-1)));
    // Wing activation alone cannot distinguish powered flight from a fall or
    // a collision launch. Likewise, surface translation need not be a gait.
    f.motion=f.airborne?(b.wingPower>.05?'airborne_wings':'falling'):1-2*(x*x+y*y)<0?'overturned':b.lastIntake>0?'feeding':f.velocity>.05?'surface_motion':b.wingPower>.05?'wing':'resting';
    f.feedback=senseBody(f,this.habitat,previous,dt);
    f.feedback.antennae=['left','right'].map((side,i)=>{
      const j=b.byJoint.get(`antenna_${side}`);return {angle:f.joints.antennae[i],speed:j?Math.abs(b.data.qvel[j.dof]):0};
    });
    f.feedback.tilt=Math.acos(clamp(1-2*(x*x+y*y),-1,1));
    f.feedback.angularVelocity=Array.from(b.data.qvel.slice(3,6));f.feedback.wingPower=b.wingPower;
    f.feedback.wingPowerLeft=b.wingPowerLeft;f.feedback.wingPowerRight=b.wingPowerRight;
    f.feedback.halterePower=b.halterePower;f.feedback.haltereSteering=b.haltereSteering;
    f.feedback.legFoodContact=b.legFoodContact;f.feedback.wingFoodContact=b.wingFoodContact;f.feedback.mouthFoodContact=b.mouthFoodContact;
    f.feedback.wingPhase=b.wingPhase;f.feedback.wingFrequency=b.wings.config.frequency_hz;
    f.feedback.legs.forEach((leg,i)=>{
      const side=i<3?'left':'right',segment=`T${i%3+1}`,tibia=b.byJoint.get(`tibia_${segment}_${side}`),coxa=b.byJoint.get(`coxa_${segment}_${side}`);
      leg.loadBodyWeights=b.legLoads[i];leg.support=clamp(b.legLoads[i]*6);leg.collision=b.legCollisions[i];
      leg.tibiaAngle=b.data.qpos[tibia.qpos];leg.tibiaVelocity=b.data.qvel[tibia.dof];leg.coxaAngle=b.data.qpos[coxa.qpos];
      leg.vibration=leg.collision*Math.abs(leg.tibiaVelocity);
    });
  }
  rates(f){return new Map(this.io.motor_neurons.map((cell,k)=>[cell.index,f.brain.motorNeuronRates?.[k]||0]));}
  poses(){return super.poses().map(p=>({...p,internal:this.bodies.get(p.id).internal}));}
  dispose(){for(const b of this.bodies.values())b.dispose();this.model.delete();}
}
