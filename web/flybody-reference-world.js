// Optional published-controller evidence in the original habitat renderer.
// This does not connect BANC motor outputs to the reference fly's actuators.
import {BodyWorld,jointPose,senseBody} from './body-world.js';
import {FlyBodyFlightReference} from './flybody-flight-reference.js';
import {createBoundedFlightTrajectory} from './flybody-reference-trajectory.js';
import {createWingLandmarks,sampleWingPose} from './flybody-wing-pose.js';
import loadMujoco from '/body-engine/mujoco.js';

const SCALE=10,TAU=2*Math.PI;
const clamp=(x,a=-1,b=1)=>Math.max(a,Math.min(b,x));
const clock=()=>performance.now();

async function get(url,json=false){
  const response=await fetch(url);
  if(!response.ok)throw new Error(`Cannot load flight reference: ${url} (${response.status})`);
  return json?response.json():response.text();
}

export async function createFlyBodyReferenceFactory({trajectory='brake-hover'}={}){
  const [mj,xml,metadata,policy]=await Promise.all([
    loadMujoco(),get('/body-model/flybody-flight-reference.xml'),
    get('/body-model/flybody-flight-reference.json',true),
    get('/body-model/flybody-flight-policy.json',true),
  ]);
  const digest=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(xml));
  const hash=Array.from(new Uint8Array(digest),x=>x.toString(16).padStart(2,'0')).join('');
  if(hash!==metadata.source.xml_sha256)throw new Error('Published FlyBody reference model checksum mismatch');
  const taskMetadata=trajectory==='straight'?metadata:createBoundedFlightTrajectory(metadata,{kind:'brake-hover',durationSeconds:3600,brakingSeconds:.3});
  return (fruit,flies,options)=>new FlyBodyReferenceWorld(fruit,flies,options,{mj,xml,metadata:taskMetadata,policy});
}

export class FlyBodyReferenceWorld extends BodyWorld{
  constructor(fruit,flies,options,assets){
    if(flies.length!==1)throw new Error('Published flight reference supports exactly one fly');
    super(fruit,flies,options);
    this.backend='mujoco-wasm-flybody-reference';
    this.reference=true;this.complete=false;this.accumulator=0;
    this.controller=new FlyBodyFlightReference(assets.mj,assets.xml,assets.metadata,assets.policy);
    this.model=this.controller.model;this.data=this.controller.data;this.metadata=assets.metadata;
    this.wingLandmarks=createWingLandmarks(this.model,this.metadata);
    this.maxWorkMilliseconds=8;this.maxControlStepsPerAdvance=64;
    const start=this.controller.sample().position,f=flies[0];
    const sceneTop=Math.max(this.habitat.surface(f.x,f.z).y,...fruit.map(p=>p.y+p.radius));
    // Translation is for rendering only. Native task coordinates and its
    // prescribed trajectory remain untouched. Fly above the visible fruit:
    // the upstream task intentionally has no habitat/floor collisions.
    this.renderOrigin=[f.x-start[0]*SCALE,sceneTop+3,f.z-start[1]*SCALE];
    this.referenceNote=this.metadata.prescribedTrajectory?'Published FlyBody policy brakes into a bounded hover using wing forces. BANC does not control this fly. Habitat is a visual backdrop; original task floor contacts are disabled.':'Published FlyBody policy follows a prescribed straight airborne trajectory. BANC does not control this fly. Habitat is a visual backdrop; original task floor contacts are disabled.';
    const bodyIds=new Map(this.metadata.bodies.map(b=>[b.name,b.id]));
    this.legBodies=[];
    for(const side of ['left','right'])for(const segment of ['T1','T2','T3']){
      const ids=['coxa','femur','claw'].map(part=>bodyIds.get(`walker/${part}_${segment}_${side}`));
      if(ids.some(id=>id===undefined))throw new Error('Reference body is missing leg landmarks');
      this.legBodies.push(ids);
    }
    this.wingJoints=['left','right'].map(side=>['roll','yaw','pitch'].map(axis=>{
      const joint=this.metadata.joints.find(j=>j.name===`walker/wing_${axis}_${side}`);
      if(!joint)throw new Error('Reference body is missing wing joint');
      return joint;
    }));
    this.copyPose(0);
  }

  setMotorCoupling(enabled){
    super.setMotorCoupling(enabled);
    if(!enabled&&this.controller)this.controller.data.ctrl.fill(0);
  }
  setFlightEnabled(enabled){
    super.setFlightEnabled(enabled);
    if(!enabled&&this.controller)this.controller.data.ctrl.fill(0);
  }
  readCommands(_brain,fly=this.flies[0]){return fly.actuators;}

  advance(seconds){
    if(!Number.isFinite(seconds)||seconds<0||this.complete)return;
    // Preserve all requested simulation time. Work is bounded per animation
    // frame; a busy browser slows the simulation rather than skipping physics.
    this.accumulator+=seconds;
    const dt=this.metadata.control_timestep,started=clock(),before=this.time;
    let steps=0;
    while(this.accumulator+1e-12>=dt&&!this.controller.done){
      this.controller.step(undefined,{actuated:this.motorCoupling&&this.flightEnabled});
      this.accumulator=Math.max(0,this.accumulator-dt);steps++;
      if(steps>=this.maxControlStepsPerAdvance||clock()-started>=this.maxWorkMilliseconds)break;
    }
    this.time=this.controller.data.time;
    this.complete=this.controller.done;
    if(steps||this.complete)this.copyPose(this.time-before);
  }

  copyPose(dt){
    const c=this.controller,d=c.data,f=this.flies[0],previous=f.joints;
    const root=c.rootBody,p=d.xpos.subarray(root*3,root*3+3),q=d.xquat.subarray(root*4,root*4+4);
    const r=d.xmat.subarray(root*9,root*9+9),[w,x,y,z]=q,o=this.renderOrigin;
    const world=[o[0]+p[0]*SCALE,o[1]+p[2]*SCALE,o[2]+p[1]*SCALE];
    f.physicsWings=sampleWingPose(d,this.wingLandmarks,p,r);
    const ground=this.habitat.surface(world[0],world[2]);
    const velocity=d.qvel.subarray(0,3),power=this.motorCoupling&&this.flightEnabled?1:0;
    Object.assign(f,{
      x:world[0],y:world[1]-.91,z:world[2],
      vx:velocity[0]*SCALE,vy:velocity[2]*SCALE,vz:velocity[1]*SCALE,
      heading:Math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)),
      pitch:Math.asin(clamp(2*(w*y-z*x))),bank:Math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),
      yawVelocity:d.qvel[5],bodyTime:this.time,velocity:Math.hypot(velocity[0],velocity[1])*SCALE,
      airborne:true,contact:false,normal:ground.normal,altitude:Math.max(0,world[1]-.91-ground.y),
      physicsQuaternion:[-x,-z,-y,w],physicsBackend:this.backend,
      physicsPosition:[world[0]-r[0]*.13-r[2]*.91,world[1]-r[6]*.13-r[8]*.91,world[2]-r[3]*.13-r[5]*.91],
      controllerSource:'Published FlyBody policy · not BANC',
      task:{phase:this.complete?'reference complete':power?'reference flight':'reference unactuated',events:[]},
      motion:power?'flying':'falling',
    });
    f.physicsLegs=this.legBodies.map(ids=>ids.map(id=>{
      const dx=d.xpos[id*3]-p[0],dy=d.xpos[id*3+1]-p[1],dz=d.xpos[id*3+2]-p[2];
      return [(r[0]*dx+r[3]*dy+r[6]*dz)*SCALE+.13,
        (r[2]*dx+r[5]*dy+r[8]*dz)*SCALE+.91,
        (r[1]*dx+r[4]*dy+r[7]*dz)*SCALE];
    }));
    // Wing values only drive renderer motion/blur here. They are not inferred
    // muscle activation or BANC firing rates.
    f.actuators={forward:0,reverse:0,turn:0,wing:power,wingLeft:power,wingRight:power,
      flightTurn:0,jump:0,landing:0,groom:0,proboscis:0,antennaLeft:0,antennaRight:0};
    const beat=c.wingbeat,phases=beat.config.trajectories[beat.frequency_index].phase;
    f.wingPhase=TAU*(phases[beat.step]%1);
    f.joints=jointPose(f);
    f.joints.wings=this.wingJoints.map((joints,side)=>joints.map(j=>d.qpos[j.qpos]*(side?1:-1)));
    f.feedback=senseBody(f,this.habitat,previous,dt);
    Object.assign(f.feedback,{tilt:Math.acos(clamp(r[8])),angularVelocity:Array.from(d.qvel.subarray(3,6)),
      wingPower:power,wingPhase:f.wingPhase,wingFrequency:beat.frequency,
      controllerSource:f.controllerSource,reference:true});
    this.diagnostics={...c.sample(),complete:this.complete,pendingSeconds:this.accumulator,
      actuated:!!power,renderOrigin:[...this.renderOrigin]};
  }
  dispose(){this.controller.dispose();}
}
