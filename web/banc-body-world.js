import {BodyWorld,jointPose,senseBody,decodeMotorOutput} from './body-world.js';
import {ReducedBody} from './banc/embodiment.js';
import {createWasmCore,WasmMuscles,WasmJoints} from '/banc-engine/src/index.js';

// The renderer's fly is about 2.3 world units long; the teacher is ~0.23 cm.
// Preserve scene dimensions while converting the distilled mechanics from cm.
const SCALE=10,clamp=(v,a,b)=>Math.max(a,Math.min(b,v));

export async function createBancBodyFactory(){
  const [core,model,io]=await Promise.all([createWasmCore(),fetch('/body-model/flybody-reduced.json').then(r=>r.json()),fetch('/banc-data/io.json').then(r=>r.json())]);
  const runtime={createMuscles:n=>new WasmMuscles(core,n),createJoints:j=>new WasmJoints(core,j)};
  return (fruit,flies,options)=>new BancBodyWorld(fruit,flies,options,{model,io,runtime});
}

export class BancBodyWorld extends BodyWorld{
  constructor(fruit,flies,options,{model,io,runtime}){
    // Each restarted population gets independent food and metabolic state.
    super(fruit.map(f=>({...f,remaining:10})),flies,options);
    this.model=model;this.io=io;this.bodies=new Map();
    const environment={surface:(x,y)=>this.habitat.surface(x*SCALE,y*SCALE).y/SCALE,
      odor:(x,y,z)=>this.habitat.odor(x*SCALE,z*SCALE,y*SCALE),
      foodAt:(x,y)=>{const s=this.habitat.surface(x*SCALE,y*SCALE);return s.fruitIndex>=0?this.habitat.fruit[s.fruitIndex]:null;}};
    for(const f of flies){
      const body=new ReducedBody(model,io,runtime,{environment});
      body.x=f.x/SCALE;body.y=f.z/SCALE;body.heading=f.heading;
      body.settleOnSurface();
      body.restHeight=-Math.min(...body.feet.map(p=>p[2]));
      this.bodies.set(f.id,body);f.internal={...body.internal};f.task={phase:'unobserved',events:[]};
    }
  }
  setMovementMode(mode){
    super.setMovementMode(mode);
    if(this.bodies)for(const f of this.flies){
      const b=this.bodies.get(f.id);b.x=f.x/SCALE;b.y=f.z/SCALE;b.z=f.y/SCALE+b.restHeight;
      b.vx=f.vx/SCALE;b.vy=f.vz/SCALE;b.vz=f.vy/SCALE;b.heading=f.heading;b.airborne=f.airborne;
      if(mode==='direct'){f.pitch=0;f.bank=0;}
    }
  }
  tick(dt){
    if(this.movementMode==='behavior'){
      super.tick(dt);
      for(const f of this.flies){
        const b=this.bodies.get(f.id),food=this.habitat.surface(f.x,f.z);
        const pumpIds=this.io.muscles.filter(m=>m.joint==='pump').flatMap(m=>m.indices);
        const rates=this.rates(f),pump=pumpIds.reduce((s,i)=>s+(rates.get(i)||0),0)/Math.max(1,pumpIds.length)/80;
        const source=food.fruitIndex>=0?this.habitat.fruit[food.fruitIndex]:null;
        const intake=source&&f.contact&&f.actuators.proboscis>.05?Math.min(source.remaining,dt*.03*clamp(pump,0,1),1-b.internal.crop):0;
        if(source)source.remaining-=intake;
        b.internal.step(dt,intake,f.actuators.forward+f.actuators.wing);
        f.internal={...b.internal};f.task={phase:'assisted behavior',events:[]};
      }
      return;
    }
    this.time+=dt;
    for(const f of this.flies){
      if(!(f.brain?.time_ms>0))continue;
      const b=this.bodies.get(f.id),previousJoints=f.joints,oldX=f.x,oldZ=f.z,wasAirborne=f.airborne;
      // The monitor gets a nearest food position for observations only.
      const food=this.habitat.fruit.reduce((best,p)=>Math.hypot(p.x-f.x,p.z-f.z)<Math.hypot(best.x-f.x,best.z-f.z)?p:best);
      b.food={x:food.x/SCALE,y:food.z/SCALE,radius:food.radius/SCALE,height:food.y/SCALE,remaining:food.remaining};
      b.step(this.rates(f),dt,{coupling:this.motorCoupling,flight:this.flightEnabled});
      let r=Math.hypot(b.x,b.y);
      if(r>6.1){b.x*=6.1/r;b.y*=6.1/r;const outward=(b.vx*b.x+b.vy*b.y)/6.1;if(outward>0){b.vx-=outward*b.x/6.1;b.vy-=outward*b.y/6.1;}}
      const ground=this.habitat.surface(b.x*SCALE,b.y*SCALE);
      if(b.z*SCALE-b.restHeight*SCALE>this.habitat.ceiling){b.z=this.habitat.ceiling/SCALE+b.restHeight;b.vz=Math.min(0,b.vz);}
      Object.assign(f,{x:b.x*SCALE,y:Math.max(ground.y,(b.z-b.restHeight)*SCALE),z:b.y*SCALE,heading:b.heading,
        vx:b.vx*SCALE,vy:b.vz*SCALE,vz:b.vy*SCALE,yawVelocity:b.yaw,airborne:b.airborne,contact:b.onFood,
        bodyTime:b.time,normal:ground.normal,altitude:Math.max(0,(b.z-b.restHeight)*SCALE-ground.y),
        internal:{...b.internal},task:{phase:b.monitor.phase,events:b.monitor.events},muscleState:b.muscleState,jointState:b.jointState});
      f.velocity=Math.hypot(f.x-oldX,f.z-oldZ)/dt;
      if(!wasAirborne&&f.airborne)this.takeoffs++;if(wasAirborne&&!f.airborne)this.landings++;
      // Antennae retain the original kinematic boundary driven by their BANC
      // motor populations; leg and wing mechanics use the distilled muscles.
      f.actuators={...decodeMotorOutput(f.brain,{connected:this.motorCoupling,flightEnabled:this.flightEnabled}),forward:clamp(f.velocity/10,0,1),wing:b.wingPower,wingLeft:b.wingPowerLeft,wingRight:b.wingPowerRight,proboscis:b.proboscis};
      f.wingPhase=b.wingPhase;
      f.antennaPhase=(f.antennaPhase+dt*9*Math.max(f.actuators.antennaLeft,f.actuators.antennaRight))%(Math.PI*2);
      f.motion=f.airborne?(wasAirborne?'flying':'takeoff'):b.internal.ingested>0&&b.mouthContact&&b.pump>.05?'feeding':b.mouthContact&&b.proboscis>.05?'proboscis':f.velocity>.02?'walking':'resting';
      // Render and sense the very joints advanced by the distilled mechanics.
      f.joints=jointPose(f);
      f.joints.legs=[];
      for(const side of ['left','right'])for(const segment of ['T1','T2','T3']){
        f.joints.legs.push(['coxa','femur','tibia'].map(name=>{
          const i=b.jointIndex.get(`${name}_${segment}_${side}`);return b.jointState[i*2]-this.model.active_joints[i].neutral;
        }));
      }
      f.feedback=senseBody(f,this.habitat,previousJoints,dt);
    }
  }
  rates(f){return new Map(this.io.motor_neurons.map((cell,k)=>[cell.index,f.brain.motorNeuronRates?.[k]||0]));}
  poses(){return super.poses().map(p=>({...p,internal:this.bodies.get(p.id).internal}));}
  dispose(){for(const b of this.bodies.values())b.dispose();}
}
