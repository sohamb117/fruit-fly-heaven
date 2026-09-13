// Brain readouts and explicitly modeled behavior programs drive body mechanics.
// The direct controller remains available without behavior-program assistance.
const TAU=Math.PI*2, STEP=1/60;
const clamp=(n,lo,hi)=>Math.max(lo,Math.min(hi,n));
const angle=n=>Math.atan2(Math.sin(n),Math.cos(n));
const approach=(n,target,rate,dt)=>n+(target-n)*(1-Math.exp(-rate*dt));

export const BODY_LABELS={waiting:'Waiting for brain',walking:'Walking',turning:'Turning',feeding:'Feeding',proboscis:'Proboscis active',antenna:'Antenna active',wing:'Wings active',landing_pose:'Landing posture',grooming:'Grooming',resting:'Resting',takeoff:'Taking off',flying:'Flying',landing:'Landing',falling:'Falling'};
Object.assign(BODY_LABELS,{airborne_wings:'Airborne · wings active',surface_motion:'Moving on surface',overturned:'Overturned'});
export const isAirborne=f=>f.airborne===true;

// These are declared transfer-function/physics assumptions, not measured muscle
// parameters. The zero-input equilibrium is stationary on a supporting surface.
export const MOTOR_DECODER=Object.freeze({quietHz:2,rangeHz:40,walkSpeed:10,reverseSpeed:7,yawRate:3,
  gravity:30,lift:54,thrust:42,airDrag:1.8,verticalDrag:1.2,jumpSpeed:12,groundResponse:15});
export function decodeMotorOutput(brain,{connected=true,flightEnabled=true}={}){
  const rates=connected&&brain?.time_ms>0?brain.motor||{}:{};
  const drive=key=>clamp(((Number.isFinite(rates[key])?rates[key]:0)-MOTOR_DECODER.quietHz)/MOTOR_DECODER.rangeHz,0,1);
  const wingLeft=flightEnabled?drive('wing_left'):0,wingRight=flightEnabled?drive('wing_right'):0;
  return {forward:drive('forward'),reverse:drive('reverse'),turn:drive('turn_left')-drive('turn_right'),
    wingLeft,wingRight,wing:(wingLeft+wingRight)/2,flightTurn:wingLeft-wingRight,
    jump:flightEnabled?drive('takeoff'):0,landing:drive('landing'),groom:drive('groom'),
    proboscis:drive('proboscis'),antennaLeft:drive('antenna_left'),antennaRight:drive('antenna_right')};
}

// Original habitat controller (commit 1f9fb8b): behavior-associated population
// rates set walking speed and yaw. Steering contributes to forward drive even
// when the walking population is quiet. This is a behavioral mapping, not a
// claim that steering neurons directly innervate forward-walking muscles.
export const BEHAVIOR_DECODER=Object.freeze({walkSpeed:8,steeringToWalk:.15,walkScaleHz:30,feedingScaleHz:35,yawRate:2.8,turnScaleHz:40});
export const FLIGHT_PROGRAM=Object.freeze({preparationSeconds:1.6,maxFlightSeconds:3.5,groundRecoverySeconds:1.5,
  targetHeight:9,heightGain:.05,verticalDamping:.075,minimumDrive:.04});
export function decodeBehaviorOutput(brain,options={}){
  const a=decodeMotorOutput(brain,options),p=BEHAVIOR_DECODER;
  const {connected=true}=options,rates=connected&&brain?.time_ms>0?brain:{};
  const hz=key=>Number.isFinite(rates[key])?Math.max(0,rates[key]):0;
  const walk=hz('walk_hz'),left=hz('left_hz'),right=hz('right_hz'),feed=hz('feed_hz');
  a.forward=p.walkSpeed*Math.tanh((walk+p.steeringToWalk*(left+right))/p.walkScaleHz)/(1+feed/p.feedingScaleHz)/MOTOR_DECODER.walkSpeed;
  a.turn=p.yawRate*Math.tanh((left-right)/p.turnScaleHz)/MOTOR_DECODER.yawRate;
  // Direct reverse, wing, jump, landing and joint outputs remain available.
  // Body-level flight programs are applied separately; brain state is untouched.
  return a;
}

const newFlightProgram=()=>({phase:'grounded',preparation:0,elapsed:0,recovery:0,airborneSeen:false});
function stepFlightProgram(f,a,dt,enabled){
  const p=FLIGHT_PROGRAM,s=f.flightProgram;
  if(!enabled){Object.assign(s,newFlightProgram());return;}
  const drive=clamp(a.forward*4+Math.abs(a.turn)*.4,0,1);
  if(s.phase==='grounded'){
    s.recovery=Math.max(0,s.recovery-dt);
    // Accumulated locomotor output requests a flight bout. Feeding raises the
    // required effort. This state/gain is a body controller, not inferred intent.
    const effort=Math.max(0,drive-.3*a.proboscis);
    if(!f.airborne&&s.recovery===0&&drive>p.minimumDrive)s.preparation=clamp(s.preparation+dt*effort/p.preparationSeconds,0,1);
    else s.preparation=Math.max(0,s.preparation-dt);
    if(s.preparation>=1&&drive>p.minimumDrive){s.phase='takeoff';s.elapsed=0;s.airborneSeen=false;}
  }else{
    s.elapsed+=dt;s.airborneSeen ||= f.airborne;
    if(s.airborneSeen&&!f.airborne){Object.assign(s,newFlightProgram(),{recovery:p.groundRecoverySeconds});return;}
    if(s.elapsed>p.maxFlightSeconds||(s.elapsed>.6&&(drive<=p.minimumDrive||a.landing>.6)))s.phase='landing';
    else if(f.airborne&&s.elapsed>.2)s.phase='cruise';
  }
}
function applyFlightProgram(f,a){
  const s=f.flightProgram,p=FLIGHT_PROGRAM;
  if(!s||s.phase==='grounded')return a;
  // Body feedback controls lift; a brief neural request must not require a
  // continuous, unrealistically high firing rate to keep wings beating.
  const power=s.phase==='landing'?.28:clamp(MOTOR_DECODER.gravity/MOTOR_DECODER.lift+
    (p.targetHeight-f.altitude)*p.heightGain-f.vy*p.verticalDamping,.25,.95);
  a.jump=s.phase==='takeoff'&&!s.airborneSeen?Math.max(a.jump,.55):a.jump;
  a.wingLeft=Math.max(a.wingLeft,power);a.wingRight=Math.max(a.wingRight,power);a.wing=(a.wingLeft+a.wingRight)/2;
  a.flightTurn=clamp(a.turn+a.flightTurn,-1,1);
  a.landing=s.phase==='landing'?Math.max(a.landing,.7):0;
  return a;
}

export function createHabitat(fruit){
  function nearest(f,x,z){
    if(f.kind==='apple')return {x:f.x,z:f.z,d:Math.hypot(x-f.x,z-f.z)};
    let best=Infinity,px=f.x,pz=f.z;
    for(let i=1;i<f.path.length;i++){
      const [ax,az]=f.path[i-1],[bx,bz]=f.path[i],dx=bx-ax,dz=bz-az;
      const t=clamp(((x-ax)*dx+(z-az)*dz)/(dx*dx+dz*dz||1),0,1);
      const qx=ax+t*dx,qz=az+t*dz,d=(x-qx)**2+(z-qz)**2;
      if(d<best){best=d;px=qx;pz=qz;}
    }
    return {x:px,z:pz,d:Math.sqrt(best)};
  }
  function surface(x,z){
    let y=1.5+.0037*(x*x+z*z),fruitIndex=-1,nx=-.0074*x,ny=1,nz=-.0074*z;
    for(let i=0;i<fruit.length;i++){
      const f=fruit[i],p=nearest(f,x,z);
      if(p.d>=f.radius)continue;
      const vertical=f.kind==='apple'?.94:1;
      const rise=Math.sqrt(f.radius*f.radius-p.d*p.d),height=f.y+vertical*rise;
      if(height>y){y=height;fruitIndex=i;nx=x-p.x;ny=Math.max(.01,rise/vertical);nz=z-p.z;}
    }
    const length=Math.hypot(nx,ny,nz);
    return {y,contact:fruitIndex>=0,fruitIndex,normal:[nx/length,ny/length,nz/length]};
  }
  function odor(x,y,z){
    let concentration=0;
    for(const f of fruit){
      if(f.remaining!==undefined&&f.remaining<=0)continue;
      const p=nearest(f,x,z),horizontal=Math.max(0,p.d-f.radius);
      const vertical=Math.max(0,y-f.y-f.radius*(f.kind==='apple'?.94:1));
      concentration+=.3*Math.exp(-Math.hypot(horizontal,vertical)/18);
    }
    return Math.min(1,concentration);
  }
  return {surface,odor,fruit,ceiling:Math.max(...fruit.map(f=>f.y+f.radius))+30};
}

export function sensoryRates(f,habitat,{odor=true,taste=true}={}){
  const c=Math.cos(f.heading),s=Math.sin(f.heading);
  return [
    odor?3+65*habitat.odor(f.x+.9*c-.4*s,f.y,f.z+.9*s+.4*c):0,
    odor?3+65*habitat.odor(f.x+.9*c+.4*s,f.y,f.z+.9*s-.4*c):0,
    taste&&f.contact?150:0,
  ];
}

// One joint pose drives both the mesh and the sensory adapter. These are the
// existing kinematic joints, not a newly inferred muscle or nerve-cord model.
export function jointPose(f){
  const a=f.actuators,air=isAirborne(f),walking=f.motion==='walking'||f.motion==='turning';
  const legs=[],wings=[],antennae=[];
  for(const side of [-1,1]){
    for(let k=0;k<3;k++){
      const phase=f.gait+(k%2)*Math.PI+(side>0?Math.PI:0);
      legs.push([air?side*.65*(1-a.landing):0,walking?Math.cos(phase)*.25:0,walking?Math.sin(phase)*.32:0]);
      const leg=legs.at(-1);
      if(k===0){leg[0]+=side*.45*a.groom;leg[1]+=Math.sin(f.groomPhase)*.45*a.groom;leg[2]-=.85*a.groom;}
      leg[2]-=a.landing*.4;
    }
    const wing=side<0?a.wingLeft:a.wingRight,antenna=side<0?a.antennaLeft:a.antennaRight;
    wings.push([side*(.35+Math.sin(f.wingPhase)*.8)*wing,side*(.12-.77*wing),0]);
    antennae.push(side*(.25+Math.sin(f.antennaPhase)*.15)*antenna);
  }
  return {legs,wings,antennae,proboscis:.15+(.29+Math.sin(f.feedPhase)*.07)*a.proboscis};
}

export function senseBody(f,habitat,previous=null,dt=0){
  const joints=f.joints,rate=(a,b)=>dt>0&&b!==undefined?Math.abs(a-b)/dt:0;
  const legs=joints.legs.map((rotation,i)=>({angle:Math.hypot(...rotation),
    speed:previous?Math.hypot(...rotation.map((n,j)=>rate(n,previous.legs[i][j]))):0,
    // Kinematic foot support proxy: swing legs unload, airborne legs have none.
    support:f.airborne?0:Math.max(0,1-Math.max(0,rotation[2])/.32)}));
  const touch=[];
  for(const side of [-1,1]){
    const x=f.x+1.3*Math.cos(f.heading)-side*.55*Math.sin(f.heading),z=f.z+1.3*Math.sin(f.heading)+side*.55*Math.cos(f.heading);
    touch.push(clamp((habitat.surface(x,z).y-(f.y+.6))/.5,0,1));
  }
  return {legs,touch,antennae:joints.antennae.map((n,i)=>({angle:n,speed:previous?rate(n,previous.antennae[i]):0})),
    speed:Math.hypot(f.vx,f.vy,f.vz),yaw:f.yawVelocity||0,tilt:Math.acos(clamp(f.airborne?Math.cos(f.pitch)*Math.cos(f.bank):f.normal[1],-1,1)),
    impact:f.impact||0,airborne:f.airborne};
}

// Worker output must never overwrite newer body poses with a stale sampled pose.
export function applyNeuralOutput(fly,output){
  if(!Number.isFinite(output.brain?.time_ms)||output.brain.time_ms<(fly.brain?.time_ms||0))return;
  fly.brain=output.brain;fly.senses=output.senses;fly.sensory=output.sensory;
  if(output.retina)fly.sensoryFrame=output.retina;
}

export class BodyWorld {
  constructor(fruit,flies,{flightEnabled=true,motorCoupling=true,movementMode='direct'}={}){
    this.habitat=createHabitat(fruit);this.flies=flies;this.flightEnabled=flightEnabled;this.motorCoupling=motorCoupling;
    this.setMovementMode(movementMode);
    this.time=0;this.accumulator=0;this.takeoffs=0;this.landings=0;
    for(const f of flies){
      const ground=this.habitat.surface(f.x,f.z);
      Object.assign(f,{y:ground.y,heading:angle(f.heading),contact:ground.contact,normal:ground.normal,motion:'waiting',velocity:0,vx:0,vy:0,vz:0,
        pitch:0,bank:0,gait:0,wingPhase:0,groomPhase:0,feedPhase:0,antennaPhase:0,bodyTime:0,altitude:0,
        airborne:false,lastJump:0,flightProgram:newFlightProgram(),actuators:decodeMotorOutput(null)});
      f.joints=jointPose(f);f.feedback=senseBody(f,this.habitat);
    }
  }
  setFlightEnabled(enabled){this.flightEnabled=enabled;}
  setMotorCoupling(enabled){this.motorCoupling=enabled;}
  setMovementMode(mode){
    if(mode!=='behavior'&&mode!=='direct')throw new RangeError('Unknown movement mode: '+mode);
    if(this.movementMode!==mode)for(const f of this.flies)if(f.flightProgram)f.flightProgram=newFlightProgram();
    this.movementMode=mode;
  }
  readCommands(brain,fly=null){
    const decode=this.movementMode==='behavior'?decodeBehaviorOutput:decodeMotorOutput;
    const a=decode(brain,{connected:this.motorCoupling,flightEnabled:this.flightEnabled});
    return fly&&this.movementMode==='behavior'&&this.motorCoupling&&this.flightEnabled?applyFlightProgram(fly,a):a;
  }
  advance(seconds){
    if(!Number.isFinite(seconds)||seconds<=0)return;
    this.accumulator+=Math.min(seconds,.25);
    while(this.accumulator+1e-10>=STEP){this.tick(STEP);this.accumulator-=STEP;}
  }
  tick(dt){
    const p=MOTOR_DECODER;
    this.time+=dt;
    for(const f of this.flies){
      if(!(f.brain?.time_ms>0))continue;
      let a=this.readCommands(f.brain);
      if(this.movementMode==='behavior'){
        stepFlightProgram(f,a,dt,this.motorCoupling&&this.flightEnabled);
        if(this.motorCoupling&&this.flightEnabled)a=applyFlightProgram(f,a);
      }
      f.actuators=a;
      const oldX=f.x,oldY=f.y,oldZ=f.z,oldHeading=f.heading,wasAirborne=f.airborne;
      const previousJoints=f.joints;f.impact=(f.impact||0)*Math.exp(-dt/.1);
      f.bodyTime+=dt;
      const lift=p.lift*a.wing*(1-.9*a.landing);
      // A jump needs a rising neural command. Holding the command cannot schedule
      // further jumps. Sustained wing force can lift off when it exceeds gravity.
      if(!f.airborne){
        if(a.jump>.15&&f.lastJump<=.15){f.vy=p.jumpSpeed*a.jump;f.airborne=true;}
        else if(lift>p.gravity){f.vy=(lift-p.gravity)*dt;f.airborne=true;}
        if(f.airborne){this.takeoffs++;f.contact=false;}
      }
      f.lastJump=a.jump;
      if(f.airborne){
        f.heading=angle(f.heading+a.flightTurn*p.yawRate*dt);
        f.vx+=(p.thrust*a.wing*Math.cos(f.heading)-p.airDrag*f.vx)*dt;
        f.vz+=(p.thrust*a.wing*Math.sin(f.heading)-p.airDrag*f.vz)*dt;
        f.vy+=(lift-p.gravity-p.verticalDrag*f.vy)*dt;
      }else{
        f.heading=angle(f.heading+a.turn*p.yawRate*dt);
        const speed=p.walkSpeed*a.forward-p.reverseSpeed*a.reverse;
        f.vx=approach(f.vx,Math.cos(f.heading)*speed,p.groundResponse,dt);
        f.vz=approach(f.vz,Math.sin(f.heading)*speed,p.groundResponse,dt);
        if(!speed&&Math.hypot(f.vx,f.vz)<.001)f.vx=f.vz=0;
      }
      let x=f.x+f.vx*dt,z=f.z+f.vz*dt,y=f.y+(f.airborne?f.vy*dt:0);
      const r=Math.hypot(x,z);
      if(r>61){
        x*=61/r;z*=61/r;
        const outward=(f.vx*x+f.vz*z)/61;
        if(outward>0){f.vx-=outward*x/61;f.vz-=outward*z/61;}
      }
      let ground=this.habitat.surface(x,z);
      // Surface contact blocks steep uphill penetration; it never turns the fly
      // or chooses a path. Leaving an unsupported edge causes a physical fall.
      if(ground.y>Math.max(f.y,y)+.6){x=f.x;z=f.z;f.vx=f.vz=0;ground=this.habitat.surface(x,z);}
      if(!f.airborne&&ground.y<f.y-.6){f.airborne=true;f.vy=0;}
      else if(!f.airborne)y=ground.y;
      if(f.airborne&&y<=ground.y&&f.vy<=0){
        f.impact=Math.abs(f.vy);y=ground.y;f.vy=0;f.airborne=false;this.landings++;
      }
      if(y>this.habitat.ceiling){y=this.habitat.ceiling;f.vy=Math.min(0,f.vy);}
      f.x=x;f.y=Math.max(y,ground.y);f.z=z;f.normal=ground.normal;
      f.altitude=Math.max(0,f.y-ground.y);f.contact=!f.airborne&&ground.contact;
      f.velocity=Math.hypot(f.x-oldX,f.z-oldZ)/dt;
      f.gait+=Math.hypot(f.x-oldX,f.z-oldZ)*3.6+Math.abs(angle(f.heading-oldHeading))*1.5;
      f.wingPhase=(f.wingPhase+dt*TAU*37*a.wing)%TAU;
      f.groomPhase=(f.groomPhase+dt*22*a.groom)%TAU;
      f.feedPhase=(f.feedPhase+dt*18*a.proboscis)%TAU;
      f.antennaPhase=(f.antennaPhase+dt*9*Math.max(a.antennaLeft,a.antennaRight))%TAU;
      f.pitch=approach(f.pitch,f.airborne?clamp((f.y-oldY)/dt*.018,-.3,.3):0,7,dt);
      f.bank=approach(f.bank,f.airborne?clamp(a.flightTurn*.5,-.5,.5):0,7,dt);
      if(f.airborne)f.motion=!wasAirborne?'takeoff':a.landing>.05?'landing':a.wing>.05?'flying':'falling';
      else f.motion=f.velocity>.02?'walking':Math.abs(a.turn)>.01?'turning':a.groom>.01?'grooming':a.proboscis>.01?(f.contact?'feeding':'proboscis'):a.wing>.01?'wing':a.landing>.01?'landing_pose':Math.max(a.antennaLeft,a.antennaRight)>.01?'antenna':'resting';
      f.yawVelocity=angle(f.heading-oldHeading)/dt;
      f.joints=jointPose(f);f.feedback=senseBody(f,this.habitat,previousJoints,dt);
    }
  }
  poses(){return this.flies.map(f=>({id:f.id,x:f.x,y:f.y,z:f.z,heading:f.heading,contact:f.contact,bodyTime:f.bodyTime,feedback:f.feedback}));}
}
