// Research interface parameters. These bounds are engineering search limits,
// not measurements of Drosophila physiology. The default reproduces v1.
export const MOTOR_INTERFACE_VERSION='banc-motor-interface-v2';
export const MOTOR_INTERFACE_PARAMETERS=Object.freeze([
  {name:'dlmRateHz',initial:80,min:1,max:200,unit:'Hz',group:'rate'},
  {name:'dvmRateHz',initial:80,min:1,max:200,unit:'Hz',group:'rate'},
  {name:'steeringRateHz',initial:80,min:1,max:300,unit:'Hz',group:'rate'},
  {name:'powerRiseSeconds',initial:.015,min:.005,max:.5,unit:'s',group:'kinetics'},
  {name:'powerFallSeconds',initial:.04,min:.01,max:2,unit:'s',group:'kinetics'},
  {name:'deploymentSeconds',initial:.012,min:.005,max:.12,unit:'s',group:'deployment'},
  {name:'deploymentThreshold',initial:.85,min:.5,max:.95,unit:'fraction',group:'deployment'},
  {name:'deploymentPowerSpan',initial:0,min:0,max:.5,unit:'fraction',group:'deployment'},
  {name:'foldedServoScale',initial:1,min:0,max:1,unit:'multiplier',group:'servo'},
  {name:'activeServoScale',initial:1,min:.25,max:1.5,unit:'multiplier',group:'servo'},
  {name:'frontGripScale',initial:1,min:0,max:1.5,unit:'multiplier',group:'adhesion'},
  {name:'middleGripScale',initial:1,min:0,max:1.5,unit:'multiplier',group:'adhesion'},
  {name:'hindGripScale',initial:1,min:0,max:1.5,unit:'multiplier',group:'adhesion'},
].map(Object.freeze));
export const DEFAULT_MOTOR_INTERFACE=Object.freeze(Object.fromEntries(MOTOR_INTERFACE_PARAMETERS.map(p=>[p.name,p.initial])));

export function validateMotorInterface(value={}){
  if(!value||typeof value!=='object'||Array.isArray(value))throw new Error('Motor interface must be an object');
  const known=new Map(MOTOR_INTERFACE_PARAMETERS.map(p=>[p.name,p]));
  for(const name of Object.keys(value))if(!known.has(name))throw new Error('Unknown motor interface parameter: '+name);
  const profile={...DEFAULT_MOTOR_INTERFACE,...value};
  for(const [name,p] of known){const n=profile[name];if(!Number.isFinite(n)||n<p.min||n>p.max)throw new Error('Motor interface parameter outside bounds: '+name);}
  return Object.freeze(profile);
}

export function motorInterfaceBuffers(mappings,value={}){
  const profile=validateMotorInterface(value),rates=new Float32Array(mappings.length).fill(80),forces=new Float32Array(mappings.length).fill(1),kinetics=new Float32Array(mappings.length*4);
  for(let i=0;i<mappings.length;i++){
    const m=mappings[i];kinetics.set([.015,.04,.08,.03],i*4);
    if(m.kind==='asynchronous_wing'){
      if(m.target==='dorsal_longitudinal_muscle')rates[i]=profile.dlmRateHz;
      else if(m.target==='dorsoventral_muscle')rates[i]=profile.dvmRateHz;
      else throw new Error('Unidentified asynchronous wing muscle: '+m.target);
      kinetics[i*4]=profile.powerRiseSeconds;kinetics[i*4+1]=profile.powerFallSeconds;
    }else if(m.kind==='wing_steering_assumption')rates[i]=profile.steeringRateHz;
    else if(m.kind==='claw_grip_assumption'){
      const segment=/^adhere_claw_T([123])_(left|right)$/.exec(m.joint)?.[1];
      if(!segment)throw new Error('Unidentified claw actuator: '+m.joint);
      forces[i]=profile[{1:'frontGripScale',2:'middleGripScale',3:'hindGripScale'}[segment]];
    }
  }
  return {profile,rates,forces,kinetics};
}

// Called only before an episode, never as an attitude/phase feedback controller.
// Mutates one body's decoder and its private wing config, not the shared model.
export function configureMotorInterface(body,profile,createConfiguredMuscles){
  if(body.time!==0||body.data.time!==0)throw new Error('Motor interface is immutable during an episode');
  if(body.motorInterface)throw new Error('Motor interface is already configured');
  const buffers=motorInterfaceBuffers(body.mappings,profile);
  const muscles=createConfiguredMuscles(buffers.kinetics);
  body.muscles.dispose();body.muscles=muscles;
  body.motorInterface=buffers.profile;body.motorRateFullHz=buffers.rates;body.motorForceScale=buffers.forces;
  body.wings.config={...body.wings.config,deployment_tau_s:buffers.profile.deploymentSeconds,deployment_before_beating:buffers.profile.deploymentThreshold,deployment_power_span:buffers.profile.deploymentPowerSpan};
  body.wings.servoScale={folded:buffers.profile.foldedServoScale,active:buffers.profile.activeServoScale};
  return buffers.profile;
}
