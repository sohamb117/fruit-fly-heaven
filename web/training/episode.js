// Pure episode/parameter rules. They observe outcomes and never issue actions.
export const PARAMETER_NAMES=['synapse_exc_log_gain','synapse_inh_log_gain','motor_wing_log_leak','motor_leg_log_leak','motor_probe_log_leak','motor_grip_log_leak','descending_log_leak','sensory_taste_log_gain','sensory_odor_log_gain','sensory_body_log_gain','muscle_wing_log_gain','muscle_leg_log_gain','muscle_probe_log_gain','muscle_grip_log_gain'];
export const STAGES=['posture','approach','feeding','flight','landing','sequence'];
export const CRITERIA=Object.freeze({maximumRadiusCm:6.5,minimumHeightCm:-.5,maximumAngularSpeed:300,overturnedSeconds:.1,postureUp:.866025403784,postureAngularSpeed:5,postureSupportedFraction:.8,postureFinalStableSeconds:.25,feedingAmount:.001,flightQualifiedSeconds:.1,approachDistanceCm:.23,sequence:['localization','approach','landing','feeding','takeoff','flight']});
export const clip=(v,a,b)=>Math.max(a,Math.min(b,v));
export function seededRandom(seed){let value=seed>>>0;return()=>{value+=0x6D2B79F5;let t=value;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return((t^(t>>>14))>>>0)/4294967296;};}
export function parameterValues(config,parameters){
 if(config.schemaVersion!==1||config.dtMs!==.5||config.bodyBlockMs!==2||config.vision!==false)throw new Error('Unsupported training configuration');
 if(config.parameters?.length!==PARAMETER_NAMES.length||!config.parameters.every((p,i)=>p.name===PARAMETER_NAMES[i]&&Number.isFinite(p.min)&&Number.isFinite(p.max)&&p.min<=p.max))throw new Error('Unknown training parameter contract');
 if(!Array.isArray(parameters)&&!(parameters instanceof Float32Array)&&!(parameters instanceof Float64Array))throw new Error('Training parameters must be a numeric vector');
 const vector=Array.from(parameters);if(vector.length!==config.parameters.length)throw new Error('Training parameter count mismatch');
 const gains={};config.parameters.forEach((p,i)=>{const v=vector[i];if(!Number.isFinite(v)||v<p.min||v>p.max)throw new Error('Training parameter outside bounds: '+p.name);gains[p.name]=Math.exp(v);});
 return {vector,gains};
}
export function muscleFamily(mapping){
 if(mapping.kind==='leg')return'leg';if(mapping.kind==='claw_grip_assumption')return'grip';
 if(mapping.kind.includes('wing')||mapping.kind.includes('haltere'))return'wing';
 if(mapping.joint==='proboscis'||mapping.kind==='pump')return'probe';return null;
}
export function motorFamilies(io){
 const families=new Map();for(const m of io.muscles){const family=muscleFamily(m);if(!family)continue;for(const index of m.indices){if(families.has(index)&&families.get(index)!==family)throw new Error('Overlapping trained motor families');families.set(index,family);}}
 return families;
}
export function parameterizedModel(base,gains,externalIndices){
 const families=motorFamilies(base.io),external=new Set(externalIndices),params=base.params.slice(),edges=base.edges.slice(),weights=new Float32Array(edges.buffer,edges.byteOffset,edges.length),counts={wing:0,leg:0,probe:0,grip:0,descending:0};
 for(const[index,family]of families){if(external.has(index))throw new Error('Motor physiology parameter overlaps external sensory input');params[index*16+1]*=gains[`motor_${family}_log_leak`];counts[family]++;}
 // Match the pinned prepared descending profile. A changed profile is a new
 // model contract, not permission to guess an anatomical class.
 for(let i=0;i<base.manifest.neuron_count;i++)if(base.params[i*16]===30&&base.params[i*16+1]===1.5&&base.params[i*16+3]===-43&&base.params[i*16+6]===120){
  if(external.has(i)||families.has(i))throw new Error('Descending physiology parameter overlaps another input/family');params[i*16+1]*=gains.descending_log_leak;counts.descending++;
 }
 if(counts.descending!==base.manifest.physiology_profiles.descending)throw new Error('Prepared descending physiology identity mismatch');
 for(let e=0;e<base.manifest.chemical_edges;e++){const receptor=edges[e*4+2];if(receptor===0||receptor===4)weights[e*4+1]*=gains.synapse_exc_log_gain;else if(receptor>=1&&receptor<=3)weights[e*4+1]*=gains.synapse_inh_log_gain;}
 return {model:{...base,params,edges},families,counts};
}
export function createEpisodeScore(stage,duration,initial){
 if(!STAGES.includes(stage)||!Number.isFinite(duration)||duration<=0)throw new Error('Invalid training stage/duration');
 const state={stage,duration,elapsed:0,uprightSeconds:0,supportedSeconds:0,stableSeconds:0,flightSeconds:0,overturnedSeconds:0,maxAngularSpeed:0,initialDistance:initial.distance,minDistance:initial.distance,initialHeight:initial.height,maxRise:0,initialIntake:initial.intake,intake:0,approachProgress:0,hasTakenOff:false,sequenceIndex:0,sequenceStarted:0,sequenceDistance:initial.distance,sequenceIntake:initial.intake,priorAirborne:initial.airborne,success:false,reason:null,return:0};
 return {state,step(o,dt){
  if(!Number.isFinite(dt)||dt<=0)throw new Error('Invalid observation interval');
  if(!o.finite||![o.angularSpeed,o.distance,o.height,o.intake,o.up,o.verticalSpeed,o.radius,o.ceiling].every(Number.isFinite)){
   state.elapsed+=dt;state.success=false;state.reason='nonfinite_state';state.return=-10;
   return {terminated:true,success:false,reason:state.reason};
  }
  state.elapsed+=dt;state.maxAngularSpeed=Math.max(state.maxAngularSpeed,o.angularSpeed);state.minDistance=Math.min(state.minDistance,o.distance);state.maxRise=Math.max(state.maxRise,o.height-state.initialHeight);state.intake=Math.max(0,o.intake-state.initialIntake);state.approachProgress=Math.max(0,state.initialDistance-o.distance);
  if(o.up>=CRITERIA.postureUp)state.uprightSeconds+=dt;
  if(!o.airborne&&o.contacts>0)state.supportedSeconds+=dt;
  const stable=o.up>=CRITERIA.postureUp&&!o.airborne&&o.contacts>0&&o.angularSpeed<CRITERIA.postureAngularSpeed;
  state.stableSeconds=stable?state.stableSeconds+dt:0;
  state.overturnedSeconds=o.up<0?state.overturnedSeconds+dt:0;
  state.flightSeconds=o.flightQualified?state.flightSeconds+dt:0;
  const departure=!state.priorAirborne&&o.airborne&&o.wingPower>.1&&o.verticalSpeed>.5;
  state.hasTakenOff ||= departure;state.priorAirborne=o.airborne;
  if(stage==='sequence'){
   const next=CRITERIA.sequence[state.sequenceIndex],events=o.events||[];
   const localized=events.some(e=>e.stage==='localization'&&e.time>=state.sequenceStarted);
   const landed=events.some(e=>e.stage==='landing'&&e.time>=state.sequenceStarted);
   const transitions={localization:localized,approach:state.sequenceDistance-o.distance>=CRITERIA.approachDistanceCm,landing:landed,feeding:o.intake-state.sequenceIntake>=CRITERIA.feedingAmount&&o.probing,takeoff:departure,flight:o.flightQualified&&state.flightSeconds>=CRITERIA.flightQualifiedSeconds};
   if(next&&transitions[next]){state.sequenceIndex++;state.sequenceStarted=state.elapsed;state.sequenceDistance=o.distance;state.sequenceIntake=o.intake;state.flightSeconds=0;}
  }
  const normalizedUpright=state.uprightSeconds/duration,normalizedSupport=state.supportedSeconds/duration;
  const phaseSuccess={posture:state.elapsed+1e-9>=duration&&normalizedSupport>=CRITERIA.postureSupportedFraction&&state.stableSeconds>=CRITERIA.postureFinalStableSeconds,approach:state.approachProgress>=CRITERIA.approachDistanceCm&&o.footFoodContacts>=2&&o.up>.5,feeding:state.intake>=CRITERIA.feedingAmount&&o.probing,flight:state.hasTakenOff&&state.flightSeconds>=CRITERIA.flightQualifiedSeconds,landing:o.landings>0,sequence:state.sequenceIndex===CRITERIA.sequence.length};
  state.success=!!phaseSuccess[stage];
  const scores={posture:3*normalizedUpright+2*normalizedSupport,approach:3*clip(state.approachProgress,0,1)+normalizedUpright,feeding:4*clip(state.intake/CRITERIA.feedingAmount,0,1)+normalizedUpright,flight:4*clip(state.flightSeconds/.5,0,1)+clip(state.maxRise,0,1)+normalizedUpright,landing:3*clip(state.stableSeconds/.25,0,1)+normalizedUpright,sequence:6*state.sequenceIndex/CRITERIA.sequence.length+normalizedUpright};
  const failed=!o.finite?'nonfinite_state':o.externalForce?'unexpected_external_force':o.radius>CRITERIA.maximumRadiusCm||o.height<CRITERIA.minimumHeightCm||o.height>o.ceiling+.2?'outside_habitat':o.angularSpeed>CRITERIA.maximumAngularSpeed?'excessive_rotation':state.overturnedSeconds>=CRITERIA.overturnedSeconds?'overturned':null;
  state.reason=failed|| (state.success?'stage_success':null);if(failed)state.success=false;
  state.return=clip(scores[stage]+(state.success?3:0)-(failed?5:0)-.5*clip(state.maxAngularSpeed/CRITERIA.maximumAngularSpeed,0,1),-10,10);
  return {terminated:!!state.reason,success:state.success,reason:state.reason};
 }};
}
