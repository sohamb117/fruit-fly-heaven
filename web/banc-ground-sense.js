// Organ identity, leg and modality come from BANC annotations. Numerical tuning
// is a prior. Load/touch are not inferred from joint angle or whole-body speed.
const clip=(v,max=100)=>Math.max(0,Math.min(max,Number.isFinite(v)?v:0));
// These explicit BANC v888 function labels occur on ten cells whose broad
// cell_class is "bristle_neuron". That class alone does not establish a
// mechanical receptive field. Keep their graph activity; add no collision
// current. Verified sugar organ/side routing is handled separately by the
// taste mapper; no pheromone or salt tuning is inferred here.
export const BANC_CONTACT_CHEMOSENSORY_FUNCTIONS=Object.freeze(['sugar','low_salt','contact_pheromone']);
const contactChemosensoryPattern=new RegExp(`(?:^|,)\\s*(?:${BANC_CONTACT_CHEMOSENSORY_FUNCTIONS.join('|')})\\s*(?:,|$)`,'i');
export function hasBancContactChemosensoryFunction(sensor){
 // This runs in the per-cell sensory loop: avoid allocating token arrays.
 return typeof sensor?.function==='string'&&contactChemosensoryPattern.test(sensor.function);
}
const jointAngleFunctionPattern=/(?:^|,)\s*joint_angle\s*(?:,|$)/i;
export function hasBancUnassignedBristlePosition(sensor){
 // v888 SNta35 is a bristle annotated joint_angle. Its particular joint and
 // tuning are unassigned; existing position receptors have a different kind.
 return sensor?.kind==='touch'&&typeof sensor.function==='string'&&jointAngleFunctionPattern.test(sensor.function);
}
const haltereSteeringTargets=['hi1_muscle','hi2_muscle','hiii2_muscle','haltere_basalare_muscle'];
export const BANC_ROTATION_MODEL={
 status:'Organ- and side-specific muscle-force-gated unsigned prior; not calibrated directional or phase tuning.',
 formula:'min(100, 5*max(power, steering) + 4*power*norm(angularVelocity)) Hz; steering=0 for wing-base afferents.',
 feedback:{halterePower:'[left, right] normalized asynchronous haltere muscle forces',haltereSteering:'{left: {annotated muscle target: force}, right: {annotated muscle target: force}}',wingPowerLeft:'normalized deployed left-wing power',wingPowerRight:'normalized deployed right-wing power',angularVelocity:'three native root angular-velocity components, rad/s; only magnitude is used'},
 limitations:['Native haltere positions, velocities and strain are not simulated; force is an activation proxy, not measured stroke amplitude.',
  'All afferents of one organ and side receive the same scalar; rotation signs, axes, sensillum receptive fields and spike phase remain unresolved.',
  'Steering contributes only a bounded basal strain proxy. No muscle-specific sign, amplitude change or gyroscopic sensitivity is inferred.',
  'Unknown organ/side or missing side-specific power does not borrow mean, opposite-side, or other-organ activation.']
};

function rotationRate(sensor,feedback){
 const side=sensor.side==='left'?0:sensor.side==='right'?1:-1;
 if(side<0)return 0;
 let power=0,activity=0;
 if(sensor.organ==='haltere'){
  power=clip(feedback.halterePower?.[side],1);activity=power;
  const steering=feedback.haltereSteering?.[sensor.side];
  for(const target of haltereSteeringTargets)activity=Math.max(activity,clip(steering?.[target],1));
 }else if(sensor.organ==='wing_base'){
  power=clip(side===0?feedback.wingPowerLeft:feedback.wingPowerRight,1);activity=power;
 }else return 0;
 const omega=feedback.angularVelocity||[],speed=Math.hypot(Number.isFinite(omega[0])?omega[0]:0,Number.isFinite(omega[1])?omega[1]:0,Number.isFinite(omega[2])?omega[2]:0);
 return clip(5*activity+4*power*speed);
}

export function bancBodyRate(sensor,feedback,enabled=true){
 if(!enabled||!feedback)return 0;
 if(hasBancContactChemosensoryFunction(sensor)||hasBancUnassignedBristlePosition(sensor))return 0;
 if(sensor.kind==='rotation')return rotationRate(sensor,feedback);
 const leg=feedback.legs?.[sensor.leg];if(!leg)return 0;
 if(sensor.kind==='load')return clip(80*(leg.loadBodyWeights||0));
 if(sensor.kind==='touch')return clip(60*(leg.collision||0));
 if(sensor.kind==='velocity')return clip(3*Math.abs(leg.tibiaVelocity||0));
 if(sensor.kind==='vibration')return clip(3*(leg.vibration||0));
 return clip(5+15*Math.abs(leg.tibiaAngle||0)+5*Math.abs(leg.coxaAngle||0));
}
