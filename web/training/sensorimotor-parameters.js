import {MOTOR_DECODER_VERSION,validateMotorDecoderContract,validateMotorDecoderVector,buildMotorDecoderContract} from '../motor-decoder.js';
import {validateLegProprioceptionConfig} from '../banc-leg-proprioception.js';
import {validateAntennaFamilyConfig} from '../banc-antenna.js';
import {validateTegulaStrainConfig} from '../banc-tegula.js';
import {validateCoupledHaltereModel,validateGeometry} from '../virtual-haltere.js';
import {validateTrainingVisionConfig} from './config-schema.js';
import {validateCompactVisionSettings} from './compact-vision.js';
import {validateRetinalSettings} from './retinal-sensor.js';

// Simulation-based engineering calibration. These coordinates expose uncertain
// transducer/mechanical scales; optimizing reward does not measure physiology.
// Anatomy, polarity, receptive directions, optics, world geometry, graph and
// neural parameters are never changed here. Motor coefficients remain absolute
// values under the existing 672-coordinate anatomical decoder contract.
export const SENSORIMOTOR_CONTRACT='banc-sensorimotor-sequence-v1';
const GROUPS=['legs','antenna','haltere','tegula','vision'];
const MOTOR_COUNT=672;
const record=x=>x!==null&&typeof x==='object'&&!Array.isArray(x)&&!ArrayBuffer.isView(x);
const finite=x=>typeof x==='number'&&Number.isFinite(x);
const fail=message=>{throw new Error('Sensorimotor parameters: '+message);};
const requireThat=(condition,message)=>{if(!condition)fail(message);};
const exactKeys=(value,keys)=>record(value)&&Reflect.ownKeys(value).length===keys.length&&keys.every(k=>Object.hasOwn(value,k));
const get=(value,path)=>path.reduce((x,k)=>x?.[k],value);
function set(value,path,next){
 let target=value;for(let k=0;k<path.length-1;k++)target=target[path[k]]??=(typeof path[k+1]==='number'?[]:{});
 target[path.at(-1)]=next;
}
const path=(group,...keys)=>[group,...keys];
const spec=(group,name,paths,description,{factor=1.5,low=0,high=Infinity}={})=>({group,name:'sensory_'+group+'_'+name,paths,description,factor,low,high});
const L='legProprioception',A='antennaFeedback',H='haltereFeedback',T='tegulaFeedback',V='visionFeedback';
const SPECS=[
 spec('legs','baseline',[path(L,'baselineRateHz')],'Claw/hook tonic rate scale'),
 spec('legs','position_gain',[path(L,'positionGainHzPerRadian')],'Claw signed-position sensitivity',{high:10000}),
 spec('legs','velocity_gain',[path(L,'velocityGainHzPerRadPerSecond')],'Hook signed-velocity sensitivity',{high:10000}),
 spec('legs','club_baseline',[path(L,'clubBaselineRateHz')],'Club tonic rate scale'),
 spec('legs','club_motion_gain',[path(L,'clubMotionGainHzPerRadPerSecond')],'Club unsigned-motion sensitivity',{high:10000}),
 spec('antenna','time_constant',[path(A,'mechanics','timeConstantSeconds')],'Virtual antenna response time',{low:.001,high:1}),
 spec('antenna','air_speed_scale',[path(A,'mechanics','referenceAirSpeedCmPerSecond')],'Airflow-to-deflection reference speed',{high:10000}),
 spec('antenna','deflection_scale',[path(A,'mechanics','maxDeflectionRadians')],'Virtual antenna deflection scale',{high:1}),
 spec('antenna','baseline',[path(A,'mechanics','baselineRateHz')],'Driven JO-family tonic rate scale'),
 spec('antenna','position_gain',[path(A,'mechanics','positionGainHzPerRadian')],'C/D/E-family position sensitivity',{high:10000}),
 spec('antenna','velocity_gain',[path(A,'mechanics','velocityGainHzPerRadPerSecond')],'JO-D mixed-response velocity sensitivity',{high:10000}),
 spec('haltere','frequency',[path(H,'mechanicalModel','naturalFrequencyHz')],'Virtual haltere resonant frequency',{factor:1.1,low:1,high:1000}),
 spec('haltere','damping',[path(H,'mechanicalModel','dampingRatio')],'Virtual haltere damping ratio',{low:.01,high:2}),
 spec('haltere','own_drive',[path(H,'mechanicalModel','ownDriveAmplitudeRadians')],'Motor-derived virtual haltere drive amplitude',{factor:4/3,high:Math.PI/4}),
 spec('haltere','coupling_stiffness',[path(H,'mechanicalModel','couplingStiffnessRatio')],'Ipsilateral wing coupling stiffness',{high:2}),
 spec('haltere','coupling_damping',[path(H,'mechanicalModel','couplingDampingRatio')],'Ipsilateral wing coupling damping',{high:2}),
 spec('haltere','wing_ratio',['left','right'].map(side=>path(H,'mechanicalModel','sides',side,'wingToHaltereRatio')),'Common wing-to-haltere transmission scale; side ratio retained',{high:2}),
 spec('haltere','current_gain',[path(H,'maxCurrentPa')],'Bending-to-neural-current sensitivity',{high:3200}),
 spec('tegula','compliance',[0,1,2].map(axis=>path(T,'complianceRadiansPerNativeMoment',axis)),'Common elastic compliance scale; axis ratios retained',{high:1e6}),
 spec('tegula','relaxation',[path(T,'relaxationSeconds')],'Virtual hinge relaxation time',{low:.00005,high:1}),
 spec('tegula','half_strain',[path(T,'halfStrain')],'Strain rectification half-saturation scale'),
 spec('tegula','rate_gain',[path(T,'maxRateHz')],'Maximum requested tegula rate',{high:200}),
 spec('vision','motion_gain',[path(V,'motion','gainHzPerRad')],'T4/T5 image-motion rate sensitivity',{high:10000}),
 spec('vision','contrast_scale',[path(V,'motion','contrastScale')],'T4/T5 temporal-contrast weighting scale',{low:.000001,high:1}),
];

function sensoryBase(config){
 requireThat(record(config),'configuration must be an object');
 requireThat(config.freezeNeuralParameters===true,'frozen neural parameters required');
 const legs=validateLegProprioceptionConfig(config.legProprioception);
 requireThat(config.legProprioception!==undefined&&legs.polarityMode==='native-positive-prior','enabled structural leg profile required');
 requireThat(record(config.antennaFeedback)&&config.antennaFeedback.schema===1&&
  Array.isArray(config.antennaFeedback.windWorldCmPerSecond)&&config.antennaFeedback.windWorldCmPerSecond.length===3&&
  config.antennaFeedback.windWorldCmPerSecond.every(finite),'structural native airflow declaration required');
 const antenna=validateAntennaFamilyConfig(config.antennaFeedback.mechanics);
 requireThat(config.antennaFeedback.mechanics?.schema===2,'structural antenna mechanics required');
 const tegula=validateTegulaStrainConfig(config.tegulaFeedback);
 const h=config.haltereFeedback,p=h?.profile;
 requireThat(record(h)&&p?.schemaVersion===2&&p.kind==='banc-haltere-mechanical-current-profile'&&
  p.dataset==='BANC'&&p.materialization===888&&p.neuronCount===175401&&
  p.orientationAssignment?.status==='population-tuning-explicit'&&p.orientationAssignment.anatomicalKnowledge===false&&
  Array.isArray(p.cells)&&p.cells.length===328&&Array.isArray(p.tuningPopulations)&&p.tuningPopulations.length>0,
  'structural BANC haltere profile required');
 const mechanical=validateCoupledHaltereModel(h.mechanicalModel);
 requireThat(Array.isArray(h.geometries)&&h.geometries.length===2&&new Set(h.geometries.map(g=>validateGeometry(g).side)).size===2,
  'bilateral native haltere geometry required');
 requireThat(finite(h.maxCurrentPa)&&h.maxCurrentPa>0&&h.maxCurrentPa<=3200,'bounded haltere current reference required');
 const visual=validateTrainingVisionConfig(config);
 requireThat(visual!==null,'compact retinal motion profile required');
 validateRetinalSettings({...visual.camera,width:visual.width,height:visual.height});
 const motion=validateCompactVisionSettings({...visual.motion,width:visual.width,height:visual.height});
 // Normalized defaults are used only to read reference values. They are not
 // inserted into the returned candidate when their coordinate is zero.
 return {...config,legProprioception:legs,antennaFeedback:{...config.antennaFeedback,mechanics:antenna},
  tegulaFeedback:tegula,haltereFeedback:{...h,mechanicalModel:mechanical},visionFeedback:{...visual,motion}};
}

function sensoryDefinitions(config){
 const base=sensoryBase(config);
 return SPECS.map(s=>{
  const values=s.paths.map(p=>get(base,p));
  requireThat(values.every(x=>finite(x)&&x>0),'positive reference required for '+s.name);
  let low=s.low,high=s.high;
  if(s.name==='sensory_legs_baseline'||s.name==='sensory_legs_club_baseline')high=base.legProprioception.maxRateHz;
  if(s.name==='sensory_antenna_baseline')high=base.antennaFeedback.mechanics.maxRateHz;
  if(s.name==='sensory_tegula_rate_gain')low=base.tegulaFeedback.baselineRateHz;
  requireThat(values.every(x=>x>=low&&x<=high),'reference outside supported bounds for '+s.name);
  const min=Math.max(-Math.log(s.factor),...values.map(x=>low>0?Math.log(low/x):-Infinity)),
    max=Math.min(Math.log(s.factor),...values.map(x=>Number.isFinite(high)?Math.log(high/x):Infinity));
  requireThat(finite(min)&&finite(max)&&min<max&&min<=0&&max>=0,'empty calibration interval for '+s.name);
  return {name:s.name,min,max,initial:0,searchScale:1,group:s.group,transform:'log-scale',
   status:'unmeasured-engineering-prior',description:s.description};
 });
}

function motorDefinitions(config,definitions,io){
 const contract=validateMotorDecoderContract(config.motorDecoderContract);
 requireThat(contract.parameters.length===MOTOR_COUNT&&Array.isArray(definitions)&&definitions.length===MOTOR_COUNT,'exact672 motor definitions required');
 requireThat(record(config.wingEventExcitation),'wing motor event interface required');
 for(let k=0;k<MOTOR_COUNT;k++){
  const actual=definitions[k],expected=contract.parameters[k];
  requireThat(record(actual)&&actual.name===expected.name&&actual.min===expected.min&&actual.max===expected.max&&
   finite(actual.initial)&&actual.initial>=actual.min&&actual.initial<=actual.max&&
   (actual.searchScale===undefined||finite(actual.searchScale)&&actual.searchScale>0&&actual.searchScale<=1),
   'motor definition mismatch at '+k);
 }
 if(io!==undefined)requireThat(JSON.stringify(contract)===JSON.stringify(buildMotorDecoderContract(io)),'motor decoder differs from prepared IO');
 return contract;
}

export function buildSensorimotorParameters(baseConfig){
 requireThat(baseConfig?.parameterContract===MOTOR_DECODER_VERSION&&baseConfig.sensorimotorSequence===undefined,'uncombined motor-decoder base required');
 const sensory=sensoryDefinitions(baseConfig);
 motorDefinitions(baseConfig,baseConfig.parameters);
 return [...sensory,...structuredClone(baseConfig.parameters)];
}

/** Builds configuration only. Callers must finalize its executable asset pins
 * and config hash before hosting. This function neither fits nor evaluates. */
export function buildSensorimotorTrainingConfig(baseConfig){
 const parameters=buildSensorimotorParameters(baseConfig),config=structuredClone(baseConfig);
 config.parameterContract=SENSORIMOTOR_CONTRACT;
 config.sensorimotorSequence={schema:1,sensoryParameterNames:SPECS.map(s=>s.name)};
 config.parameters=parameters;config.freezeNeuralParameters=true;
 return config;
}

function layout(config,io){
 requireThat(config?.parameterContract===SENSORIMOTOR_CONTRACT,'unsupported parameter contract');
 const marker=config.sensorimotorSequence,names=SPECS.map(s=>s.name);
 requireThat(exactKeys(marker,['schema','sensoryParameterNames'])&&marker.schema===1&&
  Array.isArray(marker.sensoryParameterNames)&&JSON.stringify(marker.sensoryParameterNames)===JSON.stringify(names),'sequence marker mismatch');
 const definitions=sensoryDefinitions(config),count=definitions.length;
 requireThat(Array.isArray(config.parameters)&&config.parameters.length===count+MOTOR_COUNT,'combined parameter count mismatch');
 for(let k=0;k<count;k++){
  const actual=config.parameters[k],expected=definitions[k];
  requireThat(exactKeys(actual,Object.keys(expected))&&Object.keys(expected).every(key=>actual[key]===expected[key]),'sensory definition mismatch at '+k);
 }
 const motorContract=motorDefinitions(config,config.parameters.slice(count),io);
 return {definitions,count,motorContract};
}

/** Legacy decoder-only configurations have no sensory prefix. */
export function sensoryParameterCount(config){return config?.parameterContract===SENSORIMOTOR_CONTRACT?layout(config).count:0;}

export function getSensorimotorGroups(config){
 const {count}=layout(config),groups=Object.fromEntries(GROUPS.map(group=>[group,[]]));
 SPECS.forEach((s,k)=>groups[s.group].push(k));
 groups.motor=Array.from({length:MOTOR_COUNT},(_,k)=>count+k);return groups;
}

export function applySensorimotorParameters(config,parameters,io){
 const {count,motorContract}=layout(config,io);
 requireThat((Array.isArray(parameters)||parameters instanceof Float32Array||parameters instanceof Float64Array)&&
  parameters.length===config.parameters.length,'full numeric candidate vector required');
 const vector=Array.from(parameters);
 for(let k=0;k<vector.length;k++)requireThat(finite(vector[k])&&vector[k]>=config.parameters[k].min&&vector[k]<=config.parameters[k].max,
  'candidate outside bounds: '+config.parameters[k].name);
 const motorDecoder=Array.from(validateMotorDecoderVector(motorContract,vector.slice(count))),sensoryConfig=structuredClone(config);
 if(vector.slice(0,count).some(x=>x!==0)){
  const base=sensoryBase(config);
  SPECS.forEach((s,k)=>{
   if(vector[k]===0)return;
   for(const p of s.paths)set(sensoryConfig,p,get(base,p)*Math.exp(vector[k]));
  });
  sensoryBase(sensoryConfig); // Candidate must also satisfy physical profile bounds.
 }
 return {vector,sensoryConfig,motorDecoder,sensoryCount:count};
}
