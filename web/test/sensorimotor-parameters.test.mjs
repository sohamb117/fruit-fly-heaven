import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {MOTOR_DECODER_VERSION,buildMotorDecoderContract} from '../motor-decoder.js';
import {ANTENNA_FAMILY_PRIOR} from '../banc-antenna.js';
import {LEG_PROPRIOCEPTION_PRIOR} from '../banc-leg-proprioception.js';
import {createTegulaStrainPrior} from '../banc-tegula.js';
import {COMPACT_VISION_DEFAULTS} from '../training/compact-vision.js';
import {SENSORIMOTOR_CONTRACT,buildSensorimotorParameters,buildSensorimotorTrainingConfig,
  sensoryParameterCount,getSensorimotorGroups,applySensorimotorParameters} from '../training/sensorimotor-parameters.js';

const [io,haltere]=await Promise.all([
 fs.readFile(new URL('../../data/prepared/banc888/io.json',import.meta.url),'utf8').then(JSON.parse),
 fs.readFile(new URL('../../models/banc-haltere-directional-v2.json',import.meta.url),'utf8').then(JSON.parse),
]);
const contract=buildMotorDecoderContract(io),{enabled,...haltereFeedback}=haltere;
assert.equal(enabled,true);
function baseConfig(){
 return structuredClone({parameterContract:MOTOR_DECODER_VERSION,freezeNeuralParameters:true,dtMs:.5,bodyBlockMs:2,
  motorDecoderContract:contract,parameters:contract.parameters.map((p,k)=>({name:p.name,min:p.min,max:p.max,
   initial:p.kind==='power'?.5+k/100:((k%7)-3)/100,searchScale:.1})),
  legProprioception:LEG_PROPRIOCEPTION_PRIOR,
  antennaFeedback:{schema:1,windWorldCmPerSecond:[0,0,0],mechanics:ANTENNA_FAMILY_PRIOR},
  haltereFeedback,tegulaFeedback:createTegulaStrainPrior({halfLoadNative:.02}),
  vision:true,visionFeedback:{schema:1,profile:'compact-retinal-motion-v1',width:256,height:128,frameIntervalMs:20,camera:{textureAntialias:true}},
  wingEventExcitation:{steering:{riseMs:1,decayMs:5,recruitmentGain:Math.LN2}},
  intrinsicModels:{profile:'fixed-neural-fixture',cells:[{index:12322,root_id:'720575941432491145'}]},
  assets:{'/banc-data/manifest.json':'1'.repeat(64)},modelFingerprint:'2'.repeat(64),
  objective:{min:-10,max:20},initialCondition:{rootQpos:[0,0,3.5,1,0,0,0]}});
}
const initial=config=>config.parameters.map(p=>p.initial);
const sensoryKeys=['legProprioception','antennaFeedback','haltereFeedback','tegulaFeedback','visionFeedback'];
const sensoryBytes=config=>JSON.stringify(Object.fromEntries(sensoryKeys.map(k=>[k,config[k]])));
const close=(a,b)=>assert(Math.abs(a-b)<=1e-12*Math.max(1,Math.abs(b)),`${a} != ${b}`);

test('the new contract owns24 sensory scales followed by the exact original672 motor definitions',()=>{
 const base=baseConfig(),before=JSON.stringify(base),definitions=buildSensorimotorParameters(base),config=buildSensorimotorTrainingConfig(base);
 assert.equal(definitions.length,696);assert.deepEqual(config.parameters,definitions);
 assert.equal(config.parameterContract,SENSORIMOTOR_CONTRACT);assert.equal(config.freezeNeuralParameters,true);
 assert.deepEqual(config.parameters.slice(24),base.parameters);assert.equal(sensoryParameterCount(base),0);assert.equal(sensoryParameterCount(config),24);
 const groups=getSensorimotorGroups(config);assert.deepEqual(Object.keys(groups),['legs','antenna','haltere','tegula','vision','motor']);
 assert.deepEqual(Object.values(groups).map(ids=>ids.length),[5,6,7,4,2,672]);
 assert.deepEqual(Object.values(groups).flat(),Array.from({length:696},(_,k)=>k));
 for(const p of config.parameters.slice(0,24)){
  assert.equal(p.initial,0);assert(p.name.startsWith('sensory_'+p.group+'_'));assert.equal(p.transform,'log-scale');
  assert.equal(p.status,'unmeasured-engineering-prior');assert(p.description.length>10);
  assert(p.min<0&&p.max>=0&&p.max<=Math.log(1.5));assert(p.min>=-Math.log(1.5));
 }
 assert.equal(config.parameters.find(p=>p.name==='sensory_haltere_own_drive').max,0);
 assert.equal(JSON.stringify(base),before);config.parameters[24].initial=0;assert.notEqual(base.parameters[0].initial,0);
});

test('zero sensory prefix reproduces exact prior settings and decoder initials without materializing defaults',()=>{
 const base=baseConfig(),config=buildSensorimotorTrainingConfig(base),before=JSON.stringify(config),values=initial(config);
 const result=applySensorimotorParameters(config,values,io);
 assert.equal(sensoryBytes(result.sensoryConfig),sensoryBytes(base));assert.equal(result.sensoryConfig.visionFeedback.motion,undefined);
 assert.deepEqual(result.motorDecoder,base.parameters.map(p=>p.initial));assert.deepEqual(result.vector,values);assert.equal(result.sensoryCount,24);
 assert.equal(JSON.stringify(config),before);assert.notEqual(result.sensoryConfig,config);
 result.vector[0]=1;result.motorDecoder[0]=0;result.sensoryConfig.haltereFeedback.profile.cells[0].side='changed';
 assert.equal(values[0],0);assert.notEqual(config.haltereFeedback.profile.cells[0].side,'changed');
});

test('each sensory coordinate changes only its own family and never graph, identity, decoder or world state',()=>{
 const config=buildSensorimotorTrainingConfig(baseConfig()),reference=applySensorimotorParameters(config,initial(config)),familyKeys={legs:'legProprioception',antenna:'antennaFeedback',haltere:'haltereFeedback',tegula:'tegulaFeedback',vision:'visionFeedback'};
 for(let k=0;k<24;k++){
  const values=initial(config);values[k]=config.parameters[k].min/2;
  const result=applySensorimotorParameters(config,values),changed=familyKeys[config.parameters[k].group];
  assert.notDeepEqual(result.sensoryConfig[changed],config[changed]);
  for(const key of Object.keys(config))if(key!==changed)assert.deepEqual(result.sensoryConfig[key],config[key]);
  assert.deepEqual(result.motorDecoder,reference.motorDecoder);
  assert.deepEqual(result.sensoryConfig.haltereFeedback.profile,config.haltereFeedback.profile);
  assert.deepEqual(result.sensoryConfig.haltereFeedback.geometries,config.haltereFeedback.geometries);
  assert.deepEqual(result.sensoryConfig.tegulaFeedback.fields,config.tegulaFeedback.fields);
  assert.deepEqual(result.sensoryConfig.legProprioception.cellOverrides,config.legProprioception.cellOverrides);
  assert.deepEqual(result.sensoryConfig.antennaFeedback.windWorldCmPerSecond,config.antennaFeedback.windWorldCmPerSecond);
  assert.deepEqual(result.sensoryConfig.visionFeedback.camera,config.visionFeedback.camera);
 }
});

test('scales apply once to the reference, preserve bilateral/axis ratios, and leave decoder coordinates absolute',()=>{
 const base=baseConfig();base.haltereFeedback.mechanicalModel.sides.right.wingToHaltereRatio=.4;
 base.tegulaFeedback.complianceRadiansPerNativeMoment=[.5,.4,.3];
 const config=buildSensorimotorTrainingConfig(base),values=initial(config),index=name=>config.parameters.findIndex(p=>p.name===name),factor=1.2;
 for(const name of ['sensory_haltere_wing_ratio','sensory_tegula_compliance','sensory_vision_motion_gain'])values[index(name)]=Math.log(factor);
 values[24]=1.25;const a=applySensorimotorParameters(config,values),b=applySensorimotorParameters(config,values);
 assert.deepEqual(a,b);assert.equal(a.motorDecoder[0],1.25);assert.deepEqual(a.motorDecoder.slice(1),base.parameters.slice(1).map(p=>p.initial));
 for(const side of ['left','right'])close(a.sensoryConfig.haltereFeedback.mechanicalModel.sides[side].wingToHaltereRatio,base.haltereFeedback.mechanicalModel.sides[side].wingToHaltereRatio*factor);
 a.sensoryConfig.tegulaFeedback.complianceRadiansPerNativeMoment.forEach((v,k)=>close(v,base.tegulaFeedback.complianceRadiansPerNativeMoment[k]*factor));
 close(a.sensoryConfig.visionFeedback.motion.gainHzPerRad,COMPACT_VISION_DEFAULTS.gainHzPerRad*factor);
 assert.deepEqual(Object.keys(a.sensoryConfig.visionFeedback.motion),['gainHzPerRad']);
 assert.equal(config.visionFeedback.motion,undefined);
});

test('all declared endpoints and simultaneous extremes satisfy their feature profiles',()=>{
 const config=buildSensorimotorTrainingConfig(baseConfig()),values=initial(config);
 for(let k=0;k<24;k++)for(const endpoint of ['min','max']){
  const candidate=values.slice();candidate[k]=config.parameters[k][endpoint];assert.doesNotThrow(()=>applySensorimotorParameters(config,candidate));
 }
 for(const endpoint of ['min','max']){
  const candidate=values.map((v,k)=>k<24?config.parameters[k][endpoint]:v);assert.doesNotThrow(()=>applySensorimotorParameters(config,candidate));
 }
 assert.deepEqual(applySensorimotorParameters(config,Float64Array.from(values)).vector,values);
});

test('invalid vectors and altered parameter contracts reject without mutating their source',()=>{
 const config=buildSensorimotorTrainingConfig(baseConfig()),before=JSON.stringify(config),values=initial(config);
 for(const bad of [values.slice(1),[...values,0],values.map((v,k)=>k? v:NaN),values.map((v,k)=>k? v:Infinity),
  values.map((v,k)=>k? v:config.parameters[0].min-1e-9),values.map((v,k)=>k? v:config.parameters[0].max+1e-9),
  values.map((v,k)=>k===24?4.1:v)])assert.throws(()=>applySensorimotorParameters(config,bad),/Sensorimotor/);
 for(const mutate of [c=>c.parameters[0].name='sensory_unregistered',c=>c.parameters[0].min=-1,c=>c.parameters[0].initial=.01,
  c=>c.parameters[0].group='motor',c=>c.parameters[0].status='measured',c=>c.parameters[24].min=-1,
  c=>c.sensorimotorSequence.sensoryParameterNames.reverse(),c=>c.sensorimotorSequence.extra=true,c=>c.freezeNeuralParameters=false]){
  const bad=structuredClone(config);mutate(bad);assert.throws(()=>applySensorimotorParameters(bad,values),/Sensorimotor/);
 }
 assert.equal(JSON.stringify(config),before);
 const wrongIo=structuredClone(io);wrongIo.muscles[contract.muscles[0].mappingIndex].root_ids[0]='1';
 assert.throws(()=>applySensorimotorParameters(config,values,wrongIo),/Motor decoder|Sensorimotor/);
});

test('unsupported or inert base features fail rather than creating no-op training parameters',()=>{
 for(const mutate of [c=>delete c.legProprioception,c=>c.legProprioception.polarityMode='require-measured',
  c=>c.legProprioception.baselineRateHz=0,c=>c.antennaFeedback.mechanics.schema=1,
  c=>c.antennaFeedback.mechanics.velocityGainHzPerRadPerSecond=0,c=>c.tegulaFeedback.schema=1,
  c=>c.haltereFeedback.profile.schemaVersion=1,c=>c.haltereFeedback.mechanicalModel.couplingStiffnessRatio=0,
  c=>c.haltereFeedback.mechanicalModel.sides.left.wingToHaltereRatio=0,c=>c.haltereFeedback.maxCurrentPa=0,
  c=>c.vision=false,c=>c.visionFeedback.profile='other',c=>c.freezeNeuralParameters=false]){
  const config=baseConfig();mutate(config);assert.throws(()=>buildSensorimotorTrainingConfig(config));
 }
 const combined=buildSensorimotorTrainingConfig(baseConfig());assert.throws(()=>buildSensorimotorTrainingConfig(combined),/uncombined/);
});
