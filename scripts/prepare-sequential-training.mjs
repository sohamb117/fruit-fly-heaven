// Assemble one immutable experiment; never rewrites the public configuration.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {buildTrainingAssetManifest} from './prepare-training-manifest.mjs';
import {readExperimentBundle} from './package-training-client.mjs';
import {buildSensorimotorTrainingConfig,getSensorimotorGroups} from '../web/training/sensorimotor-parameters.js';
import {validateConfig} from '../web/training/optimizer.js';
import {validateFlightTeacherCalibration} from '../web/training/flight-teacher.js';
import {DEFAULT_RECOVERY_DISTURBANCE} from '../web/training/recovery-objective.js';
import {prepareTrainingBrain} from './prepare-training-brain.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const sha=value=>createHash('sha256').update(value).digest('hex');
const teacherURL='/body-model/flight-teacher-calibration-v1.json';
const prefixes=[['/body-engine/','packages/flybody-runtime/node_modules/@mujoco/mujoco/'],['/banc-engine/','packages/banc-runtime/'],['/banc-data/','data/prepared/banc888/'],['/body-model/','models/']];
const local=url=>{const match=prefixes.find(([prefix])=>url.startsWith(prefix));return path.join(root,match?match[1]+url.slice(match[0].length):'web'+url);};
export async function prepareSequence({baseBundle='reports/structural-proprioception-20260915/v2/structural.bundle.json',
 output='reports/sequential-training-20260915',calibration='reports/flight-decoder-feasibility-20260914/calibration.json'}={}){
 const baseBytes=await fs.readFile(path.resolve(root,baseBundle)),base=JSON.parse(baseBytes);readExperimentBundle(baseBytes);
 const config=buildSensorimotorTrainingConfig(JSON.parse(base.configText)),groups=getSensorimotorGroups(config);
 const owned=Object.fromEntries(Object.entries(base.assets).filter(([url])=>!url.endsWith('.js')));
 config.environmentVersion='banc888-sensorimotor-sequence-v1';
 config.stage='maintained_flight';config.durationSeconds=5;
 config.stages=[{id:'maintained_flight',label:'Maintain flight',durationSeconds:5},
  {id:'recovery',label:'Recover',durationSeconds:5},{id:'takeoff',label:'Take off',durationSeconds:3},
  {id:'landing',label:'Take off, fly and land',durationSeconds:8}];
 config.recoveryDisturbance=structuredClone(DEFAULT_RECOVERY_DISTURBANCE);
 const descriptions={legs:'Tune joint position and movement sensing.',antenna:'Tune airflow sensing.',
  haltere:'Tune rotation feedback.',tegula:'Tune wing strain feedback.',vision:'Tune image-motion sensitivity.',
  decoder_fit:'Learn motor-to-wing commands from demonstrations.',recovery:'Regain controlled flight after a disturbance.',
  takeoff:'Lift off and gain height.',landing:'Take off, stay airborne, then land on your feet.'};
 const phase=(id,label,stage,indices,kind='search')=>({id,label,description:descriptions[id],kind,stage,parameterIndices:indices,
  maxRounds:6,validationCount:3,minimumSuccesses:2,minimumMeanImprovement:0.01});
 config.trainingSequence={schema:1,calibrationMode:'simulation-engineering',teacherAsset:teacherURL,
  progressAcceptance:'paired-improvement-before-stage-completion-v1',
  fitOptions:{ridge:1e-5,maxSweeps:2000,tolerance:1e-9},phases:[
   ...[['legs','Leg feedback'],['antenna','Antenna feedback'],['haltere','Haltere feedback'],
    ['tegula','Wing strain feedback'],['vision','Visual feedback']].map(([id,label])=>phase(id,label,'maintained_flight',groups[id])),
   phase('decoder_fit','Fit movement','recovery',groups.motor,'decoder-fit'),
   phase('recovery','Recover','recovery',groups.motor),phase('takeoff','Take off','takeoff',groups.motor),
   phase('landing','Land','landing',groups.motor)]};
 config.notes=[...config.notes??[],
  'Simulation-based engineering calibration, not measured stimulus-response physiology. Sensory priors and the existing fitted motor decoder initialize the sequence.',
  'One family changes at a time. Teacher-driven command fitting requires whole-episode validation; autonomous paired comparisons and fresh-seed gates determine acceptance.',
  'The sequence stops for review when its declared trial budget fails. Passed anonymous trials remain unverified biological evidence.'];
 const raw=await fs.readFile(path.resolve(root,calibration)),measured=JSON.parse(raw);
 if(measured.complete!==true)throw new Error('Incomplete native teacher calibration');
 const sourceDirectory=path.join(root,'dist/training-lease-client/fruit-fly-training-client-'+measured.identity.configHash.slice(0,12));
 const sourceConfigText=await fs.readFile(path.join(sourceDirectory,'training/config.json'));
 if(sha(sourceConfigText)!==measured.identity.configHash)throw new Error('Native calibration source config mismatch');
 const sourceConfig=JSON.parse(sourceConfigText),mechanics={};
 for(const url of ['/body-model/flybody-mujoco.xml','/body-model/flybody-mujoco.json','/body-engine/mujoco.wasm','/flybody-wings.js','/flybody-physics.js']){
  const current=Object.hasOwn(owned,url)?Buffer.from(owned[url]):await fs.readFile(local(url));
  if(url==='/flybody-physics.js'){
   // Reviewed observer-only change; reject any other actuator/integrator drift.
   const previous=await fs.readFile(path.join(sourceDirectory,url.slice(1)),'utf8');
   if(sha(previous)!==sourceConfig.assets[url]||current.toString()!==previous
    .replace('enableWingLoadFeedback(){','enableWingLoadFeedback({localFrame=false}={}){')
    .replace('createWingLoadSampler({mj:this.mj,model:this.model})','createWingLoadSampler({mj:this.mj,model:this.model,localFrame})'))
     throw new Error('Teacher mechanical transfer needs a new review: '+url);
  }else if(sha(current)!==sourceConfig.assets[url])throw new Error('Teacher mechanical calibration changed: '+url);
  mechanics[url]=sha(current);
 }
 const teacher=validateFlightTeacherCalibration({schemaVersion:1,kind:'flight-state-teacher-calibration-v1',
  source:{kind:'native-control-calibration',sha256:sha(raw),configHash:measured.identity.configHash,
   description:'Native finite-difference wing control Jacobian. XML, metadata, native binary and wing mechanics unchanged; physics wrapper differs only by read-only local-frame force observation. New sensory state requires new successful demonstrations.'},
  mechanics,controlNames:measured.controlNames,trim:measured.trim,J:measured.J,response:measured.response,
  inertia:measured.inertia,massG:measured.massG,gains:{naturalFrequency:12,dampingRatio:.9}});
 owned[teacherURL]=JSON.stringify(teacher,null,2)+'\n';
 Object.assign(config,await buildTrainingAssetManifest({config,assetOverrides:owned}));
 validateConfig(config);
 for(const url of Object.keys(config.assets))if(url.endsWith('.js'))owned[url]=await fs.readFile(local(url),'utf8');
 const configText=JSON.stringify(config,null,2)+'\n',bundle={schemaVersion:1,kind:'flight-development-bundle',
  configText,configHash:sha(configText),modelFingerprint:config.modelFingerprint,assets:owned,
  provenance:{baseBundleSha256:sha(baseBytes),teacherCalibrationSha256:sha(raw),calibrationMode:'simulation-engineering'}};
 const encoded=JSON.stringify(bundle,null,2)+'\n';readExperimentBundle(encoded);
 const destination=path.resolve(root,output);
 if(!destination.startsWith(path.join(root,'reports')+path.sep))throw new Error('Sequence output must be inside reports');
 await fs.mkdir(destination,{recursive:true});
 await fs.writeFile(path.join(destination,'sequence.bundle.json'),encoded,{flag:'wx'});
 await fs.writeFile(path.join(destination,'config.json'),configText,{flag:'wx'});
 await fs.writeFile(path.join(destination,'brain-sample.json'),JSON.stringify(await prepareTrainingBrain(config))+'\n',{flag:'wx'});
 return {bundle:path.join(destination,'sequence.bundle.json'),configHash:bundle.configHash,modelFingerprint:bundle.modelFingerprint,
  parameters:config.parameters.length,sensoryParameters:24,motorParameters:672,phases:config.trainingSequence.phases.map(p=>p.id),nativeExecution:config.optimizer.acceptance.nativeExecution};
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 const options={};for(const arg of process.argv.slice(2)){const match=/^--(baseBundle|output|calibration)=(.+)$/.exec(arg);if(!match)throw new Error('Use --output=reports/NAME, --baseBundle=PATH or --calibration=PATH');options[match[1]]=match[2];}
 console.log(JSON.stringify(await prepareSequence(options),null,2));
}
