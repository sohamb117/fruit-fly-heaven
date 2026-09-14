// Repository-model preparation only; no previous experiment/report bundle reads.
// Preexisting local reports/native-webgpu-tooling is hashed, never installed/run.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {buildMotorDecoderContract,validateMotorDecoderContract,validateMotorDecoderVector,
 MOTOR_DECODER_VERSION} from '../web/motor-decoder.js';
import {parameterValues} from '../web/training/episode.js';
import {selectFlightInitialCondition} from '../web/training/airborne-reset-contract.js';
import {matchMaintainedScene} from '../web/flight-scene-profile.js';
import {buildTrainingAssetManifest} from './prepare-training-manifest.mjs';

const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const TEMPLATE='configs/training-motor-decoder-v1.json',SELF='scripts/prepare-motor-decoder-experiment.mjs';
const sha=x=>createHash('sha256').update(x).digest('hex'),clone=x=>JSON.parse(JSON.stringify(x));
const requireThat=(condition,message)=>{if(!condition)throw new Error(message);};

/** Importable pure validation. A calibration checkpoint initializes a NEW
 * unverified experiment; it is not a coordinator/behaviorally accepted state. */
export function validateInitialMotorDecoderCheckpoint(checkpoint,{contract,ioSha256}){
 const checked=validateMotorDecoderContract(contract),contractSha256=sha(JSON.stringify(checked));
 requireThat(checkpoint?.schemaVersion===1&&checkpoint.kind==='motor-decoder-calibration-checkpoint'&&
  checkpoint.profile===MOTOR_DECODER_VERSION,'Unsupported motor decoder initialization checkpoint');
 requireThat(checkpoint.ioSha256===ioSha256&&checkpoint.contractSha256===contractSha256,
  'Initial checkpoint IO or decoder contract hash mismatch');
 for(const field of ['parameterNames','names'])if(Object.hasOwn(checkpoint,field))
  requireThat(Array.isArray(checkpoint[field])&&checkpoint[field].length===checked.parameters.length&&
   checkpoint[field].every((name,i)=>name===checked.parameters[i].name),'Initial checkpoint parameter names/order mismatch');
 return Array.from(validateMotorDecoderVector(checked,checkpoint.parameters));
}
function argumentsFor(argv){
 let output,initial,backend='dawn-metal',backendSet=false;
 for(const argument of argv){
  if(argument.startsWith('--backend=')){
   requireThat(!backendSet&&['dawn-metal','wasm'].includes(argument.slice('--backend='.length)),'Use one --backend=dawn-metal or --backend=wasm');
   backend=argument.slice('--backend='.length);backendSet=true;
  }else if(argument.startsWith('--initial=')){
   requireThat(initial===undefined&&argument.length>'--initial='.length,'Use one nonempty --initial=checkpoint.json');
   initial=argument.slice('--initial='.length);
  }else{requireThat(!argument.startsWith('--')&&output===undefined,'Unknown or repeated preparation argument');output=argument;}
 }
 return {output:path.resolve(ROOT,output||'reports/motor-decoder-v1/experiment'),backend,
  initial:initial===undefined?null:path.resolve(ROOT,initial)};
}
export async function prepareMotorDecoderExperiment(argv=[]){
 const options=argumentsFor(argv),output=options.output;
 requireThat(output.startsWith(path.join(ROOT,'reports')+path.sep),'Experiment must be inside this checkout reports');
 try{await fs.access(output);throw new Error('Experiment already exists; choose a fresh output directory');}
 catch(error){if(error.code!=='ENOENT')throw error;}
 const files=[SELF,TEMPLATE,'scripts/prepare-training-manifest.mjs','models/flybody-mujoco.xml',
  'models/flybody-mujoco.json','data/prepared/banc888/io.json'];
 const bytes=Object.fromEntries(await Promise.all(files.map(async file=>[file,await fs.readFile(path.join(ROOT,file))])));
 const sourceHashes=Object.fromEntries(files.map(file=>[file,sha(bytes[file])]));
 const template=JSON.parse(bytes[TEMPLATE]);
 requireThat(template.schemaVersion===1&&template.kind==='motor-decoder-experiment-template','Unsupported experiment template');
 const config=clone(template.config),fixture=template.modelFixture,prerequisites=template.nativeBackendPrerequisites;
 requireThat(config.parameterContract===MOTOR_DECODER_VERSION&&config.freezeNeuralParameters===true&&
  config.stage==='maintained_flight'&&config.durationSeconds===5&&config.stages.length===1&&
  config.stages[0].id===config.stage&&config.stages[0].durationSeconds===5,'Template must declare the frozen maintained-flight decoder experiment');
 requireThat(config.initialCondition.bodyVariantHash===undefined&&config.assets===undefined&&
  config.modelFingerprint===undefined&&config.motorDecoderContract===undefined&&config.parameters===undefined,
  'Template must not contain stale generated hashes, anatomy or parameter vectors');
 requireThat(sourceHashes['models/flybody-mujoco.xml']===fixture.sourceXmlSha256&&
  sourceHashes['models/flybody-mujoco.json']===fixture.sourceMetadataSha256,'Repository model differs from the explicit fixture source pins');
 requireThat(fixture.clawAdhesionGainScale===0,'Only the explicit no-claw-adhesion diagnostic fixture is supported');
 requireThat(prerequisites.backend==='dawn-metal'&&prerequisites.module==='reports/native-webgpu-tooling/backend.mjs'&&
  prerequisites.packageLock==='reports/native-webgpu-tooling/package-lock.json','Unsupported local native backend prerequisite');
 for(const file of options.backend==='dawn-metal'?[prerequisites.module,prerequisites.packageLock]:[]){
  try{bytes[file]=await fs.readFile(path.join(ROOT,file));}
  catch(error){throw new Error('Missing preexisting native Dawn tooling prerequisite: '+file+' (preparation does not install it)',{cause:error});}
  sourceHashes[file]=sha(bytes[file]);
 }
 const native=options.backend==='dawn-metal'?{moduleSha256:sourceHashes[prerequisites.module],packageLockSha256:sourceHashes[prerequisites.packageLock]}:null;
 if(native)config.optimizer.acceptance.nativeExecution={backend:'dawn-metal',...native};
 else config.notes[0]='Browser experiment with one WASM fly per evaluation and a fixed BANC neural model; all results go to the hosted coordinator.';
 // Sole XML change: preserve all joints, inertials, collision/friction and
 // actuator identities, disabling only phenomenological claw adhesion.
 const sourceXml=bytes['models/flybody-mujoco.xml'].toString('utf8');let replacements=0;
 const xml=sourceXml.replace(/(<default class="adhesion_claw">\s*<adhesion[^>]*gain=")([^"]+)(")/g,
  (_,prefix,_gain,suffix)=>{replacements++;return prefix+'0'+suffix;});
 requireThat(replacements===1&&sha(xml)===fixture.expectedXmlSha256,'No-adhesion XML fixture differs from its declared digest');
 const metadata=JSON.parse(bytes['models/flybody-mujoco.json']);
 requireThat(metadata.xml_sha256===fixture.sourceXmlSha256,'Repository metadata/XML source mismatch');
 metadata.xml_sha256=sha(xml);
 metadata.diagnosticVariant={clawAdhesionGainScale:0,sourceXmlSha256:fixture.sourceXmlSha256,
  wingEventExcitation:clone(config.wingEventExcitation),intrinsicModels:clone(config.intrinsicModels),
  tegulaFeedback:clone(config.tegulaFeedback),tegulaReference:clone(fixture.tegulaReference)};
 // Historical tegula paths above are documentary provenance, never inputs.
 // Normalized decoder power bypasses this retained compatibility profile.
 metadata.wing_actuation.power_transfer=clone(fixture.powerTransfer);
 metadata.maintained_scene=clone(config.maintainedScene);
 const metadataText=JSON.stringify(metadata,null,2)+'\n',metadataSha256=sha(metadataText);
 requireThat(metadataSha256===fixture.expectedMetadataSha256,'Rebuilt metadata differs from its explicit fixture');
 config.initialCondition.bodyVariantHash=metadataSha256;
 matchMaintainedScene(config.maintainedScene,metadata.maintained_scene);
 selectFlightInitialCondition(config.stage,config.initialCondition,{bodyMetadataSha256:metadataSha256});
 const io=JSON.parse(bytes['data/prepared/banc888/io.json']),ioSha256=sourceHashes['data/prepared/banc888/io.json'];
 const contract=buildMotorDecoderContract(io),contractSha256=sha(JSON.stringify(contract));
 let initial=contract.parameters.map(p=>p.initial),initialSource=null,initialBytes=null;
 if(options.initial){
  initialBytes=await fs.readFile(options.initial);
  initial=validateInitialMotorDecoderCheckpoint(JSON.parse(initialBytes),{contract,ioSha256});
  initialSource={path:path.relative(ROOT,options.initial),sha256:sha(initialBytes),status:'unverified-fit-initialization',
   note:'New experiment initialization only; no coordinator acceptance or behavioral promotion.'};
  config.notes.push('Initial coefficients come from an explicitly supplied compatible offline calibration checkpoint. This unverified fit initializes a new experiment, not a promoted behavioral checkpoint.');
 }
 config.motorDecoderContract=contract;
 config.parameters=contract.parameters.map((p,i)=>{
  const searchScale=template.parameterSearchScales[p.kind];
  requireThat(Number.isFinite(searchScale)&&searchScale>0&&searchScale<=1,'Invalid template parameter search scale');
  return {name:p.name,min:p.min,max:p.max,initial:initial[i],searchScale};
 });
 const manifest=await buildTrainingAssetManifest();
 const assets={'/body-model/flybody-mujoco.xml':xml,'/body-model/flybody-mujoco.json':metadataText},hashes={...manifest.assets};
 for(const [url,text]of Object.entries(assets))hashes[url]=sha(text);
 config.assets=Object.fromEntries(Object.entries(hashes).sort(([a],[b])=>a.localeCompare(b)));
 if(options.backend==='wasm')config.optimizer.acceptance.nativeExecution={backend:'wasm',moduleSha256:config.assets['/banc-engine/dist/core.wasm']};
 config.modelFingerprint=sha(Object.entries(config.assets).map(([url,digest])=>`${url}:${digest}\n`).join(''));
 parameterValues(config,initial,io);
 const configText=JSON.stringify(config,null,2)+'\n',configHash=sha(configText);
 const provenance={sourceTemplate:TEMPLATE,sourceHashes,ioSha256,contractSha256,initialCheckpoint:initialSource,
  backend:options.backend,nativeBackendPrerequisites:native?clone(prerequisites):null,modelFixture:{clawAdhesionGainScale:0,
   sourceXmlSha256:fixture.sourceXmlSha256,sourceMetadataSha256:fixture.sourceMetadataSha256,xmlSha256:sha(xml),metadataSha256}};
 const bundle={schemaVersion:1,kind:'flight-development-bundle',createdAt:new Date().toISOString(),scope:config.notes.join(' '),
  preparation:provenance,configText,configHash,modelFingerprint:config.modelFingerprint,assets};
 const backend={schemaVersion:1,backend:options.backend,configHash,modelFingerprint:config.modelFingerprint,
  ...(native?{nativeWebGPU:native}:{}),stopAfterGeneration:template.stopAfterGeneration,
  scope:native?'One isolated guarded generation of the new constrained decoder; fit initialization remains unverified.':'Browser WASM cohort. The site uses its coordinator for all candidate assignments, comparisons and checkpoints.'};
 requireThat(backend.stopAfterGeneration===1,'This experiment is limited to one guarded generation');
 const baseline={schemaVersion:1,backend:options.backend,...(native?{nativeWebGPU:native}:{}),configHash,modelFingerprint:config.modelFingerprint,
  scope:'Full-horizon live BANC baseline of the new decoder; not an optimizer update.',
  jobs:[{name:'decoder-baseline',seed:config.optimizer.seed,stage:config.stage,durationSeconds:config.durationSeconds,
   parameters:initial,captureMotorEvents:true,captureFlightHistory:true}]};
 // Re-read the manifest and inputs before creating any output, so concurrent
 // source edits cannot yield a mixed preparation snapshot.
 requireThat(JSON.stringify(await buildTrainingAssetManifest())===JSON.stringify(manifest),'Executable sources changed during preparation');
 for(const [file,digest]of Object.entries(sourceHashes))requireThat(sha(await fs.readFile(path.join(ROOT,file)))===digest,'Preparation input changed: '+file);
 if(options.initial)requireThat(sha(await fs.readFile(options.initial))===initialSource.sha256,'Initial checkpoint changed during preparation');
 const readme=`Prepared from ${TEMPLATE} and repository models; no previous experiment bundle is read.\n\n`+
  `Recreate: node ${SELF} reports/motor-decoder-v1/NEW_DIRECTORY --backend=${options.backend}`+(initialSource?' --initial=PATH_TO_CALIBRATION_CHECKPOINT':'')+'\n\n'+
  'Exact preparation source/template are archived here. Config/model hashes are deterministic for unchanged inputs; the bundle timestamp is informational.\n\n'+
  (native?'Native evaluation requires preexisting reports/native-webgpu-tooling/backend.mjs, package-lock.json and installed Dawn dependencies. Preparation only hashes them; it does not install/run them.\n\n':'Open the deployed training page and press Start to run WASM in the browser. Results and selected checkpoints are stored by the hosted coordinator. This cohort does not require native Dawn tooling and does not combine native GPU scores with WASM evaluations.\n\n')+
  'The full articulated body retains ordinary collisions/friction but disables phenomenological claw adhesion: no gripping/climbing claim. The old power-transfer profile remains for compatibility; normalized decoder power bypasses it. Historical tegula paths are provenance labels and are not read.\n\n'+
  'Optional --initial requires exact IO/contract hashes, schema/profile and 672 bounded values. Its bytes/hash are archived as unverified fit initialization, never as a promoted coordinator checkpoint.\n';
 await fs.mkdir(path.dirname(output),{recursive:true});await fs.mkdir(output);
 const outputs={'bundle.json':JSON.stringify(bundle,null,2)+'\n','config.json':configText,
  'backend-plan.json':JSON.stringify(backend,null,2)+'\n','baseline-plan.json':JSON.stringify(baseline,null,2)+'\n',
  'preparation-source.mjs':bytes[SELF],'preparation-template.json':bytes[TEMPLATE],
  'preparation-provenance.json':JSON.stringify(provenance,null,2)+'\n','README.md':readme};
 if(initialBytes)outputs['initial-checkpoint.json']=initialBytes;
 for(const [name,content]of Object.entries(outputs))await fs.writeFile(path.join(output,name),content,{flag:'wx'});
 return {output,configHash,modelFingerprint:config.modelFingerprint,parameters:config.parameters.length,
  metadataSha256,initialCheckpointSha256:initialSource?.sha256??null};
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 if(process.argv.slice(2).length===1&&process.argv[2]==='--help')console.log(
  'node scripts/prepare-motor-decoder-experiment.mjs [NEW_REPORT_DIRECTORY] [--backend=wasm|dawn-metal] [--initial=checkpoint.json]\nRequires prepared BANC data and runtime assets; only dawn-metal requires preexisting reports/native-webgpu-tooling. Never starts training or physics.');
 else prepareMotorDecoderExperiment(process.argv.slice(2)).then(result=>console.log(JSON.stringify(result)))
  .catch(error=>{console.error(error.stack||error);process.exitCode=1;});
}
