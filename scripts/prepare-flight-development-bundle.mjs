// Reduced flight benchmark, deliberately separate from the hosted/full model.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {PARAMETER_NAMES,parameterValues} from '../web/training/episode.js';
import {createMotorExcitation} from '../web/flybody-motor-excitation.js';
import {DEFAULT_WING_EVENT_PRIORS} from '../web/flybody-wing-event-excitation.js';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const output=path.resolve(process.argv[2]||'reports/flight-development/bundle.json');
if(!output.startsWith(path.join(root,'reports')+path.sep))throw new Error('Bundle must be inside reports');
const sha=value=>createHash('sha256').update(value).digest('hex');
const variantArgs=process.argv.slice(3),referenceArg=variantArgs.find(value=>value.startsWith('--force-reference='));
const wingEvents=variantArgs.includes('--wing-events');
const dlmIonic=variantArgs.includes('--dlm-ionic');
const tegulaLoad=variantArgs.includes('--tegula-load');
const flags=['--wing-events','--dlm-ionic','--tegula-load'];
const candidateArgs=variantArgs.filter(value=>!value.startsWith('--force-reference=')&&!flags.includes(value));
if(candidateArgs.length>1||candidateArgs.some(value=>value.startsWith('--'))||variantArgs.filter(value=>value.startsWith('--force-reference=')).length>1||flags.some(flag=>variantArgs.filter(value=>value===flag).length>1))throw new Error('Unknown or duplicate variant argument');
if((wingEvents||dlmIonic||tegulaLoad)&&(candidateArgs.length||referenceArg))throw new Error('Test event/ionic/sensory models independently of recruitment or force-reference changes');
let intrinsicModels;
if(dlmIonic){
 const {DLM_PROFILE,DLM_CELLS}=await import('../packages/banc-runtime/src/cell-models.js');
 intrinsicModels={schema:1,profile:DLM_PROFILE,cells:structuredClone(DLM_CELLS),ionic_step_ms:.1,initial_gates:{h:.146,b:.146},event_policy:'threshold-10ms-guard'};
}
let tegulaFeedback,tegulaProvenance;
if(tegulaLoad){
 const priorFile='reports/flight-tegula-inputs/recruitment-prior.json',priorBytes=await fs.readFile(path.join(root,priorFile)),prior=JSON.parse(priorBytes);
 if(prior.schemaVersion!==1||prior.kind!=='tegula-fluid-load-prior')throw new Error('Invalid tegula recruitment prior');
 const referencePath=path.resolve(root,prior.referenceFile);
 if(!(await fs.realpath(referencePath)).startsWith((await fs.realpath(path.join(root,'reports')))+path.sep))throw new Error('Tegula reference escaped reports');
 const referenceBytes=await fs.readFile(referencePath),reference=JSON.parse(referenceBytes);
 if(sha(referenceBytes)!==prior.referenceSha256||!reference.completed||!reference.sourceUnchanged||reference.explicitDiagnosticForwardEvaluations!==112)throw new Error('Unverified tegula reference');
 const values=reference.states.flatMap(s=>s.evaluations.find(e=>e.axis===null).wings.map(w=>w.magnitude)).sort((a,b)=>a-b);
 if(values.length!==32||prior.profile.schema!==1||prior.profile.profile!=='tegula-fluid-load-v1'||prior.profile.maxRateHz!==100||prior.profile.halfLoadNative!==(values[15]+values[16])/2)throw new Error('Tegula prior differs from fixed mechanical reference');
 for(const file of ['web/flybody-wings.js','web/training/flight-parameters.js','models/flybody-mujoco.xml','models/flybody-mujoco.json','models/flybody-wing-actuation.json'])if(sha(await fs.readFile(path.join(root,file)))!==reference.sourceHashes[file])throw new Error('Tegula mechanical reference changed: '+file);
 tegulaFeedback=structuredClone(prior.profile);tegulaProvenance={priorFile,priorSha256:sha(priorBytes),referenceFile:prior.referenceFile,referenceSha256:prior.referenceSha256};
}
const candidatePath=candidateArgs[0]?path.resolve(candidateArgs[0]):null;
let candidate=null,calibrationProvenance=null,calibration=null;
if(candidatePath){
 if(!(await fs.realpath(candidatePath)).startsWith((await fs.realpath(path.join(root,'reports')))+path.sep))throw new Error('Calibration candidate must be inside reports');
 const candidateBytes=await fs.readFile(candidatePath),calibrationPath=path.join(path.dirname(candidatePath),'result.json'),calibrationBytes=await fs.readFile(calibrationPath);
 candidate=JSON.parse(candidateBytes);calibration=JSON.parse(calibrationBytes);
 if(candidate.schemaVersion!==1||candidate.kind!=='mean-force-calibrated-steering-candidate'||candidate.sourceCalibrationSha256!==sha(calibrationBytes))throw new Error('Invalid or unpinned calibration candidate');
 if(JSON.stringify(candidate.parameterNames)!==JSON.stringify(PARAMETER_NAMES)||JSON.stringify(candidate.parameters)!==JSON.stringify(calibration.parameters)||JSON.stringify(candidate.recruitment)!==JSON.stringify(calibration.recruitment))throw new Error('Candidate differs from native calibration');
 if(!['legacyStatesByteExact','nonSteeringStatesByteExact','unchangedNonExcitationInputs','unchangedPowerMappings'].every(key=>calibration.gate[key]===true)||calibration.gate.nativeReplay.passed!==true)throw new Error('Native calibration checks did not pass');
 if(calibration.helperSha256!==sha(await fs.readFile(path.join(root,'web/flybody-motor-excitation.js'))))throw new Error('Recruitment implementation changed since calibration');
 createMotorExcitation(candidate.recruitment);
 calibrationProvenance={candidateFile:path.relative(root,candidatePath),candidateSha256:sha(candidateBytes),calibrationFile:path.relative(root,calibrationPath),calibrationSha256:sha(calibrationBytes),sourceCaptureSha256:calibration.captureSha256,window:calibration.window};
}
const original=await fs.readFile(path.join(root,'web/training/config.json'));
const config=JSON.parse(original),xml=await fs.readFile(path.join(root,'models/flybody-mujoco.xml'),'utf8');
if(calibration)for(const [url,digest]of Object.entries(calibration.sourceHashes))if(config.assets[url]!==digest)throw new Error('Native calibration source differs from current manifest: '+url);
let forceReference=null,referenceProvenance=null;
if(referenceArg){
 const referencePath=path.resolve(referenceArg.slice('--force-reference='.length));
 if(!(await fs.realpath(referencePath)).startsWith((await fs.realpath(path.join(root,'reports')))+path.sep))throw new Error('Force reference must be inside reports');
 const bytes=await fs.readFile(referencePath),reference=JSON.parse(bytes);
 const calibrationPath=path.resolve(root,reference.calibrationFile);
 if(!(await fs.realpath(calibrationPath)).startsWith((await fs.realpath(path.join(root,'reports')))+path.sep))throw new Error('Force reference calibration escaped reports');
 const calibrationBytes=await fs.readFile(calibrationPath),source=JSON.parse(calibrationBytes);
 if(reference.schemaVersion!==1||reference.kind!=='native-mean-steering-force-reference'||reference.calibrationSha256!==sha(calibrationBytes))throw new Error('Unpinned force reference');
 const field=candidate?'newMeanForce':'oldMeanForce';
 const expected=Object.fromEntries(source.perType.map(row=>[row.type,row[field]]));
 if(reference.recruitment!==(candidate?'hill80-n1':'legacy')||JSON.stringify(reference.values)!==JSON.stringify(expected))throw new Error('Force reference differs from the native calibration/recruitment');
 if(source.gate.legacyStatesByteExact!==true||source.gate.nonSteeringStatesByteExact!==true||source.gate.nativeReplay.passed!==true)throw new Error('Force reference native calibration failed');
 for(const [url,digest]of Object.entries(source.sourceHashes))if(config.assets[url]!==digest)throw new Error('Force reference source changed: '+url);
 forceReference=reference.values;referenceProvenance={referenceFile:path.relative(root,referencePath),referenceSha256:sha(bytes),calibrationFile:reference.calibrationFile,calibrationSha256:sha(calibrationBytes),window:source.window};
}
if(JSON.stringify(config.parameters.map(p=>p.name))!==JSON.stringify(PARAMETER_NAMES))throw new Error('Expected canonical 27 parameters');
const metadataText=await fs.readFile(path.join(root,'models/flybody-mujoco.json'),'utf8');
if(config.assets['/body-model/flybody-mujoco.xml']!==sha(xml)||config.assets['/body-model/flybody-mujoco.json']!==sha(metadataText))throw new Error('Source model differs from the canonical manifest');
let replacements=0;
const reduced=xml.replace(/(<default class="adhesion_claw">\s*<adhesion[^>]*gain=")([^"]+)(")/g,(_,a,_b,c)=>{replacements++;return a+'0'+c;});
if(replacements!==1)throw new Error('Expected one adhesion class');
const metadata=JSON.parse(metadataText);
metadata.xml_sha256=sha(reduced);
metadata.diagnosticVariant={clawAdhesionGainScale:0,sourceXmlSha256:sha(xml)};
if(wingEvents)metadata.diagnosticVariant.wingEventExcitation=structuredClone(DEFAULT_WING_EVENT_PRIORS);
if(dlmIonic)metadata.diagnosticVariant.intrinsicModels=structuredClone(intrinsicModels);
if(tegulaLoad){metadata.diagnosticVariant.tegulaFeedback=structuredClone(tegulaFeedback);metadata.diagnosticVariant.tegulaReference=tegulaProvenance;}
if(candidate){metadata.motor_excitation=candidate.recruitment;metadata.diagnosticVariant.steeringCalibration=calibrationProvenance;}
if(forceReference){metadata.wing_actuation.steering_force_reference=forceReference;metadata.diagnosticVariant.steeringForceReference=referenceProvenance;}
const assets={'/body-model/flybody-mujoco.xml':reduced,'/body-model/flybody-mujoco.json':JSON.stringify(metadata)+'\n'};
for(const [url,bytes]of Object.entries(assets))config.assets[url]=sha(bytes);
config.environmentVersion+='-no-adhesion-benchmark';
if(candidate)config.environmentVersion+='-hill80-n1-mean-matched';
if(forceReference)config.environmentVersion+='-observed-force-reference';
if(wingEvents){config.wingEventExcitation=structuredClone(DEFAULT_WING_EVENT_PRIORS);config.environmentVersion+='-wing-events-v1';}
if(dlmIonic){config.intrinsicModels=structuredClone(intrinsicModels);config.environmentVersion+='-dlm-snl-2023-v1';}
if(tegulaLoad){config.tegulaFeedback=structuredClone(tegulaFeedback);config.environmentVersion+='-tegula-fluid-load-v1';}
config.modelFingerprint=sha(Object.entries(config.assets).sort(([a],[b])=>a.localeCompare(b)).map(([url,digest])=>`${url}:${digest}\n`).join(''));
for(let i=0;i<config.parameters.length;i++)config.parameters[i].initial=candidate?candidate.parameters[i]:i===0?Math.log(1.5):i>=3?config.parameters[i].min:0;
parameterValues(config,config.parameters.map(p=>p.initial));
config.notes.push('Development-only reduced flight benchmark: normal foot collision/friction retained; phenomenological claw adhesion disabled. No gripping or climbing claim. The reference uses power 1.5 and steering 0.05, fixed operator-selected controls from the causal assay, not learned parameters.');
if(candidate)config.notes.push('Steering-only Hill recruitment is an explicit modeling prior. Per-type bilateral gains preserve aggregate mean native muscle force over the frozen 100–280 ms legacy calibration window; individual sides, temporal variation and closed-loop forces can differ. No reward fitting or flight validation was used to set these gains.');
if(forceReference)config.notes.push('A tied per-type steering-force reference centers the local response on mean native muscle force from the frozen 100–280 ms legacy trajectory. It removes the aggregate target-space operating-point offset, not the independent left/right difference. These are modeled reference values from an unstable diagnostic trajectory, not measured physiology or validated flight trim.');
if(wingEvents)config.notes.push('Development event interface: each of48 wing motor neurons drives an individual excitation kernel, then28 native wing muscle groups update at each completed1ms boundary. Other muscles keep their existing rate interface. Kernel dynamics are fixed modeled priors; no corrective controller or synthetic spikes enter this BANC evaluation.');
if(dlmIonic)config.notes.push('Experimental DLM intrinsic profile: exactly10 annotated DLM motor neurons use the published sodium/Shab conductance model with0.1ms substeps. Chemical graph, delays and other cell profiles stay fixed. No tonic current or electrical coupling from the reference network is added. Substep motor event times are retained; chemical release uses the existing0.5ms graph clock.');
if(tegulaLoad)config.notes.push('Experimental tegula input:26 existing annotated wing strain sensory neurons receive side-specific native aerodynamic hinge-load feedback. One common saturating recruitment prior uses the fixed mechanical trim median and100Hz cap. This is an isotropic load proxy, not measured receptor strain, total joint reaction, or cell-specific phase tuning. No desired posture or reward enters the sensory signal.');
const configText=JSON.stringify(config,null,2)+'\n';
const bundle={schemaVersion:1,kind:'flight-development-bundle',createdAt:new Date().toISOString(),
 sourceConfigHash:sha(original),configHash:sha(configText),modelFingerprint:config.modelFingerprint,
 scope:config.notes.slice(-(1+Number(!!candidate)+Number(!!forceReference)+Number(wingEvents)+Number(dlmIonic)+Number(tegulaLoad))).join(' '),calibrationProvenance,referenceProvenance,tegulaProvenance,configText,assets};
await fs.mkdir(path.dirname(output),{recursive:true});
await fs.writeFile(output,JSON.stringify(bundle,null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({output,configHash:bundle.configHash,modelFingerprint:bundle.modelFingerprint}));
