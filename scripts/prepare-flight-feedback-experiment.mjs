import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {buildTrainingAssetManifest} from './prepare-training-manifest.mjs';
import {ANTENNA_PRIOR} from '../web/banc-antenna.js';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const output=path.resolve(process.argv[2]||path.join(root,'reports/sensory-feedback-20260915/current'));
if(!output.startsWith(path.join(root,'reports')+path.sep))throw new Error('Experiment output must be inside reports');
const sha=x=>createHash('sha256').update(x).digest('hex');
const originalDirectory=path.join(root,'dist/training-lease-client/fruit-fly-training-client-1f3b0935af59');
const originalBytes=await fs.readFile(path.join(originalDirectory,'training/config.json'));
const original=JSON.parse(originalBytes);
if(original.parameters.length!==672||original.stage!=='maintained_flight')throw new Error('Expected frozen 672-parameter flight experiment');
const {assets}=await buildTrainingAssetManifest({extraUrls:['/banc-data/console/visual-projections.json']});
const overrides={};
for(const url of ['/body-model/flybody-mujoco.xml','/body-model/flybody-mujoco.json']){
  overrides[url]=await fs.readFile(path.join(originalDirectory,url.slice(1)),'utf8');
  if(sha(overrides[url])!==original.assets[url])throw new Error('Original body asset differs from its configuration');
  assets[url]=original.assets[url];
}
const checkpointPath='reports/flight-decoder-feasibility-20260914/refinement-v2/checkpoint.json';
const checkpointBytes=await fs.readFile(path.join(root,checkpointPath)),checkpoint=JSON.parse(checkpointBytes);
if(checkpoint.parameters.length!==672)throw new Error('Wrong decoder checkpoint');
const sorted=Object.fromEntries(Object.entries(assets).sort(([a],[b])=>a.localeCompare(b)));
const fingerprint=sha(Object.entries(sorted).map(([url,digest])=>`${url}:${digest}\n`).join(''));
await fs.mkdir(output,{recursive:true});
for(const variant of ['off','vision','airflow','vision-airflow']){
  const config=structuredClone(original);
  config.assets=sorted;config.modelFingerprint=fingerprint;
  config.environmentVersion+='-sensory-feedback-v2-'+variant;
  config.parameters.forEach((p,i)=>p.initial=checkpoint.parameters[i]);
  config.vision=variant.includes('vision');
  if(config.vision)config.visionFeedback={schema:1,profile:'compact-retinal-motion-v1',width:256,height:128,frameIntervalMs:20,camera:{textureAntialias:true}};
  if(variant.includes('airflow'))config.antennaFeedback={schema:1,windWorldCmPerSecond:[0,0,0],mechanics:ANTENNA_PRIOR};
  config.notes=['Local sensory identification candidate. Same original native body, BANC graph and fitted-v2 decoder; only declared sensory feedback differs.',
    'Camera optics, receptive fields and antenna mechanics are explicit priors. Only mapped T4/T5 receive image-derived input; antenna feedback replaces the legacy speed/tilt proxy.',
    'Local evaluations only. This bundle does not submit coordinator results, fit parameters, or establish biological calibration or reliable flight.'];
  const configText=JSON.stringify(config,null,2)+'\n';
  const bundle={schemaVersion:1,kind:'flight-development-bundle',createdAt:new Date().toISOString(),sourceConfigHash:sha(originalBytes),
    configHash:sha(configText),modelFingerprint:fingerprint,variant,configText,assets:overrides,
    checkpoint:{file:checkpointPath,sha256:sha(checkpointBytes)},scope:'Local only; no coordinator writes. Full native body and 672 fitted decoder coefficients retained.'};
  await fs.writeFile(path.join(output,variant+'.bundle.json'),JSON.stringify(bundle,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({variant,configHash:bundle.configHash,modelFingerprint:fingerprint,output}));
}
