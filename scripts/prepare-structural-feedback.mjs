// Build owned, reproducible local evaluations without changing canonical
// training config or any coordinator state. All numerical tuning is a prior.
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {buildTrainingAssetManifest} from './prepare-training-manifest.mjs';
import {root,localAsset,sha} from './flight-feedback-runtime.mjs';
import {ANTENNA_FAMILY_PRIOR} from '../web/banc-antenna.js';
import {LEG_PROPRIOCEPTION_PRIOR} from '../web/banc-leg-proprioception.js';
import {createTegulaStrainPrior} from '../web/banc-tegula.js';

const args=Object.fromEntries(process.argv.slice(2).map(value=>{
 const m=/^--(base-bundle|output-dir)=(.+)$/.exec(value);assert(m,'Use --base-bundle=... --output-dir=...');return [m[1],m[2]];
}));
assert(args['base-bundle']&&args['output-dir'],'Explicit source bundle and fresh output directory required');
const input=path.resolve(root,args['base-bundle']),output=path.resolve(root,args['output-dir']);
assert(output.startsWith(path.join(root,'reports')+path.sep),'Output must stay inside reports');
const sourceBytes=await fs.readFile(input),source=JSON.parse(sourceBytes),base=JSON.parse(source.configText);
assert.equal(sha(source.configText),source.configHash,'Base config digest');
assert.equal(base.tegulaFeedback?.schema,1,'Base reference moment must come from the declared v1 tegula profile');
const profiles=await Promise.all(['directional','marginalized'].map(name=>fs.readFile(path.join(root,`models/banc-haltere-${name}-v2.json`),'utf8').then(JSON.parse)));
const bodyAssets={};
for(const url of ['/body-model/flybody-mujoco.xml','/body-model/flybody-mujoco.json']){
 const bytes=source.assets[url]===undefined?await fs.readFile(localAsset(url)):Buffer.from(source.assets[url]);
 assert.equal(sha(bytes),base.assets[url],'Source native body digest');bodyAssets[url]=bytes.toString('utf8');
}
await fs.mkdir(output,{recursive:true});
for(const [variant,haltere]of [['structural',profiles[0]],['marginalized',profiles[1]],['prior-feedback',null]]){
 const config=structuredClone(base);
 if(haltere){
  config.legProprioception=LEG_PROPRIOCEPTION_PRIOR;
  config.antennaFeedback={schema:1,windWorldCmPerSecond:[0,0,0],mechanics:ANTENNA_FAMILY_PRIOR};
  config.tegulaFeedback=createTegulaStrainPrior({halfLoadNative:base.tegulaFeedback.halfLoadNative});
  assert.equal(haltere.enabled,true,'Canonical haltere profile must be explicitly enabled');
  const {enabled,...settings}=haltere;config.haltereFeedback=settings;
 }
 config.environmentVersion+='-structural-proprioception-v1-'+variant;
 config.notes=[
  'Local structural-feedback evaluation. Biological graph, neural parameters, decoder coefficients and native body are retained from the source bundle.',
  'Sensory family mappings are annotation-constrained; individual polarity, mechanical gains, compliance and phase coefficients remain explicit unmeasured priors.',
  'Native-positive leg and mirrored tegula/haltere directions are configured priors. Unknown antenna families and unsupported leg joint/axis mappings abstain.',
  'No calibration, decoder fitting, coordinator contribution or broad behavioral success is established by this bundle.'
 ];
 const {assets}=await buildTrainingAssetManifest({config});
 for(const [url,value]of Object.entries(bodyAssets))assets[url]=sha(value);
 config.assets=Object.fromEntries(Object.entries(assets).sort(([a],[b])=>a.localeCompare(b)));
 config.modelFingerprint=sha(Object.entries(config.assets).map(([url,digest])=>`${url}:${digest}\n`).join(''));
 const owned={...bodyAssets};
 for(const url of Object.keys(config.assets))if(url.endsWith('.js')||url==='/body-model/banc-leg-proprioception-v1.json')
  owned[url]=(await fs.readFile(localAsset(url))).toString('utf8');
 const configText=JSON.stringify(config,null,2)+'\n',bundle={schemaVersion:1,kind:'flight-development-bundle',variant,
  createdAt:new Date().toISOString(),sourceConfigHash:source.configHash,sourceBundleSha256:sha(sourceBytes),
  configHash:sha(configText),modelFingerprint:config.modelFingerprint,configText,assets:owned,
  sourceArchive:{javascriptSources:Object.keys(owned).filter(url=>url.endsWith('.js')).length,
   externalAssets:'Prepared graph/data and native WASM remain mandatory hash-verified dependencies.'},
  scope:'One-fly local structural evaluation; no training or coordinator writes.'};
 const file=path.join(output,variant+'.bundle.json');await fs.writeFile(file,JSON.stringify(bundle,null,2)+'\n',{flag:'wx'});
 console.log(JSON.stringify({file,variant,configHash:bundle.configHash,modelFingerprint:bundle.modelFingerprint,ownedSources:bundle.sourceArchive.javascriptSources}));
}
