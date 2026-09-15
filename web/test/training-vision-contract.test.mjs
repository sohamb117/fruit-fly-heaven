import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {validateTrainingVisionConfig,validateTrainingConfigSchema} from '../training/config-schema.js';
import {parameterValues} from '../training/episode.js';
import {buildTrainingAssetManifest} from '../../scripts/prepare-training-manifest.mjs';
const base=JSON.parse(await fs.readFile(new URL('../training/config.json',import.meta.url)));
const enabled=()=>({...structuredClone(base),vision:true,visionFeedback:{schema:1,profile:'compact-retinal-motion-v1',width:256,height:128,frameIntervalMs:20}});

test('retinal declaration enables vision without changing the motor parameter contract',()=>{
  const config=enabled(),before=structuredClone(config),parameters=config.parameters.map(p=>p.initial);
  assert.equal(validateTrainingVisionConfig(config),config.visionFeedback);
  assert.equal(validateTrainingConfigSchema(config),config);
  assert.deepEqual(parameterValues(config,parameters),parameterValues(base,parameters));
  assert.deepEqual(config,before);
  assert.equal(validateTrainingVisionConfig(base),null);
});

test('a mismatched flag, unsupported profile, or nondeterministic sampling cadence fails before parameter application',()=>{
  const changes=[
    c=>{c.vision=false;},c=>{delete c.vision;},c=>{c.vision='true';},c=>{delete c.visionFeedback;},
    c=>{c.visionFeedback=null;},c=>{c.visionFeedback=[];},c=>{c.visionFeedback.schema=2;},
    c=>{c.visionFeedback.profile='unknown';},c=>{c.visionFeedback.width=31;},
    c=>{c.visionFeedback.height=15;},c=>{c.visionFeedback.width=1025;},c=>{c.visionFeedback.height=128.5;},
    c=>{c.visionFeedback.frameIntervalMs=1;},c=>{c.visionFeedback.frameIntervalMs=101;},
    c=>{c.visionFeedback.frameIntervalMs=3;},c=>{c.visionFeedback.frameIntervalMs=NaN;},
    c=>{c.visionFeedback.camera=[];},c=>{c.visionFeedback.motion=null;},c=>{c.visionFeedback.framIntervalMs=20;},
  ];
  for(const change of changes){const config=enabled();change(config);assert.throws(()=>validateTrainingVisionConfig(config),/retinal feedback/);assert.throws(()=>parameterValues(config,config.parameters.map(p=>p.initial)));}
  const absent=structuredClone(base);delete absent.vision;
  assert.throws(()=>parameterValues(absent,absent.parameters.map(p=>p.initial)),/vision flag/);
});

test('config-aware asset discovery includes retinal projection and static sensory dependencies',async()=>{
  const result=await buildTrainingAssetManifest({config:enabled()});
  for(const url of ['/banc-data/console/visual-projections.json','/training/sensory-feedback.js','/training/retinal-sensor.js','/training/compact-vision.js','/banc-antenna.js','/training/config-schema.js'])
    assert.match(result.assets[url]??'',/^[a-f0-9]{64}$/);
});
