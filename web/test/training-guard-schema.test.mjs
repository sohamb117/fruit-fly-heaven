import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {validateTrainingConfigSchema} from '../training/config-schema.js';
import {validateConfig, makeGeneration, updateGeneration, checkpoint, readCheckpoint} from '../training/optimizer.js';
import {parameterValues} from '../training/episode.js';

const base = JSON.parse(fs.readFileSync(new URL('../training/config.json', import.meta.url)));
const guarded = () => {
  const config = structuredClone(base);
  config.schemaVersion = 2;
  config.optimizer.acceptance = {profile: 1, proposal: 'best-search-job', seedCount: 3,
    nativeExecution: {backend: 'dawn-metal', moduleSha256: 'a'.repeat(64), packageLockSha256: 'b'.repeat(64)}};
  return config;
};

test('legacy schema remains unchanged and cannot silently request guarded acceptance', () => {
  const original = structuredClone(base);
  assert.equal(validateTrainingConfigSchema(original), original);
  assert.deepEqual(original, base);
  for (const acceptance of [null, false, {}, guarded().optimizer.acceptance]) {
    const config = structuredClone(base); config.optimizer.acceptance = acceptance;
    assert.throws(() => validateConfig(config), /version 2/);
    assert.throws(() => parameterValues(config, config.parameters.map(p => p.initial)), /version 2/);
  }
});

test('guarded assignments apply exactly the same 27 physical parameters', () => {
  const config = guarded(), parameters = config.parameters.map(p => p.initial);
  assert.equal(validateConfig(config), config);
  assert.deepEqual(parameterValues(config, parameters), parameterValues(base, parameters));
  const saved = checkpoint(config, 'config-pin', parameters, 3, config.stage);
  assert.equal(saved.schemaVersion, 1);
  assert.deepEqual(readCheckpoint(saved, config, 'config-pin').parameters, parameters);
});

test('guarded search and retention cannot run through the legacy local optimizer', () => {
  const config = guarded(), parameters = config.parameters.map(p => p.initial);
  assert.throws(() => makeGeneration(parameters, 0, config), /coordinator/);
  assert.throws(() => updateGeneration({baseline: parameters, pairs: []}, config), /coordinator/);
  assert.ok(makeGeneration(parameters, 0, base).pairs.length > 0);
});

test('unsupported or incomplete guard profiles reject before applying parameters', () => {
  const changes = [
    c => { c.schemaVersion = 3; }, c => { delete c.optimizer.acceptance; },
    c => { c.optimizer.acceptance = null; }, c => { c.optimizer.acceptance = []; },
    c => { c.optimizer.acceptance.profile = true; }, c => { c.optimizer.acceptance.profile = 2; },
    c => { c.optimizer.acceptance.seedCount = 2; }, c => { c.optimizer.acceptance.seedCount = '3'; },
    c => { c.optimizer.acceptance.proposal = 'es'; }, c => { c.optimizer.acceptance.minimumImprovement = 0; },
    c => { delete c.optimizer.acceptance.nativeExecution; },
    c => { c.optimizer.acceptance.nativeExecution.backend = 'wasm'; },
    c => { c.optimizer.acceptance.nativeExecution.moduleSha256 = 'A'.repeat(64); },
    c => { c.optimizer.acceptance.nativeExecution.packageLockSha256 = 'b'.repeat(63); },
    c => { c.optimizer.acceptance.nativeExecution.fallbackAllowed = true; },
  ];
  for (const change of changes) {
    const config = guarded(); change(config);
    assert.throws(() => validateTrainingConfigSchema(config));
    assert.throws(() => parameterValues(config, config.parameters.map(p => p.initial)));
  }
});

test('WASM guard pins exactly the neural binary already in the model manifest', () => {
  const config=guarded(),moduleSha256=base.assets['/banc-engine/dist/core.wasm'];
  config.optimizer.acceptance.nativeExecution={backend:'wasm',moduleSha256};
  const original=structuredClone(config);
  assert.equal(validateConfig(config),config);assert.deepEqual(config,original);
  assert.deepEqual(parameterValues(config,config.parameters.map(p=>p.initial)),parameterValues(base,base.parameters.map(p=>p.initial)));
  assert.throws(()=>makeGeneration(config.parameters.map(p=>p.initial),0,config),/coordinator/);
  const invalid=[
    c=>{delete c.assets['/banc-engine/dist/core.wasm'];},
    c=>{c.assets['/banc-engine/dist/core.wasm']='f'.repeat(64);},
    c=>{c.optimizer.acceptance.nativeExecution.moduleSha256='f'.repeat(64);},
    c=>{c.optimizer.acceptance.nativeExecution.packageLockSha256='b'.repeat(64);},
    c=>{c.optimizer.acceptance.nativeExecution.backend='auto';},
    c=>{c.optimizer.acceptance.nativeExecution.fallbackAllowed=true;},
    c=>{c.optimizer.acceptance.nativeExecution={backend:'wasm'};},
  ];
  for(const change of invalid){const candidate=structuredClone(config);change(candidate);assert.throws(()=>validateTrainingConfigSchema(candidate),/execution pin/);}
});
