import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import test from 'node:test';
import {readExperimentBundle,manifestFingerprint} from '../../scripts/package-training-client.mjs';

const digest=value=>createHash('sha256').update(value).digest('hex');
function fixture(){
  const xml='<mujoco model="experiment"/>\n';
  const metadata=JSON.stringify({xml_sha256:digest(xml)});
  const assets={'/body-model/flybody-mujoco.xml':xml,'/body-model/flybody-mujoco.json':metadata};
  const config={schemaVersion:2,assets:Object.fromEntries(Object.entries(assets).map(([url,value])=>[url,digest(value)]))};
  config.assets['/training/episode.js']=digest('pinned executable');
  config.modelFingerprint=manifestFingerprint(config.assets);
  const configText=JSON.stringify(config,null,2)+'\n';
  return {schemaVersion:1,kind:'flight-development-bundle',configText,configHash:digest(configText),modelFingerprint:config.modelFingerprint,assets};
}

test('experiment packaging preserves exact scientific configuration and only overrides physical model assets',()=>{
  const bundle=fixture(),parsed=readExperimentBundle(JSON.stringify(bundle));
  assert.equal(parsed.configBytes.toString(),bundle.configText);
  assert.equal(parsed.overrides.get('/training/config.json').toString(),bundle.configText);
  assert.equal(parsed.overrides.size,3);
  assert.equal(parsed.overrides.has('/training/episode.js'),false);
  for(const [url,value] of Object.entries(bundle.assets))assert.equal(parsed.overrides.get(url).toString(),value);
});

test('changed configuration or model bytes cannot silently enter the packaged experiment',()=>{
  for(const mutate of [b=>{b.configText+=' ';},b=>{b.modelFingerprint='a'.repeat(64);},b=>{b.assets['/body-model/flybody-mujoco.xml']+=' ';}]){
    const bundle=fixture();mutate(bundle);assert.throws(()=>readExperimentBundle(JSON.stringify(bundle)),/mismatch/);
  }
});

test('experiment bundle cannot replace executable assets or add private files',()=>{
  for(const extra of ['/training/episode.js','/backup.sqlite3','/../secrets.env']){
    const bundle=fixture();bundle.assets[extra]='replacement';
    assert.throws(()=>readExperimentBundle(JSON.stringify(bundle)),/identity mismatch/);
  }
});

test('matching config hashes cannot disguise disagreement between model XML and its metadata',()=>{
  const bundle=fixture(),config=JSON.parse(bundle.configText);
  bundle.assets['/body-model/flybody-mujoco.json']=JSON.stringify({xml_sha256:'0'.repeat(64)});
  config.assets['/body-model/flybody-mujoco.json']=digest(bundle.assets['/body-model/flybody-mujoco.json']);
  config.modelFingerprint=manifestFingerprint(config.assets);
  bundle.configText=JSON.stringify(config);bundle.configHash=digest(bundle.configText);bundle.modelFingerprint=config.modelFingerprint;
  assert.throws(()=>readExperimentBundle(JSON.stringify(bundle)),/metadata\/XML mismatch/);
});
