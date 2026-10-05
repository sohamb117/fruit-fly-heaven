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
function repin(bundle,config=JSON.parse(bundle.configText)){
  config.modelFingerprint=manifestFingerprint(config.assets);
  bundle.configText=JSON.stringify(config,null,2)+'\n';bundle.configHash=digest(bundle.configText);bundle.modelFingerprint=config.modelFingerprint;
  return bundle;
}
function sourceFixture(){
  const bundle=fixture(),config=JSON.parse(bundle.configText);
  Object.assign(bundle.assets,{
    '/training/episode.js':'export const profile = "frozen experiment";\n',
    '/banc-leg-proprioception.js':'export const receptor = "fixture";\n',
    '/banc-engine/src/index.js':'export const backend = "fixture";\n',
    '/body-engine/mujoco.js':'export const body = "fixture";\n',
    '/vendor/three.core.js':'export const rendering = "fixture";\n',
    '/body-model/banc-leg-proprioception-v1.json':'{"schema":1,"cells":[]}\n',
  });
  for(const [url,value]of Object.entries(bundle.assets))config.assets[url]=digest(value);
  return repin(bundle,config);
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

test('explicit experiment sources and leg catalog retain exact pinned bytes without changing configuration',()=>{
  const bundle=sourceFixture(),before=JSON.stringify(bundle),parsed=readExperimentBundle(before);
  assert.equal(parsed.configBytes.toString(),bundle.configText);
  assert.equal(parsed.overrides.size,Object.keys(bundle.assets).length+1);
  for(const [url,value]of Object.entries(bundle.assets))assert.deepEqual(parsed.overrides.get(url),Buffer.from(value));
  assert.equal(parsed.overrides.get('/training/config.json').toString(),bundle.configText);
  assert.equal(JSON.stringify(bundle),before);
  parsed.overrides.get('/training/episode.js')[0]=0;
  assert.equal(readExperimentBundle(before).overrides.get('/training/episode.js').toString(),bundle.assets['/training/episode.js']);
});

test('owned executable sources must be explicitly pinned and match their hashes',()=>{
  const unpinned=fixture();unpinned.assets['/banc-antenna.js']='replacement';
  assert.throws(()=>readExperimentBundle(JSON.stringify(unpinned)),/Unpinned experiment asset/);
  for(const url of ['/training/episode.js','/banc-engine/src/index.js','/body-model/banc-leg-proprioception-v1.json']){
    const bundle=sourceFixture();bundle.assets[url]+=' ';
    assert.throws(()=>readExperimentBundle(JSON.stringify(bundle)),/asset checksum mismatch/);
  }
});

test('even hash-pinned overrides cannot replace UI, coordinator, private data or native binaries',()=>{
  for(const extra of ['/training/view.js','/training/client.js','/training/optimizer.js','/training/config.json',
    '/train.html','/scripts/training_coordinator.py','/backup.sqlite3','/private.js','/banc-data/manifest.json',
    '/banc-engine/dist/core.wasm','/body-engine/mujoco.wasm','/body-model/unlisted.json']){
    const bundle=sourceFixture(),config=JSON.parse(bundle.configText);bundle.assets[extra]='replacement';config.assets[extra]=digest('replacement');repin(bundle,config);
    assert.throws(()=>readExperimentBundle(JSON.stringify(bundle)),/Unsupported experiment asset/);
  }
});

test('malformed envelopes, manifest digests and unsafe paths reject before yielding any overrides',()=>{
  for(const value of [null,[],{},'bundle'])assert.throws(()=>readExperimentBundle(JSON.stringify(value)),/Invalid experiment bundle/);
  for(const mutate of [b=>{delete b.assets['/body-model/flybody-mujoco.xml'];},b=>{b.assets=[];},
    b=>{b.assets['/training/episode.js']=42;}]){
    const bundle=sourceFixture();mutate(bundle);assert.throws(()=>readExperimentBundle(JSON.stringify(bundle)),/mismatch/);
  }
  for(const value of [null,[],{},'not-a-digest',42]){
    const bundle=sourceFixture(),config=JSON.parse(bundle.configText);config.assets['/training/episode.js']=value;repin(bundle,config);
    assert.throws(()=>readExperimentBundle(JSON.stringify(bundle)),/configuration identity mismatch/);
  }
  for(const url of ['/../secrets.js','/training/../private.js','/training//episode.js','/training/episode.js?x=1']){
    const bundle=sourceFixture();bundle.assets[url]='replacement';
    assert.throws(()=>readExperimentBundle(JSON.stringify(bundle)),/Unsafe bundle URL/);
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
