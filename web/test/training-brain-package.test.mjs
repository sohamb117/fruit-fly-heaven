import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {sampleBrain} from '../../scripts/prepare-training-brain.mjs';
import {openModelBundle,packageTrainingClient,manifestFingerprint} from '../../scripts/package-training-client.mjs';
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const record=bytes=>({bytes:bytes.length,sha256:sha(bytes)});

function anatomy(){
  const idsBytes=Buffer.alloc(5*8),positionsBytes=Buffer.alloc(4*12),ownersBytes=Buffer.alloc(4*4);
  for(let i=0;i<5;i++)idsBytes.writeBigUInt64LE(BigInt(100+i),i*8);
  [0,1,3,4].forEach((owner,k)=>{ownersBytes.writeUInt32LE(owner,k*4);for(let axis=0;axis<3;axis++)positionsBytes.writeFloatLE(owner*10+axis,k*12+axis*4);});
  const neuronsBytes=Buffer.from(JSON.stringify({ids:['100','101','102','103','104'],labels:['a','b','c','d','e']}));
  return {config:{modelFingerprint:'a'.repeat(64)},graph:{dataset:'BANC',materialization:888,neuron_count:5,files:{'ids.bin':record(idsBytes)}},
    metadata:{dataset:'BANC v888',neuronCount:5,positionMeaning:'Measured representative point',files:{'positions.bin':record(positionsBytes),'neuron-indices.bin':record(ownersBytes),'neurons.json':record(neuronsBytes)}},
    idsBytes,positionsBytes,ownersBytes,neuronsBytes,count:3};
}

test('brain sample preserves measured locations and exact graph identities; missing anchors stay omitted',()=>{
  const value=sampleBrain(anatomy());
  assert.deepEqual(value.indices,[0,1,4]);assert.deepEqual(value.ids,['100','101','104']);assert.deepEqual(value.labels,['a','b','e']);
  assert.deepEqual(value.positions,[0,1,2,10,11,12,40,41,42]);assert.equal(value.neuronCount,5);assert.equal(value.sampleCount,3);
});
test('brain geometry rejects stale hashes, neuron reorder, duplicate owners and invalid coordinates',()=>{
  for(const mutate of [
    a=>{a.positionsBytes[0]^=1;},
    a=>{a.neuronsBytes=Buffer.from(JSON.stringify({ids:['101','100','102','103','104'],labels:['a','b','c','d','e']}));a.metadata.files['neurons.json']=record(a.neuronsBytes);},
    a=>{a.ownersBytes.writeUInt32LE(0,4);a.metadata.files['neuron-indices.bin']=record(a.ownersBytes);},
    a=>{a.positionsBytes.writeFloatLE(NaN,0);a.metadata.files['positions.bin']=record(a.positionsBytes);},
    a=>{a.metadata.dataset='FlyWire';},
  ]){const a=anatomy();mutate(a);assert.throws(()=>sampleBrain(a));}
});
test('brain sample size is bounded and includes no invented missing neurons',()=>{
  assert.equal(sampleBrain({...anatomy(),count:4096}).sampleCount,4);
  for(const count of [0,4097,NaN,1.5])assert.throws(()=>sampleBrain({...anatomy(),count}));
});

async function bundle(t){
  const root=await fs.mkdtemp(path.join(os.tmpdir(),'training-frozen-'));t.after(()=>fs.rm(root,{recursive:true,force:true}));
  const config=Buffer.from('{"fixture":true}'),asset=Buffer.from('released runtime');
  const configHash=sha(config),modelFingerprint='b'.repeat(64);
  await fs.mkdir(path.join(root,'training'));await fs.writeFile(path.join(root,'training/config.json'),config);await fs.writeFile(path.join(root,'runtime.js'),asset);
  const manifest={kind:'local-contributor-client',configHash,modelFingerprint,files:[{path:'training/config.json',...record(config)},{path:'runtime.js',...record(asset)}]};
  await fs.writeFile(path.join(root,'bundle-manifest.json'),JSON.stringify(manifest));
  return {root,configHash,modelFingerprint,manifest};
}
test('frozen model source uses verified release bytes and refuses missing or modified assets',async t=>{
  const fixture=await bundle(t),frozen=await openModelBundle(fixture.root,fixture);
  assert.equal((await fs.readFile((await frozen.file('/runtime.js')).path)).toString(),'released runtime');
  await assert.rejects(()=>frozen.file('/missing.js'),/missing/);
  await fs.writeFile(path.join(fixture.root,'runtime.js'),'modified runtime');
  await assert.rejects(()=>frozen.file('/runtime.js'),/checksum/);
});
test('frozen model identity cannot be relabeled and paths cannot escape the bundle',async t=>{
  const fixture=await bundle(t);
  await assert.rejects(()=>openModelBundle(fixture.root,{...fixture,configHash:'0'.repeat(64)}),/identity/);
  fixture.manifest.files.push({path:'../private',bytes:1,sha256:'0'.repeat(64)});
  await fs.writeFile(path.join(fixture.root,'bundle-manifest.json'),JSON.stringify(fixture.manifest));
  await assert.rejects(()=>openModelBundle(fixture.root,fixture),/Unsafe/);
});
test('frozen model rejects symlinks and detects manifest changes during packaging',async t=>{
  const fixture=await bundle(t),frozen=await openModelBundle(fixture.root,fixture);
  await fs.unlink(path.join(fixture.root,'runtime.js'));await fs.symlink('training/config.json',path.join(fixture.root,'runtime.js'));
  await assert.rejects(()=>frozen.file('/runtime.js'),/Symlink/);
  await fs.appendFile(path.join(fixture.root,'bundle-manifest.json'),' ');
  await assert.rejects(()=>frozen.verifyManifest(),/changed/);
});
test('a symlink output alias cannot replace the frozen input bundle',async t=>{
  const root=await fs.mkdtemp(path.join(os.tmpdir(),'training-output-alias-'));t.after(()=>fs.rm(root,{recursive:true,force:true}));
  const xml='<mujoco/>',metadata=JSON.stringify({xml_sha256:sha(xml)});
  const overrides={'/body-model/flybody-mujoco.json':metadata,'/body-model/flybody-mujoco.xml':xml};
  const assets=Object.fromEntries(Object.entries(overrides).map(([url,text])=>[url,sha(text)]));
  const configText=JSON.stringify({assets,modelFingerprint:manifestFingerprint(assets)}),configHash=sha(configText),modelFingerprint=manifestFingerprint(assets);
  const output=path.join(root,'output'),frozen=path.join(output,'fruit-fly-training-client-'+configHash.slice(0,12)),alias=path.join(root,'alias');
  await fs.mkdir(path.join(frozen,'training'),{recursive:true});await fs.symlink(output,alias);
  await fs.writeFile(path.join(frozen,'training/config.json'),configText);
  await fs.writeFile(path.join(frozen,'bundle-manifest.json'),JSON.stringify({kind:'local-contributor-client',configHash,modelFingerprint,
    files:[{path:'training/config.json',...record(Buffer.from(configText))}]}));
  const experiment=path.join(root,'experiment.json');
  await fs.writeFile(experiment,JSON.stringify({schemaVersion:1,kind:'flight-development-bundle',configText,configHash,modelFingerprint,assets:overrides}));
  await assert.rejects(()=>packageTrainingClient({experimentBundle:experiment,modelBundle:frozen,outputDirectory:alias}),/preserve the frozen model bundle/);
  assert.equal(await fs.readFile(path.join(frozen,'training/config.json'),'utf8'),configText);
});
