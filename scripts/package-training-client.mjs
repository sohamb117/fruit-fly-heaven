#!/usr/bin/env node
// Explicit, verified contributor bundle. Never copies a repository directory.
import fs from 'node:fs/promises';
import {createReadStream} from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {prepareTrainingBrain} from './prepare-training-brain.mjs';

const repo=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const defaultOutput=path.join(repo,'dist/training-client');
const digest=bytes=>createHash('sha256').update(bytes).digest('hex');
const json=value=>JSON.stringify(value,null,2)+'\n';
const prefixes=[['/body-engine/','packages/flybody-runtime/node_modules/@mujoco/mujoco/'],['/banc-engine/','packages/banc-runtime/'],['/banc-data/','data/prepared/banc888/'],['/body-model/','models/']];

export function safeUrl(url){
  if(typeof url!=='string'||!/^\/[A-Za-z0-9_./-]+$/.test(url)||url.includes('//')||url.split('/').slice(1).some(s=>!s||s==='.'||s==='..'||s.startsWith('.')))throw new Error('Unsafe bundle URL: '+url);
  return url;
}
export function manifestFingerprint(assets){return digest(Object.entries(assets).map(([url,sha])=>`${safeUrl(url)}:${sha}\n`).join(''));}
async function hashFile(file){const hash=createHash('sha256');for await(const chunk of createReadStream(file))hash.update(chunk);return hash.digest('hex');}
function repositoryPath(url){const match=prefixes.find(([prefix])=>url.startsWith(prefix));return match?match[1]+url.slice(match[0].length):'web'+url;}
async function sourceFile(relative){
  if(path.isAbsolute(relative)||relative.split('/').some(s=>s==='..'||s===''))throw new Error('Unsafe source path: '+relative);
  const source=path.join(repo,relative),real=await fs.realpath(source);
  if(!real.startsWith(repo+path.sep)||!(await fs.stat(real)).isFile())throw new Error('Source is not a regular repository file: '+relative);
  // Disallow symlink substitution, including any parent directory.
  for(let current=source;current!==repo;current=path.dirname(current))if((await fs.lstat(current)).isSymbolicLink())throw new Error('Symlink in bundle source: '+relative);
  return source;
}
function run(command,args,{input}={}){
  return new Promise((resolve,reject)=>{
    const child=spawn(command,args,{cwd:repo,stdio:['pipe','pipe','pipe']});let stdout='',stderr='';
    child.stdout.on('data',data=>stdout+=data);child.stderr.on('data',data=>stderr+=data);
    child.on('error',reject);child.on('close',code=>code===0?resolve(stdout):reject(new Error(`${command} failed (${code}): ${stderr||stdout}`)));
    child.stdin.end(input);
  });
}

const serverSource=`"""Serve only this verified contributor bundle on localhost. Python 3.9+."""
import argparse,hashlib,json
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote,urlsplit

ROOT=Path(__file__).resolve().parent
MANIFEST=json.loads((ROOT/'bundle-manifest.json').read_text())
ALLOWED={record['path'] for record in MANIFEST['files']}|{'bundle-manifest.json'}
def verify():
    for record in MANIFEST['files']:
        target=(ROOT/record['path']).resolve()
        if ROOT not in target.parents or not target.is_file():raise ValueError('Invalid bundle path: '+record['path'])
        sha=hashlib.sha256()
        with target.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):sha.update(chunk)
        if target.stat().st_size!=record['bytes'] or sha.hexdigest()!=record['sha256']:raise ValueError('Bundle checksum mismatch: '+record['path'])

class Handler(SimpleHTTPRequestHandler):
    extensions_map={**SimpleHTTPRequestHandler.extensions_map,'.wasm':'application/wasm','.js':'text/javascript','.mjs':'text/javascript','.json':'application/json','.bin':'application/octet-stream','.wgsl':'text/plain'}
    def send_head(self):
        route=unquote(urlsplit(self.path).path)
        relative='index.html' if route=='/' else route.removeprefix('/')
        target=(ROOT/relative).resolve()
        if relative not in ALLOWED or ROOT not in target.parents or not target.is_file():
            self.send_error(404);return None
        self.path='/'+relative
        return super().send_head()
    def end_headers(self):
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Cross-Origin-Opener-Policy','same-origin')
        self.send_header('Cross-Origin-Embedder-Policy','require-corp')
        super().end_headers()
    def log_message(self,*args):pass

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=7842)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args();verify()
    if args.verify_only:print(json.dumps({'verifiedFiles':len(MANIFEST['files']),'configHash':MANIFEST['configHash']}))
    else:
        server=ThreadingHTTPServer(('127.0.0.1',args.port),partial(Handler,directory=str(ROOT)))
        print(json.dumps({'url':'http://127.0.0.1:'+str(server.server_port)+'/train.html','configHash':MANIFEST['configHash'],'verifiedFiles':len(MANIFEST['files'])}),flush=True)
        try:server.serve_forever()
        except KeyboardInterrupt:pass
        finally:server.server_close()
`;

const archiveSource=`import hashlib,json,pathlib,sys,zipfile
root=pathlib.Path(sys.argv[1]);destination=pathlib.Path(sys.argv[2]);manifest=json.loads((root/'bundle-manifest.json').read_text())
paths=sorted([record['path'] for record in manifest['files']]+['bundle-manifest.json'])
with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as archive:
    for name in paths:
        source=root/name
        info=zipfile.ZipInfo(root.name+'/'+name,date_time=(2026,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
        with source.open('rb') as incoming,archive.open(info,'w',force_zip64=True) as outgoing:
            for block in iter(lambda:incoming.read(1024*1024),b''):outgoing.write(block)
with zipfile.ZipFile(destination) as archive:
    assert archive.testzip() is None
    for record in manifest['files']:
        sha=hashlib.sha256();count=0
        with archive.open(root.name+'/'+record['path']) as incoming:
            for block in iter(lambda:incoming.read(1024*1024),b''):sha.update(block);count+=len(block)
        assert sha.hexdigest()==record['sha256'] and count==record['bytes'],record['path']
print(json.dumps({'archiveEntries':len(paths),'archiveIntegrity':'passed'}))
`;

async function httpVerify(bundle,manifest){
  const child=spawn('uv',['run','--offline','python',path.join(bundle,'serve.py'),'--port','0'],{cwd:repo,stdio:['ignore','pipe','pipe']});
  const closed=new Promise(resolve=>child.once('close',resolve));
  let stderr='';child.stderr.on('data',data=>stderr+=data);
  try{
    const ready=await new Promise((resolve,reject)=>{
      let output='';const timeout=setTimeout(()=>reject(new Error('Bundle server startup timeout: '+stderr)),30000);
      child.on('error',error=>{clearTimeout(timeout);reject(error);});child.on('exit',code=>{clearTimeout(timeout);reject(new Error('Bundle server exited '+code+': '+stderr));});
      child.stdout.on('data',data=>{output+=data;if(output.includes('\n')){clearTimeout(timeout);try{resolve(JSON.parse(output.split('\n')[0]));}catch(error){reject(error);}}});
    });
    const origin=new URL(ready.url).origin;
    for(const record of manifest.files){
      const response=await fetch(origin+'/'+record.path);if(!response.ok)throw new Error('HTTP bundle asset missing: '+record.path);
      if(record.path.endsWith('.wasm')&&response.headers.get('content-type')!=='application/wasm')throw new Error('Wrong WASM MIME');
      if(response.headers.get('cross-origin-opener-policy')!=='same-origin'||response.headers.get('cross-origin-embedder-policy')!=='require-corp')throw new Error('Missing isolation headers');
      const hash=createHash('sha256');let bytes=0;for await(const block of response.body){hash.update(block);bytes+=block.length;}
      if(bytes!==record.bytes||hash.digest('hex')!==record.sha256)throw new Error('HTTP checksum mismatch: '+record.path);
    }
    const rootResponse=await fetch(origin+'/');if(!rootResponse.ok||!(await rootResponse.text()).includes('/train.html'))throw new Error('Bundle root does not lead to training');
    for(const forbidden of ['/scripts/training_coordinator.py','/.env','/data/training/coordinator.sqlite3','/../LICENSE-not-allowed'])if((await fetch(origin+forbidden)).status!==404)throw new Error('Unexpected exposed path: '+forbidden);
    return {httpVerifiedFiles:manifest.files.length,headers:'passed',root:'training redirect',unlistedPaths:'rejected',neuralExecution:false};
  }finally{
    child.kill('SIGTERM');await closed;
  }
}

// An explicit experiment may own the runtime it pins, including sources that
// differ from the current checkout. Keep this execution-only allowlist separate
// from presentation/coordinator code and from arbitrary data or native binaries.
const experimentSourceUrls=new Set([
  '/banc-antenna.js','/banc-ground-sense.js','/banc-haltere.js','/banc-leg-proprioception.js',
  '/banc-proboscis.js','/banc-sensory-current.js','/banc-taste.js','/banc-tegula.js','/banc/embodiment.js',
  '/body-world.js','/color-vision.js','/flight-scene-profile.js','/motor-decoder.js','/sensory-encoder.js','/virtual-haltere.js',
  ...['contact-environment','habitat-collision','leg-actuation','motor-excitation','mouth-pose','physics','stance',
    'wing-event-excitation','wing-load','wing-pose','wings','world'].map(name=>'/flybody-'+name+'.js'),
  ...['airborne-reset-contract','airborne-reset','brain-resources','compact-vision','config-schema','environment',
    'episode','flight-objective','flight-observation','flight-parameters','flight-telemetry','maintained-flight-objective',
    'retinal-sensor','sensory-feedback','worker','sensorimotor-parameters','recovery-objective',
    'sequential-environment','decoder-calibration','flight-teacher'].map(name=>'/training/'+name+'.js'),
  ...['cell-models','index','model','motor-events','wasm','webgpu'].map(name=>'/banc-engine/src/'+name+'.js'),
  '/banc-engine/dist/core.js','/body-engine/mujoco.js','/vendor/three.core.js','/vendor/three.module.js',
]);
const experimentModelUrls=['/body-model/flybody-mujoco.json','/body-model/flybody-mujoco.xml'];
const experimentCatalogUrl='/body-model/banc-leg-proprioception-v1.json';
const record=value=>value!==null&&typeof value==='object'&&!Array.isArray(value);
const sha256=value=>typeof value==='string'&&/^[a-f0-9]{64}$/.test(value);

export function readExperimentBundle(bytes){
  const bundle=JSON.parse(bytes);
  if(!record(bundle)||bundle.schemaVersion!==1||bundle.kind!=='flight-development-bundle'||typeof bundle.configText!=='string')throw new Error('Invalid experiment bundle');
  const configBytes=Buffer.from(bundle.configText),config=JSON.parse(configBytes),configHash=digest(configBytes);
  if(!record(config)||!record(config.assets)||!Object.keys(config.assets).length||!Object.values(config.assets).every(sha256)||
    !record(bundle.assets)||!experimentModelUrls.every(url=>Object.hasOwn(bundle.assets,url))||
    !sha256(config.modelFingerprint)||
    bundle.configHash!==configHash||bundle.modelFingerprint!==config.modelFingerprint||
    manifestFingerprint(config.assets)!==config.modelFingerprint)throw new Error('Experiment configuration identity mismatch');
  const overrides=new Map();
  for(const [url,value]of Object.entries(bundle.assets)){
    safeUrl(url);
    if(!experimentModelUrls.includes(url)&&!experimentSourceUrls.has(url)&&url!==experimentCatalogUrl&&url!=='/body-model/flight-teacher-calibration-v1.json')
      throw new Error('Unsupported experiment asset: '+url);
    if(!Object.hasOwn(config.assets,url))throw new Error('Unpinned experiment asset: '+url);
    if(typeof value!=='string'||digest(value)!==config.assets[url])throw new Error('Experiment asset checksum mismatch: '+url);
    overrides.set(url,Buffer.from(value));
  }
  if(JSON.parse(bundle.assets[experimentModelUrls[0]]).xml_sha256!==digest(bundle.assets[experimentModelUrls[1]]))throw new Error('Experiment metadata/XML mismatch');
  overrides.set('/training/config.json',configBytes);
  return {configBytes,overrides};
}

// UI-only releases can retain the exact released simulation, including its
// matching native source. Never substitute locally edited model files.
export async function openModelBundle(directory,{configHash,modelFingerprint}){
  const root=await fs.realpath(directory),manifestPath=path.join(root,'bundle-manifest.json');
  const bytes=await fs.readFile(manifestPath),manifest=JSON.parse(bytes),manifestHash=digest(bytes);
  if(manifest.kind!=='local-contributor-client'||manifest.configHash!==configHash||manifest.modelFingerprint!==modelFingerprint||!Array.isArray(manifest.files))throw new Error('Frozen model bundle identity mismatch');
  const records=new Map();
  for(const record of manifest.files){
    const url=safeUrl('/'+record.path);
    if(records.has(url)||!Number.isSafeInteger(record.bytes)||record.bytes<0||!/^[a-f0-9]{64}$/.test(record.sha256))throw new Error('Invalid frozen model bundle manifest');
    records.set(url,record);
  }
  async function file(url){
    safeUrl(url);const record=records.get(url);if(!record)throw new Error('Frozen model bundle is missing '+url);
    const target=path.join(root,url.slice(1));
    for(let current=target;current!==root;current=path.dirname(current))if((await fs.lstat(current)).isSymbolicLink())throw new Error('Symlink in frozen model bundle');
    if(!(await fs.stat(target)).isFile()||(await fs.stat(target)).size!==record.bytes||await hashFile(target)!==record.sha256)throw new Error('Frozen model bundle checksum mismatch: '+url);
    return {path:target,record};
  }
  const config=await file('/training/config.json');
  if(config.record.sha256!==configHash)throw new Error('Frozen model bundle configuration mismatch');
  return {root,file,has:url=>records.has(safeUrl(url)),async verifyManifest(){if(await hashFile(manifestPath)!==manifestHash)throw new Error('Frozen model manifest changed during packaging');}};
}

export async function packageTrainingClient({experimentBundle=null,modelBundle=null,outputDirectory=defaultOutput}={}){
  await fs.mkdir(path.resolve(outputDirectory),{recursive:true});
  const output=await fs.realpath(outputDirectory);
  const configSource=experimentBundle?path.resolve(experimentBundle):path.join(repo,'web/training/config.json');
  const inputBytes=await fs.readFile(configSource),inputHash=digest(inputBytes);
  const {configBytes,overrides}=experimentBundle?readExperimentBundle(inputBytes):{configBytes:inputBytes,overrides:new Map()};
  const config=JSON.parse(configBytes),configHash=digest(configBytes);
  if(!config.assets||Object.keys(config.assets).length<1||manifestFingerprint(config.assets)!==config.modelFingerprint)throw new Error('Finalize the canonical training asset manifest first');
  const id='fruit-fly-training-client-'+configHash.slice(0,12),bundle=path.join(output,id);
  const frozen=modelBundle?await openModelBundle(modelBundle,{configHash,modelFingerprint:config.modelFingerprint}):null;
  if(frozen&&path.resolve(bundle)===frozen.root)throw new Error('Choose a different --output-dir to preserve the frozen model bundle');
  await fs.mkdir(output,{recursive:true});const staging=await fs.mkdtemp(path.join(output,'.pack-')),stage=path.join(staging,id);await fs.mkdir(stage);
  const records=new Map(),frozenSources=new Map();
  async function add(url,{relative=repositoryPath(url),expectedHash,expectedBytes,kind='model'}={}){
    safeUrl(url);const existing=records.get(url);
    if(existing){if(expectedHash&&existing.sha256!==expectedHash||expectedBytes!==undefined&&existing.bytes!==expectedBytes)throw new Error('Conflicting bundle identity: '+url);return;}
    const target=path.join(stage,url.slice(1));await fs.mkdir(path.dirname(target),{recursive:true});
    const useFrozen=frozen&&(Object.hasOwn(config.assets,url)||kind==='graph'||kind==='source'||kind==='license'||url.startsWith('/banc-engine/')||url.startsWith('/body-engine/'));
    if(useFrozen){const source=await frozen.file(url);await fs.copyFile(source.path,target);frozenSources.set(url,source);}
    else if(overrides.has(url))await fs.writeFile(target,overrides.get(url));
    else await fs.copyFile(await sourceFile(relative),target);
    const sha256=await hashFile(target),bytes=(await fs.stat(target)).size;
    if(expectedHash&&sha256!==expectedHash||expectedBytes!==undefined&&bytes!==expectedBytes)throw new Error('Source checksum mismatch: '+relative);
    if(useFrozen&&sha256!==frozenSources.get(url).record.sha256)throw new Error('Frozen model bundle changed during copy: '+url);
    records.set(url,{path:url.slice(1),bytes,sha256,kind,source:useFrozen?'frozen model bundle '+path.relative(repo,frozen.root)+url:overrides.has(url)?'generated experiment asset from '+path.relative(repo,configSource):relative});
  }
  async function generated(url,content,kind='instructions',source='generated by scripts/package-training-client.mjs'){
    safeUrl(url);if(records.has(url))throw new Error('Duplicate generated path: '+url);const target=path.join(stage,url.slice(1));await fs.mkdir(path.dirname(target),{recursive:true});await fs.writeFile(target,content);
    records.set(url,{path:url.slice(1),bytes:Buffer.byteLength(content),sha256:digest(content),kind,source});
  }
  try{
    for(const [url,sha] of Object.entries(config.assets)){if(!/^[a-f0-9]{64}$/.test(sha))throw new Error('Invalid asset digest: '+url);await add(url,{expectedHash:sha});}
    const graph=JSON.parse(await fs.readFile(path.join(stage,'banc-data/manifest.json'),'utf8'));
    if(graph.dataset!=='BANC'||graph.materialization!==888||!graph.files)throw new Error('BANC v888 graph manifest required');
    for(const [name,record] of Object.entries(graph.files)){
      if(!/^[a-zA-Z0-9_-]+\.(bin|json)$/.test(name)||!/^[a-f0-9]{64}$/.test(record.sha256)||!Number.isSafeInteger(record.bytes)||record.bytes<0)throw new Error('Unsafe graph file: '+name);
      await add('/banc-data/'+name,{expectedHash:record.sha256,expectedBytes:record.bytes,kind:'graph'});
    }
    for(const url of ['/train.html','/training/config.json','/training/view.js','/training/client.js','/training/optimizer.js','/training/preview.js','/training/brain-preview.js','/training/eye-preview.js','/training/eye-observer-runtime.js','/training/observer-worker.js','/training/observer-runtime.js','/training/train.css','/training/package.json'])await add(url,{kind:'client',...(url==='/training/config.json'?{expectedHash:configHash}:{})});
    await generated('/training/brain-sample.json',JSON.stringify(await prepareTrainingBrain(config))+'\n','presentation');
    // Explicit entry points above define the code allowlist. Reject missing code
    // imports; local HTML/CSS images/fonts may be included only below /assets,
    // /fonts, /training or /vendor. No directory recursion and no remote assets.
    for(const record of records.values()){
      if(!/\.(html|css|js)$/.test(record.path))continue;
      const text=await fs.readFile(path.join(stage,record.path),'utf8'),url='/'+record.path,dependencies=[];
      if(record.path.endsWith('.js')){
        for(const re of [/(?:import|export)\s+(?:[^;'"\n]*?\s+from\s*)?['"]([^'"]+)['"]/g,/import\(\s*['"]([^'"]+)['"]\s*\)/g,/new URL\(\s*['"]([^'"]+)['"]\s*,\s*import\.meta\.url\s*\)/g])for(const match of text.matchAll(re))if(match[1].startsWith('.')||match[1].startsWith('/'))dependencies.push(match[1]);
      }else if(record.path.endsWith('.html')){
        for(const tag of text.matchAll(/<(?:script|link|img|source|video|audio)\b[^>]*>/gi))for(const match of tag[0].matchAll(/(?:src|href|poster)\s*=\s*['"]([^'"]+)['"]/gi))dependencies.push(match[1]);
      }else for(const match of text.matchAll(/url\(\s*['"]?([^)'"\s]+)['"]?\s*\)|@import\s+['"]([^'"]+)['"]/g))dependencies.push(match[1]||match[2]);
      for(const dependency of dependencies){
        if(dependency.startsWith('data:')||dependency.startsWith('#'))continue;
        const resolved=new URL(dependency,'http://bundle'+url);if(resolved.origin!=='http://bundle'||resolved.search||resolved.hash)throw new Error('External or ambiguous asset: '+dependency);
        const target=safeUrl(resolved.pathname);if(records.has(target))continue;
        if(!/^\/(assets|fonts|training|vendor)\/[A-Za-z0-9_./-]+\.(woff2?|ttf|otf|png|jpg|jpeg|svg|webp|ico|css)$/.test(target))throw new Error('Dependency missing from explicit allowlist: '+target+' in '+url);
        await add(target,{kind:'presentation'});
      }
    }
    for(const [url,relative] of [
      ['/LICENSE','LICENSE'],['/licenses/FlyBody-LICENSE','models/FlyBody-LICENSE'],['/licenses/Three-LICENSE','web/vendor/LICENSE.three'],
      ['/banc-engine/LICENSE','packages/banc-runtime/LICENSE'],['/banc-engine/native/core.cpp','packages/banc-runtime/native/core.cpp'],['/banc-engine/native/dlm-ionic.h','packages/banc-runtime/native/dlm-ionic.h'],['/banc-engine/build.sh','packages/banc-runtime/build.sh'],['/banc-engine/package.json','packages/banc-runtime/package.json'],
      ['/licenses/mujoco-package.json','packages/flybody-runtime/node_modules/@mujoco/mujoco/package.json'],
    ])await add(url,{relative,kind:/\.(?:cpp|h|sh)$/.test(url)?'source':'license'});
    for(const file of ['emscripten.txt','libcxx.txt','musl.txt','libcxxabi.txt','compiler-rt.txt'])await add('/banc-engine/third-party-licenses/'+file,{relative:'packages/banc-runtime/third-party-licenses/'+file,kind:'license'});
    const mujoco=JSON.parse(await fs.readFile(path.join(stage,'licenses/mujoco-package.json'),'utf8'));
    if(!/^\d+\.\d+\.\d+$/.test(mujoco.version)||mujoco.license!=='Apache-2.0')throw new Error('Unexpected MuJoCo license/version; review attribution');
    for(const name of ['LICENSE']){
      if(frozen){await add('/licenses/MuJoCo-'+name,{kind:'license'});continue;}
      const url=`https://raw.githubusercontent.com/google-deepmind/mujoco/${mujoco.version}/${name}`,response=await fetch(url,{signal:AbortSignal.timeout(15000)});
      if(!response.ok)throw new Error('Missing pinned MuJoCo notice: '+url);const bytes=Buffer.from(await response.arrayBuffer());
      if(bytes.length<100||bytes.length>1024*1024)throw new Error('Unexpected license size');await generated('/licenses/MuJoCo-'+name,bytes,'license',url);
    }
    // PyPI wheels carry a generated third-party notice not present at the
    // upstream repository root or in the npm package. Include only the notice
    // from the exact same installed release; do not guess another license.
    let thirdPartyNotice=false;
    if(frozen?.has('/licenses/MuJoCo-LICENSES_THIRD_PARTY.md')){await add('/licenses/MuJoCo-LICENSES_THIRD_PARTY.md',{kind:'license'});thirdPartyNotice=true;}
    for(const python of frozen?[]:await fs.readdir(path.join(repo,'.venv/lib')).catch(()=>[])){
      if(!/^python[0-9.]+$/.test(python))continue;
      const relative=`.venv/lib/${python}/site-packages/mujoco-${mujoco.version}.dist-info/licenses/LICENSES_THIRD_PARTY.md`;
      if(!(await fs.stat(path.join(repo,relative)).catch(()=>null))?.isFile())continue;
      await add('/licenses/MuJoCo-LICENSES_THIRD_PARTY.md',{relative,kind:'license'});thirdPartyNotice=true;break;
    }
    await generated('/ATTRIBUTION.md',`# Sources and attribution\n\n- Fruit Fly Heaven and BANC runtime: repository GPL-2.0-only license in LICENSE and banc-engine/LICENSE. JavaScript/WGSL source is included; the BANC native C++ source and build script are included under banc-engine/. Rebuilding requires Emscripten; ordinary use requires no compiler.\n- BANC v888: Alexander S. Bates, Jasper S. Phelps, Minsu Kim, Helen H. Yang and the BANC-FlyWire Consortium, “Distributed control circuits across a brain-and-cord connectome” (2026), https://doi.org/10.1038/s41586-026-10735-w. Data deposit: https://doi.org/10.7910/DVN/7WTH1N. Project and license statement: https://github.com/htem/BANC-project#license. The data are CC BY 4.0, https://creativecommons.org/licenses/by/4.0/. This bundle contains derived, filtered runtime graph buffers and added mechanistic physiology priors; these changes are not measured BANC physiology and imply no author endorsement. The exact original source URLs and digests are retained in banc-data/manifest.json.\n- FlyBody: https://github.com/TuragaLab/flybody, source revision d015e9bfe441bd90ae431bac24c55cb74bdbce26; Apache-2.0 text in licenses/FlyBody-LICENSE. The reduced model and local contact/actuation modifications are described by body-model/flybody-mujoco.json and the included JavaScript source.\n- MuJoCo ${mujoco.version}, Google DeepMind: https://github.com/google-deepmind/mujoco/tree/${mujoco.version}, Apache-2.0; the official versioned license is in licenses/MuJoCo-LICENSE. ${thirdPartyNotice?'The matching Python release provides the included MuJoCo-LICENSES_THIRD_PARTY.md; it describes wheel dependencies and may include components absent from WASM.':'No additional third-party notice was available in the installed matching release.'} The npm package's own metadata is also included.\n- Three.js: MIT, copyright 2010–2026 Three.js authors; see licenses/Three-LICENSE.\n- Emscripten/C++ runtime notices are retained under banc-engine/third-party-licenses/.\n\nNo learned FlyBody policy checkpoint, observation-console model, unrelated repository data, credentials, or coordinator database is included. License texts retain their original terms; this attribution does not relicense third-party components.\n`);
    await generated('/serve.py',serverSource,'server');
    await generated('/index.html','<!doctype html><html lang="en"><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=/train.html"><title>Fruit Fly training</title><a href="/train.html">Open the training console</a></html>\n','entry');
    const usage=config.schemaVersion===2&&config.optimizer?.acceptance?.nativeExecution?.backend==='dawn-metal'
      ? '4. The page monitors the connected native trainer and its latest recorded frame. This experiment requires the pinned native backend; opening it in a browser does not contribute compute.\n5. Use Download checkpoint to save the currently selected parameters. Run the native contributor from the project checkout with this experiment\'s pinned training plan to contribute.'
      : '4. Press Start. Use Pause or Stop whenever you want, and adjust Intensity to suit your computer. Training continues when you switch tabs.\n5. Results upload automatically. Use Download checkpoint to save the latest model parameters.';
    await generated('/README.md',`# Fruit Fly Heaven\n\nPython 3.9+ and a browser are required.\n\n1. Extract the ZIP and open a terminal in its folder.\n2. Run \`python3 serve.py --port 7842\` (Windows: \`py -3 serve.py --port 7842\`; with uv: \`uv run --offline python serve.py --port 7842\`).\n3. Open http://127.0.0.1:7842/train.html. If that port is occupied, use 7843 in both places.\n${usage}\n\nAn internet connection is required. If the app reports that it is out of date, download the latest archive.\n\nRun \`python3 serve.py --verify-only\` to check file integrity. Do not edit files while training is running. Model and build identifiers are recorded in bundle-manifest.json; source attribution and licenses are included in ATTRIBUTION.md.\n`);
    const files=[...records.values()].sort((a,b)=>a.path.localeCompare(b.path));
    const manifest={schemaVersion:1,kind:'local-contributor-client',modelFingerprint:config.modelFingerprint,configHash,environmentVersion:config.environmentVersion,parameterCount:config.parameters.length,vision:config.vision,modelAssetCount:Object.keys(config.assets).length,graphFileCount:Object.keys(graph.files).length,files,totalBytes:files.reduce((sum,file)=>sum+file.bytes,0),manifestSelfExcluded:true};
    await fs.writeFile(path.join(stage,'bundle-manifest.json'),json(manifest));
    const verification=await httpVerify(stage,manifest);
    const stagedZip=path.join(staging,id+'.zip');const archive=JSON.parse(await run('uv',['run','--offline','python','-',stage,stagedZip],{input:archiveSource}));
    // Refuse a snapshot assembled while any local source changed.
    for(const record of files){
      const frozenSource=frozenSources.get('/'+record.path);
      if(frozenSource){if(await hashFile(frozenSource.path)!==record.sha256)throw new Error('Frozen model source changed during packaging: '+record.path);}
      else if(!record.source.startsWith('generated ')&&!record.source.startsWith('https://')&&await hashFile(await sourceFile(record.source))!==record.sha256)throw new Error('Source changed during packaging: '+record.source);
    }
    await frozen?.verifyManifest();
    if(digest(await fs.readFile(configSource))!==inputHash)throw new Error('Configuration source changed during packaging');
    const zip=path.join(output,id+'.zip'),sha256=await hashFile(stagedZip),bytes=(await fs.stat(stagedZip)).size;
    // Existing output is generated content only; never remove arbitrary paths.
    await fs.rm(bundle,{recursive:true,force:true});await fs.rename(stage,bundle);await fs.rename(stagedZip,zip);
    await fs.writeFile(zip+'.sha256',sha256+'  '+path.basename(zip)+'\n');await fs.writeFile(path.join(output,id+'.manifest.json'),json(manifest));
    const report={zip:path.relative(repo,zip),sha256,bytes,uncompressedBytes:manifest.totalBytes,fileCount:files.length+1,modelFingerprint:config.modelFingerprint,configHash,...verification,...archive};
    await fs.writeFile(path.join(output,id+'.verification.json'),json(report));console.log(json(report));return report;
  }finally{await fs.rm(staging,{recursive:true,force:true});}
}

if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  const args=process.argv.slice(2);
  const options={},names={'experiment-bundle':'experimentBundle','model-bundle':'modelBundle','output-dir':'outputDirectory'};
  for(const arg of args){const match=/^--(experiment-bundle|model-bundle|output-dir)=(.+)$/.exec(arg);if(!match||Object.hasOwn(options,names[match?.[1]]))throw new Error('Usage: package-training-client.mjs [--experiment-bundle=PATH] [--model-bundle=PATH] [--output-dir=PATH]');options[names[match[1]]]=match[2];}
  packageTrainingClient(options).catch(error=>{console.error(error.stack||error);process.exitCode=1;});
}
