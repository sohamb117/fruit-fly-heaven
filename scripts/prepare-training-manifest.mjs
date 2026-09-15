// Pins the exact executable environment and source model assets shared by contributors.
// Rebuild after changing the training environment, neural/body code, or model data.
import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const prefixes=[['/body-engine/','packages/flybody-runtime/node_modules/@mujoco/mujoco/'],['/banc-engine/','packages/banc-runtime/'],['/banc-data/','data/prepared/banc888/'],['/body-model/','models/']];
const resolve=url=>{const match=prefixes.find(([prefix])=>url.startsWith(prefix));return path.join(root,match?match[1]+url.slice(match[0].length):'web'+url);};
const digest=bytes=>createHash('sha256').update(bytes).digest('hex');
export async function buildTrainingAssetManifest({extraUrls=[],config=null}={}){
 const urls=new Set([
  '/training/worker.js','/training/environment.js','/motor-decoder.js','/banc-engine/src/index.js',
  '/banc-engine/src/neural.wgsl','/banc-engine/src/neural-dlm.wgsl','/banc-engine/dist/core.js','/banc-engine/dist/core.wasm',
  '/body-engine/mujoco.js','/body-engine/mujoco.wasm',
  '/body-model/flybody-mujoco.xml','/body-model/flybody-mujoco.json',
  '/banc-data/manifest.json','/banc-data/io.json','/banc-data/console/groups.json','/banc-data/console/sensory-inputs.json',
  '/body-model/banc-taste-peg-annotations.json',
  '/habitat.json',...extraUrls,
  ...(config?.vision===true?['/banc-data/console/visual-projections.json']:[]),
 ]);
 const assets={};
 for(const url of urls){
  const bytes=await fs.readFile(resolve(url));assets[url]=digest(bytes);
  if(!url.endsWith('.js'))continue;
  // ES module source dependencies, including literal dynamic imports. Fetch
  // assets with computed paths are pinned explicitly above/by model manifest.
  const source=bytes.toString('utf8');
  const patterns=[/(?:import|export)\s+(?:[^;'"\n]*?\s+from\s*)?['"]([^'"]+)['"]/g,/import\(\s*['"]([^'"]+)['"]\s*\)/g];
  for(const pattern of patterns)for(const match of source.matchAll(pattern)){
    const dependency=match[1];if(dependency.startsWith('.')||dependency.startsWith('/'))urls.add(new URL(dependency,'http://local'+url).pathname);
  }
 }
 const assetsSorted=Object.fromEntries(Object.entries(assets).sort(([a],[b])=>a.localeCompare(b)));
 const modelFingerprint=digest(Object.entries(assetsSorted).map(([url,sha])=>`${url}:${sha}\n`).join(''));
 return {assets:assetsSorted,modelFingerprint};
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 const configPath=path.join(root,'web/training/config.json'),config=JSON.parse(await fs.readFile(configPath,'utf8'));
 const {assets,modelFingerprint}=await buildTrainingAssetManifest({config});
 config.assets=assets;config.modelFingerprint=modelFingerprint;
 const encoded=JSON.stringify(config,null,2)+'\n';await fs.writeFile(configPath,encoded);
 console.log(JSON.stringify({assets:Object.keys(assets).length,modelFingerprint,configHash:digest(encoded),configuration:'web/training/config.json'}));
}
