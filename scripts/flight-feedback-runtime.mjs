// Read-only local asset host for the actual browser-WASM training environment.
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL,fileURLToPath} from 'node:url';
import {registerHooks} from 'node:module';
import {createHash} from 'node:crypto';
export const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
export const sha=x=>createHash('sha256').update(x).digest('hex');
const prefixes=[['/body-engine/','packages/flybody-runtime/node_modules/@mujoco/mujoco/'],['/banc-engine/','packages/banc-runtime/'],['/banc-data/','data/prepared/banc888/'],['/body-model/','models/']];
export function localAsset(url){
  const prefix=prefixes.find(([p])=>url.startsWith(p));
  const file=path.resolve(root,prefix?prefix[1]+url.slice(prefix[0].length):'web'+url);
  if(!file.startsWith(root+path.sep))throw new Error('Asset escaped checkout');return file;
}
export async function installFeedbackRuntime(bundleFile){
  const bytes=await fs.readFile(bundleFile),bundle=JSON.parse(bytes),config=JSON.parse(bundle.configText);
  if(sha(bundle.configText)!==bundle.configHash)throw new Error('Configuration hash mismatch');
  const read=async url=>bundle.assets[url]!==undefined?Buffer.from(bundle.assets[url]):fs.readFile(localAsset(url));
  const graph=JSON.parse(await read('/banc-data/manifest.json'));
  const pinned={...config.assets,...Object.fromEntries(Object.entries(graph.files).map(([name,value])=>['/banc-data/'+name,value.sha256]))};
  const assets=new Map(Object.keys(pinned).map(url=>[localAsset(url),url]));
  async function verify(){for(const [url,digest]of Object.entries(pinned))if(sha(await read(url))!==digest)throw new Error('Changed experiment asset: '+url);}
  await verify();
  const hooks=registerHooks({resolve(specifier,context,next){
    if(specifier.startsWith('/')&&config.assets[specifier])return {url:pathToFileURL(localAsset(specifier)).href,shortCircuit:true};
    return next(specifier,context);
  },load(url,context,next){
    // Archived experiments own their exact JS sources as well as native XML.
    // Binaries/data remain external, with the same mandatory hash checks.
    if(url.startsWith('file:')){
      const asset=assets.get(fileURLToPath(url));
      if(asset?.endsWith('.js')&&Object.hasOwn(bundle.assets,asset))
        return {format:'module',source:bundle.assets[asset],shortCircuit:true};
    }
    return next(url,context);
  }});
  const previousFetch=globalThis.fetch,previousLocation=globalThis.location;
  globalThis.location={href:'http://127.0.0.1/flight-feedback/'};
  globalThis.fetch=async(input,options={})=>{
    const u=new URL(typeof input==='string'?input:input.url||String(input),location.href),method=String(options.method||'GET').toUpperCase();
    if(!['GET','HEAD'].includes(method))throw new Error('Diagnostic runtime permits asset reads only');
    let url;if(u.protocol==='file:'){url=assets.get(fileURLToPath(u));if(!url)throw new Error('Unpinned file request');}
    else{if(u.origin!=='http://127.0.0.1')throw new Error('External request from local diagnostic');url=u.pathname;}
    if(url!=='/training/config.json'&&!pinned[url])throw new Error('Unpinned asset request: '+url);
    return new Response(method==='HEAD'?null:url==='/training/config.json'?bundle.configText:await read(url),
      {headers:{'Content-Type':url.endsWith('.wasm')?'application/wasm':'application/octet-stream'}});
  };
  return {bundle,config,read,verify,identity:{bundleSha256:sha(bytes),configHash:bundle.configHash,modelFingerprint:config.modelFingerprint,
    neuralWasmSha256:config.assets['/banc-engine/dist/core.wasm'],bodyWasmSha256:config.assets['/body-engine/mujoco.wasm']},
    dispose(){globalThis.fetch=previousFetch;globalThis.location=previousLocation;hooks.deregister();}};
}
