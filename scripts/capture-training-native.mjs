// Actual BANC/WASM + native MuJoCo diagnostic without a browser. This is not a
// contributor or optimizer, and is explicitly distinct from Safari/WebGPU.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {registerHooks} from 'node:module';
import {createHash} from 'node:crypto';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const output=path.resolve(process.argv[2]||path.join(root,'reports/flight-motor-capture-wasm/result.json'));
const args=Object.fromEntries(process.argv.slice(3).map(value=>value.replace(/^--/,'').split('=')));
const limitSeconds=Number(args['limit-seconds']??.5),clawScale=Number(args['claw-scale']??1),power=Number(args.power??1),steering=Number(args.steering??1);
if(Object.keys(args).some(key=>!['limit-seconds','claw-scale','power','steering'].includes(key))||!Number.isFinite(limitSeconds)||limitSeconds<=0||limitSeconds>8||!Number.isFinite(clawScale)||clawScale<0||clawScale>1||!Number.isFinite(power)||power<=0||power>2||!Number.isFinite(steering)||steering<.05||steering>2)throw new Error('Invalid diagnostic variant');
try{await fs.access(output);throw new Error('Output already exists: '+output);}catch(error){if(error.code!=='ENOENT')throw error;}
const virtualAssets=new Map();
const prefixes=[['/body-engine/','packages/flybody-runtime/node_modules/@mujoco/mujoco/'],['/banc-engine/','packages/banc-runtime/'],['/banc-data/','data/prepared/banc888/'],['/body-model/','models/']];
function resolveAsset(url){const entry=prefixes.find(([prefix])=>url.startsWith(prefix));const file=path.resolve(root,entry?entry[1]+url.slice(entry[0].length):'web/'+url.slice(1));if(!file.startsWith(root+path.sep))throw new Error('Asset escaped checkout');return file;}
registerHooks({resolve(specifier,context,nextResolve){if(prefixes.some(([prefix])=>specifier.startsWith(prefix)))return {url:pathToFileURL(resolveAsset(specifier)).href,shortCircuit:true};return nextResolve(specifier,context);}});
globalThis.location={href:'http://127.0.0.1/native-capture'};
globalThis.fetch=async input=>{const url=new URL(typeof input==='string'?input:input.url||String(input),location.href);if(url.origin!==new URL(location.href).origin)throw new Error('Native capture refuses external asset: '+url);return new Response(virtualAssets.get(url.pathname)??await fs.readFile(resolveAsset(url.pathname)));};
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const [{createTrainingEnvironment},{createMotorReplayCapture},canonicalRaw,canonicalXml]=await Promise.all([
 import('../web/training/environment.js'),import('../web/test/training-motor-capture.js'),fs.readFile(path.join(root,'web/training/config.json')),fs.readFile(path.join(root,'models/flybody-mujoco.xml'),'utf8')]);
const config=JSON.parse(canonicalRaw),observerUrls=['/test/training-motor-capture-worker.js','/test/training-motor-capture.js','/test/training-safari-check.html'];
let raw=canonicalRaw,sourceXml=canonicalXml;
if(clawScale!==1){
 const pattern=/(<default class="adhesion_claw">\s*<adhesion[^>]*gain=")([^"]+)(")/g;let replacements=0;
 sourceXml=canonicalXml.replace(pattern,(_,prefix,gain,suffix)=>{replacements++;return prefix+(Number(gain)*clawScale)+suffix;});
 if(replacements!==1)throw new Error('Expected exactly one native claw adhesion class');
 const metadata=JSON.parse(await fs.readFile(resolveAsset('/body-model/flybody-mujoco.json')));metadata.xml_sha256=sha(sourceXml);
 metadata.diagnosticVariant={clawAdhesionGainScale:clawScale,sourceXmlSha256:sha(canonicalXml)};
 virtualAssets.set('/body-model/flybody-mujoco.xml',sourceXml);virtualAssets.set('/body-model/flybody-mujoco.json',JSON.stringify(metadata)+'\n');
 for(const [url,bytes]of virtualAssets)config.assets[url]=sha(bytes);
 config.modelFingerprint=sha(Object.entries(config.assets).sort(([a],[b])=>a.localeCompare(b)).map(([url,digest])=>`${url}:${digest}\n`).join(''));
 config.notes=[...config.notes,'Diagnostic-only model variant: native claw adhesion gain multiplied by '+clawScale+'. Canonical workspace and hosted models are unchanged.'];
 raw=Buffer.from(JSON.stringify(config,null,2)+'\n');virtualAssets.set('/training/config.json',raw);
}
const observerAssets=Object.fromEntries(await Promise.all(observerUrls.map(async url=>[url,sha(await fs.readFile(resolveAsset(url)))])));
const report={kind:'node-native-motor-capture',date:new Date().toISOString(),scope:'Actual BANC/WASM and native MuJoCo. Distinct from the Safari/WebGPU baseline; no coordinator or optimization.',configHash:sha(raw),modelFingerprint:config.modelFingerprint,scriptSha256:sha(await fs.readFile(fileURLToPath(import.meta.url))),nodeVersion:process.version,
 canonicalConfigHash:sha(canonicalRaw),diagnostic:{clawScale,power,steering,limitSeconds,liveClosedLoop:true},virtualAssets:Object.fromEntries([...virtualAssets].map(([url,bytes])=>[url,typeof bytes==='string'?bytes:bytes.toString('utf8')])),frames:[]};
let lastMessage='',environment;
try{
 environment=await createTrainingEnvironment(config,{onProgress:value=>{if(value.message!==lastMessage){lastMessage=value.message;console.log(value.message);}}});
 report.ready=await environment.ready();if(report.ready.backend!=='wasm')throw new Error('Expected the actual WASM neural backend');
 const capture=createMotorReplayCapture({sourceXml,seconds:.5,bodyBlockMs:config.bodyBlockMs,sourceAssets:observerAssets});
 const job={name:'native-wasm-baseline',parameters:config.parameters.map(p=>p.initial),seed:888,stage:config.stage,durationSeconds:config.durationSeconds};
 for(const [name,scale]of [['flight_power_log_gain',power],...config.parameters.slice(3).map(p=>[p.name,steering])]){const i=config.parameters.findIndex(p=>p.name===name);job.parameters[i]=Math.max(config.parameters[i].min,Math.min(config.parameters[i].max,Math.log(scale)));}
 let simulated=0,lastProgress=0;const started=performance.now();
 report.evaluation=await environment.evaluate(job,{previewHz:1,dutyCycle:1,
  checkpoint:async()=>{await new Promise(resolve=>setTimeout(resolve,0));if(simulated>=limitSeconds-1e-10||performance.now()-started>300000){const error=new Error('Diagnostic evaluation limit');error.name='AbortError';throw error;}},
  onFrame:frame=>{const saved=structuredClone(frame);if(report.frames.length<400)report.frames.push(saved);else report.frames[399]=saved;},
  onInitialState:capture.onInitialState,
  onPhysicsStep:value=>{capture.onPhysicsStep(value);simulated=value.body.time;},
  onProgress:value=>{if(performance.now()-lastProgress>3000){lastProgress=performance.now();console.log(JSON.stringify({simSeconds:value.simSeconds,score:value.return}));}}});
 report.evaluation.motorReplay=capture.finish(report.evaluation);
 if(report.evaluation.reason==='simulation_error')throw new Error(report.evaluation.metrics.error);
 if(report.evaluation.steps<1)throw new Error('No native steps captured');
 report.captured=true;
} catch(error){report.error=error.stack||error.message;process.exitCode=1;}
finally{
 environment?.dispose();await fs.mkdir(path.dirname(output),{recursive:true});await fs.writeFile(output,JSON.stringify(report)+'\n',{flag:'wx'});
 console.log(JSON.stringify({output,captured:report.captured,error:report.error,reason:report.evaluation?.reason,simSeconds:report.evaluation?.simSeconds,backend:report.evaluation?.backend,steps:report.evaluation?.steps}));
}
