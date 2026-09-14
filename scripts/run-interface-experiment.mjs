// Operator-owned, independent validation. This does not issue or upload public
// training jobs. Anonymous contributor results cannot authorize promotion.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash,randomUUID} from 'node:crypto';
import {gzipSync} from 'node:zlib';
import {interfaceParameterValues,createOperatorPromotionGate} from '../web/training/interface-parameters.js';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const args=Object.fromEntries(process.argv.slice(2).map(arg=>{const index=arg.indexOf('=');if(!arg.startsWith('--')||index<3)throw new Error('Use --name=value arguments');return [arg.slice(2,index),arg.slice(index+1)];}));
const known=['config','site','report','stage','candidates','seeds','headed'];
if(Object.keys(args).some(key=>!known.includes(key)))throw new Error('Unknown experiment argument');
const configPath=path.resolve(root,args.config||'configs/training-interface-v2.json');
const configText=await fs.readFile(configPath,'utf8'),config=JSON.parse(configText);
if(config.parameterContract!=='banc-motor-interface-v2'||config.freezeNeuralParameters!==true||config.behaviorCriteriaVersion!==2)throw new Error('This experiment requires the frozen-neural v2 contract');
const sha=value=>createHash('sha256').update(value).digest('hex');
const configHash=sha(configText),stage=args.stage||'posture';
if(!config.stages.some(s=>s.id===stage))throw new Error('Unknown experiment stage');
const seeds=args.seeds?JSON.parse(args.seeds):config.validation.seeds;
if(!Array.isArray(seeds)||!seeds.length||new Set(seeds).size!==seeds.length||seeds.some(s=>!Number.isInteger(s)||s<0||s>0xffffffff||s===config.optimizer.seed))throw new Error('Expected distinct held-out uint32 seeds');
if(seeds.some(s=>!config.validation.seeds.includes(s)))throw new Error('Seeds must come from the configuration validation split, not the final test split');
const candidates=args.candidates?JSON.parse(args.candidates):[{name:'hind_025',parameters:{hindGripScale:.25}}];
if(!Array.isArray(candidates)||!candidates.length)throw new Error('At least one candidate is required');
const baseValues=config.parameters.map(p=>p.initial),names=new Set(['baseline']);
const specifications=[{name:'baseline',values:baseValues}];
for(const candidate of candidates){
 if(!candidate||!/^[-a-zA-Z0-9_]+$/.test(candidate.name)||names.has(candidate.name)||!candidate.parameters||typeof candidate.parameters!=='object'||Array.isArray(candidate.parameters))throw new Error('Invalid or duplicate candidate');
 if(Object.keys(candidate.parameters).some(name=>!config.parameters.some(p=>p.name===name)))throw new Error('Candidate changes a value outside the experiment contract');
 names.add(candidate.name);const values=config.parameters.map(p=>candidate.parameters[p.name]??p.initial);
 interfaceParameterValues(config,values);specifications.push({name:candidate.name,values});
}
const directory=path.resolve(root,args.report||'reports/interface-training-validation');
await fs.mkdir(directory,{recursive:true});
const site=new URL(args.site||process.env.TRAINING_SITE_URL||'http://127.0.0.1:7842');
// Avoid accidentally executing a local experiment against a public coordinator.
if(!['127.0.0.1','localhost','[::1]'].includes(site.hostname))throw new Error('Independent validation requires a local static server');
const report={schemaVersion:1,executionId:randomUUID(),date:new Date().toISOString(),stage,configHash,modelFingerprint:config.modelFingerprint,
 scope:'Real BANC v888 and native MuJoCo/WASM muscles; frozen neural parameters; matched held-out operator evaluations. No public training jobs are issued or uploaded.',
 deploymentGate:config.deploymentGate,validationSeeds:seeds,runs:[],errors:[],promotion:[],passed:false};
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'playwright');
const browser=await chromium.launch({executablePath:process.env.CHROME_PATH||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:args.headed!=='true',args:['--enable-unsafe-webgpu']});
const gate=createOperatorPromotionGate(),localEvidence=[];
try{
 const context=await browser.newContext({viewport:{width:800,height:600}});
 // The harness page is an inert observer, not train.html (which starts a
 // coordinator session). All /api requests are blocked as a second guard.
 await context.route('**/api/**',route=>route.abort('blockedbyclient'));
 await context.route('**/__interface-config.json',route=>route.fulfill({status:200,contentType:'application/json',body:configText}));
 await context.route('**/__interface-experiment.html',route=>route.fulfill({status:200,contentType:'text/html',body:'<!doctype html><meta charset="utf-8"><title>Motor interface experiment</title><style>body{margin:0;background:#102017;color:#deebd9;font:16px system-ui;padding:24px}#preview{width:720px;height:405px}canvas{width:100%;height:100%;image-rendering:auto}p{margin:12px 0}</style><div id="preview"></div><p id="status">Initializing</p>'}));
 const page=await context.newPage();page.on('pageerror',error=>report.errors.push(error.message));
 await page.goto(new URL('/__interface-experiment.html',site).href);
 await page.evaluate(async()=>{
  const {NativeFlyPreview}=await import('/training/preview.js');
  window.experimentPreview=new NativeFlyPreview(document.querySelector('#preview'));experimentPreview.setQuality('low');
  window.experimentEvents=[];window.experimentWorker=new Worker('/training/worker.js',{type:'module'});
  experimentWorker.onmessage=event=>{const message=event.data;experimentEvents.push(message);
   if(message.type==='ready'||message.type==='frame'){experimentPreview.setFrame(message.frame);document.querySelector('#status').textContent=`${message.frame.stage} · ${message.frame.simSeconds.toFixed(3)} s`;}
  };
  experimentWorker.onerror=error=>experimentEvents.push({type:'error',message:error.message});
  experimentWorker.postMessage({type:'initialize',id:'init',configUrl:'/__interface-config.json'});
 });
 const wait=async(type,id,timeout=600000)=>{
  await page.waitForFunction(({type,id})=>experimentEvents.some(event=>(event.type===type&&(!id||event.id===id))||event.type==='error'),{type,id},{timeout});
  const error=await page.evaluate(()=>experimentEvents.find(event=>event.type==='error'));
  if(error)throw new Error(error.message);
  return page.evaluate(({type,id})=>experimentEvents.find(event=>event.type===type&&(!id||event.id===id)),{type,id});
 };
 const ready=await wait('ready','init');
 if(ready.configHash!==configHash||ready.modelFingerprint!==config.modelFingerprint||!ready.neuralParametersFrozen||ready.bodyBackend!=='mujoco-wasm')throw new Error('Unexpected runtime identity');
 report.ready={backend:ready.backend,bodyBackend:ready.bodyBackend,neuralParametersFrozen:ready.neuralParametersFrozen,parameterCount:ready.parameterCount,criteria:ready.criteria};
 console.log(JSON.stringify({type:'ready',...report.ready,criteria:undefined}));
 for(const specification of specifications){
  const profile=interfaceParameterValues(config,specification.values).profile;
  const run={name:specification.name,executionId:randomUUID(),configHash,modelFingerprint:config.modelFingerprint,interfaceHash:sha(JSON.stringify(profile)),stage,split:'validation',parameters:specification.values,profile,results:[]};
  report.runs.push(run);
  for(const seed of seeds){
   const id=`${specification.name}-${seed}`,job={stage,seed,parameters:specification.values,durationSeconds:config.stages.find(s=>s.id===stage).durationSeconds};
   console.log(JSON.stringify({type:'starting',id,stage,durationSeconds:job.durationSeconds}));
   await page.evaluate(({id,job})=>{experimentEvents.length=0;experimentWorker.postMessage({type:'evaluate',id,job,previewHz:3,dutyCycle:1});},{id,job});
   const result=(await wait('evaluation',id)).result;
   if(result.configHash!==configHash||result.modelFingerprint!==config.modelFingerprint||JSON.stringify(result.parameters)!==JSON.stringify(specification.values)||result.seed!==seed)throw new Error('Evaluation identity mismatch');
   run.results.push(result);
   const frames=await page.evaluate(id=>experimentEvents.filter(event=>event.type==='frame'&&event.id===id).map(event=>event.frame),id);
   await fs.writeFile(path.join(directory,`${id}.frames.json.gz`),gzipSync(JSON.stringify(frames)));
   if(frames.length){await page.evaluate(()=>{experimentPreview.ensureRenderer();experimentPreview.draw();});await page.screenshot({path:path.join(directory,`${id}.png`)});}
   console.log(JSON.stringify({type:'evaluation',id,success:result.success,reason:result.reason,simSeconds:result.simSeconds,return:result.return,wallSeconds:result.metrics.wallSeconds,spikes:result.metrics.finalNeuralSpikes,error:result.metrics.error}));
   await fs.writeFile(path.join(directory,'result.json'),JSON.stringify(report,null,2)+'\n');
   if(['simulation_error','invalid_observation'].includes(result.reason)||result.cancelled||!result.steps||!result.metrics.neuralParametersFrozen)throw new Error(result.metrics.error||result.metrics.invalidObservation||'Incomplete real evaluation');
  }
  localEvidence.push(gate.recordLocalEvidence(run));
 }
 // These experiments cannot certify the motor/wing transfer by themselves.
 // A failed mechanical record is explicit; candidate success must not bypass it.
 const requiredPriorStages=config.stages.slice(0,config.stages.findIndex(item=>item.id===stage)).map(item=>item.id);
 for(const candidate of localEvidence.slice(1))report.promotion.push({name:candidate.name,...gate.evaluate({candidate,incumbent:localEvidence[0],priorStages:[],requiredPriorStages,requiredSeeds:seeds,trainingSeeds:[config.optimizer.seed],stage,minimumImprovement:config.validation.minimumImprovement,mechanicalGate:{passed:false}})});
 await page.evaluate(()=>experimentWorker.postMessage({type:'stop',id:'stop'}));await wait('stopped');
 report.configUnchanged=sha(await fs.readFile(configPath))===configHash;
 report.passed=report.configUnchanged&&report.errors.length===0;
 if(!report.passed)throw new Error('Config changed during validation or browser reported an error');
}catch(error){report.failure=error.stack;process.exitCode=1;}
finally{await browser.close();await fs.writeFile(path.join(directory,'result.json'),JSON.stringify(report,null,2)+'\n');}
console.log(JSON.stringify({passed:report.passed,behaviorValidated:report.runs.some(run=>run.name!=='baseline'&&run.results.length===seeds.length&&run.results.every(result=>result.success)),promotion:report.promotion,report:path.relative(root,path.join(directory,'result.json')),failure:report.failure}));
