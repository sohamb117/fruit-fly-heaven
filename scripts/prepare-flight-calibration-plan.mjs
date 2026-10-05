// Fixed operator interventions to characterize the current wing interpreter.
// This is not an optimizer: all jobs are chosen before any result is observed.
import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {parameterValues,PARAMETER_NAMES} from '../web/training/episode.js';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const output=path.resolve(process.argv[2]||path.join(root,'reports/flight-calibration-27/plan.json'));
const raw=await fs.readFile(path.join(root,'web/training/config.json')),config=JSON.parse(raw);
const parameterNames=config.parameters.map(p=>p.name);
if(JSON.stringify(parameterNames)!==JSON.stringify(PARAMETER_NAMES)||parameterNames.length!==27)throw new Error('Expected the 27-parameter interpreter contract');
const baseline=config.parameters.map(p=>p.initial),jobs=[];
function add(name,power,tau,{steeringScale=1}={}){
 const parameters=[...baseline];
 for(const [key,scale]of [['flight_power_log_gain',power],['flight_deployment_tau_log_scale',tau]])parameters[parameterNames.indexOf(key)]=Math.log(scale);
 if(steeringScale!==1)for(let i=3;i<parameters.length;i++)parameters[i]=Math.max(config.parameters[i].min,Math.log(steeringScale));
 parameterValues(config,parameters);
 jobs.push({name,seed:888,stage:config.stage,durationSeconds:config.durationSeconds,parameters});
}
add('baseline',1,1);
add('baseline-repeat',1,1);
for(const power of [1,1.25,1.5])for(const tau of [1,2])if(power!==1||tau!==1)add(`power-${power}-deployment-${tau}`,power,tau);
add('power-1.5-deployment-2-steering-0.05',1.5,2,{steeringScale:.05});
const plan={schemaVersion:1,kind:'motor-interface-causal-sweep',createdAt:new Date().toISOString(),
 configHash:createHash('sha256').update(raw).digest('hex'),modelFingerprint:config.modelFingerprint,parameterNames,
 purpose:'Check repeatable episode reset, then the physical effects of power/deployment and a bounded steering-attenuation control. Neural physiology, sensory transduction and native muscle/body parameters stay fixed. Complete outcomes are retained; these predeclared probes do not demonstrate optimizer learning.',jobs};
await fs.mkdir(path.dirname(output),{recursive:true});await fs.writeFile(output,JSON.stringify(plan,null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({output,jobs:jobs.length,configHash:plan.configHash,modelFingerprint:plan.modelFingerprint}));
