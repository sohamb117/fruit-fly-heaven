// Real HTTPS transport checks without starting a simulation or accepting a job.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdir,readFile,writeFile} from 'node:fs/promises';
import path from 'node:path';

const origin=new URL(process.env.TRAINING_URL||'https://flytrain.morisoba.moe').origin;
assert.equal(new URL(origin).protocol,'https:');
const directory=path.resolve(process.env.REPORT_DIR||'reports/cloudrun-http');
await mkdir(directory,{recursive:true});
const report={date:new Date().toISOString(),origin,checks:[],passed:false};
async function request(route,options={}){
  const response=await fetch(origin+route,{...options,signal:AbortSignal.timeout(20000)});
  report.checks.push({path:route,method:options.method||'GET',status:response.status});
  return response;
}
try{
  const page=await request('/train.html');assert.equal(page.status,200);
  assert.equal(page.headers.get('cross-origin-opener-policy'),'same-origin');
  assert.equal(page.headers.get('cross-origin-embedder-policy'),'require-corp');
  await page.arrayBuffer();
  const configResponse=await request('/training/config.json');assert.equal(configResponse.status,200);
  const raw=Buffer.from(await configResponse.arrayBuffer()),config=JSON.parse(raw);
  const digest=value=>createHash('sha256').update(value).digest('hex');
  report.configHash=digest(raw);assert.equal(report.configHash,digest(await readFile(process.env.TRAINING_CONFIG||'web/training/config.json')));
  const statusResponse=await request('/api/training/status');assert.equal(statusResponse.status,200);
  const status=await statusResponse.json();
  assert.equal(status.configHash,report.configHash);assert.equal(status.modelFingerprint,config.modelFingerprint);
  report.acceptedResults=status.acceptedResults;report.generation=status.generation;
  const checkpointResponse=await request('/api/training/checkpoint');assert.equal(checkpointResponse.status,200);
  assert.equal(checkpointResponse.headers.get('cache-control'),'no-store');
  const checkpoint=await checkpointResponse.json();assert.equal(checkpoint.configHash,report.configHash);
  assert(checkpoint.generation>=status.generation);
  assert.equal(checkpointResponse.headers.get('content-disposition'),`attachment; filename="heaven-checkpoint-generation-${checkpoint.generation}.json"`);
  for(const route of ['/config.json','/server/training_coordinator.py','/.git/config','/migration.sqlite3','/public-manifest.json']){
    const response=await request(route);assert.equal(response.status,404);await response.arrayBuffer();
  }
  for(const allowed of [origin,'https://flytrain.morisoba.moe','http://127.0.0.1:7842','http://localhost:7843']){
    const response=await request('/api/training/lease',{method:'OPTIONS',headers:{Origin:allowed,'Access-Control-Request-Method':'POST','Access-Control-Request-Headers':'content-type'}});
    assert.equal(response.status,204);assert.equal(response.headers.get('access-control-allow-origin'),allowed);
  }
  const denied=await request('/api/training/lease',{method:'POST',headers:{Origin:'https://example.invalid','Content-Type':'application/json'},body:'{}'});
  assert.equal(denied.status,403);await denied.arrayBuffer();
  const oversized=await request('/api/training/lease',{method:'POST',headers:{Origin:origin,'Content-Type':'application/json'},body:'x'.repeat(config.contribution.maxRequestBytes+1)});
  assert.equal(oversized.status,413,'The hosting proxy must preserve oversized-request rejection');
  assert.equal((await oversized.json()).error,'body_too_large');
  report.passed=true;
}catch(error){report.failure={message:error.message,stack:error.stack};process.exitCode=1;}
await writeFile(path.join(directory,'http-verification.json'),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({passed:report.passed,origin,acceptedResults:report.acceptedResults,checks:report.checks.length,failure:report.failure?.message}));
