// Client orchestration only: synthetic anatomy, fake workers and fake HTTP.
// These fixtures never allocate a brain/body or provide evidence of flight.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {TrainingClient,EVALUATION_STALL_MS} from '../training/client.js';
import {checkpoint} from '../training/optimizer.js';
import {wasmConfigFixture} from './fixtures/training-native-config.mjs';
import {buildSensorimotorTrainingConfig,getSensorimotorGroups} from '../training/sensorimotor-parameters.js';
import {LEG_PROPRIOCEPTION_PRIOR} from '../banc-leg-proprioception.js';
import {ANTENNA_FAMILY_PRIOR} from '../banc-antenna.js';
import {createTegulaStrainPrior} from '../banc-tegula.js';

const [base,haltere]=await Promise.all([
  readFile(new URL('../training/config.json',import.meta.url),'utf8').then(JSON.parse),
  readFile(new URL('../../models/banc-haltere-directional-v2.json',import.meta.url),'utf8').then(JSON.parse),
]);
const origin='https://sequence-fixture.example';
const deferred=()=>{let resolve;const promise=new Promise(r=>resolve=r);return {promise,resolve};};

function sequenceConfig(){
  const native=wasmConfigFixture(base),{enabled,...haltereFeedback}=haltere;
  assert.equal(enabled,true);
  Object.assign(native,{
    legProprioception:structuredClone(LEG_PROPRIOCEPTION_PRIOR),
    antennaFeedback:{schema:1,windWorldCmPerSecond:[0,0,0],mechanics:structuredClone(ANTENNA_FAMILY_PRIOR)},
    haltereFeedback:structuredClone(haltereFeedback),tegulaFeedback:createTegulaStrainPrior({halfLoadNative:.02}),
    vision:true,visionFeedback:{schema:1,profile:'compact-retinal-motion-v1',width:256,height:128,frameIntervalMs:20,camera:{textureAntialias:true}},
  });
  const config=buildSensorimotorTrainingConfig(native),groups=getSensorimotorGroups(config);
  config.stage='maintained_flight';config.durationSeconds=5;
  config.stages=[{id:'maintained_flight',durationSeconds:5},{id:'recovery',durationSeconds:5},
    {id:'takeoff',durationSeconds:3},{id:'landing',durationSeconds:8}];
  const phase=(id,stage,kind='search')=>({id,label:id,description:id,kind,stage,
    parameterIndices:groups[id]||groups.motor,maxRounds:6,validationCount:3,minimumSuccesses:2,minimumMeanImprovement:.01});
  config.trainingSequence={schema:1,calibrationMode:'simulation-engineering',teacherAsset:'/fixture-teacher.json',
    fitOptions:{ridge:1e-5,maxSweeps:2000,tolerance:1e-9},phases:[
      ...['legs','antenna','haltere','tegula','vision'].map(id=>phase(id,'maintained_flight')),
      phase('decoder_fit','recovery','decoder-fit'),phase('recovery','recovery'),phase('takeoff','takeoff'),phase('landing','landing'),
    ]};
  config.assets['/fixture-teacher.json']='d'.repeat(64);
  return config;
}

function fixture(t,{mode='decoder-fit',terminal='complete',terminalOnly=false,automatic=true,
  mutateResult=value=>value}={}){
  const config=sequenceConfig(),text=JSON.stringify(config),configHash=createHash('sha256').update(text).digest('hex');
  const parameters=config.parameters.map(p=>p.initial),candidate=parameters.slice();candidate[24]+=.125;candidate[48]=-.01;
  const execution={backend:'wasm',neuralEngine:'wasm',wasmExecution:{...config.optimizer.acceptance.nativeExecution}};
  const job={jobId:'fixture-sequence-job',leaseToken:'fixture-sequence-lease',generation:5,pairId:'fit-0',sign:0,seed:12345,
    mode,phaseId:mode==='decoder-fit'?'decoder_fit':'recovery',stage:'recovery',durationSeconds:5,
    modelFingerprint:config.modelFingerprint,configHash,parameters,parametersHash:'a'.repeat(64),
    ...(mode==='decoder-fit'?{trainingSeeds:[12345,12346],validationSeeds:[12347]}:{})};
  const current=()=>checkpoint(config,configHash,parameters,5,'recovery');
  const curriculum=done=>config.trainingSequence.phases.map(p=>({id:p.id,label:p.label,
    status:done?(terminal==='complete'?'passed':p.id==='decoder_fit'?'needs-review':'pending'):'pending'}));
  const calls=[],workers=[],uploads=[],started=deferred();let accepted=0,leaseCount=0,atTerminal=false;
  const status=()=>({modelFingerprint:config.modelFingerprint,configHash,generation:5,stage:'recovery',acceptedResults:accepted,
    checkpoint:current(),curriculum:curriculum(atTerminal),sequenceStatus:atTerminal?terminal:'running'});
  function result(j=job){
    const provenance={...execution,environmentVersion:config.environmentVersion,configHash,modelFingerprint:config.modelFingerprint,
      bodyBackend:'mujoco-wasm',dtMs:config.dtMs,bodyBlockMs:config.bodyBlockMs,seed:j.seed,stage:j.stage,
      generation:j.generation,pairId:j.pairId,sign:j.sign,parametersHash:j.parametersHash,mode:j.mode,phaseId:j.phaseId};
    const common={...provenance,provenance,parameters:[...j.parameters],cancelled:false,metrics:{wallSeconds:1}};
    return mutateResult(mode==='decoder-fit'?{
      ...common,return:0,success:false,terminated:false,steps:7500,simSeconds:15,reason:'fit_candidate',
      candidateParameters:[...candidate],calibration:{passed:true,trainingTrials:2,validationTrials:1,
        teacherUsedForEvaluation:false,teacherOnlyDuringDemonstrations:true,teacherCalibrationSha256:'d'.repeat(64),
        fit:{beforeValidationRmse:.1,afterValidationRmse:.05}},
    }:{...common,return:1,success:true,terminated:true,steps:2500,simSeconds:5,reason:'recovery_success'});
  }
  class FixtureWorker extends EventTarget{
    constructor(){super();this.messages=[];workers.push(this);}
    send(data){if(!this.terminated)this.dispatchEvent(new MessageEvent('message',{data}));}
    postMessage(message){
      this.messages.push(message);
      if(message.type==='initialize')queueMicrotask(()=>this.send({type:'ready',id:message.id,...execution,configHash,modelFingerprint:config.modelFingerprint}));
      if(message.type==='evaluate'){
        this.evaluation=message;started.resolve();
        if(automatic)queueMicrotask(()=>this.complete());
      }
    }
    complete(){this.send({type:'evaluation',id:this.evaluation.id,result:result(this.evaluation.job)});}
    progress(values){this.send({type:'progress',id:this.evaluation.id,...values});}
    terminate(){this.terminated=true;}
  }
  const storage=new Map();
  const client=new TrainingClient({sharedOnly:true,coordinatorUrl:origin,workerFactory:()=>new FixtureWorker(),
    storage:{getItem:key=>storage.get(key),setItem:(key,value)=>storage.set(key,value)},fetcher:async(url,options={})=>{
      const path=new URL(url).pathname;calls.push(path);
      if(path.endsWith('/config.json'))return new Response(text);
      if(path.endsWith('/status'))return Response.json(status());
      if(path.endsWith('/lease')){
        if(++leaseCount===1&&!terminalOnly)return Response.json({job});
        atTerminal=true;return Response.json({job:null,status:terminal});
      }
      if(path.endsWith('/result')){uploads.push(JSON.parse(options.body));accepted++;return Response.json({accepted:true});}
      if(path.endsWith('/heartbeat'))return Response.json({renewed:true});
      if(path.endsWith('/release'))return Response.json({released:true});
      throw new Error('Unexpected fixture request '+path);
    }});
  t.after(async()=>{await client.dispose();await client.loop;});
  return {client,workers,uploads,calls,job,candidate,config,curriculum,started,result,
    count:name=>calls.filter(path=>path.endsWith('/'+name)).length};
}

async function run(f){await f.client.initialize();await f.client.start();await f.client.loop;}

test('fit uploads preserve the full candidate and calibration while the saved checkpoint stays unchanged',async t=>{
  const f=fixture(t);await run(f);
  assert.equal(f.uploads.length,1);const payload=f.uploads[0];
  assert.equal(payload.candidateParameters.length,696);assert.deepEqual(payload.candidateParameters,f.candidate);
  assert.deepEqual(payload.candidateParameters.slice(0,24),f.job.parameters.slice(0,24));
  assert.deepEqual(payload.calibration,f.result().calibration);
  assert.deepEqual(payload.provenance.parameters,f.job.parameters);assert.equal(payload.provenance.parametersHash,f.job.parametersHash);
  assert.equal(payload.provenance.mode,'decoder-fit');assert.deepEqual(payload.provenance.wasmExecution,f.config.optimizer.acceptance.nativeExecution);
  assert.equal(payload.metrics.success,false);assert.equal(payload.metrics.terminated,false);assert.equal(payload.metrics.cancelled,false);
  assert.equal(payload.metrics.simSeconds,15,'Three demonstration trajectories are not one five-second autonomous trial');
  assert.deepEqual(f.client.parameters,f.job.parameters);assert.deepEqual(f.client.exportCheckpoint().parameters,f.job.parameters);
  assert.equal(f.client.state.completedEpisodes,1);assert.equal(f.client.state.contributedEpisodes,1);
  assert.equal(f.client.state.history[0].mode,'decoder-fit');assert.equal(f.client.state.history[0].success,false);
});

test('failed teacher fitting uploads an attempted outcome with no candidate and does not stall the client',async t=>{
  const f=fixture(t,{terminal:'needs-review',mutateResult:r=>({...r,simSeconds:.2,steps:100,reason:'fit_failed',candidateParameters:null,
    calibration:{...r.calibration,passed:false,trainingTrials:1,validationTrials:0}})});await run(f);
  assert.equal(f.uploads.length,1);assert.equal(f.uploads[0].candidateParameters,null);
  assert.deepEqual([f.uploads[0].calibration.trainingTrials,f.uploads[0].calibration.validationTrials],[1,0]);
  assert.equal(f.uploads[0].metrics.success,false);assert.equal(f.client.state.phase,'ready');assert.equal(f.client.state.error,null);
  assert.match(f.client.state.message,/Review this stage/);assert.deepEqual(f.client.state.curriculum,f.curriculum(true));
});

test('demonstrations cannot claim autonomous success or alter sensory coefficients',async t=>{
  const cases=[
    ['autonomous success',r=>({...r,success:true}),/Demonstration cannot claim autonomous success/],
    ['sensory update',r=>({...r,candidateParameters:r.candidateParameters.map((v,k)=>k===0?.01:v)}),/Fit changed sensory parameters/],
    ['missing evidence',r=>({...r,calibration:undefined}),/Missing calibration evidence/],
    ['wrong fit provenance',r=>({...r,provenance:{...r.provenance,mode:'episode'}}),/Missing calibration evidence/],
    ['out-of-bounds motor',r=>({...r,candidateParameters:r.candidateParameters.map((v,k)=>k===24?5:v)}),/Checkpoint parameters/],
    ['failed fit with weights',r=>({...r,calibration:{...r.calibration,passed:false}}),/Failed fit supplied weights/],
  ];
  for(const [name,mutateResult,expected] of cases)await t.test(name,async sub=>{
    const f=fixture(sub,{mutateResult});await run(f);
    assert.equal(f.client.state.phase,'error');assert.match(f.client.state.error,expected);
    assert.equal(f.count('result'),0);assert.equal(f.client.state.completedEpisodes,0);assert.equal(f.client.state.contributedEpisodes,0);
  });
});

test('a complete autonomous recovery_success is accepted without demonstration fields',async t=>{
  const f=fixture(t,{mode:'episode'});await run(f);
  assert.equal(f.uploads.length,1);const payload=f.uploads[0];
  assert.equal(payload.metrics.reason,'recovery_success');assert.equal(payload.metrics.success,true);assert.equal(payload.metrics.simSeconds,5);
  assert.equal(Object.hasOwn(payload,'candidateParameters'),false);assert.equal(Object.hasOwn(payload,'calibration'),false);
  assert.equal(f.client.state.completedEpisodes,1);assert.equal(f.client.state.contributedEpisodes,1);
});

test('recovery cannot use another stage success reason or stop before its assigned horizon',async t=>{
  for(const mutateResult of [r=>({...r,reason:'stage_success'}),r=>({...r,simSeconds:4.998,steps:2499})]){
    const f=fixture(t,{mode:'episode',mutateResult});await run(f);
    assert.match(f.client.state.error,/complete trial or physical failure/);assert.equal(f.count('result'),0);assert.equal(f.client.state.completedEpisodes,0);
  }
});

for(const terminal of ['complete','needs-review'])test(`terminal ${terminal} refreshes curriculum and ends without polling or evaluating`,async t=>{
  const f=fixture(t,{terminal,terminalOnly:true});await run(f);
  assert.equal(f.count('lease'),1);assert.equal(f.count('status'),2);assert.equal(f.count('result'),0);assert.equal(f.count('heartbeat'),0);
  assert.equal(f.workers.length,1);assert.equal(f.workers[0].messages.some(message=>message.type==='evaluate'),false);
  assert.equal(f.workers[0].terminated,true);assert.equal(f.client.worker,null);assert.equal(f.client.workerReady,null);
  assert.equal(f.client.running,false);assert.equal(f.client.state.phase,'ready');assert.equal(f.client.state.activity,null);assert.equal(f.client.state.error,null);
  assert.deepEqual(f.client.state.curriculum,f.curriculum(true));assert.equal(f.client.state.coordinator.sequenceStatus,terminal);
});

test('cumulative calibration time keeps restarted demonstrations live, but repeated or regressing clocks do not',async t=>{
  let now=0;const timers=new Map();let timerId=0;
  t.mock.method(performance,'now',()=>now);
  t.mock.method(globalThis,'setInterval',(callback,ms)=>{const id=++timerId;timers.set(id,{callback,ms});return id;});
  t.mock.method(globalThis,'clearInterval',id=>timers.delete(id));
  const tick=()=>{for(const timer of [...timers.values()])if(timer.ms===1000)timer.callback();};
  const f=fixture(t,{automatic:false});await f.client.initialize();await f.client.start();await f.started.promise;
  const worker=f.workers[0],watch=f.client.evaluationWatch;
  worker.progress({calibrationNeuralMs:5500,neuralMs:5500,nativeTimeSeconds:5.5,steps:2500,simSeconds:5});
  for(const cumulative of [5502,11002]){
    now+=EVALUATION_STALL_MS-1;
    worker.progress({calibrationNeuralMs:cumulative,neuralMs:2,nativeTimeSeconds:.002,steps:0,simSeconds:0});tick();
    assert.equal(f.client.running,true);assert.equal(watch.lastAdvance,now);assert.equal(watch.clocks.neuralMs,5500);
    assert.equal(watch.clocks.calibrationNeuralMs,cumulative);
  }
  const lastAdvance=watch.lastAdvance;now+=EVALUATION_STALL_MS-1;
  for(const calibrationNeuralMs of [11002,5500,NaN,Infinity])worker.progress({calibrationNeuralMs,neuralMs:2,simSeconds:0});
  worker.send({type:'progress',id:worker.evaluation.id+1,calibrationNeuralMs:99999});
  assert.equal(watch.lastAdvance,lastAdvance);assert.equal(watch.clocks.calibrationNeuralMs,11002);tick();assert.equal(f.client.running,true);
  now++;tick();await f.client.loop;
  assert.match(f.client.state.error,/no progress for 3 minutes/);assert.equal(worker.terminated,true);
  assert.equal(f.count('result'),0);assert.equal(f.count('release'),1);assert.equal(f.client.state.completedEpisodes,0);
});
