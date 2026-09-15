// Dispatch/provenance tests with a declared runner fixture. These do not claim
// to train, simulate biology, or validate the quality of the diagnostic teacher.
import test from 'node:test';
import assert from 'node:assert/strict';
import {registerHooks} from 'node:module';
const url=new URL('../training/sequential-environment.js',import.meta.url).href;
const mocks=new Map([
 ['./environment.js','export const createTrainingEnvironment=()=>{throw new Error("inject test environment");};'],
 ['./decoder-calibration.js','export const createDecoderCalibrationRunner=()=>{throw new Error("inject test runner");};'],
 ['./sensorimotor-parameters.js','export function sensoryParameterCount(config){if(config.invalidSensory)throw new Error("invalid sensory contract");return 24;}'],
]);
const hooks=registerHooks({resolve(specifier,context,next){if(context.parentURL===url&&mocks.has(specifier))return {url:'data:text/javascript,'+encodeURIComponent(mocks.get(specifier)),shortCircuit:true};return next(specifier,context);}});
const {createSequentialTrainingEnvironment}=await import(url);
const sha='a'.repeat(64),teacherURL='/training/teacher.json';
function fixture({legacy=false,status='awaiting-autonomous-validation',assetObject=false}={}){
 const config={environmentVersion:'fixture-v1',modelFingerprint:'b'.repeat(64),dtMs:.5,bodyBlockMs:2,parameterContract:'banc-sensorimotor-sequence-v1',parameters:Array.from({length:696},()=>({initial:0})),
  assets:{[teacherURL]:assetObject?{sha256:sha,url:teacherURL}:sha},stages:[{id:'maintained_flight',durationSeconds:5},{id:'recovery',durationSeconds:5}],
  ...(legacy?{}:{trainingSequence:{teacherAsset:teacherURL,fitOptions:{ridge:1e-5}}})};
 const f={config,fetches:0,readyCalls:0,disposed:0,calls:[],runners:[],checkpoints:0,status};
 const environment={config,configHash:'c'.repeat(64),modelFingerprint:config.modelFingerprint,
  async ready(){f.readyCalls++;return {backend:'wasm',neuralEngine:'wasm',wasmExecution:{backend:'wasm',moduleSha256:'d'.repeat(64)}};},
  async evaluate(job,options){f.calls.push({job,options});
   for(const neuralMs of [0,100,80])options.onProgress?.({neuralMs,stage:job.stage});
   options.onFrame?.({recordedFixture:true});
   return {success:true,return:9,simSeconds:5,nativeTimeSeconds:5.5,neuralMs:5500,stage:job.stage};
  },dispose(){f.disposed++;}};
 const trials=[{seed:101,split:'train',stage:'maintained_flight',durationSeconds:5},{seed:102,split:'train',stage:'maintained_flight',durationSeconds:5},{seed:103,split:'validation',stage:'maintained_flight',durationSeconds:5}];
 const parameters=Array.from({length:696},(_,i)=>i<24?.1:0);
 const job={mode:'decoder-fit',stage:'maintained_flight',seed:99,durationSeconds:5,parameters,calibrationTrials:trials,generation:2,pairId:'pair',sign:0,parametersHash:'e'.repeat(64)};
 const createRunner=options=>{f.runners.push(options);return {async run(run){
  f.run=run;if(f.status==='cancelled')return {status:'cancelled',trials:[]};
  const selected=f.status==='teacher_failed'?run.trials.slice(0,1):run.trials;
  for(const trial of selected){await options.checkpoint?.();await options.environment.evaluate({stage:trial.stage,seed:trial.seed,durationSeconds:5,parameters:options.evaluationParameters},
   {checkpoint:options.checkpoint,onFrame:options.onFrame,onProgress:options.onProgress});}
  const result={status:f.status,parameters:Array.from({length:672},()=>.02),trials:selected.map(t=>({...t,success:f.status!=='teacher_failed',simSeconds:f.status==='teacher_failed'?.2:5})),
   ...(f.status==='teacher_failed'?{}:{metrics:{before:{validation:{mixedUnitRawRmse:.3}},after:{validation:{mixedUnitRawRmse:.2}},changedParameterCount:672,fits:Array.from({length:8},()=>({converged:true}))}})};
  f.mutateFit?.(result);return result;
 }};};
 const dependencies={createEnvironment:async()=>environment,createRunner,fetcher:async()=>{f.fetches++;return {ok:true,text:async()=>'{"fixture":"teacher"}'};}};
 return {...f,environment,job,dependencies,state:f};
}

test('legacy configurations and ordinary assigned evaluations bypass the fit runner exactly',async()=>{
 const f=fixture({legacy:true}),wrapped=await createSequentialTrainingEnvironment(f.config,{},f.dependencies);
 assert.strictEqual(wrapped,f.environment);assert.equal(f.state.fetches,0);assert.equal(f.state.runners.length,0);
 const g=fixture(),enabled=await createSequentialTrainingEnvironment(g.config,{},g.dependencies),job={...g.job,mode:'evaluation'},options={dutyCycle:.4};
 const expected=await enabled.evaluate(job,options);assert.equal(expected.return,9);assert.equal(g.state.fetches,0);assert.equal(g.state.runners.length,0);
 assert.strictEqual(g.state.calls[0].job,job);assert.strictEqual(g.state.calls[0].options,options);
});

test('assigned fitting preserves budgets and the sensory prefix while reporting no autonomous score',async()=>{
 const f=fixture(),wrapped=await createSequentialTrainingEnvironment(f.config,{},f.dependencies),progress=[],frames=[],budget=()=>({dutyCycle:.35,previewHz:2}),checkpoint=async()=>{f.state.checkpoints++;};
 const result=await wrapped.evaluate(f.job,{dutyCycle:.35,previewHz:2,getBudget:budget,checkpoint,onFrame:frame=>frames.push(frame),onProgress:p=>progress.push(p)});
 assert.equal(f.state.readyCalls,1);assert.equal(f.state.calls.length,3);assert.equal(f.state.checkpoints,3);assert.equal(frames.length,3);
 for(const call of f.state.calls){assert.strictEqual(call.options.getBudget,budget);assert.equal(call.options.dutyCycle,.35);assert.equal(call.options.previewHz,2);assert.strictEqual(call.options.checkpoint,checkpoint);assert.deepEqual(call.job.parameters,f.job.parameters);}
 const r=f.state.runners[0];assert.equal(r.initialParameters.length,672);assert.equal(r.evaluationParameters.length,696);assert.equal(r.teacherCalibrationSha256,sha);assert.deepEqual(f.state.run.fitOptions,{ridge:1e-5});
 assert.equal(result.reason,'fit_candidate');assert.equal(result.success,false);assert.equal(result.return,0);assert.equal(result.simSeconds,15);assert.equal(result.steps,7500);
 assert.equal(result.calibration.passed,true);assert.equal(result.calibration.trainingTrials,2);assert.equal(result.calibration.validationTrials,1);
 assert.equal(result.calibration.teacherUsedForEvaluation,false);assert.equal(result.calibration.teacherUsedForDemonstrations,true);assert.equal(result.calibration.autonomousEvaluationPerformed,false);
 assert.deepEqual(result.parameters,f.job.parameters);assert.deepEqual(result.candidateParameters.slice(0,24),f.job.parameters.slice(0,24));assert(result.candidateParameters.slice(24).every(x=>x===.02));
 assert.equal(result.provenance.configHash,f.environment.configHash);assert.equal(result.provenance.mode,'decoder-fit');assert.equal(result.provenance.neuralEngine,'wasm');
 const clocks=progress.map(p=>p.calibrationNeuralMs);assert(clocks.every(Number.isFinite));assert(clocks.every((x,i)=>i===0||x>=clocks[i-1]));assert.equal(clocks.at(-1),16500);
 assert.deepEqual(new Set(progress.map(p=>p.calibrationTrial)),new Set([1,2,3]));
 // Teacher text is cached, but each subsequent assignment gets its own runner.
 await wrapped.evaluate(f.job);assert.equal(f.state.fetches,1);assert.equal(f.state.runners.length,2);
});

test('early teacher failure reports actual attempted counts and never supplies a candidate',async()=>{
 const f=fixture({status:'teacher_failed'}),wrapped=await createSequentialTrainingEnvironment(f.config,{},f.dependencies),result=await wrapped.evaluate(f.job);
 assert.equal(result.reason,'fit_failed');assert.equal(result.success,false);assert.equal(result.return,0);assert.equal(result.candidateParameters,null);assert.equal(result.calibration.passed,false);
 assert.equal(result.calibration.trainingTrials,1);assert.equal(result.calibration.validationTrials,0);assert.equal(result.calibration.assignedTrainingTrials,2);assert.equal(result.calibration.assignedValidationTrials,1);
 assert.equal(result.calibration.autonomousEvaluationPerformed,false);assert.equal(result.simSeconds,.2);
});

test('non-improving, unconverged or incomplete fits cannot become candidates',async()=>{
 for(const mutate of [fit=>fit.metrics.after.validation.mixedUnitRawRmse=.3,fit=>fit.metrics.changedParameterCount=0,fit=>fit.metrics.fits[0].converged=false,fit=>fit.metrics.fits=[],fit=>fit.parameters.pop(),fit=>fit.trials[0].success=false]){
  const f=fixture();f.state.mutateFit=mutate;const wrapped=await createSequentialTrainingEnvironment(f.config,{},f.dependencies),result=await wrapped.evaluate(f.job);
  assert.equal(result.reason,'fit_failed');assert.equal(result.candidateParameters,null);assert.equal(result.success,false);
 }
});

test('cancellation propagates as AbortError rather than a completed fit result',async()=>{
 const f=fixture({status:'cancelled'}),wrapped=await createSequentialTrainingEnvironment(f.config,{},f.dependencies);
 await assert.rejects(wrapped.evaluate(f.job),error=>error.name==='AbortError');assert.equal(f.state.calls.length,0);
});

test('malformed assignments fail before ready, fetch or demonstrations',async()=>{
 for(const mutate of [j=>j.calibrationTrials.pop(),j=>j.calibrationTrials[0].split='validation',j=>j.calibrationTrials[0].seed=103,j=>j.calibrationTrials[0].stage='takeoff',j=>j.calibrationTrials[0].durationSeconds=1,j=>j.parameters.pop(),j=>j.parameters[0]=NaN]){
  const f=fixture(),wrapped=await createSequentialTrainingEnvironment(f.config,{},f.dependencies),job=structuredClone(f.job);mutate(job);
  await assert.rejects(wrapped.evaluate(job),/Invalid demonstration/);assert.equal(f.state.readyCalls,0);assert.equal(f.state.fetches,0);assert.equal(f.state.calls.length,0);
 }
});

test('teacher asset descriptors normalize to hashes and failed wrapper construction releases its environment',async()=>{
 const f=fixture({assetObject:true}),wrapped=await createSequentialTrainingEnvironment(f.config,{},f.dependencies);await wrapped.evaluate(f.job);
 assert.equal(f.state.runners[0].teacherCalibrationSha256,sha);
 for(const mutate of [c=>c.invalidSensory=true,c=>delete c.assets[teacherURL],c=>c.assets[teacherURL]={sha256:'not-a-digest'}]){
  const g=fixture();mutate(g.config);await assert.rejects(createSequentialTrainingEnvironment(g.config,{},g.dependencies));assert.equal(g.state.disposed,1);
 }
});
test.after(()=>hooks.deregister());
