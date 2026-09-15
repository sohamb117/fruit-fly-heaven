import {createTrainingEnvironment} from './environment.js';
import {createDecoderCalibrationRunner} from './decoder-calibration.js';
import {sensoryParameterCount} from './sensorimotor-parameters.js';

/** Fits are assigned work, but their teacher-driven scores never become
 * autonomous evaluation scores. Only the coordinator can promote a proposal. */
export async function createSequentialTrainingEnvironment(requestedConfig,options={},
 {createEnvironment=createTrainingEnvironment,createRunner=createDecoderCalibrationRunner,fetcher=fetch}={}){
 const environment=await createEnvironment(requestedConfig,options),config=environment.config;
 if(!config.trainingSequence)return environment;
 let sensoryCount,url,teacherHash;
 try{
  sensoryCount=sensoryParameterCount(config);url=config.trainingSequence.teacherAsset;
  const asset=config.assets?.[url];teacherHash=typeof asset==='string'?asset:asset?.sha256;
  if(typeof url!=='string'||typeof teacherHash!=='string'||!/^[0-9a-f]{64}$/.test(teacherHash))throw new Error('Missing pinned flight teacher');
 }catch(error){environment.dispose();throw error;}
 let teacherText;
 return {...environment,async evaluate(job,runOptions={}){
  if(job.mode!=='decoder-fit')return environment.evaluate(job,runOptions);
  if(!Array.isArray(job.calibrationTrials)||job.calibrationTrials.length!==3||
   job.calibrationTrials.filter(t=>t?.split==='train').length!==2||job.calibrationTrials.filter(t=>t?.split==='validation').length!==1||
   new Set(job.calibrationTrials.map(t=>t.seed)).size!==3||job.calibrationTrials.some(t=>!Number.isInteger(t.seed)||t.seed<0||t.seed>0xffffffff||
    !['maintained_flight','recovery'].includes(t.stage)||t.durationSeconds!==5||!config.stages?.some(s=>s.id===t.stage&&s.durationSeconds===5)))throw new Error('Invalid demonstration assignment');
  if(!Array.isArray(job.parameters)||job.parameters.length!==config.parameters.length||!job.parameters.every(Number.isFinite))throw new Error('Invalid demonstration parameter vector');
  if(teacherText===undefined){const response=await fetcher(url,{cache:'no-store'});if(!response.ok)throw new Error('Flight teacher unavailable');teacherText=await response.text();}
  const info=await environment.ready();let trialIndex=0,lastClock=0,clockOffset=0;
  // Each underlying episode resets its clocks. Accumulate real advances so
  // fitting three trajectories cannot look like a frozen worker to the page.
  const wrappedEnvironment={...environment,async evaluate(trial,trialOptions){
   trialIndex++;lastClock=0;
   const result=await environment.evaluate(trial,{...runOptions,...trialOptions,
    onProgress(progress){const clock=progress.neuralMs??progress.nativeTimeSeconds*1000;
     if(Number.isFinite(clock))lastClock=Math.max(lastClock,clock);
     runOptions.onProgress?.({...progress,calibrationNeuralMs:clockOffset+lastClock,calibrationTrial:trialIndex,
      message:`Recording movement ${trialIndex} / ${job.calibrationTrials.length}`});}});
   // Progress callbacks can precede the final completed block. Accumulate
   // actual final clocks, never elapsed wall time or invented neural steps.
   for(const clock of [result.neuralMs,result.nativeTimeSeconds*1000,(result.totalSimSeconds??result.simSeconds)*1000])
    if(Number.isFinite(clock))lastClock=Math.max(lastClock,clock);
   clockOffset+=lastClock;
   runOptions.onProgress?.({calibrationNeuralMs:clockOffset,calibrationTrial:trialIndex,
    message:`Recorded movement ${trialIndex} / ${job.calibrationTrials.length}`});
   return result;
  }};
  const runner=createRunner({config,environment:wrappedEnvironment,teacherCalibrationText:teacherText,
   teacherCalibrationSha256:teacherHash,initialParameters:job.parameters.slice(sensoryCount),
   evaluationParameters:job.parameters,checkpoint:runOptions.checkpoint,onFrame:runOptions.onFrame,onProgress:runOptions.onProgress});
  const fitted=await runner.run({trials:job.calibrationTrials,fitOptions:config.trainingSequence.fitOptions??{}});
  if(fitted.status==='cancelled')throw new DOMException('Calibration cancelled','AbortError');
  if(!Array.isArray(fitted.trials)||fitted.trials.some(t=>!Number.isFinite(t.simSeconds)||t.simSeconds<0))throw new Error('Invalid demonstration fit result');
  const before=fitted.metrics?.before?.validation?.mixedUnitRawRmse,after=fitted.metrics?.after?.validation?.mixedUnitRawRmse;
  const passed=fitted.status==='awaiting-autonomous-validation'&&Number.isFinite(before)&&Number.isFinite(after)&&
   after<before&&fitted.metrics.changedParameterCount>0&&fitted.metrics.fits?.length===8&&fitted.metrics.fits.every(f=>f.converged)&&
   fitted.trials.length===3&&fitted.trials.every(t=>t.success===true&&t.simSeconds===5)&&
   Array.isArray(fitted.parameters)&&fitted.parameters.length===job.parameters.length-sensoryCount&&fitted.parameters.every(Number.isFinite);
  const calibration={passed,trainingTrials:fitted.trials.filter(t=>t.split==='train').length,
   validationTrials:fitted.trials.filter(t=>t.split==='validation').length,teacherUsedForEvaluation:false,
   assignedTrainingTrials:2,assignedValidationTrials:1,teacherOnlyDuringDemonstrations:true,teacherUsedForDemonstrations:true,autonomousEvaluationPerformed:false,teacherCalibrationSha256:teacherHash,
   interpretation:'Simulation-based command imitation; separate autonomous tests required',fit:fitted};
  const provenance={environmentVersion:config.environmentVersion,modelFingerprint:config.modelFingerprint,
   configHash:environment.configHash,seed:job.seed,stage:job.stage,dtMs:config.dtMs,bodyBlockMs:config.bodyBlockMs,
   bodyBackend:'mujoco-wasm',backend:info.backend,neuralEngine:info.neuralEngine,wasmExecution:info.wasmExecution,
   generation:job.generation,pairId:job.pairId,sign:job.sign,parametersHash:job.parametersHash,durationSeconds:job.durationSeconds,
   parameterContract:config.parameterContract,mode:'decoder-fit'};
  const simSeconds=fitted.trials.reduce((sum,t)=>sum+t.simSeconds,0);
  return {...provenance,parameters:job.parameters.slice(),provenance,return:0,success:false,terminated:false,
   cancelled:false,truncated:false,reason:passed?'fit_candidate':'fit_failed',simSeconds,
   steps:Math.round(simSeconds*1000/config.bodyBlockMs),metrics:{calibration:true},calibration,
   candidateParameters:passed?[...job.parameters.slice(0,sensoryCount),...fitted.parameters]:null};
 },};
}
