// Browser-only demonstration collection and bounded command imitation.
// No coordinator, network, filesystem, optimizer or deployed-decoder mutation.
import {validateMotorDecoderContract,validateMotorDecoderVector} from '../motor-decoder.js';
import {createFlightTeacher,validateFlightTeacherCalibration} from './flight-teacher.js';
import {measureFlightKinematics} from './flight-observation.js';

const finite=Number.isFinite,clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const requireThat=(ok,message)=>{if(!ok)throw new Error('Decoder calibration: '+message);};
const vector=(v,n,label)=>requireThat(v?.length===n&&Array.from(v).every(finite),`invalid ${label}`);
const equal=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const sha=async value=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',
  typeof value==='string'?new TextEncoder().encode(value):value)),b=>b.toString(16).padStart(2,'0')).join('');
const assetHash=(config,url)=>typeof config.assets?.[url]==='string'?config.assets[url]:config.assets?.[url]?.sha256;
const REQUIRED_MECHANICS=['/body-model/flybody-mujoco.xml','/body-model/flybody-mujoco.json',
  '/body-engine/mujoco.wasm','/flybody-wings.js','/flybody-physics.js'];

function layout(value){
  const contract=validateMotorDecoderContract(value),families=[];
  for(const kind of ['power','steering'])for(let side=0;side<2;side++){
    const groups=(kind==='power'?[null]:[0,1,2]).map(axis=>({kind,side,axis,
      name:kind+'_'+['left','right'][side]+(axis===null?'':'_'+['yaw','roll','pitch'][axis]),
      indices:contract.parameters.flatMap((p,i)=>p.kind===kind&&p.sideIndex===side&&(axis===null||p.axisIndex===axis)?[i]:[])}));
    const size=kind==='power'?12:108;
    requireThat(groups.every(g=>g.indices.length===size),'unsupported decoder feature layout');
    families.push({kind,side,size,groups});
  }
  requireThat(new Set(families.flatMap(f=>f.groups.flatMap(g=>g.indices))).size===672,'decoder layout must cover672 coefficients');
  return {contract,families};
}

/** Streaming normal equations use actual production features at EVERY applied
 * 0.2 ms phase sample. Three steering axes share the same108 input columns;
 * retain one covariance per side. No full trajectory/state capture is kept. */
export function createDecoderStatistics(contract,{maximumSamples=40000}={}){
  requireThat(Number.isInteger(maximumSamples)&&maximumSamples>0&&maximumSamples<=100000,'sample cap');
  const {families}=layout(contract),blocks=families.map(f=>({...f,count:0,gram:new Float64Array(f.size*f.size),
    rhs:f.groups.map(()=>new Float64Array(f.size)),targetSum:new Float64Array(f.groups.length),
    targetSquare:new Float64Array(f.groups.length),clippedTargets:new Uint32Array(f.groups.length)}));
  let count=0;
  function add(features,targets){
    vector(features,672,'actual decoder features');vector(targets?.power,2,'power targets');
    requireThat(Array.from(features).every(x=>Math.abs(x)<=1),'feature outside production excitation/phase bounds');
    requireThat(targets.power.every(x=>x>=0&&x<=4),'unclipped power target range');
    requireThat(targets.steering?.length===2,'steering targets');targets.steering.forEach(row=>vector(row,3,'steering targets'));
    requireThat(count<maximumSamples,'demonstration sample limit exceeded');
    for(const f of blocks){
      const x=f.groups[0].indices.map(j=>features[j]),p=f.size;
      // Verify the expected same-side feature sharing rather than silently
      // fitting the wrong axis after a future feature-contract change.
      for(const g of f.groups)for(let j=0;j<p;j++)requireThat(features[g.indices[j]]===x[j],'axis feature layout changed');
      for(let j=0;j<p;j++)for(let k=0;k<=j;k++)f.gram[j*p+k]+=x[j]*x[k];
      for(let a=0;a<f.groups.length;a++){
        const g=f.groups[a],y=g.kind==='power'?targets.power[g.side]:targets.steering[g.side][g.axis];
        f.targetSum[a]+=y;f.targetSquare[a]+=y*y;
        f.clippedTargets[a]+=Number(y!==clamp(y,g.kind==='power'?0:-.25,g.kind==='power'?1:.25));
        for(let j=0;j<p;j++)f.rhs[a][j]+=x[j]*y;
      }
      f.count++;
    }
    count++;
  }
  function finish(){
    requireThat(count>0,'no applied teacher/motor pairs');
    return {count,blocks:blocks.map(f=>{const gram=f.gram.slice(),p=f.size;
      for(let j=0;j<p;j++)for(let k=0;k<j;k++)gram[k*p+j]=gram[j*p+k];
      return {...f,gram,rhs:f.rhs.map(x=>x.slice()),targetSum:f.targetSum.slice(),targetSquare:f.targetSquare.slice(),clippedTargets:f.clippedTargets.slice()};})};
  }
  return Object.freeze({add,finish,get count(){return count;}});
}

/** Same weighted ridge objective and box-coordinate minimizer as the offline
 * calibrator, evaluated from sufficient statistics instead of retaining rows.
 * Gram/rhs/y2 already carry equal-trajectory normalized weights. */
export function fitBoundedRidgeStatistics(gram,rhs,y2,initial,bounds,
  {ridge=1e-5,maxSweeps=2000,tolerance=1e-9}={}){
  const p=initial.length;vector(gram,p*p,'Gram matrix');vector(rhs,p,'target covariance');vector(initial,p,'prior');
  requireThat(p>0&&bounds.length===p&&finite(y2)&&y2>=0,'regression dimensions');
  requireThat(finite(ridge)&&ridge>0&&Number.isInteger(maxSweeps)&&maxSweeps>0&&maxSweeps<=10000&&finite(tolerance)&&tolerance>0,'regression options');
  for(let j=0;j<p;j++){
    const b=bounds[j];vector(b,2,'bounds');requireThat(b[0]<b[1]&&initial[j]>=b[0]&&initial[j]<=b[1]&&gram[j*p+j]>=0,'bounds or Gram diagonal');
    for(let k=0;k<p;k++)requireThat(Math.abs(gram[j*p+k]-gram[k*p+j])<1e-10,'asymmetric Gram matrix');
  }
  const w=Float64Array.from(initial),gradient=Float64Array.from(rhs,(y,j)=>-y+initial.reduce((s,v,k)=>s+gram[j*p+k]*v,0));
  const objective=()=>{
    let value=y2;for(let j=0;j<p;j++){value-=2*w[j]*rhs[j];value+=ridge*(w[j]-initial[j])**2;
      for(let k=0;k<p;k++)value+=w[j]*gram[j*p+k]*w[k];}return value;
  };
  const before=objective();let sweeps=0,maxDelta=Infinity;
  for(;sweeps<maxSweeps;sweeps++){
    maxDelta=0;
    for(let j=0;j<p;j++){
      const delta=clamp(w[j]-(gradient[j]+ridge*(w[j]-initial[j]))/(gram[j*p+j]+ridge),...bounds[j])-w[j];
      if(delta!==0){w[j]+=delta;for(let k=0;k<p;k++)gradient[k]+=gram[k*p+j]*delta;}
      maxDelta=Math.max(maxDelta,Math.abs(delta));
    }
    if(maxDelta<=tolerance){sweeps++;break;}
  }
  let projectedGradient=0;
  for(let j=0;j<p;j++)projectedGradient=Math.max(projectedGradient,Math.abs(w[j]-clamp(w[j]-gradient[j]-ridge*(w[j]-initial[j]),...bounds[j])));
  const after=objective();requireThat(finite(after)&&after<=before+1e-9*Math.max(1,Math.abs(before)),'ridge objective increased');
  return {values:Array.from(w),sweeps,converged:maxDelta<=tolerance,maxCoordinateChange:maxDelta,projectedGradient,
    objectiveBefore:Math.max(0,before),objectiveAfter:Math.max(0,after),
    unexcitedColumns:Array.from({length:p},(_,j)=>j).filter(j=>gram[j*p+j]===0)};
}

function combined(trials,split){
  const selected=trials.filter(t=>t.split===split);requireThat(selected.length>0,'whole-trajectory training and validation required');
  return selected[0].statistics.blocks.map((base,index)=>{
    const f={...base,gram:new Float64Array(base.gram.length),rhs:base.rhs.map(x=>new Float64Array(x.length)),
      targetSum:new Float64Array(base.groups.length),targetSquare:new Float64Array(base.groups.length),sampleCount:0};
    for(const trial of selected){const b=trial.statistics.blocks[index],weight=1/(selected.length*b.count);f.sampleCount+=b.count;
      for(let j=0;j<f.gram.length;j++)f.gram[j]+=b.gram[j]*weight;
      for(let a=0;a<f.groups.length;a++){f.targetSum[a]+=b.targetSum[a]*weight;f.targetSquare[a]+=b.targetSquare[a]*weight;
        for(let j=0;j<f.size;j++)f.rhs[a][j]+=b.rhs[a][j]*weight;}
    }
    return f;
  });
}
function metrics(blocks,parameters){
  const outputs=[];
  for(const f of blocks)for(let a=0;a<f.groups.length;a++){
    const g=f.groups[a],w=g.indices.map(j=>parameters[j]);let mse=f.targetSquare[a];
    for(let j=0;j<f.size;j++){mse-=2*w[j]*f.rhs[a][j];for(let k=0;k<f.size;k++)mse+=w[j]*f.gram[j*f.size+k]*w[k];}
    requireThat(finite(mse)&&mse>-1e-8,'invalid held-out fit error');
    outputs.push({name:g.name,rawRmse:Math.sqrt(Math.max(0,mse)),targetMean:f.targetSum[a],
      targetStandardDeviation:Math.sqrt(Math.max(0,f.targetSquare[a]-f.targetSum[a]**2)),samples:f.sampleCount});
  }
  return {outputs,mixedUnitRawRmse:Math.sqrt(outputs.reduce((s,x)=>s+x.rawRmse**2,0)/outputs.length)};
}

export async function fitDecoderStatistics(contract,trials,initial,{checkpoint=async()=>{},...options}={}){
  contract=validateMotorDecoderContract(contract);initial=Array.from(validateMotorDecoderVector(contract,initial));
  requireThat(Array.isArray(trials)&&trials.length>=2,'at least two whole trajectories required');
  const seeds=new Set();for(const t of trials){requireThat(Number.isInteger(t.seed)&&!seeds.has(t.seed)&&['train','validation'].includes(t.split)&&t.statistics?.count>0,'independent trial seeds/splits required');seeds.add(t.seed);}
  const train=combined(trials,'train'),validation=combined(trials,'validation');
  requireThat(train.some(f=>f.gram.some(x=>x!==0)),'no motor-feature information');
  requireThat(train.some(f=>f.targetSquare.some(x=>x>0)),'no nonzero teacher target information');
  const parameters=initial.slice(),fits=[];
  for(const f of train)for(let a=0;a<f.groups.length;a++){
    await checkpoint();const g=f.groups[a],result=fitBoundedRidgeStatistics(f.gram,f.rhs[a],f.targetSquare[a],
      g.indices.map(j=>initial[j]),g.indices.map(j=>[contract.parameters[j].min,contract.parameters[j].max]),options);
    g.indices.forEach((j,k)=>{parameters[j]=result.values[k];});
    const {values,unexcitedColumns,...summary}=result;fits.push({output:g.name,...summary,unexcitedParameterCount:unexcitedColumns.length});
  }
  validateMotorDecoderVector(contract,parameters);
  return {parameters,fits,changedParameterCount:parameters.filter((v,i)=>v!==initial[i]).length,
    updateL2:Math.hypot(...parameters.map((v,i)=>v-initial[i])),
    before:{train:metrics(train,initial),validation:metrics(validation,initial)},
    after:{train:metrics(train,parameters),validation:metrics(validation,parameters)},
    unexcitedParameterCount:fits.reduce((s,f)=>s+f.unexcitedParameterCount,0),
    note:'Equal trajectory-weighted raw command imitation; clipping error cannot be reconstructed from moments. No closed-loop performance claim.'};
}

/** Install only on a locally owned demonstration body. The original decoder
 * still owns all48 excitation histories and advances at its normal1ms clock.
 * Never patch a prototype or write root/control arrays. */
export function createDemonstrationCollector({body,contract,initialParameters,calibration,targetHeight,maximumSamples=40000}){
  const original=body?.motorDecoder,descriptor=Object.getOwnPropertyDescriptor(body??{},'motorDecoder');
  requireThat(original&&descriptor&&descriptor.writable&&equal(original.contract,contract),'own writable production decoder required');
  requireThat(equal(original.snapshot().weights,Array.from(initialParameters)),'demonstration decoder differs from prior');
  requireThat(finite(body.metadata?.mass_g)&&Math.abs(body.metadata.mass_g-calibration.massG)<=calibration.massG*1e-8,'teacher/body mass mismatch');
  const statistics=createDecoderStatistics(contract,{maximumSamples});let teacher=createFlightTeacher({calibration,targetHeight}),phase='warmup',active=true;
  let lastSampleTime=null,sampleCount=0,warmupSamples=0,scoredSamples=0,lastUpdateTime=null,clippedTeacherChannels=0;
  const samples=[];
  function noForces(){for(const name of ['xfrc_applied','qfrc_applied'])requireThat(body.data[name]?.every(v=>v===0),'unexpected external force during demonstration');}
  const wrapper=Object.freeze({...original,get timeMs(){return original.timeMs;},sample(wingPhase){
    requireThat(active,'disposed demonstration collector');noForces();
    // body.time is refreshed only after each completed1ms control interval.
    // MuJoCo data.time advances at50us and identifies this applied0.2ms sample.
    const time=body.data.time,interval=original.timeMs,offset=time*1000-interval;
    requireThat(finite(time)&&offset>=-1e-7&&offset<1-1e-7,'applied decoder feature/native clock mismatch');
    requireThat(lastSampleTime===null||Math.abs(time-lastSampleTime-.0002)<1e-8,'native wing sample schedule changed');
    const pair=teacher.sample(wingPhase),features=original.features(wingPhase);statistics.add(features,pair.targets);
    clippedTeacherChannels+=pair.clippedChannels;sampleCount++;phase==='warmup'?warmupSamples++:scoredSamples++;lastSampleTime=time;
    // Small owned audit samples only; neither brain buffers nor physical state
    // enter the returned candidate. Targets are before output clipping.
    if(samples.length<8)samples.push({nativeTimeSeconds:time,decoderTimeMs:interval,wingPhaseRadians:wingPhase,
      phase,targetPower:pair.targets.power.slice(),targetSteering:pair.targets.steering.map(a=>a.slice()),featureL2:Math.hypot(...features)});
    return pair.controls;
  }});
  Object.defineProperty(body,'motorDecoder',{...descriptor,value:wrapper});
  return Object.freeze({
    release(height){requireThat(active&&finite(height),'release height');teacher=createFlightTeacher({calibration,targetHeight:height});phase='scored';},
    beforeStep(context){
      requireThat(active&&context.body===body&&context.durationSeconds===.002,'demonstration body/block identity');
      requireThat(Math.abs(context.neuralMs-(body.time+.002)*1000)<1e-7,'teacher requires completed neural packet before native step');
      requireThat(lastUpdateTime===null||Math.abs(body.time-lastUpdateTime-.002)<1e-8,'teacher update schedule changed');
      noForces();phase=context.phase;
      if(phase==='scored'){
        const k=measureFlightKinematics(body);teacher.update({quaternion:Array.from(body.data.qpos.slice(3,7)),
          omegaRoot:Array.from(body.data.qvel.slice(3,6)),height:k.height,verticalSpeed:k.verticalSpeed,dt:.002});
      }else requireThat(phase==='warmup','unsupported demonstration phase');
      lastUpdateTime=body.time;
    },
    finish(){requireThat(scoredSamples>0,'no scored demonstration pairs');return {statistics:statistics.finish(),
      summary:{samples:sampleCount,warmupSamples,scoredSamples,clippedTeacherChannels,lastSampleTimeSeconds:lastSampleTime,auditSamples:samples}};},
    restore(){if(!active)return;requireThat(body.motorDecoder===wrapper,'demonstration decoder replaced by another owner');Object.defineProperty(body,'motorDecoder',descriptor);active=false;}
  });
}

/** Real environment orchestration; one evaluation at a time, train/validation
 * seeds explicit. Failed teachers produce no candidate, not a false success or
 * a permanently failed worker. The environment remains caller-owned. */
export function createDecoderCalibrationRunner({config,environment,teacherCalibrationText,teacherCalibrationSha256,
  initialParameters,evaluationParameters=initialParameters,checkpoint=async()=>{},onFrame,onProgress}={}){
  requireThat(environment&&typeof environment.evaluate==='function'&&config,'environment/config required');
  const contract=validateMotorDecoderContract(config.motorDecoderContract),prior=Array.from(validateMotorDecoderVector(contract,initialParameters));
  requireThat(Array.isArray(evaluationParameters)&&evaluationParameters.length===config.parameters.length&&evaluationParameters.every(finite),'full evaluation parameter vector required');
  requireThat(typeof teacherCalibrationText==='string'&&/^[0-9a-f]{64}$/.test(teacherCalibrationSha256),'exact teacher asset text/hash required');
  const calibration=validateFlightTeacherCalibration(JSON.parse(teacherCalibrationText));
  let running=false;
  async function run({trials,fitOptions={}}={}){
    requireThat(!running,'concurrent calibration run');running=true;
    const summaries=[],records=[];let collector=null;
    try{
      requireThat(await sha(teacherCalibrationText)===teacherCalibrationSha256,'teacher asset hash mismatch');
      requireThat(Object.keys(config.assets??{}).some(url=>assetHash(config,url)===teacherCalibrationSha256),'teacher asset is not pinned by config');
      for(const url of REQUIRED_MECHANICS)requireThat(calibration.mechanics[url]&&calibration.mechanics[url]===assetHash(config,url),'teacher mechanics mismatch: '+url);
      for(const [url,digest]of Object.entries(calibration.mechanics))requireThat(assetHash(config,url)===digest,'teacher mechanics mismatch: '+url);
      requireThat(environment.modelFingerprint===config.modelFingerprint,'environment model fingerprint mismatch');
      requireThat(Array.isArray(trials)&&trials.length>=2&&trials.length<=8,'use2–8 declared demonstration trajectories');
      const seeds=new Set(),reserved=new Set(config.validation?.testSeeds??[]);
      for(const trial of trials){
        requireThat(Number.isInteger(trial.seed)&&trial.seed>=0&&trial.seed<=0xffffffff&&!seeds.has(trial.seed)&&!reserved.has(trial.seed),'demonstration seeds must be independent and not reserved test seeds');seeds.add(trial.seed);
        requireThat(['train','validation'].includes(trial.split)&&['maintained_flight','recovery'].includes(trial.stage),'unsupported split or teacher stage');
        requireThat(trial.durationSeconds===5&&config.stages?.some(s=>s.id===trial.stage&&s.durationSeconds===5),'teacher requires configured full5s horizon');
      }
      requireThat(trials.some(t=>t.split==='train')&&trials.some(t=>t.split==='validation'),'whole-trajectory validation required');
      for(const [index,trial]of trials.entries()){
        await checkpoint();let released=false;
        const job={seed:trial.seed,stage:trial.stage,durationSeconds:5,parameters:evaluationParameters.slice()};
        let result;
        try{
          result=await environment.evaluate(job,{checkpoint,onFrame,onProgress,
            onBeforePhysicsStep(context){
              if(!collector){requireThat(context.phase==='warmup'&&context.body.time===0,'demonstration must include initial live warmup');
                collector=createDemonstrationCollector({body:context.body,contract,initialParameters:prior,calibration,
                  targetHeight:config.initialCondition?.rootQpos?.[2]??3.5});}
              collector.beforeStep(context);
            },
            onInitialState(context){
              requireThat(collector&&context.phase==='scored-release','missing warmup collection/release hook');
              collector.release(context.initialObservation.height);released=true;
            }});
          const valid=result&&result.cancelled!==true&&result.metrics?.error==null&&result.reason!=='simulation_error'&&
            result.success===true&&result.simSeconds===5&&released;
          const summary={index,seed:trial.seed,split:trial.split,stage:trial.stage,success:result?.success===true,
            reason:result?.reason??'missing_result',simSeconds:result?.simSeconds??0,return:result?.return??null,
            cancelled:result?.cancelled===true,teacherIntervention:true};
          summaries.push(summary);
          if(!valid)return {schemaVersion:1,kind:'motor-decoder-fit-attempt',status:summary.cancelled?'cancelled':'teacher_failed',candidate:null,
            trials:summaries,reason:'Successful full-horizon teacher demonstrations are required before fitting; no autonomous result or promotion.'};
          requireThat(result.modelFingerprint===config.modelFingerprint&&result.configHash===environment.configHash&&
            result.seed===trial.seed&&equal(result.parameters,evaluationParameters),'demonstration result identity mismatch');
          const captured=collector.finish();
          requireThat(captured.summary.scoredSamples===25000&&
            Math.abs(captured.summary.lastSampleTimeSeconds+.0002-result.nativeTimeSeconds)<1e-7,
            'complete5s demonstration sample/native clock gate failed');
          records.push({...trial,...captured});summary.collection=captured.summary;
        }finally{collector?.restore();collector=null;}
      }
      const fit=await fitDecoderStatistics(contract,records,prior,{...fitOptions,checkpoint});
      const statisticsSha256=await sha(JSON.stringify(records.map(t=>({seed:t.seed,split:t.split,count:t.statistics.count,
        blocks:t.statistics.blocks.map(f=>({gram:Array.from(f.gram),rhs:f.rhs.map(x=>Array.from(x)),targetSquare:Array.from(f.targetSquare)}))}))));
      const output={schemaVersion:1,kind:'motor-decoder-calibration-candidate',status:'awaiting-autonomous-validation',
        parameters:fit.parameters,metrics:{...fit,parameters:undefined},trials:summaries,
        provenance:{configHash:environment.configHash,modelFingerprint:config.modelFingerprint,teacherCalibrationSha256,
          priorParametersSha256:await sha(JSON.stringify(prior)),statisticsSha256,profile:contract.version,
          teacherOnlyDuringDemonstrations:true,containsPrivilegedDecoderInputs:false,
          sampling:'All actual0.2ms applied phases with unchanged1ms production MN feature history; warmup and scored pairs included.',
          targetSpace:'pre-clamp-decoder',promotion:'Separate autonomous matched-seed evaluation and coordinator acceptance required'}};
      requireThat(new TextEncoder().encode(JSON.stringify(output)).length<65536,'candidate summary exceeds64KiB');return output;
    }catch(error){
      if(error?.name==='AbortError')return {schemaVersion:1,kind:'motor-decoder-fit-attempt',status:'cancelled',candidate:null,trials:summaries};
      throw error;
    }finally{collector?.restore();running=false;}
  }
  return Object.freeze({run});
}
