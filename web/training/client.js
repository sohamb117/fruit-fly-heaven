import {validateConfig,validateParameters,validateResult,makeGeneration,updateGeneration,checkpoint,readCheckpoint} from './optimizer.js';

const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const hash=async bytes=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),x=>x.toString(16).padStart(2,'0')).join('');
const detailEvent=(type,detail)=>new CustomEvent(type,{detail});
const abortError=()=>new DOMException('Training stopped','AbortError');
const codedError=(message,code)=>Object.assign(new Error(message),{code});
export const browserTrainingAvailable=config=>!!config&&(config.schemaVersion!==2||config.optimizer?.acceptance?.nativeExecution?.backend==='wasm');

// The explicit development server marks its own HTML. Ordinary downloaded
// clients keep using the hosted pool; this never enables an unshared mode.
export function trainingCoordinatorURL(location,developmentSameOrigin=false){
  const loopback=location.hostname==='localhost'||location.hostname.endsWith('.localhost')||location.hostname==='[::1]'||/^127(?:\.\d{1,3}){3}$/.test(location.hostname);
  const hosted=location.protocol==='https:'&&!loopback;
  return hosted||(loopback&&developmentSameOrigin===true)?location.origin:'https://flytrain.morisoba.moe';
}

/** Local and shared training orchestration. Loading this class starts no compute. */
export class TrainingClient extends EventTarget {
  constructor({workerFactory=()=>new Worker(new URL('./worker.js',import.meta.url),{type:'module'}),storage=globalThis.localStorage,fetcher=globalThis.fetch?.bind(globalThis),sharedOnly=false,coordinatorUrl}={}) {
    super();Object.assign(this,{workerFactory,storage,fetcher});
    const requiredCoordinatorUrl=sharedOnly?this.coordinatorUrl(coordinatorUrl):null;
    Object.defineProperties(this,{sharedOnly:{value:!!sharedOnly},requiredCoordinatorUrl:{value:requiredCoordinatorUrl}});
    this.pending=new Map();this.nextId=0;this.runToken=0;this.running=false;this.paused=false;this.worker=null;this.round=null;this.options={dutyCycle:.6,previewHz:6,mode:sharedOnly?'shared':'local',...(sharedOnly?{coordinatorUrl:requiredCoordinatorUrl}:{})};
    this.state={phase:'idle',message:'Ready',backend:null,stage:null,episode:0,generation:0,completedEpisodes:0,contributedEpisodes:0,simSeconds:0,wallSeconds:0,
      bestReturn:null,lastReturn:null,history:[],parameters:[],checkpointStatus:'unverified',coordinator:{connected:false},curriculum:[],error:null};
    this.contributorId=crypto.randomUUID();
    this.visibility=()=>{if(globalThis.document?.hidden&&this.running&&!this.paused)this.pause('Paused');};
    globalThis.document?.addEventListener('visibilitychange',this.visibility);
  }
  emit(patch={}) {Object.assign(this.state,patch);this.dispatchEvent(detailEvent('state',structuredClone(this.state)));}
  async initialize() {
    this.emit({phase:'loading',message:'Loading'});
    try {
      const response=await this.fetcher(new URL('./config.json',import.meta.url),{cache:'no-store'});if(!response.ok)throw new Error('Training settings are unavailable');
      const bytes=await response.arrayBuffer();this.config=validateConfig(JSON.parse(new TextDecoder().decode(bytes)));this.configHash=await hash(bytes);
      this.key=`heaven-training-v1:${this.configHash}`;this.parameters=this.config.parameters.map(p=>p.initial);this.state.stage=this.config.stage;this.completedStages=new Set();this.savedValidation=null;
      try {
        const raw=this.storage?.getItem(this.key);if(raw){const saved=JSON.parse(raw),c=readCheckpoint(saved.checkpoint,this.config,this.configHash);this.parameters=c.parameters;
          this.state.stage=c.stage;this.state.generation=c.generation;this.state.completedEpisodes=Number.isSafeInteger(saved.completedEpisodes)?saved.completedEpisodes:0;
          this.state.contributedEpisodes=Number.isSafeInteger(saved.contributedEpisodes)?saved.contributedEpisodes:0;
          this.state.history=Array.isArray(saved.history)?saved.history.filter(x=>Number.isFinite(x.return)&&typeof x.stage==='string').slice(-120):[];
          if(typeof saved.contributorId==='string'&&/^[a-zA-Z0-9-]{8,80}$/.test(saved.contributorId))this.contributorId=saved.contributorId;
          if(saved.outbox?.configHash===this.configHash&&saved.outbox?.modelFingerprint===this.config.modelFingerprint&&Number.isFinite(saved.outbox.objective)){
            this.outbox=saved.outbox;this.outboxUrl=typeof saved.outboxUrl==='string'?saved.outboxUrl:null;
          }
        }
      } catch {this.emit({message:'Loading'});}
      this.emit({phase:'ready',message:'Ready',modelFingerprint:this.config.modelFingerprint,configHash:this.configHash,parameters:this.parameters.slice(),
        curriculum:this.config.stages.map(s=>({...s,status:'available'})),config:this.config,checkpointStatus:'unverified',browserTrainingAvailable:browserTrainingAvailable(this.config)});
      return this;
    }catch(error){this.fail(error);throw error;}
  }
  async ensureWorker() {
    if(this.config&&!browserTrainingAvailable(this.config))throw codedError('This run is using the connected trainer.','native_trainer_required');
    if(this.workerReady)return this.workerReady;
    if(!this.config)await this.initialize();
    if(!browserTrainingAvailable(this.config))throw codedError('This run is using the connected trainer.','native_trainer_required');
    if(this.sharedOnly&&(!this.state.coordinator.connected||this.state.coordinator.url!==this.requiredCoordinatorUrl))throw new Error('The shared coordinator must be connected before training can start');
    const worker=this.worker=this.workerFactory();
    worker.addEventListener('message',event=>{
      if(this.worker!==worker)return;
      const message=event.data;
      if(message.type==='frame'){this.dispatchEvent(detailEvent('frame',message.frame));return;}
      if(message.type==='progress'){
        this.emit({message:message.message||message.phase||this.state.message,...(Number.isFinite(message.simSeconds)?{episodeSimSeconds:message.simSeconds}:{})});return;
      }
      if(message.type==='error') {
        const error=new Error(message.message||message.error||'Training worker failed');
        if(this.pending.has(message.id)){this.pending.get(message.id).reject(error);this.pending.delete(message.id);}
        else {for(const p of this.pending.values())p.reject(error);this.pending.clear();this.fail(error);}
        return;
      }
      const pending=this.pending.get(message.id);
      if(pending&&['ready','evaluation','cancelled','stopped'].includes(message.type)){
        this.pending.delete(message.id);message.type==='cancelled'||message.type==='stopped'?pending.reject(abortError()):pending.resolve(message);
      }
    });
    worker.addEventListener('error',event=>{if(this.worker!==worker)return;const error=new Error(event.message||'Training worker stopped unexpectedly');for(const p of this.pending.values())p.reject(error);this.pending.clear();this.workerReady=null;this.fail(error);});
    this.emit({phase:'loading',message:'Loading simulation'});
    this.workerReady=this.rpc('initialize',{configUrl:new URL('./config.json',import.meta.url).href}).then(message=>{
      if(this.worker!==worker)throw abortError();
      const info=message.info||message;
      if(info.configHash!==this.configHash||info.modelFingerprint!==this.config.modelFingerprint)throw new Error('Worker and page loaded different training builds; reload the page');
      this.validateExecution(info);
      this.emit({backend:info.backend,modelFingerprint:info.modelFingerprint,message:'Ready'});
      if(message.frame)this.dispatchEvent(detailEvent('frame',message.frame));return info;
    }).catch(error=>{worker.terminate();if(this.worker===worker){this.worker=null;this.workerReady=null;}throw error;});
    return this.workerReady;
  }
  rpc(type,payload={}) {
    const id=++this.nextId;
    return new Promise((resolve,reject)=>{this.pending.set(id,{resolve,reject});this.worker.postMessage({type,id,...payload});});
  }
  setBudget({dutyCycle=this.options.dutyCycle,previewHz=this.options.previewHz}={}) {
    if(!Number.isFinite(dutyCycle)||dutyCycle<.1||dutyCycle>1||!Number.isFinite(previewHz)||previewHz<0||previewHz>15)throw new Error('Invalid compute or preview budget');
    Object.assign(this.options,{dutyCycle,previewHz});this.worker?.postMessage({type:'budget',dutyCycle,previewHz});
  }
  async start(options={}) {
    if(this.running||this.starting)throw new Error('Training is already running');
    const runToken=++this.runToken;this.starting=true;this.state.error=null;
    try{return await this.startRun(options,runToken);}
    catch(error){if(runToken===this.runToken&&error.name!=='AbortError')this.fail(error);throw error;}
    finally{if(runToken===this.runToken)this.starting=false;}
  }
  async startRun(options,runToken) {
    if(this.sharedOnly){
      if(options.mode&&options.mode!=='shared')throw new Error('This training page always contributes to shared training');
      if(options.coordinatorUrl&&this.coordinatorUrl(options.coordinatorUrl)!==this.requiredCoordinatorUrl)throw new Error('This training page uses its fixed shared coordinator');
      options={...options,mode:'shared',coordinatorUrl:this.requiredCoordinatorUrl};
    }
    if(!this.config)await this.initialize();
    if(!browserTrainingAvailable(this.config))throw codedError('This run is using the connected trainer.','native_trainer_required');
    const previousMode=this.options.mode,nextMode=options.mode||previousMode;
    if(this.config.schemaVersion===2&&nextMode!=='shared')throw new Error('This run requires assigned training jobs');
    if(previousMode!=='shared'&&nextMode==='shared')this.localView=this.localViewSnapshot();
    if(previousMode==='shared'&&nextMode==='local'&&this.localView){this.emit(this.localView);this.localView=null;}
    this.options={...this.options,...options};this.setBudget(this.options);
    if(!['local','shared'].includes(this.options.mode))throw new Error('Unknown training mode');
    if(this.options.mode==='shared')await this.connectCoordinator(this.options.coordinatorUrl||this.state.coordinator.url);
    else if(options.stage&&options.stage!==this.state.stage){
      if(!this.config.stages.some(s=>s.id===options.stage))throw new Error('Unknown curriculum stage');
      this.emit({stage:options.stage,checkpointStatus:'unverified',bestReturn:null,validation:null});this.round=null;this.savedValidation=null;this.testValidation=null;
    }
    if(runToken!==this.runToken)return;
    this.running=true;this.paused=false;this.started=performance.now();this.wallBefore=this.state.wallSeconds;
    try{await this.ensureWorker();if(!this.running)return;this.emit({phase:'training',message:'Running',error:null});
      this.loop=(this.options.mode==='shared'?this.runShared(runToken):this.runLocal(runToken)).catch(error=>{if(runToken===this.runToken&&error.name!=='AbortError')this.fail(error);}).finally(()=>{if(runToken===this.runToken){this.running=false;this.persist();}});
    }catch(error){if(runToken!==this.runToken||error.name==='AbortError')return;this.running=false;this.fail(error);throw error;}
  }
  async runnable(token=this.runToken) {while(this.paused&&this.running&&token===this.runToken)await delay(50);if(!this.running||token!==this.runToken)throw abortError();}
  pause(message='Paused') {if(!this.running||this.paused)return;this.paused=true;this.worker?.postMessage({type:'pause'});this.emit({phase:'paused',message});}
  resume() {if(!this.running||!this.paused)return;this.paused=false;this.worker?.postMessage({type:'resume'});this.emit({phase:'training',message:'Running'});}
  async stop() {
    this.runToken++;this.starting=false;this.running=false;this.paused=false;this.worker?.postMessage({type:'cancel'});
    // Cancellation is cooperative at the next neural/body boundary. Resolve all
    // waiting callers too, so a stop during initialization cannot leave a loop.
    for(const p of this.pending.values())p.reject(abortError());this.pending.clear();
    if(this.heartbeat){clearInterval(this.heartbeat);this.heartbeat=null;}
    // Termination also releases model memory and makes a subsequent start clean.
    this.worker?.terminate();this.worker=null;this.workerReady=null;
    const lease=this.activeLease;this.activeLease=null;
    if(this.started)this.state.wallSeconds=this.wallBefore+(performance.now()-this.started)/1000;
    this.persist();this.emit({phase:'stopped',message:this.outbox?'Stopped · upload pending':'Stopped'});
    if(lease&&!this.hasPendingResult(lease))await this.releaseLease(lease);
  }
  async evaluate(job,role='exploration',token=this.runToken) {
    if(this.sharedOnly&&(role!=='contribution'||!this.activeLease||this.activeLease.jobId!==job.jobId))throw new Error('Shared training only evaluates jobs assigned by its coordinator');
    await this.runnable(token);validateParameters(job.parameters,this.config);
    this.emit({episode:this.state.completedEpisodes+1,stage:job.stage,message:`${role==='validation'?'Evaluating validation seed':role==='test'?'Evaluating independent test seed':'Running episode'} ${job.seed}`,episodeSimSeconds:0});
    const reply=await this.rpc('evaluate',{job,previewHz:this.options.previewHz,dutyCycle:this.options.dutyCycle});
    await this.runnable(token);if(reply.result?.cancelled)throw abortError();const result=validateResult(reply.result,this.config);
    this.validateProvenance(result,job);
    if(result.reason==='simulation_error')throw new Error(result.metrics?.error||'Simulation failed; this episode was not scored');
    this.state.completedEpisodes++;this.state.simSeconds+=result.simSeconds;
    this.state.wallSeconds=this.wallBefore+(performance.now()-this.started)/1000;
    const row={episode:this.state.completedEpisodes,return:result.return,success:result.success,stage:job.stage,role,reason:result.reason,simSeconds:result.simSeconds};
    this.state.history.push(row);if(this.state.history.length>120)this.state.history.shift();
    this.emit({lastReturn:result.return,metrics:result.metrics,lastResult:row});this.persist();return result;
  }
  validateProvenance(result,job) {
    const expected={environmentVersion:this.config.environmentVersion,modelFingerprint:this.config.modelFingerprint,configHash:this.configHash,
      seed:job.seed,stage:job.stage,dtMs:this.config.dtMs,bodyBlockMs:this.config.bodyBlockMs,bodyBackend:'mujoco-wasm'};
    for(const [name,value] of Object.entries(expected))if(result[name]!==value||result.provenance?.[name]!==value)throw new Error(`Episode provenance mismatch: ${name}`);
    if(!['webgpu','wasm'].includes(result.backend)||result.provenance.backend!==result.backend||result.backend!==this.state.backend)throw new Error('Episode neural backend mismatch');
    this.validateExecution(result.provenance);
    validateParameters(result.parameters,this.config);
    if(result.parameters.some((v,i)=>v!==job.parameters[i]))throw new Error('Episode parameters differ from assigned job');
    for(const name of ['generation','pairId','sign','parametersHash'])if(job[name]!==undefined&&result.provenance[name]!==job[name])throw new Error(`Episode job mismatch: ${name}`);
    if(Math.abs(result.simSeconds-result.steps*this.config.bodyBlockMs/1000)>1e-8||result.simSeconds>job.durationSeconds+1e-8)throw new Error('Episode time does not match its physics steps');
    if(this.config.schemaVersion===2){
      const physicalFailure=result.terminated===true&&result.success===false&&['outside_habitat','excessive_rotation','overturned'].includes(result.reason);
      const fullOutcome=(result.reason==='stage_success'&&result.success===true&&result.terminated===true)||(result.reason==='time_limit'&&result.success===false&&result.terminated===false);
      if(result.steps<=0||result.simSeconds<=0||typeof result.terminated!=='boolean'||result.cancelled!==false||
        !(physicalFailure||(Math.abs(result.simSeconds-job.durationSeconds)<=1e-8&&fullOutcome)))throw new Error('This run requires a complete trial or physical failure');
    }
  }
  validateExecution(value){
    if(this.config.schemaVersion!==2)return;
    const expected=this.config.optimizer.acceptance.nativeExecution,actual=value?.wasmExecution;
    if(expected.backend!=='wasm'||value?.backend!=='wasm'||value?.neuralEngine!=='wasm'||Object.hasOwn(value,'nativeWebGPU')||
      !actual||Object.keys(actual).length!==2||actual.backend!=='wasm'||actual.moduleSha256!==expected.moduleSha256)
      throw new Error('Training execution does not match this run');
  }
  async validateCandidate(parameters,stage,{independent=false,token=this.runToken}={}) {
    const definition=this.config.stages.find(s=>s.id===stage),results=[];
    const seeds=independent?this.config.validation.testSeeds:this.config.validation.seeds;
    for(const seed of seeds)results.push(await this.evaluate({parameters,seed,stage,durationSeconds:definition.durationSeconds},independent?'test':'validation',token));
    return {stage,meanReturn:results.reduce((sum,r)=>sum+r.return,0)/results.length,successes:results.filter(r=>r.success).length,
      passed:results.every(r=>r.success),independent,seeds:seeds.slice(),results:results.map(r=>({return:r.return,success:r.success,reason:r.reason}))};
  }
  async runLocal(token) {
    if(this.sharedOnly)throw new Error('This training page always contributes to shared training');
    if(this.config.schemaVersion===2)throw new Error('This run requires assigned training jobs');
    while(this.running&&token===this.runToken){
      await this.runnable(token);const stage=this.state.stage;
      if(!this.savedValidation){this.savedValidation=await this.validateCandidate(this.parameters,stage,{token});this.emit({bestReturn:this.savedValidation.meanReturn,validation:this.savedValidation});}
      this.round??=makeGeneration(this.parameters,this.state.generation,this.config,stage);
      for(const pair of this.round.pairs)for(const job of pair.jobs){if(!pair.results[job.sign])pair.results[job.sign]=await this.evaluate(job,'exploration',token);}
      const candidate=updateGeneration(this.round,this.config),validation=await this.validateCandidate(candidate,stage,{token});
      const better=validation.meanReturn>this.savedValidation.meanReturn+this.config.validation.minimumImprovement;
      if(better){this.parameters=candidate;this.savedValidation=validation;this.testValidation=null;this.completedStages.clear();}
      this.state.generation++;this.round=null;
      const accepted=this.savedValidation;
      if(accepted.passed)this.completedStages.add(stage);
      this.emit({parameters:this.parameters.slice(),bestReturn:accepted.meanReturn,validation:accepted,checkpointStatus:accepted.passed?'locally-validated':'unverified',
        message:better?'Candidate improved on validation seeds':'Candidate did not improve; retaining the previous controller',
        curriculum:this.config.stages.map(s=>({...s,status:this.completedStages.has(s.id)?'validated':s.id===stage?'training':'available'}))});
      this.persist();
      if(accepted.passed){const next=this.config.stages[this.config.stages.findIndex(s=>s.id===stage)+1];
        if(next){this.emit({stage:next.id,message:`Validation goals passed; advancing to ${next.label}`,checkpointStatus:'unverified',bestReturn:null,validation:null});this.savedValidation=null;this.testValidation=null;}
        else {this.running=false;this.emit({phase:'ready',message:`${this.config.stages.find(s=>s.id===stage).label} passed local validation. Evaluate independent test seeds before export.`});}
      }
    }
  }
  coordinatorUrl(value) {
    const url=new URL(value);
    if(!['http:','https:'].includes(url.protocol)||url.username||url.password||url.search||url.hash)throw new Error('Use an HTTP(S) coordinator address without credentials or query parameters');
    if(url.protocol==='http:'&&!['localhost','127.0.0.1','[::1]'].includes(url.hostname))throw new Error('Remote coordinators require HTTPS');
    return url.href.replace(/\/$/,'').replace(/\/api\/training$/,'');
  }
  async api(path,body,url=this.state.coordinator.url) {
    if(!url)throw new Error('No coordinator selected');
    const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),15000);
    try{const response=await this.fetcher(`${url}/api/training${path}`,{method:body?'POST':'GET',headers:body?{'Content-Type':'application/json'}:undefined,
      body:body?JSON.stringify(body):undefined,signal:controller.signal,credentials:'omit',cache:'no-store'});
      const value=await response.json();if(!response.ok){const error=new Error(value.error||`Coordinator returned ${response.status}`);error.status=response.status;throw error;}return value;
    }finally{clearTimeout(timer);}
  }
  async connectCoordinator(value) {
    if(this.running)throw new Error('Stop training before changing coordinators');
    const url=this.coordinatorUrl(value||this.requiredCoordinatorUrl);
    if(this.sharedOnly&&url!==this.requiredCoordinatorUrl)throw new Error('This training page uses its fixed shared coordinator');
    const token=this.runToken,request=this.connectionRequest=(this.connectionRequest||0)+1;
    const statusRequest=this.statusRequest=(this.statusRequest||0)+1;
    if(!this.config)await this.initialize();let status;
    try{status=await this.api('/status',null,url);}
    catch(error){if(token===this.runToken&&request===this.connectionRequest&&statusRequest===this.statusRequest&&!this.running)this.emit({coordinator:{...this.state.coordinator,connected:false,url},syncError:{message:error.message,code:error.code}});throw error;}
    if(token!==this.runToken||request!==this.connectionRequest||this.running)throw abortError();
    if(statusRequest!==this.statusRequest)throw abortError();
    try{this.applyCoordinatorStatus(status,url,{message:'Ready'});}
    catch(error){this.emit({coordinator:{...this.state.coordinator,connected:false,url},syncError:{message:error.message,code:error.code}});throw error;}
    return status;
  }
  applyCoordinatorStatus(status,url,extra={}) {
    if(status.modelFingerprint!==this.config.modelFingerprint||status.configHash!==this.configHash)
      throw codedError('Coordinator uses a different build. Install the matching training site and model assets.','build_mismatch');
    let current;
    try{if(status.checkpoint)current=readCheckpoint(status.checkpoint,this.config,this.configHash);}
    catch(error){throw codedError(error.message,'build_mismatch');}
    if(this.sharedOnly&&!current)throw codedError('The shared coordinator did not provide its current checkpoint','build_mismatch');
    const previous=this.state.coordinator;
    // Responses can finish out of order while a trainer advances the run.
    if(previous.url===url&&((Number.isFinite(previous.generation)&&status.generation<previous.generation)||
      (Number.isFinite(previous.acceptedResults)&&status.acceptedResults<previous.acceptedResults)))return false;
    if(current)this.sharedCheckpoint=current;
    if(this.sharedOnly&&current)this.parameters=current.parameters.slice();
    this.emit({coordinator:{...status,connected:true,url},syncError:null,
      ...(this.sharedOnly&&current?{parameters:current.parameters.slice(),stage:current.stage,generation:current.generation,checkpointStatus:'shared-unverified'}:{}),...extra});
    return true;
  }
  async refreshCoordinatorStatus() {
    if(!this.config)await this.initialize();
    if(this.starting)throw abortError();
    const url=this.requiredCoordinatorUrl||this.state.coordinator.url;
    const token=this.runToken,request=this.statusRequest=(this.statusRequest||0)+1;
    try{
      const status=await this.api('/status?compact=1',null,url);
      if(token!==this.runToken||request!==this.statusRequest)throw abortError();
      this.applyCoordinatorStatus(status,url);return status;
    }catch(error){
      if(error.name!=='AbortError'&&token===this.runToken&&request===this.statusRequest)
        this.emit({coordinator:{...this.state.coordinator,connected:false,url},syncError:{message:error.message,code:error.code}});
      throw error;
    }
  }
  async disconnectCoordinator() {if(this.sharedOnly)throw new Error('Sharing is always enabled; use Pause or Stop to control compute');this.connectionRequest=(this.connectionRequest||0)+1;this.statusRequest=(this.statusRequest||0)+1;if(this.running&&this.options.mode==='shared')await this.stop();this.emit({coordinator:{connected:false},message:'Coordinator disconnected'});}
  async downloadSharedCheckpoint() {
    if(!this.config)await this.initialize();
    const url=this.requiredCoordinatorUrl||this.state.coordinator.url;
    const value=await this.api('/checkpoint',null,url);
    try{readCheckpoint(value,this.config,this.configHash);}catch(error){throw codedError(error.message,'build_mismatch');}
    return structuredClone(value);
  }
  leaseIdentity(job) {return {jobId:job.jobId,leaseToken:job.leaseToken,contributorId:this.contributorId,modelFingerprint:this.config.modelFingerprint,configHash:this.configHash};}
  hasPendingResult(job) {return !!this.outbox&&this.outbox.jobId===job.jobId&&this.outbox.leaseToken===job.leaseToken&&this.outboxUrl===job.coordinatorUrl;}
  async releaseLease(job) {try{await this.api('/release',this.leaseIdentity(job),job.coordinatorUrl);}catch{/* The server also expires disconnected leases. */}}
  async runShared(token) {
    if(!browserTrainingAvailable(this.config))throw codedError('This run is using the connected trainer.','native_trainer_required');
    const url=this.state.coordinator.url;
    if(this.outbox&&this.outboxUrl===this.state.coordinator.url){
      try{const accepted=await this.api('/result',this.outbox,url);await this.runnable(token);if(accepted.accepted!==true)throw new Error('Coordinator did not accept the saved result');this.outbox=null;this.state.contributedEpisodes++;this.persist();this.emit({message:'Uploaded'});}
      catch(error){if([400,404,409,410,422].includes(error.status)){this.outbox=null;this.persist();this.emit({message:'Starting a new run'});}else throw error;}
    }
    while(this.running&&token===this.runToken){
      await this.runnable(token);const response=await this.api('/lease',{contributorId:this.contributorId,modelFingerprint:this.config.modelFingerprint,configHash:this.configHash},url);
      const job=response.job===undefined?response:response.job;
      if(job)job.coordinatorUrl=url;
      if(!this.running||token!==this.runToken){if(job)await this.releaseLease(job);throw abortError();}
      if(job)this.activeLease=job;
      await this.runnable(token);
      if(!job){this.emit({message:'Waiting'});await delay(Math.min(response.waitMs||2000,10000));continue;}
      if(job.modelFingerprint!==this.config.modelFingerprint||job.configHash!==this.configHash)throw new Error('Incompatible shared job');
      validateParameters(job.parameters,this.config);if(!this.config.stages.some(s=>s.id===job.stage))throw new Error('Unknown shared task');
      const heartbeat=this.heartbeat=setInterval(()=>{if(this.running&&!this.paused)this.api('/heartbeat',this.leaseIdentity(job),url).catch(error=>this.emit({message:`Lease renewal: ${error.message}`}));},60000);
      try{
        const result=await this.evaluate(job,'contribution',token);
        const evidence={};
        for(const key of ['hasTakenOff','takeoffTime','flightSeconds','bestFlightSeconds','landingSeconds','landingTime','diagnostics','initialCondition','finalObservation','wallSeconds','setupWallSeconds','executionWallSeconds'])
          if(result.metrics?.[key]!==undefined)evidence[key]=result.metrics[key];
        const payload={...this.leaseIdentity(job),objective:result.return,metrics:{...evidence,success:result.success,terminated:result.terminated,cancelled:result.cancelled,simSeconds:result.simSeconds,steps:result.steps,reason:result.reason},
          provenance:{...result.provenance,parameters:result.parameters}};
        // Keep an unsent complete result locally; it is never silently counted as
        // contributed. Repeated accepted submissions are idempotent at the server.
        this.outbox=payload;this.outboxUrl=url;this.persist();let accepted;
        for(let attempt=0;attempt<3;attempt++){
          await this.runnable(token);try{accepted=await this.api('/result',payload,url);await this.runnable(token);if(accepted.accepted!==true)throw new Error('Coordinator did not accept this result');break;}catch(error){if(attempt===2||(error.status&&error.status<500)||error.name==='AbortError')throw error;this.emit({message:'Retrying upload'});await delay(1000*(attempt+1));}
        }
        this.outbox=null;this.state.contributedEpisodes++;this.activeLease=null;
        const statusRequest=this.statusRequest=(this.statusRequest||0)+1,status=await this.api('/status',null,url);
        await this.runnable(token);
        if(statusRequest===this.statusRequest){
          this.applyCoordinatorStatus(status,url,{generation:status.generation,checkpointStatus:'shared-unverified',message:'Uploaded'});
          if(this.sharedCheckpoint)this.emit({parameters:this.sharedCheckpoint.parameters.slice()});
        }
        this.persist();
      }finally{clearInterval(heartbeat);if(this.heartbeat===heartbeat)this.heartbeat=null;}
    }
  }
  exportCheckpoint() {
    if(!this.config)throw new Error('Training settings are not loaded');
    if(this.sharedOnly&&!this.sharedCheckpoint)throw new Error('Download the current checkpoint from the shared coordinator');
    if(this.options.mode==='shared'&&this.sharedCheckpoint)return structuredClone(this.sharedCheckpoint);
    return checkpoint(this.config,this.configHash,this.parameters,this.state.generation,this.state.stage,{status:this.state.checkpointStatus,validation:this.savedValidation,testValidation:this.testValidation||null,completedStages:[...this.completedStages]});
  }
  importCheckpoint(value) {
    if(this.sharedOnly)throw new Error('Shared training uses the coordinator checkpoint; local checkpoint imports are disabled');
    if(this.running||this.starting)throw new Error('Stop training before importing a checkpoint');
    const c=readCheckpoint(value,this.config,this.configHash);this.parameters=c.parameters;this.round=null;this.savedValidation=null;this.testValidation=null;this.completedStages.clear();this.sharedCheckpoint=null;
    this.options.mode='local';this.localView=null;this.emit({parameters:c.parameters,generation:c.generation,stage:c.stage,checkpointStatus:'unverified',bestReturn:null,validation:null,curriculum:this.config.stages.map(s=>({...s,status:'available'})),message:'Checkpoint imported; independent evaluation is required'});this.persist();
  }
  async evaluateCheckpoint() {
    if(this.sharedOnly)throw new Error('This training page only runs shared coordinator jobs');
    if(this.running||this.starting)throw new Error('Stop training before independent evaluation');
    const token=++this.runToken;this.running=true;this.paused=false;this.started=performance.now();this.wallBefore=this.state.wallSeconds;
    try{await this.ensureWorker();await this.runnable(token);this.emit({phase:'training',message:'Evaluating checkpoint on independent test seeds'});
      const c=this.options.mode==='shared'&&this.sharedCheckpoint?this.sharedCheckpoint:this.exportCheckpoint();
      const validation=await this.validateCandidate(c.parameters,c.stage,{independent:true,token});this.parameters=c.parameters.slice();this.state.stage=c.stage;this.savedValidation=null;
      this.state.generation=c.generation;this.options.mode='local';this.localView=null;this.sharedCheckpoint=null;this.round=null;this.completedStages.clear();
      this.testValidation=validation;this.emit({phase:'ready',validation,parameters:this.parameters.slice(),curriculum:this.config.stages.map(s=>({...s,status:'available'})),checkpointStatus:validation.passed?'locally-tested':'unverified',bestReturn:validation.meanReturn,message:validation.passed?'Independent test goals passed locally':'Independent test goals not yet achieved'});this.persist();return validation;
    }catch(error){if(token===this.runToken&&error.name!=='AbortError')this.fail(error);throw error;}finally{if(token===this.runToken)this.running=false;}
  }
  persist() {
    if(!this.config||!this.key)return;
    const local=this.options.mode==='shared'&&this.localView?this.localView:this.state;
    try{this.storage?.setItem(this.key,JSON.stringify({checkpoint:checkpoint(this.config,this.configHash,this.parameters,local.generation,local.stage,{status:'unverified'}),
      completedEpisodes:this.state.completedEpisodes,contributedEpisodes:this.state.contributedEpisodes,history:this.state.history,contributorId:this.contributorId,outbox:this.outbox||null,outboxUrl:this.outboxUrl||null}));}
    catch{this.emit({storageWarning:'Local storage is unavailable. Export a checkpoint before closing this page.'});}
  }
  localViewSnapshot() {return structuredClone(Object.fromEntries(['stage','generation','parameters','checkpointStatus','bestReturn','validation','curriculum'].map(key=>[key,this.state[key]])));}
  fail(error) {this.running=false;this.paused=false;this.worker?.postMessage({type:'cancel'});if(this.heartbeat)clearInterval(this.heartbeat);this.heartbeat=null;
    if(this.activeLease){const lease=this.activeLease;this.activeLease=null;if(!this.hasPendingResult(lease))this.releaseLease(lease);}
    this.persist();this.emit({phase:'error',message:error.message,error:error.message});}
  async dispose() {await this.stop();globalThis.document?.removeEventListener('visibilitychange',this.visibility);}
}
