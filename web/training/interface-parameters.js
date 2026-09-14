import {MOTOR_INTERFACE_VERSION,MOTOR_INTERFACE_PARAMETERS,validateMotorInterface} from '../motor-interface.js';

// v2 uses named physical/interface values, never aliases v1's logarithmic
// neural gains. Which values are identifiable/trainable is experiment-specific.
export function interfaceParameterValues(config,values){
  if(config.parameterContract!==MOTOR_INTERFACE_VERSION||config.freezeNeuralParameters!==true)throw new Error('Interface experiments must freeze neural parameters');
  if(config.environmentVersion!=='banc-flybody-interface-v2'||config.behaviorCriteriaVersion!==2)throw new Error('Interface experiments require the v2 environment and behavior criteria');
  if(!Array.isArray(values)&&!(values instanceof Float32Array)&&!(values instanceof Float64Array))throw new Error('Invalid interface parameter vector');
  const known=new Map(MOTOR_INTERFACE_PARAMETERS.map(p=>[p.name,p])),names=new Set();
  if(!Array.isArray(config.parameters)||!config.parameters.length||values.length!==config.parameters.length)throw new Error('Interface parameter count mismatch');
  const profile={...validateMotorInterface(config.referenceInterface||{})};
  for(let i=0;i<config.parameters.length;i++){
    const p=config.parameters[i],limit=known.get(p.name),value=values[i];
    if(!limit||names.has(p.name)||![p.min,p.max,p.initial,value].every(Number.isFinite)||p.min<limit.min||p.max>limit.max||p.min>=p.max||p.initial<p.min||p.initial>p.max||value<p.min||value>p.max)throw new Error('Invalid interface parameter: '+p.name);
    names.add(p.name);profile[p.name]=value;
  }
  return {vector:Array.from(values),profile:validateMotorInterface(profile),neuralParametersFrozen:true};
}

// Reports cannot confer trust on themselves. This capability lives only in an
// operator process, is not serializable, and is not available to contributors.
const operatorEvidence=new WeakMap();
export function createOperatorPromotionGate(){
  const authority=Object.freeze({}),trusted=new WeakSet();operatorEvidence.set(authority,trusted);
  const freeze=(value,seen=new WeakSet())=>{
    if(value&&typeof value==='object'&&!seen.has(value)){seen.add(value);for(const child of Object.values(value))freeze(child,seen);Object.freeze(value);}
    return value;
  };
  return Object.freeze({
    // Call only for observations made by this operator's actual evaluation or
    // independently authenticated calibration process. Do not call this on
    // anonymous uploads or because JSON contains trustedExecution: true.
    recordLocalEvidence(evidence){
      if(!evidence||typeof evidence!=='object'||Array.isArray(evidence))throw new Error('Invalid operator evidence');
      const snapshot=freeze(structuredClone(evidence));trusted.add(snapshot);return snapshot;
    },
    evaluate:contract=>evaluatePromotionGate(contract,authority),
  });
}

export function evaluatePromotionGate({candidate,incumbent,priorStages=[],requiredSeeds,trainingSeeds,
  requiredPriorStages,stage,minimumImprovement=0,mechanicalGate,mechanicalProtocolHash}={},authority){
  const denied=reason=>({passed:false,reason,biologicalSuccessValidated:false});
  // A failed/missing mechanical report must be reportable without creating a
  // trusted pass or even evaluating the rest of the promotion contract.
  if(mechanicalGate?.passed!==true)return denied('mechanical_calibration_required');
  const trusted=operatorEvidence.get(authority);
  if(!trusted?.has(mechanicalGate))return denied('untrusted_mechanical_evidence');
  const hash=value=>typeof value==='string'&&/^[a-f0-9]{64}$/.test(value);
  const label=value=>typeof value==='string'&&value.length>0&&value.length<=128&&value.trim()===value;
  const seeds=values=>Array.isArray(values)&&values.length>0&&values.every(s=>Number.isInteger(s)&&s>=0&&s<=0xffffffff)&&new Set(values).size===values.length;
  if(!Number.isFinite(minimumImprovement)||minimumImprovement<0||!label(stage)||
    !seeds(requiredSeeds)||!seeds(trainingSeeds)||requiredSeeds.some(s=>trainingSeeds.includes(s))||
    !Array.isArray(requiredPriorStages)||!requiredPriorStages.every(label)||new Set(requiredPriorStages).size!==requiredPriorStages.length||requiredPriorStages.includes(stage)||
    !Array.isArray(priorStages)||!hash(mechanicalProtocolHash))return denied('invalid_validation_contract');
  const identity=run=>!!run&&label(run.executionId)&&hash(run.configHash)&&hash(run.modelFingerprint)&&hash(run.interfaceHash);
  if(!identity(candidate)||!identity(incumbent)||candidate.configHash!==incumbent.configHash||candidate.modelFingerprint!==incumbent.modelFingerprint||
    candidate.stage!==stage||incumbent.stage!==stage)return denied('validation_identity_mismatch');
  if(!identity(mechanicalGate)||mechanicalGate.configHash!==candidate.configHash||mechanicalGate.modelFingerprint!==candidate.modelFingerprint||
    mechanicalGate.interfaceHash!==candidate.interfaceHash||mechanicalGate.protocolHash!==mechanicalProtocolHash)return denied('mechanical_identity_mismatch');
  const executionIds=new Set();
  for(const run of [mechanicalGate,candidate,incumbent,...priorStages]){
    if(!trusted.has(run))return denied('untrusted_execution_evidence');
    if(!identity(run)||executionIds.has(run.executionId))return denied('validation_identity_mismatch');
    executionIds.add(run.executionId);
  }
  const validate=(run,requireSuccess)=>{
    if(run.split!=='validation'||!Array.isArray(run.results)||run.results.length!==requiredSeeds.length)return false;
    const seen=new Set();
    return run.results.every(r=>{
      if(!r||!requiredSeeds.includes(r.seed)||seen.has(r.seed)||!Number.isFinite(r.return)||r.return< -10||r.return>10||
        typeof r.success!=='boolean'||(r.cancelled!==undefined&&r.cancelled!==false)||r.error||(requireSuccess&&r.success!==true))return false;
      seen.add(r.seed);return true;
    });
  };
  if(!validate(candidate,true)||!validate(incumbent,false))return denied('matched_held_out_evaluation_required');
  const seenStages=new Set();
  if(priorStages.length!==requiredPriorStages.length||priorStages.some(run=>{
    if(!requiredPriorStages.includes(run.stage)||seenStages.has(run.stage)||run.configHash!==candidate.configHash||
      run.modelFingerprint!==candidate.modelFingerprint||run.interfaceHash!==candidate.interfaceHash||!validate(run,true))return true;
    seenStages.add(run.stage);return false;
  }))return denied('prior_stage_regression');
  const mean=run=>run.results.reduce((sum,r)=>sum+r.return,0)/run.results.length;
  const improvement=mean(candidate)-mean(incumbent);
  if(!(improvement>0)||improvement<minimumImprovement)return {...denied('candidate_did_not_improve'),improvement};
  return {passed:true,reason:'operator_validation_passed',improvement,biologicalSuccessValidated:false};
}
