// Reward-based, antithetic policy search over bounded BANC calibration parameters.
// This module never supplies motor actions or changes the physical environment.
export function validateConfig(config) {
  if (config?.schemaVersion !== 1 || config.algorithm !== 'antithetic-evolution-strategies') throw new Error('Unsupported training configuration');
  if (!Array.isArray(config.parameters) || !config.parameters.length || config.parameters.length > 256) throw new Error('Invalid parameter schema');
  const names = new Set();
  for (const p of config.parameters) {
    if (typeof p.name !== 'string' || names.has(p.name) || ![p.min,p.max,p.initial].every(Number.isFinite) || p.min >= p.max || p.initial < p.min || p.initial > p.max) throw new Error('Invalid parameter bounds');
    names.add(p.name);
  }
  const o = config.optimizer;
  if (!o || !Number.isInteger(o.populationPairs) || o.populationPairs < 1 || o.populationPairs > 128 || ![o.sigma,o.learningRate,o.maximumUpdate].every(x => Number.isFinite(x) && x > 0)) throw new Error('Invalid optimizer');
  if (!Array.isArray(config.stages) || !config.stages.length || !config.stages.some(x => x.id === config.stage)) throw new Error('Invalid curriculum');
  if (!Number.isFinite(config.objective?.min) || !Number.isFinite(config.objective?.max) || config.objective.min >= config.objective.max) throw new Error('Invalid objective bounds');
  return config;
}

export function validateParameters(parameters, config) {
  if (!Array.isArray(parameters) || parameters.length !== config.parameters.length || parameters.some((x,i) => !Number.isFinite(x) || x < config.parameters[i].min || x > config.parameters[i].max)) throw new Error('Checkpoint parameters do not match this model');
  return parameters.slice();
}

export function validateResult(result, config) {
  if (!result || !Number.isFinite(result.return) || result.return < config.objective.min || result.return > config.objective.max || typeof result.success !== 'boolean' || !Number.isFinite(result.simSeconds) || result.simSeconds < 0 || !Number.isInteger(result.steps) || result.steps < 0) throw new Error('Invalid episode result');
  return result;
}

export function random(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ state >>> 15, 1 | state);
    t ^= t + Math.imul(t ^ t >>> 7, 61 | t);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
const clamp = (x,a,b) => Math.max(a,Math.min(b,x));
export function makeGeneration(parameters, generation, config, stage = config.stage) {
  validateParameters(parameters,config);
  if (!Number.isInteger(generation) || generation < 0) throw new Error('Invalid generation');
  const task = config.stages.find(x => x.id === stage);
  if (!task) throw new Error('Unknown training stage');
  const rng = random((config.optimizer.seed + Math.imul(generation + 1, 2654435761)) >>> 0), pairs = [];
  for (let i=0; i<config.optimizer.populationPairs; i++) {
    const noise = parameters.map(() => Math.sqrt(-2*Math.log(Math.max(rng(),1e-12)))*Math.cos(2*Math.PI*rng()));
    const seed = (config.optimizer.seed + generation*1009 + i*65537) >>> 0;
    const jobs = [-1,1].map(sign => ({id:`local-${stage}-${generation}-${i}-${sign}`,pairId:i,sign,seed,stage,durationSeconds:task.durationSeconds,
      parameters:parameters.map((x,k) => clamp(x+sign*config.optimizer.sigma*noise[k],config.parameters[k].min,config.parameters[k].max))}));
    pairs.push({noise,jobs,results:{}});
  }
  return {generation,stage,baseline:parameters.slice(),pairs};
}

export function updateGeneration(round, config) {
  validateParameters(round.baseline,config);
  if (round.pairs.length !== config.optimizer.populationPairs) throw new Error('Incomplete generation');
  const delta = new Float64Array(round.baseline.length);
  for (const pair of round.pairs) {
    const minus=validateResult(pair.results[-1],config),plus=validateResult(pair.results[1],config);
    if (!Array.isArray(pair.noise) || pair.noise.length !== delta.length || pair.noise.some(x=>!Number.isFinite(x))) throw new Error('Invalid exploration noise');
    for (let k=0;k<delta.length;k++) delta[k] += (plus.return-minus.return)*pair.noise[k];
  }
  const scale=config.optimizer.learningRate/(2*round.pairs.length*config.optimizer.sigma);
  return round.baseline.map((x,k)=>clamp(x+clamp(scale*delta[k],-config.optimizer.maximumUpdate,config.optimizer.maximumUpdate),config.parameters[k].min,config.parameters[k].max));
}

export function readCheckpoint(value, config, configHash) {
  if (!value || value.schemaVersion !== 1 || value.algorithm !== config.algorithm || value.modelFingerprint !== config.modelFingerprint || value.configHash !== configHash) throw new Error('Checkpoint belongs to a different training model or configuration');
  if (!Number.isInteger(value.generation) || value.generation < 0 || !config.stages.some(s=>s.id===value.stage)) throw new Error('Invalid checkpoint progress');
  if (value.parameterNames?.join('\n') !== config.parameters.map(p=>p.name).join('\n')) throw new Error('Checkpoint parameter names differ');
  const parameters=validateParameters(value.parameters,config);
  // Imported scores and success claims are untrusted. Evaluate locally before promotion.
  return {schemaVersion:1,algorithm:config.algorithm,modelFingerprint:config.modelFingerprint,configHash,parameterNames:config.parameters.map(p=>p.name),parameters,
    generation:value.generation,stage:value.stage,status:'unverified',createdAt:new Date().toISOString()};
}

export function checkpoint(config, configHash, parameters, generation, stage, extra={}) {
  return {...extra,schemaVersion:1,algorithm:config.algorithm,modelFingerprint:config.modelFingerprint,configHash,parameterNames:config.parameters.map(p=>p.name),
    parameters:validateParameters(parameters,config),generation,stage,createdAt:new Date().toISOString()};
}
