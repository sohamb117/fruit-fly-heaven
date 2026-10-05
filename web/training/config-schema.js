// The guarded coordinator owns proposal evaluation and checkpoint retention.
// Version 2 prevents an older coordinator from silently skipping that phase.
const record = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const keys = (value, expected) => record(value) && Object.keys(value).length === expected.length && expected.every(key => Object.hasOwn(value, key));
const digest = value => typeof value === 'string' && /^[0-9a-f]{64}$/.test(value);

/** The flag describes delivered sensory input, independently of observer images.
 * Legacy vision:false requires no retinal declaration. Experimental vision must
 * opt into the supported, simulation-clocked T4/T5 profile explicitly. */
export function validateTrainingVisionConfig(config) {
  const visual = config?.visionFeedback;
  if (config?.vision === false && visual === undefined) return null;
  if (config?.vision !== true || !record(visual) || visual.schema !== 1 ||
      visual.profile !== 'compact-retinal-motion-v1' ||
      Object.keys(visual).some(key => !['schema', 'profile', 'width', 'height', 'frameIntervalMs', 'camera', 'motion'].includes(key)) ||
      !Number.isInteger(visual.width) || visual.width < 32 || visual.width > 1024 ||
      !Number.isInteger(visual.height) || visual.height < 16 || visual.height > 1024 ||
      !Number.isFinite(config.bodyBlockMs) || config.bodyBlockMs <= 0 ||
      !Number.isFinite(visual.frameIntervalMs) || visual.frameIntervalMs < config.bodyBlockMs || visual.frameIntervalMs > 100 ||
      Math.abs(visual.frameIntervalMs / config.bodyBlockMs - Math.round(visual.frameIntervalMs / config.bodyBlockMs)) > 1e-9 ||
      (visual.camera !== undefined && !record(visual.camera)) || (visual.motion !== undefined && !record(visual.motion)))
    throw new Error('Invalid training retinal feedback profile or vision flag');
  return visual;
}

export function validateTrainingConfigSchema(config) {
  // Generic optimizer fixtures may omit sensory fields. Actual execution also
  // calls the strict validator and always requires an explicit boolean flag.
  if (record(config) && (Object.hasOwn(config, 'vision') || Object.hasOwn(config, 'visionFeedback')))
    validateTrainingVisionConfig(config);
  if (config?.schemaVersion === 1) {
    if (config.optimizer && Object.hasOwn(config.optimizer, 'acceptance'))
      throw new Error('Checkpoint acceptance requires training configuration version 2');
    return config;
  }
  if (config?.schemaVersion !== 2) throw new Error('Unsupported training configuration');
  const acceptance = config.optimizer?.acceptance;
  if (!keys(acceptance, ['profile', 'proposal', 'seedCount', 'nativeExecution']) ||
      acceptance.profile !== 1 || acceptance.proposal !== 'best-search-job' || acceptance.seedCount !== 3)
    throw new Error('Invalid checkpoint acceptance configuration');
  const execution = acceptance.nativeExecution;
  const dawn = keys(execution, ['backend', 'moduleSha256', 'packageLockSha256']) && execution.backend === 'dawn-metal' &&
    digest(execution.moduleSha256) && digest(execution.packageLockSha256);
  const wasm = keys(execution, ['backend', 'moduleSha256']) && execution.backend === 'wasm' &&
    digest(execution.moduleSha256) && config.assets?.['/banc-engine/dist/core.wasm'] === execution.moduleSha256;
  if (!dawn && !wasm)
    throw new Error('Invalid checkpoint acceptance execution pin');
  return config;
}
