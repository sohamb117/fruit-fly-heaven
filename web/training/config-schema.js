// The guarded coordinator owns proposal evaluation and checkpoint retention.
// Version 2 prevents an older coordinator from silently skipping that phase.
const record = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const keys = (value, expected) => record(value) && Object.keys(value).length === expected.length && expected.every(key => Object.hasOwn(value, key));
const digest = value => typeof value === 'string' && /^[0-9a-f]{64}$/.test(value);

export function validateTrainingConfigSchema(config) {
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
