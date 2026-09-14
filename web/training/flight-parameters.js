// A fixed, auditable calibration contract. Bilateral homologues share learned
// coefficients; their incoming muscle-force signals remain independent.
export const STEERING_MUSCLE_TYPES = Object.freeze([
  'b1_muscle', 'b2_muscle', 'b3_muscle', 'i1_muscle', 'i2_muscle',
  'iii1_muscle', 'iii3_muscle', 'iii4_muscle',
  'iv1_muscle', 'iv2_muscle', 'iv3_muscle', 'iv4_muscle',
]);
export const FLIGHT_PARAMETER_NAMES = Object.freeze([
  'flight_power_log_gain', 'flight_deployment_tau_log_scale', 'flight_frequency_log_scale',
  ...STEERING_MUSCLE_TYPES.flatMap(name => [`flight_${name}_bias_log_gain`, `flight_${name}_amplitude_log_gain`]),
]);
export const DEFAULT_FLIGHT_INTERPRETER = Object.freeze({
  powerGain: 1, deploymentTauScale: 1, frequencyScale: 1,
  steering: Object.freeze(Object.fromEntries(STEERING_MUSCLE_TYPES.map(name =>
    [name, Object.freeze({biasGain: 1, amplitudeGain: 1})]))),
});
const scalarNames = Object.freeze(['powerGain', 'deploymentTauScale', 'frequencyScale']);
const record = value => value !== null && typeof value === 'object' && !Array.isArray(value) && !ArrayBuffer.isView(value);
function positive(value, name) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0)
    throw new TypeError('Invalid flight interpreter parameter: ' + name);
  return value;
}

/** Own and validate all coefficients before changing a live interpreter.
 * Top-level scales can be patched. If steering is supplied, all twelve types
 * and both gains must be explicit; silently defaulting a missing learned value
 * would reinterpret a malformed checkpoint. No free global steering gains exist.
 */
export function validateFlightInterpreter(values = {}, previous = DEFAULT_FLIGHT_INTERPRETER) {
  if (!record(values)) throw new TypeError('Flight interpreter parameters must be an object');
  for (const name of Object.keys(values)) if (name !== 'steering' && !scalarNames.includes(name))
    throw new TypeError('Unknown flight interpreter parameter: ' + name);
  const next = Object.fromEntries(scalarNames.map(name =>
    [name, positive(Object.hasOwn(values, name) ? values[name] : previous[name], name)]));
  const source = Object.hasOwn(values, 'steering') ? values.steering : previous.steering;
  if (!record(source)) throw new TypeError('Missing flight steering coefficient map');
  for (const name of Object.keys(source)) if (!STEERING_MUSCLE_TYPES.includes(name))
    throw new TypeError('Unknown flight steering muscle: ' + name);
  const steering = {};
  for (const name of STEERING_MUSCLE_TYPES) {
    if (!Object.hasOwn(source, name) || !record(source[name])) throw new TypeError('Missing flight steering muscle: ' + name);
    const gains = source[name];
    for (const key of Object.keys(gains)) if (key !== 'biasGain' && key !== 'amplitudeGain')
      throw new TypeError('Unknown flight steering coefficient: ' + name + '.' + key);
    if (!Object.hasOwn(gains, 'biasGain') || !Object.hasOwn(gains, 'amplitudeGain'))
      throw new TypeError('Missing flight steering coefficient: ' + name);
    steering[name] = Object.freeze({biasGain: positive(gains.biasGain, name + '.biasGain'),
      amplitudeGain: positive(gains.amplitudeGain, name + '.amplitudeGain')});
  }
  return Object.freeze({...next, steering: Object.freeze(steering)});
}

/** Map the complete ordered 27-log-parameter vector to physical coefficients.
 * Training configuration bounds are checked by the caller; this helper also
 * rejects nonfinite values and exponential overflow/underflow independently.
 */
export function flightParametersToInterpreter(parameters) {
  if ((!Array.isArray(parameters) && !ArrayBuffer.isView(parameters)) || parameters.length !== FLIGHT_PARAMETER_NAMES.length)
    throw new TypeError('Flight interpreter requires exactly 27 log parameters');
  const values = Array.from(parameters, (value, index) => {
    if (typeof value !== 'number' || !Number.isFinite(value))
      throw new TypeError('Invalid log parameter: ' + FLIGHT_PARAMETER_NAMES[index]);
    return positive(Math.exp(value), FLIGHT_PARAMETER_NAMES[index]);
  });
  return validateFlightInterpreter({powerGain: values[0], deploymentTauScale: values[1], frequencyScale: values[2],
    steering: Object.fromEntries(STEERING_MUSCLE_TYPES.map((name, index) =>
      [name, {biasGain: values[3 + 2 * index], amplitudeGain: values[4 + 2 * index]}]))});
}
