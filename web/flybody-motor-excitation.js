// Explicit, opt-in recruitment priors for the reduced motor interface.
// These constants are model assumptions, not measured receptor physiology or
// learned parameters. Power muscles and every other muscle kind stay legacy.
export const STEERING_RECRUITMENT_BOUNDS = Object.freeze({
  halfActivationHz: Object.freeze([1, 1000]),
  exponent: Object.freeze([0.25, 4]),
});

const legacy = rateHz => Math.max(0, Math.min(1, rateHz / 80));
const legacyDecoder = Object.freeze({steering: null, fromRate: (_kind, rateHz) => legacy(rateHz)});

function exactRecord(value, keys, label) {
  if (value === null || typeof value !== 'object' ||
      (Object.getPrototypeOf(value) !== Object.prototype && Object.getPrototypeOf(value) !== null))
    throw new TypeError(label + ' must be an object');
  for (const key of Reflect.ownKeys(value)) if (!keys.includes(key))
    throw new TypeError('Unknown ' + label + ' option: ' + String(key));
  for (const key of keys) if (!Object.hasOwn(value, key))
    throw new TypeError('Missing ' + label + ' option: ' + key);
}

function bounded(value, key) {
  const [minimum, maximum] = STEERING_RECRUITMENT_BOUNDS[key];
  if (typeof value !== 'number' || !Number.isFinite(value) || value < minimum || value > maximum)
    throw new RangeError(`Steering recruitment ${key} must be finite and within [${minimum}, ${maximum}]`);
  return value;
}

/** Validate once, then own an immutable copy of the model's optional prior.
 * Absent metadata.motor_excitation preserves the original clamp byte for byte.
 * The Hill curve is mathematically strictly increasing for positive rates;
 * finite-precision values can coincide very near its zero/one asymptotes.
 */
export function createMotorExcitation(config) {
  if (config === undefined) return legacyDecoder;
  exactRecord(config, ['steering'], 'motor excitation');
  exactRecord(config.steering, ['kind', 'halfActivationHz', 'exponent'], 'steering recruitment');
  if (config.steering.kind !== 'hill') throw new TypeError('Unknown steering recruitment kind');
  const halfActivationHz = bounded(config.steering.halfActivationHz, 'halfActivationHz');
  const exponent = bounded(config.steering.exponent, 'exponent');
  const steering = Object.freeze({kind: 'hill', halfActivationHz, exponent});
  return Object.freeze({steering, fromRate(kind, rateHz) {
    if (kind !== 'wing_steering_assumption') return legacy(rateHz);
    if (!Number.isFinite(rateHz)) throw new TypeError('Steering recruitment requires a finite rate');
    if (rateHz <= 0) return 0;
    // Raise only a ratio <= 1, avoiding overflow at large rates/exponents.
    if (rateHz <= halfActivationHz) {
      const fraction = (rateHz / halfActivationHz) ** exponent;
      return fraction / (1 + fraction);
    }
    return 1 / (1 + (halfActivationHz / rateHz) ** exponent);
  }});
}
