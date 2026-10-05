// Diagnostic sensor-motion positive control only. This is not autonomous
// motor recruitment, a physical haltere actuator, or a proposed live coupling.
const MAX_AMPLITUDE=Math.PI/4;
const ONSET_TOLERANCE_SECONDS=1e-9;

export function prescribeLeftHaltereSensorMotion({body,fly}={}, {onsetSeconds,leftAmplitudeRadians}={}){
 if(!Number.isFinite(onsetSeconds)||onsetSeconds<0)throw new RangeError('onsetSeconds must be finite and nonnegative');
 if(!Number.isFinite(leftAmplitudeRadians)||leftAmplitudeRadians<0||leftAmplitudeRadians>MAX_AMPLITUDE)
  throw new RangeError('leftAmplitudeRadians must be finite within [0,pi/4]');
 if(!body||!Number.isFinite(body.time))throw new TypeError('A finite body.time is required');
 const nativePower=body.halterePower;
 if((!Array.isArray(nativePower)&&!ArrayBuffer.isView(nativePower))||nativePower.length!==2)
  throw new TypeError('body.halterePower must contain two native muscle powers');
 const actualMusclePower=Array.from(nativePower);
 if(actualMusclePower.some(value=>!Number.isFinite(value)||value<0||value>1))
  throw new RangeError('Native haltere muscle powers must be finite within [0,1]');
 if(!fly||!fly.feedback||typeof fly.feedback!=='object'||Array.isArray(fly.feedback))
  throw new TypeError('Existing fly.feedback is required after body/world copy');
 // Native time accumulation can lie just below an exact block boundary.
 // Use the declared 1 ns clock tolerance, not a whole-block onset delay.
 const active=body.time>=onsetSeconds-ONSET_TOLERANCE_SECONDS;
 const transducerAmplitudeEquivalentPower=active?
  [leftAmplitudeRadians/MAX_AMPLITUDE,actualMusclePower[1]]:actualMusclePower.slice();
 const diagnostic={kind:'externally-prescribed-left-virtual-haltere-motion',active,
  actualMusclePower,transducerAmplitudeEquivalentPower,
  amplitudeRadians:leftAmplitudeRadians,onsetSeconds,bodyTimeSeconds:body.time};
 // All validation precedes the sole mutation. Preserve every other feedback
 // field and break the native power-array alias only when the control is on.
 if(active)fly.feedback={...fly.feedback,halterePower:transducerAmplitudeEquivalentPower.slice()};
 return diagnostic;
}
