// BANC's abstract sign is +1 for flexors and -1 for extensors. The native
// FlyBody femur and tibia coordinates increase their physical internal angle
// (extension) on both sides. Keep source annotations intact and convert only
// at the native actuator boundary. Other joints retain the existing mapping
// until their anatomical motion has an independently verified conversion.
export function nativeLegDirection(mapping){
  const extensionCoordinate=/^(femur|tibia)_T[123]_(left|right)$/.test(mapping.joint);
  return mapping.sign*(extensionCoordinate?-1:1);
}
