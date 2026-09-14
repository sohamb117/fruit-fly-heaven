// Explicit locomotion curriculum, not a biological or learned parameter.
// Version 1 fixes finite geometry so a typo cannot quietly relax task bounds.
export const SPACIOUS_MAINTAINED_SCENE=Object.freeze({schemaVersion:1,profile:'spacious-maintained-flight-v1',radiusCm:50,ceilingCm:50,floorCapRadiusCm:6.5});
export function validateMaintainedScene(value){
 if(value===undefined)return null;
 const keys=Object.keys(SPACIOUS_MAINTAINED_SCENE);
 if(value===null||typeof value!=='object'||(Object.getPrototypeOf(value)!==Object.prototype&&Object.getPrototypeOf(value)!==null)||
  Reflect.ownKeys(value).length!==keys.length||keys.some(k=>!Object.hasOwn(value,k)||value[k]!==SPACIOUS_MAINTAINED_SCENE[k]))
  throw new TypeError('Unsupported maintained scene: require the complete finite spacious-maintained-flight-v1 profile');
 return Object.freeze({...SPACIOUS_MAINTAINED_SCENE});
}
export function matchMaintainedScene(configValue,metadataValue){
 const config=validateMaintainedScene(configValue),metadata=validateMaintainedScene(metadataValue);
 if(!!config!==!!metadata)throw new Error('Maintained scene config and body metadata must agree');
 return config;
}
export function maintainedFloorHeightScene(x,z,scene){
 // Scene coordinates are ten units per native centimetre; preserve the old
 // bowl arithmetic exactly inside the declared cap radius.
 const square=x*x+z*z,cap=scene.floorCapRadiusCm*10;
 return 1.5+.0037*Math.min(square,cap*cap);
}
export function assertMaintainedObservationScene(value,scene){
 if(!scene){if(value.curriculumScene!==undefined)throw new TypeError('Curriculum observation requires explicit scorer scene');return;}
 const observed=validateMaintainedScene(value.curriculumScene);
 if(!observed||value.ceiling!==scene.ceilingCm)throw new TypeError('Observation bounds differ from the configured curriculum');
}
