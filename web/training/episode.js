// Pure episode/parameter rules. They observe outcomes and never issue actions.
export {FLIGHT_CRITERIA as CRITERIA,createFlightScore as createEpisodeScore} from './flight-objective.js';
import {FLIGHT_PARAMETER_NAMES,flightParametersToInterpreter,DEFAULT_FLIGHT_INTERPRETER} from './flight-parameters.js';
import {validateTrainingConfigSchema,validateTrainingVisionConfig} from './config-schema.js';
import {MOTOR_DECODER_VERSION,buildMotorDecoderContract,validateMotorDecoderContract,validateMotorDecoderVector} from '../motor-decoder.js';
import {SENSORIMOTOR_CONTRACT,applySensorimotorParameters} from './sensorimotor-parameters.js';
export const PARAMETER_NAMES=FLIGHT_PARAMETER_NAMES;
export const STAGES=['takeoff','flight','landing','maintained_flight','recovery'];
export const clip=(v,a,b)=>Math.max(a,Math.min(b,v));
export function seededRandom(seed){let value=seed>>>0;return()=>{value+=0x6D2B79F5;let t=value;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return((t^(t>>>14))>>>0)/4294967296;};}
/** Validate the explicit decoder declaration without reinterpreting old vectors.
 * A loaded io adds the authoritative identity check; the serialized contract
 * alone is sufficient for coordinator/client schema and bounds validation. */
export function validateMotorDecoderTrainingConfig(config,io){
 if(config.parameterContract!==MOTOR_DECODER_VERSION){
  if(config.motorDecoderContract!==undefined)throw new Error('Unknown training motor decoder contract');
  return null;
 }
 if(config.freezeNeuralParameters!==true)throw new Error('Motor decoder training requires frozen neural parameters');
 if(!config.wingEventExcitation||typeof config.wingEventExcitation!=='object'||Array.isArray(config.wingEventExcitation))throw new Error('Motor decoder training requires wingEventExcitation');
 const contract=validateMotorDecoderContract(config.motorDecoderContract);
 if(!Array.isArray(config.parameters)||config.parameters.length!==contract.parameters.length||!config.parameters.every((p,i)=>{
  const expected=contract.parameters[i];return p&&p.name===expected.name&&p.min===expected.min&&p.max===expected.max&&
   Number.isFinite(p.initial)&&p.initial>=p.min&&p.initial<=p.max&&
   (p.searchScale===undefined||(Number.isFinite(p.searchScale)&&p.searchScale>0&&p.searchScale<=1));
 }))throw new Error('Unknown training motor decoder parameter contract');
 if(io!==undefined&&JSON.stringify(contract)!==JSON.stringify(buildMotorDecoderContract(io)))throw new Error('Training motor decoder contract differs from prepared io');
 return contract;
}
export function parameterValues(config,parameters,io){
 validateTrainingConfigSchema(config);
 if(config.dtMs!==.5||config.bodyBlockMs!==2)throw new Error('Unsupported training configuration');
 validateTrainingVisionConfig(config);
 if(config.parameterContract===SENSORIMOTOR_CONTRACT)
  return {...applySensorimotorParameters(config,parameters,io),interpreter:DEFAULT_FLIGHT_INTERPRETER};
 const decoder=validateMotorDecoderTrainingConfig(config,io);
 if(decoder){
  const vector=Array.from(validateMotorDecoderVector(decoder,parameters));
  return {vector,motorDecoder:vector,interpreter:DEFAULT_FLIGHT_INTERPRETER};
 }
 if(config.parameters?.length!==PARAMETER_NAMES.length||!config.parameters.every((p,i)=>p.name===PARAMETER_NAMES[i]&&Number.isFinite(p.min)&&Number.isFinite(p.max)&&p.min<=p.max))throw new Error('Unknown training parameter contract');
 if(!Array.isArray(parameters)&&!(parameters instanceof Float32Array)&&!(parameters instanceof Float64Array))throw new Error('Training parameters must be a numeric vector');
 const vector=Array.from(parameters);if(vector.length!==config.parameters.length)throw new Error('Training parameter count mismatch');
 const gains={};config.parameters.forEach((p,i)=>{const v=vector[i];if(!Number.isFinite(v)||v<p.min||v>p.max)throw new Error('Training parameter outside bounds: '+p.name);gains[p.name]=Math.exp(v);});
 return {vector,gains,interpreter:flightParametersToInterpreter(vector)};
}
