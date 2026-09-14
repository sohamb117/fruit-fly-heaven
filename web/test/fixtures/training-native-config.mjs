import {MOTOR_DECODER_VERSION,buildMotorDecoderContract} from '../../motor-decoder.js';
import {STEERING_MUSCLE_TYPES} from '../../training/flight-parameters.js';

// Synthetic anatomy for metadata/UI tests. No prepared asset or simulation.
export function nativeConfigFixture(base){
  let index=1000;
  const muscles=[];
  for(const side of ['left','right']){
    for(const [target,count]of [['dorsal_longitudinal_muscle',5],['dorsoventral_muscle',7],...STEERING_MUSCLE_TYPES.map(target=>[target,1])]){
      const steering=count===1,indices=Array.from({length:count},()=>index++);
      muscles.push({kind:steering?'wing_steering_assumption':'asynchronous_wing',joint:(steering?'wing_steer_':'wing_power_')+side,
        target,sign:1,indices,root_ids:indices.map(value=>(720575940000000000n+BigInt(value)).toString())});
    }
  }
  const contract=buildMotorDecoderContract({muscles}),config=structuredClone(base);
  return {...config,schemaVersion:2,parameterContract:MOTOR_DECODER_VERSION,motorDecoderContract:contract,
    freezeNeuralParameters:true,wingEventExcitation:{schemaVersion:1},
    parameters:contract.parameters.map(({name,min,max,initial})=>({name,min,max,initial})),
    optimizer:{...config.optimizer,acceptance:{profile:1,proposal:'best-search-job',seedCount:3,
      nativeExecution:{backend:'dawn-metal',moduleSha256:'a'.repeat(64),packageLockSha256:'b'.repeat(64)}}}};
}

export function wasmConfigFixture(base){
  const config=nativeConfigFixture(base),moduleSha256='c'.repeat(64);
  config.assets={...config.assets,'/banc-engine/dist/core.wasm':moduleSha256};
  config.optimizer.acceptance.nativeExecution={backend:'wasm',moduleSha256};
  return config;
}
