// Explicit environment presets. The reusable WASM module accepts either
// precision independently of these timestep and delay choices.
export const SIMULATION_MODES=Object.freeze({
  reference:Object.freeze({precision:'float64',parameters:Object.freeze({}),dtMs:.1,label:'Reference · Float64',description:'0.1 ms steps · original timing and precision.'}),
  fast:Object.freeze({precision:'float32',parameters:Object.freeze({dtMs:1,delayMs:2,refractoryMs:2}),dtMs:1,label:'Fast · approximate',description:'Float32 · 1 ms steps · delay and refractory rounded to 2 ms. Activity and spike timing can change.'}),
});
