export type ActivationField = 'voltage' | 'synapticDrive' | 'spikeCount';
export type StatePrecision = 'float64' | 'float32';
export interface NeuronParameters {
  dtMs?:number; tauMembraneMs?:number; tauSynapseMs?:number;
  restMv?:number; thresholdMv?:number; resetMv?:number;
  refractoryMs?:number; delayMs?:number; weightScaleMv?:number;
  spikeHistoryCapacity?:number;
}
export interface BrainOptions extends NeuronParameters { seed?:number|bigint; }
export interface CSRConnectome {neuronCount:number; rowOffsets:Uint32Array; targets:Uint32Array; weights:Float32Array;}
export interface ActivationOptions {field?:ActivationField;indices?:Uint32Array;}
export interface SpikeHistory {timesMs:Float64Array;neuronIndices:Uint32Array;total:number;dropped:number;}
export interface Brain {
  readonly precision:StatePrecision;
  readonly neuronCount:number; readonly parameters:Readonly<Required<NeuronParameters>>;
  readonly timeMs:number; readonly totalSpikes:number; readonly connectome:Connectome;
  setPoissonInputs(input:{indices:Uint32Array;ratesHz:Float32Array;amplitudesMv?:Float32Array}):this;
  clearPoissonInputs():this;
  setCurrentInputs(indices:Uint32Array,driveMv:Float32Array):this;
  injectVoltage(indices:Uint32Array,deltaMv:Float32Array):this;
  setRefractoryPeriod(indices:Uint32Array,ms:number):this;
  step(durationMs:number):number;
  readActivations(options?:ActivationOptions):Float64Array;
  readSpikes():SpikeHistory;
  dispose():void;
}
export interface Connectome {
  readonly neuronCount:number;readonly edgeCount:number;
  createBrain(options?:BrainOptions):Brain;
  createPopulation(count:number,options?:BrainOptions):Brain[];
  dispose():void;
}
export interface ActivationMatrix {
  precision:StatePrecision;
  values:Float64Array;shape:[number,number];order:'row-major';field:ActivationField;
  unit:'mV'|'spikes';timesMs:Float64Array;
}
export interface BrainModule {
  readonly precision:StatePrecision;
  createConnectome(csr:CSRConnectome):Connectome;
  readActivationMatrix(brains:Brain[],options?:ActivationOptions):ActivationMatrix;
  readonly allocatedHeapBytes:number;
}
export declare const DEFAULT_PARAMETERS:Readonly<Required<NeuronParameters>>;
export declare function createBrainModule(options?:{precision?:StatePrecision;wasmUrl?:string|URL;wasmBinary?:Uint8Array}):Promise<BrainModule>;
export declare function loadConnectome(url:string|URL,options?:{fetch?:typeof fetch}):Promise<CSRConnectome & {metadata:Record<string,unknown>}>;
