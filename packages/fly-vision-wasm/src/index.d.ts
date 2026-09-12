export interface GradedNetwork {
  readonly timeSeconds: number;
  step(input: Float32Array, steps?: number): number;
  readActivations(): Float32Array;
  reset(): void;
  dispose(): void;
}
export interface GradedModel {
  readonly neuronCount: number;
  readonly inputCount: number;
  readonly dtSeconds: number;
  createNetwork(): GradedNetwork;
  dispose(): void;
}
export function createVisionModule(options?: {wasmBinary?: Uint8Array; wasmUrl?: string | URL}): Promise<{
  createModel(data: {rowOffsets: Uint32Array; sources: Uint32Array; weights: Float32Array; bias: Float32Array; tauSeconds: Float32Array; inputIndices: Uint32Array; dtSeconds?: number}): GradedModel;
  readonly allocatedHeapBytes: number;
}>;
