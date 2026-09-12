export interface ClipPlane {normal?:ArrayLike<number>;offset?:number;halfThickness?:number;mode?:'all'|'cutaway'|'slab';}
export interface GeometryData {positions:Float32Array;neuronIndices:Uint32Array;neuronCount:number;}
export interface Activity {voltage:Float64Array;lastSpikeMs?:Float64Array;timeMs?:number;restMv?:number;thresholdMv?:number;}
export interface Geometry {
  readonly vertexCount:number;
  updateActivity(activity:Activity):Float32Array;
  filter(plane?:ClipPlane):Uint32Array;
  pick(options:ClipPlane & {matrix:Float32Array;x:number;y:number;width:number;height:number;radius?:number}):number;
  dispose():void;
}
export interface VolumeData {values:Uint8Array;dimensions:[number,number,number];}
export interface SamplePlane {origin:ArrayLike<number>;u:ArrayLike<number>;v:ArrayLike<number>;width:number;height:number;}
export interface Volume {samplePlane(plane:SamplePlane):Uint8Array;dispose():void;}
export interface BrainViewModule {
  createGeometry(data:GeometryData):Geometry;
  createVolume(data:VolumeData):Volume;
  sliceMesh(mesh:{positions:Float32Array;triangles:Uint32Array;normal?:ArrayLike<number>;offset?:number}):Float32Array;
  readonly allocatedHeapBytes:number;
}
export declare function createBrainViewModule(options?:{wasmUrl?:string|URL;wasmBinary?:Uint8Array}):Promise<BrainViewModule>;
