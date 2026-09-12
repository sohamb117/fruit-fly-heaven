export interface ColorLUT { size:number; channels:number; lut:Float32Array; }
export interface Mapper { readonly channels:number; readonly maxPixels:number; map(rgb:Uint8Array, output?:Float32Array):Float32Array; dispose():void; }
export function decodeSRGB(value:number):number;
export function loadFlyColorModel(base?:URL|string):Promise<ColorLUT & {metadata:Record<string,unknown>}>;
export function createColorModule(options?:{wasmBinary?:Uint8Array;wasmUrl?:URL|string}):Promise<{createMapper(model:ColorLUT & {maxPixels?:number}):Mapper}>;
