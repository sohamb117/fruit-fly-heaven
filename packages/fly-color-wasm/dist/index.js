import createCore from './core.js';
const assert=(ok,message)=>{if(!ok)throw new TypeError(message);};
export const decodeSRGB=value=>value<=.04045?value/12.92:((value+.055)/1.055)**2.4;

export async function loadFlyColorModel(base=new URL('./',import.meta.url)){
  const read=async name=>{
    const url=new URL(name,base);
    if(url.protocol==='file:')return new Uint8Array(await (await import('node:fs/promises')).readFile(url));
    const response=await fetch(url);if(!response.ok)throw new Error(`Cannot load ${name}: ${response.status}`);
    return new Uint8Array(await response.arrayBuffer());
  };
  const metadata=JSON.parse(new TextDecoder().decode(await read('fly-color-model.json')));
  assert(metadata.schema_version===1&&metadata.lut_file==='fly-color-lut.bin','Unsupported color model');
  const bytes=await read(metadata.lut_file);
  const sha=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),n=>n.toString(16).padStart(2,'0')).join('');
  assert(sha===metadata.lut_sha256,'Color LUT checksum mismatch');
  return {size:metadata.size,channels:metadata.channels.length,lut:new Float32Array(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength)),metadata};
}

/** Batched sRGB -> arbitrary receptor-channel LUT. One mapper can serve many eyes. */
export async function createColorModule(options={}){
  const core=await createCore({...options.wasmBinary?{wasmBinary:options.wasmBinary}:{},locateFile:p=>options.wasmUrl?String(options.wasmUrl):new URL(p,import.meta.url).href});
  return Object.freeze({createMapper({size,channels,lut,maxPixels=1024}){
    assert(Number.isInteger(size)&&size>=2&&size<=129,'Invalid LUT size');
    assert(Number.isInteger(channels)&&channels>=1&&channels<=16,'Invalid channel count');
    assert(lut instanceof Float32Array&&lut.length===size**3*channels&&lut.every(v=>Number.isFinite(v)&&v>=0),'Invalid LUT');
    assert(Number.isInteger(maxPixels)&&maxPixels>=1&&maxPixels<=1048576,'Invalid pixel capacity');
    const pointers=[];
    const alloc=bytes=>{const p=core._malloc(bytes);if(!p)throw new Error('WASM allocation failed');pointers.push(p);return p;};
    let lp,sp,ip,op;
    try{
      lp=alloc(lut.byteLength);sp=alloc(256*4);ip=alloc(maxPixels*3);op=alloc(maxPixels*channels*4);
      core.HEAPF32.set(lut,lp/4);
      core.HEAPF32.set(Float32Array.from({length:256},(_,i)=>decodeSRGB(i/255)*(size-1)),sp/4);
    }catch(e){pointers.forEach(p=>core._free(p));throw e;}
    let disposed=false;
    return Object.freeze({
      get channels(){return channels;}, get maxPixels(){return maxPixels;},
      map(rgb,output){
        assert(!disposed,'Mapper has been disposed');
        assert(rgb instanceof Uint8Array&&rgb.length%3===0&&rgb.length/3<=maxPixels,'Expected packed RGB bytes within pixel capacity');
        const n=rgb.length/3;
        if(output===undefined)output=new Float32Array(n*channels);
        assert(output instanceof Float32Array&&output.length===n*channels,'Output must have one Float32 per pixel/channel');
        core.HEAPU8.set(rgb,ip);core._color_map(n,ip,size,channels,lp,sp,op);
        output.set(core.HEAPF32.subarray(op/4,op/4+output.length));return output;
      },
      dispose(){if(!disposed){pointers.forEach(p=>core._free(p));disposed=true;}},
    });
  }});
}
