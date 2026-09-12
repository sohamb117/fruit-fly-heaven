import createCore from './core.js';
const assert=(ok,message)=>{if(!ok)throw new TypeError(message);};
const finite=(x,name)=>assert(Number.isFinite(x),`${name} must be finite`);
const vector=(v,name)=>{assert(v?.length===3,`${name} must have 3 elements`);for(let i=0;i<3;i++)finite(v[i],name);return Float64Array.from(v);};
const integer=(x,name,max=0x7fffffff)=>assert(Number.isInteger(x)&&x>0&&x<=max,`${name} must be a positive integer <= ${max}`);

export async function createBrainViewModule(options={}){
  const core=await createCore({...(options.wasmBinary?{wasmBinary:options.wasmBinary}:{}),locateFile:path=>options.wasmUrl?String(options.wasmUrl):new URL(path,import.meta.url).href});
  function alloc(array){const p=core._malloc(Math.max(8,array.byteLength));if(!p)throw new Error('WASM allocation failed');core.HEAPU8.set(new Uint8Array(array.buffer,array.byteOffset,array.byteLength),p);return p;}
  function buffers(arrays,fn){const ps=[];try{for(const a of arrays)ps.push(alloc(a));return fn(...ps);}finally{for(const p of ps)core._free(p);}}
  function plane(options={}){
    const n=vector(options.normal??[0,0,1],'normal'),length=Math.hypot(...n);assert(length>0,'normal must be nonzero');
    const offset=options.offset??0,halfThickness=options.halfThickness??10;finite(offset,'offset');finite(halfThickness,'halfThickness');assert(halfThickness>=0,'halfThickness must be nonnegative');
    const modes={all:0,cutaway:1,slab:2},mode=options.mode??'all';assert(Object.hasOwn(modes,mode),'Unknown clipping mode');
    return {n:Float64Array.from(n,x=>x/length),offset:offset/length,halfThickness,mode:modes[mode]};
  }
  function positions(p){assert(p instanceof Float32Array&&p.length>0&&p.length%3===0,'positions must be Float32Array xyz triplets');for(const x of p)finite(x,'position');}
  class Geometry{
    #positions;#owners;#colors;#n;#neurons;#disposed=false;
    constructor({positions:p,neuronIndices,neuronCount}){
      positions(p);integer(neuronCount,'neuronCount');assert(neuronIndices instanceof Uint32Array&&neuronIndices.length===p.length/3,'One Uint32 neuron index per vertex is required');for(const i of neuronIndices)assert(i<neuronCount,'Neuron index outside activity array');
      this.#n=neuronIndices.length;this.#neurons=neuronCount;
      const ps=[];try{ps.push(alloc(p));ps.push(alloc(neuronIndices));ps.push(alloc(new Float32Array(p.length)));}catch(error){for(const q of ps)core._free(q);throw error;}
      [this.#positions,this.#owners,this.#colors]=ps;
    }
    #live(){assert(!this.#disposed,'Geometry has been disposed');}
    get vertexCount(){return this.#n;}
    updateActivity({voltage,lastSpikeMs,timeMs=0,restMv=-52,thresholdMv=-45}){
      this.#live();assert(voltage instanceof Float64Array&&voltage.length===this.#neurons,'voltage must contain one Float64 per neuron');for(const v of voltage)finite(v,'voltage');
      lastSpikeMs=lastSpikeMs??new Float64Array(this.#neurons).fill(-1e30);assert(lastSpikeMs instanceof Float64Array&&lastSpikeMs.length===this.#neurons,'lastSpikeMs length/type mismatch');for(const v of lastSpikeMs)finite(v,'lastSpikeMs');
      finite(timeMs,'timeMs');finite(restMv,'restMv');finite(thresholdMv,'thresholdMv');assert(thresholdMv>restMv,'thresholdMv must exceed restMv');
      buffers([voltage,lastSpikeMs],(v,s)=>core._nv_colors(this.#n,this.#neurons,this.#owners,v,s,timeMs,restMv,thresholdMv,this.#colors));
      return core.HEAPF32.slice(this.#colors/4,this.#colors/4+this.#n*3);
    }
    filter(options={}){this.#live();const p=plane(options);return buffers([p.n,new Uint32Array(this.#n)],(n,out)=>{const count=core._nv_filter(this.#n,this.#positions,n,p.offset,p.halfThickness,p.mode,out);return core.HEAPU32.slice(out/4,out/4+count);});}
    pick({matrix,x,y,width,height,radius=8,...options}){
      this.#live();assert(matrix instanceof Float32Array&&matrix.length===16,'matrix must be 16 Float32 column-major elements');for(const v of matrix)finite(v,'matrix');
      for(const [name,v]of Object.entries({x,y,width,height,radius}))finite(v,name);assert(width>0&&height>0&&radius>0,'Viewport and radius must be positive');
      const p=plane(options);return buffers([matrix,p.n],(m,n)=>core._nv_pick(this.#n,this.#positions,m,n,p.offset,p.halfThickness,p.mode,x,y,width,height,radius));
    }
    dispose(){if(!this.#disposed){for(const p of [this.#positions,this.#owners,this.#colors])core._free(p);this.#disposed=true;}}
  }
  class Volume{
    #data;#dims;#disposed=false;
    constructor({values,dimensions}){
      assert(dimensions?.length===3,'dimensions must contain x,y,z');for(const d of dimensions)integer(d,'dimension',65536);
      assert(values instanceof Uint8Array&&values.length===dimensions[0]*dimensions[1]*dimensions[2],'Volume size/type mismatch');this.#dims=[...dimensions];this.#data=alloc(values);
    }
    samplePlane({origin,u,v,width,height}){
      assert(!this.#disposed,'Volume has been disposed');integer(width,'width',4096);integer(height,'height',4096);
      return buffers([vector(origin,'origin'),vector(u,'u'),vector(v,'v'),new Uint8Array(width*height)],(o,a,b,out)=>{core._nv_sample(this.#data,...this.#dims,o,a,b,width,height,out);return core.HEAPU8.slice(out,out+width*height);});
    }
    dispose(){if(!this.#disposed){core._free(this.#data);this.#disposed=true;}}
  }
  return Object.freeze({
    createGeometry:data=>new Geometry(data),createVolume:data=>new Volume(data),
    sliceMesh({positions:p,triangles,normal=[0,0,1],offset=0}){
      positions(p);assert(triangles instanceof Uint32Array&&triangles.length%3===0,'triangles must be Uint32 triplets');for(const i of triangles)assert(i<p.length/3,'Triangle index outside geometry');
      const cut=plane({normal,offset});return buffers([p,triangles,cut.n,new Float32Array(triangles.length*2)],(v,t,n,out)=>{const count=core._nv_contour(v,triangles.length/3,t,n,cut.offset,out);return core.HEAPF32.slice(out/4,out/4+count*6);});
    },
    get allocatedHeapBytes(){return core.HEAPU8.buffer.byteLength;},
  });
}
