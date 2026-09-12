// Includes RGB decode lookup, copies, interpolation and caller-owned output.
import fs from 'node:fs';
import {createColorModule,loadFlyColorModel,decodeSRGB} from '../packages/fly-color-wasm/dist/index.js';
const model=await loadFlyColorModel(),runtime=await createColorModule();
const {size,channels,lut}=model,scale=Float32Array.from({length:256},(_,i)=>decodeSRGB(i/255)*(size-1));
function javascript(rgb,out){
  for(let p=0;p<rgb.length/3;p++){
    const r=scale[rgb[p*3]],g=scale[rgb[p*3+1]],b=scale[rgb[p*3+2]],ri=Math.min(Math.floor(r),size-2),gi=Math.min(Math.floor(g),size-2),bi=Math.min(Math.floor(b),size-2);
    const x=r-ri,y=g-gi,z=b-bi,bs=channels,gs=size*bs,rs=size*gs,at=ri*rs+gi*gs+bi*bs;
    for(let c=0;c<channels;c++){
      const a=lut[at+c]*(1-z)+lut[at+bs+c]*z,d=lut[at+gs+c]*(1-z)+lut[at+gs+bs+c]*z,e=lut[at+rs+c]*(1-z)+lut[at+rs+bs+c]*z,f=lut[at+rs+gs+c]*(1-z)+lut[at+rs+gs+bs+c]*z;
      out[p*channels+c]=(a*(1-y)+d*y)*(1-x)+(e*(1-y)+f*y)*x;
    }
  }return out;
}
const results=[];
for(const pixels of [1024,102400]){
  const mapper=runtime.createMapper({...model,maxPixels:pixels}),rgb=Uint8Array.from({length:pixels*3},(_,i)=>(Math.imul(i,1103515245)>>>16)&255),out=new Float32Array(pixels*4);
  const wasm=()=>mapper.map(rgb,out),js=()=>javascript(rgb,out);
  for(let i=0;i<100;i++){wasm();js();}
  const rounds=9,iterations=pixels===1024?300:10,times={wasm:[],javascript:[]};
  for(let r=0;r<rounds;r++)for(const [name,fn] of r%2?[['wasm',wasm],['javascript',js]]:[['javascript',js],['wasm',wasm]]){
    const start=performance.now();for(let i=0;i<iterations;i++)fn();times[name].push((performance.now()-start)/iterations);
  }
  const median=a=>a.sort((a,b)=>a-b)[Math.floor(a.length/2)];
  const w=median(times.wasm),j=median(times.javascript);results.push({pixels,wasmMilliseconds:w,javascriptMilliseconds:j,speedup:j/w});mapper.dispose();
}
const report={scope:'Mapping only, not the full-brain simulation. Median of nine alternating warmed batches; WASM includes input/output copies. 1024 pixels = both 32×16 eyes of one fly.',runtime:process.version,platform:process.platform,architecture:process.arch,results};
fs.writeFileSync(new URL('../reports/color-performance.json',import.meta.url),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
