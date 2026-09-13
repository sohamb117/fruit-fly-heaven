// Exact feed-forward inference for an exported published FlyBody controller.
// This is an explicitly learned controller, not the BANC neural simulation.
// The exported weight artifact carries its own license and source provenance.
export function flattenFlyBodyObservation(model,observation,out=new Float32Array(model.input.size)){
 if(out.length!==model.input.size)throw new Error('Wrong observation buffer size');
 for(const field of model.input.fields){
  const value=observation[field.name];
  if(value===undefined&&field.length!==0)throw new Error('Missing observation: '+field.name);
  const flatten=v=>ArrayBuffer.isView(v)?Array.from(v):Array.isArray(v)?v.flatMap(flatten):[v];
  const flat=value===undefined?[]:flatten(value);
  if(flat.length!==field.length||!flat.every(Number.isFinite))throw new Error('Invalid observation: '+field.name);
  out.set(flat,field.offset);
 }
 return out;
}
export class FlyBodyPolicy{
 constructor(model){
  if(model.format!=='flybody-flight-mlp-v1'||!Number.isInteger(model.input?.size)||!model.layers?.length)throw new Error('Unsupported FlyBody policy export');
  this.model=model;this.inputSize=model.input.size;
  let size=this.inputSize;
  this.layers=model.layers.map(layer=>{
   if(layer.type==='dense'){
    if(layer.input_size!==size||layer.weights.length!==size||layer.bias.length!==layer.output_size)throw new Error('Invalid dense layer shape');
    const source=Float32Array.from(layer.weights.flat()),bias=Float32Array.from(layer.bias);
    if(source.length!==size*layer.output_size||!source.every(Number.isFinite)||!bias.every(Number.isFinite))throw new Error('Invalid dense weights');
    // Keep each output's dot product contiguous for the CPU reference path.
    const weights=new Float32Array(source.length);
    for(let j=0;j<layer.output_size;j++)for(let i=0;i<size;i++)weights[j*size+i]=source[i*layer.output_size+j];
    const compiled={...layer,weights,bias,output:new Float32Array(layer.output_size)};size=layer.output_size;return compiled;
   }
   if(layer.type==='layer_norm'){
    if(layer.axis!==-1||!(layer.epsilon>0)||layer.scale.length!==size||layer.offset.length!==size)throw new Error('Unsupported layer normalization');
    return {...layer,scale:Float32Array.from(layer.scale),offset:Float32Array.from(layer.offset),output:new Float32Array(size)};
   }
   if(!['tanh','elu','relu'].includes(layer.type))throw new Error('Unsupported policy operation: '+layer.type);
   return {...layer,output:new Float32Array(size)};
  });
  if(size!==model.output.size)throw new Error('Policy output shape mismatch');
 }
 // Returned buffer is reused; copy it when retaining a trajectory sample.
 predict(input){
  if(input.length!==this.inputSize)throw new Error('Wrong FlyBody observation size');
  let x=input;
  for(const layer of this.layers){
   const y=layer.output;
   if(layer.type==='dense'){
    const n=layer.output_size,w=layer.weights,size=layer.input_size;
    for(let j=0;j<n;j++){
     let sum=layer.bias[j],offset=j*size;for(let i=0;i<size;i++)sum+=x[i]*w[offset+i];y[j]=sum;
    }
   }else if(layer.type==='layer_norm'){
    let mean=0;for(let i=0;i<x.length;i++)mean+=x[i];mean/=x.length;
    let variance=0;for(let i=0;i<x.length;i++)variance+=(x[i]-mean)**2;variance/=x.length;
    const inverse=1/Math.sqrt(variance+layer.epsilon);
    for(let i=0;i<x.length;i++)y[i]=(x[i]-mean)*inverse*layer.scale[i]+layer.offset[i];
   }else if(layer.type==='tanh')for(let i=0;i<x.length;i++)y[i]=Math.tanh(x[i]);
   else if(layer.type==='elu')for(let i=0;i<x.length;i++)y[i]=x[i]>=0?x[i]:Math.expm1(x[i]);
   else for(let i=0;i<x.length;i++)y[i]=Math.max(0,x[i]);
   x=y;
  }
  return x;
 }
}
