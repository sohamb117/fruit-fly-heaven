// Compile the published retinotopic graph once; each eye has independent state.
export function compileVisualModel(spec){
  if(spec.schema_version!==1)throw new Error('Unsupported visual model');
  const nodes=[],layers={},lookup=new Map(),radius=spec.extent;
  for(const type of spec.nodes){
    layers[type.type]=[];
    for(let u=-radius;u<=radius;u++)for(let v=Math.max(-radius,-radius-u);v<=Math.min(radius,radius-u);v++){
      if(u%type.stride[0]||v%type.stride[1])continue;
      const i=nodes.length;nodes.push({type:type.type,u,v,bias:type.bias,tau:type.tauSeconds,
        x:(u+v/2)/radius*.5+.5,y:.5-v/radius*.5});
      lookup.set([type.type,u,v].join(','),i);layers[type.type].push(i);
    }
  }
  const n=nodes.length,counts=new Uint32Array(n),edges=[];
  for(const [source,target,du,dv,weight]of spec.kernels)for(const s of layers[source]){
    const p=nodes[s],t=lookup.get([target,p.u+du,p.v+dv].join(','));
    if(t!==undefined){edges.push(s,t,weight);counts[t]++;}
  }
  const rowOffsets=new Uint32Array(n+1);for(let i=0;i<n;i++)rowOffsets[i+1]=rowOffsets[i]+counts[i];
  const cursor=rowOffsets.slice(),sources=new Uint32Array(edges.length/3),weights=new Float32Array(sources.length);
  for(let k=0;k<edges.length;k+=3){const p=cursor[edges[k+1]]++;sources[p]=edges[k];weights[p]=edges[k+2];}
  const inputIndices=Uint32Array.from(spec.inputs.flatMap(t=>layers[t]));
  return {nodes,layers,inputIndices,csr:{rowOffsets,sources,weights,inputIndices,
    bias:Float32Array.from(nodes,n=>n.bias),tauSeconds:Float32Array.from(nodes,n=>n.tau),dtSeconds:spec.stepSeconds}};
}

export function eyeInput(compiled,pixels,width=32,height=16,side=0){
  const input=new Float32Array(compiled.inputIndices.length);
  compiled.inputIndices.forEach((id,k)=>{const n=compiled.nodes[id];
    const x=Math.max(0,Math.min(width-1,Math.round(n.x*(width-1)))),y=Math.max(0,Math.min(height-1,Math.round(n.y*(height-1))));
    input[k]=pixels[side*width*height+y*width+x]/255;
  });return input;
}

export function compileVisualProjection(compiled,mapping){
  if(mapping.schema_version!==1||!Number.isInteger(mapping.neuron_count)||!mapping.cells.length||!Number.isFinite(mapping.rate_scale_hz)||mapping.rate_scale_hz<=0||!Number.isFinite(mapping.max_rate_hz)||mapping.max_rate_hz<=0)throw new Error('Invalid visual mapping');
  const seen=new Set();
  return mapping.cells.map(c=>{
    if(!Number.isInteger(c.index)||c.index<0||c.index>=mapping.neuron_count||seen.has(c.index))throw new Error('Invalid visual projection index');seen.add(c.index);
    if(!compiled.layers[c.type]?.length||!['left','right'].includes(c.side)||![c.u,c.v].every(v=>Number.isFinite(v)&&v>=0&&v<=1))throw new Error('Invalid visual projection');
    let best=-1,distance=Infinity;
    for(const i of compiled.layers[c.type]){const n=compiled.nodes[i],d=(n.x-c.u)**2+(n.y-c.v)**2;if(d<distance){best=i;distance=d;}}
    return {index:c.index,node:best,side:c.side==='left'?0:1};
  });
}

export class GradedVision{
  constructor(model,compiled,mapping,projection=compileVisualProjection(compiled,mapping)){
    this.compiled=compiled;this.mapping=mapping;this.projection=projection;
    this.eyes=[model.createNetwork(),model.createNetwork()];
    this.inputs=[new Float32Array(model.inputCount).fill(.5),new Float32Array(model.inputCount).fill(.5)];
    this.serial=0;this.nextMs=20;this.sequence=-1;this.enabled=false;
    this.ratesHz=new Float32Array(projection.length);
    this.groups=['R1','L1','Mi1','Tm1','T2','T3','T4a','T4b','T4c','T4d','T5a','T5b','T5c','T5d'];
    this.summary={ready:false,model:'FlyVis · graded WASM',activity:{},leftMotion:0,rightMotion:0};
  }
  update(frame,timeMs,enabled=true){
    if(!enabled){
      if(this.enabled){this.ratesHz.fill(0);this.serial++;this.enabled=false;}
      this.summary={ready:false,model:'FlyVis · graded WASM',activity:{},leftMotion:0,rightMotion:0,meanDriveHz:0};this.nextMs=timeMs+20;return this;
    }
    if(!frame||frame.pixels?.length!==1024)return this;
    if(!this.enabled){
      this.eyes.forEach((eye,i)=>{eye.reset();this.inputs[i].fill(.5);eye.step(this.inputs[i],50);});
      this.enabled=true;this.sequence=-1;this.nextMs=timeMs;
    }
    if(frame.sequence!==this.sequence){this.inputs=this.eyes.map((_,side)=>eyeInput(this.compiled,frame.pixels,32,16,side));this.sequence=frame.sequence;}
    if(timeMs+1e-8<this.nextMs)return this;
    const steps=Math.max(1,Math.floor((timeMs-this.nextMs)/20)+1);this.nextMs+=steps*20;
    this.eyes.forEach((eye,i)=>eye.step(this.inputs[i],steps));
    const activities=this.eyes.map(eye=>eye.readActivations());
    this.projection.forEach((p,k)=>{this.ratesHz[k]=Math.min(this.mapping.max_rate_hz,Math.max(0,activities[p.side][p.node])*this.mapping.rate_scale_hz);});
    const groups={};
    for(const type of this.groups)groups[type]=activities.map(values=>this.compiled.layers[type].reduce((s,i)=>s+Math.max(0,values[i]),0)/this.compiled.layers[type].length);
    // A dimensionless neural population readout; this does not decode pixels.
    // Direction calibration is measured separately with reversed gratings.
    const motion=side=>groups.T4b[side]+groups.T5a[side]-groups.T4a[side]-groups.T5b[side];
    this.summary={ready:true,model:'FlyVis · graded WASM',activity:groups,leftMotion:motion(0),rightMotion:motion(1),neuralTimeMs:timeMs,
      frameSequence:frame.sequence,meanDriveHz:this.ratesHz.reduce((s,n)=>s+n,0)/this.ratesHz.length};
    this.serial++;return this;
  }
  dispose(){this.eyes.forEach(eye=>eye.dispose());}
}
