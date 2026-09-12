export const COLOR_CHANNELS=Object.freeze(['Rh3','Rh4','Rh5','Rh6']);
export function validateColorMapping(mapping,neuronCount){
  if(mapping.schema_version!==1||mapping.neuron_count!==neuronCount||!mapping.cells?.length)throw new Error('Color mapping does not match this connectome');
  const seen=new Set();
  for(const c of mapping.cells){
    if(!Number.isInteger(c.index)||c.index<0||c.index>=neuronCount||seen.has(c.index)||!['left','right'].includes(c.side)||![c.u,c.v].every(v=>Number.isFinite(v)&&v>=0&&v<=1)||!['R7','R8'].includes(c.type)||!(c.type==='R7'?['Rh3','Rh4']:['Rh5','Rh6']).includes(c.opsin))throw new Error('Invalid color receptor mapping');
    seen.add(c.index);
  }
  if(![mapping.base_rate_hz,mapping.rate_scale_hz,mapping.max_rate_hz].every(v=>Number.isFinite(v)&&v>=0)||mapping.max_rate_hz>200||mapping.base_rate_hz>mapping.max_rate_hz)throw new Error('Invalid color rate gains');
}

// Spectral conversion is stateless and shared; frames and input rates are per fly.
export class ColorVision{
  constructor(mapper,mapping){
    validateColorMapping(mapping,mapping.neuron_count);
    if(mapper.channels!==4||mapper.maxPixels<1024)throw new Error('Expected four-channel binocular color mapper');
    this.mapper=mapper;this.mapping=mapping;this.ratesHz=new Float32Array(mapping.cells.length);
    this.captures=new Float32Array(1024*4);this.serial=0;this.sequence=-1;
    this.addresses=mapping.cells.map(c=>((c.side==='left'?0:512)+Math.round(c.v*15)*32+Math.round(c.u*31))*4+COLOR_CHANNELS.indexOf(c.opsin));
    this.summary={ready:false,channels:COLOR_CHANNELS,left:[0,0,0,0],right:[0,0,0,0]};
  }
  update(frame,enabled=true){
    const valid=frame?.rgb instanceof Uint8Array&&frame.rgb.length===3072&&Number.isInteger(frame.sequence)&&frame.sequence>=0&&Number.isFinite(frame.bodyTime);
    if(!enabled||!valid){
      if(this.sequence!==-1){this.ratesHz.fill(0);this.captures.fill(0);this.sequence=-1;this.serial++;}
      this.summary={ready:false,channels:COLOR_CHANNELS,left:[0,0,0,0],right:[0,0,0,0]};return this;
    }
    if(frame.sequence===this.sequence)return this;
    this.mapper.map(frame.rgb,this.captures);
    this.addresses.forEach((address,i)=>{this.ratesHz[i]=Math.min(this.mapping.max_rate_hz,this.mapping.base_rate_hz+this.mapping.rate_scale_hz*this.captures[address]);});
    const mean=side=>COLOR_CHANNELS.map((_,c)=>{let sum=0;for(let p=side*512;p<(side+1)*512;p++)sum+=this.captures[p*4+c];return sum/512;});
    this.sequence=frame.sequence;this.serial++;
    this.summary={ready:true,channels:COLOR_CHANNELS,left:mean(0),right:mean(1),frameSequence:frame.sequence,
      meanDriveHz:this.ratesHz.reduce((s,v)=>s+v,0)/this.ratesHz.length,backend:'WASM',uv:'No incident light below 400 nm'};
    return this;
  }
}
