/** Displays copied sensor pixels. This view never renders a scene or supplies vision. */
export function validateEyeSnapshot(value){
  const integer=(x,min,max)=>Number.isSafeInteger(x)&&x>=min&&x<=max;
  if(!value||value.source!=='sensory-retina'||value.sensoryInput!=='luminance-derived-from-rgb'||value.format!=='rgba8'||
    typeof value.jobId!=='string'||!value.jobId||value.jobId.length>200||
    !integer(value.width,1,256)||!integer(value.height,1,128)||
    !integer(value.sourceWidth,value.width,16384)||!integer(value.sourceHeight,value.height,16384)||
    ['sampleSequence','trialSequence','sequence'].some(key=>!integer(value[key],0,Number.MAX_SAFE_INTEGER))||
    ['frameTimeSeconds','nativeTimeSeconds','neuralTimeMs'].some(key=>!Number.isFinite(value[key])||value[key]<0)||
    ['left','right'].some(key=>!(value[key] instanceof Uint8Array)||value[key].length!==value.width*value.height*4))
    throw new Error('Invalid eye image');
  return value;
}

export class TrainingEyePreview{
  constructor(left,right,{onStatus=()=>{}}={}){
    this.canvases=[left,right];this.contexts=[];this.onStatus=onStatus;
    this.active=false;this.snapshot=null;this.jobId=null;this.instance=null;this.drawn=null;this.disposed=false;
  }
  setJob(jobId,instance){
    if(jobId===this.jobId&&instance===this.instance)return;
    this.jobId=jobId;this.instance=instance;this.snapshot=null;this.drawn=null;
    this.contexts.forEach((context,i)=>context.clearRect(0,0,this.canvases[i].width,this.canvases[i].height));
    this.onStatus({hasFrame:false});
  }
  setActive(active){this.active=!!active;if(this.active)this.draw();}
  setSnapshot(value){
    try{validateEyeSnapshot(value);}catch{return false;}
    const previous=this.snapshot;
    if(this.disposed||value.jobId!==this.jobId||previous&&(value.sampleSequence<=previous.sampleSequence||
      value.trialSequence<previous.trialSequence||value.trialSequence===previous.trialSequence&&
      (value.sequence<=previous.sequence||value.frameTimeSeconds<previous.frameTimeSeconds)))return false;
    this.snapshot=value;this.draw();return true;
  }
  draw(){
    if(!this.active||!this.snapshot||this.disposed||this.drawn===this.snapshot)return;
    try{
      const value=this.snapshot;
      for(let i=0;i<2;i++){
        const canvas=this.canvases[i],context=this.contexts[i]??=canvas.getContext('2d',{alpha:false});
        if(!context)throw new Error('Eye canvas unavailable');
        if(canvas.width!==value.width)canvas.width=value.width;
        if(canvas.height!==value.height)canvas.height=value.height;
        const pixels=context.createImageData(value.width,value.height);
        pixels.data.set(i===0?value.left:value.right);context.putImageData(pixels,0,0);
      }
      this.drawn=value;this.onStatus({hasFrame:true,timeSeconds:value.frameTimeSeconds});
    }catch{this.onStatus({error:true});}
  }
  dispose(){this.disposed=true;this.active=false;this.snapshot=null;this.drawn=null;
    this.canvases.forEach(canvas=>{canvas.width=0;canvas.height=0;});this.contexts=[];}
}
