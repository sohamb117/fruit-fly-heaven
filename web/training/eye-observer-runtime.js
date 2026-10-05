// Presentation-only copies of retina frames already consumed by the sensory
// pipeline. Never renders, advances a sensor/brain/body, or uploads an image.
export const EYE_SAMPLE_INTERVAL_MS=500;
export const MAX_EYE_WIDTH=256;
export const MAX_EYE_HEIGHT=128;
const check=(ok,message)=>{if(!ok)throw new Error('Eye preview: '+message);};

function copyEyes(frame){
 const {width:sourceWidth,height:sourceHeight,rgb}=frame;
 check(Number.isInteger(sourceWidth)&&sourceWidth>=8&&sourceWidth<=1024&&
  Number.isInteger(sourceHeight)&&sourceHeight>=8&&sourceHeight<=1024,'invalid retinal dimensions');
 check(rgb instanceof Uint8Array&&rgb.length===sourceWidth*sourceHeight*6,'invalid retinal RGB data');
 check(frame.eyes?.length===2&&frame.eyes[0].side==='left'&&frame.eyes[1].side==='right','invalid retinal eye order');
 const scale=Math.min(1,MAX_EYE_WIDTH/sourceWidth,MAX_EYE_HEIGHT/sourceHeight),
  width=Math.max(1,Math.floor(sourceWidth*scale)),height=Math.max(1,Math.floor(sourceHeight*scale));
 const eyes=[new Uint8Array(width*height*4),new Uint8Array(width*height*4)];
 for(let eye=0;eye<2;eye++)for(let y=0;y<height;y++)for(let x=0;x<width;x++){
  // Nearest source pixel only. The current256x128 retina is copied exactly;
  // larger future retina settings are reduced solely for presentation.
  const sx=Math.min(sourceWidth-1,Math.floor((x+.5)*sourceWidth/width)),
   sy=Math.min(sourceHeight-1,Math.floor((y+.5)*sourceHeight/height)),
   from=(eye*sourceWidth*sourceHeight+sy*sourceWidth+sx)*3,to=(y*width+x)*4;
  eyes[eye][to]=rgb[from];eyes[eye][to+1]=rgb[from+1];eyes[eye][to+2]=rgb[from+2];eyes[eye][to+3]=255;
 }
 return {width,height,sourceWidth,sourceHeight,left:eyes[0],right:eyes[1]};
}

export function createEyeObserver({postMessage,now=()=>performance.now()}){
 check(typeof postMessage==='function'&&typeof now==='function','message sink and clock required');
 let enabled=false,disposed=false,job=null,lastWall=-Infinity,sampleSequence=0,trialSequence=0,sources=new WeakMap();
 const report=error=>{try{postMessage({type:'eyes-error',...(job?{id:job.requestId}:{}),message:error?.message||'Eye preview unavailable'});}catch{}};
 function endJob(token){if(token!==undefined&&token!==job)return;job=null;sources=new WeakMap();trialSequence=0;}
 return {
  configure(message){
   check(!disposed,'observer disposed');
   if(typeof message?.enabled!=='boolean'){enabled=false;throw new TypeError('Eye preview enabled must be boolean');}
   enabled=message.enabled;
  },
  beginJob(id,requestId){
   endJob();lastWall=-Infinity;
   if(typeof id!=='string'||!id.length||id.length>256||!Number.isSafeInteger(requestId)||requestId<0)return null;
   job={id,requestId};return job;
  },
  observe(context){
   // In particular, the disabled path does not even read the sensor cache.
   if(!enabled||disposed||!job)return;
   try{
    const wall=now();check(Number.isFinite(wall),'invalid wall clock');
    if(wall-lastWall<EYE_SAMPLE_INTERVAL_MS)return;
    const source=context?.sensoryFeedback,frame=source?.lastFrame;
    if(!frame)return; // Vision disabled or no completed retinal sample yet.
    check(Number.isSafeInteger(frame.sequence)&&frame.sequence>=0,'invalid retinal sequence');
    check(Number.isFinite(frame.bodyTime)&&frame.bodyTime>=0,'invalid retinal capture time');
    const nativeTimeSeconds=context.body?.time,neuralTimeMs=context.neuralMs;
    check(Number.isFinite(nativeTimeSeconds)&&Number.isFinite(neuralTimeMs)&&
     frame.bodyTime<=nativeTimeSeconds+1e-8&&Math.abs(nativeTimeSeconds*1000-neuralTimeMs)<1e-5,
     'frame must belong to a completed native/neural block');
    let state=sources.get(source);
    if(!state){state={trialSequence:++trialSequence,sequence:-1};sources.set(source,state);}
    if(frame.sequence===state.sequence)return;
    check(frame.sequence>state.sequence,'retinal sequence regressed within a trial');
    const images=copyEyes(frame);
    const snapshot={jobId:job.id,sampleSequence:++sampleSequence,trialSequence:state.trialSequence,
     sequence:frame.sequence,frameTimeSeconds:frame.bodyTime,nativeTimeSeconds,neuralTimeMs,
     format:'rgba8',source:'sensory-retina',sensoryInput:'luminance-derived-from-rgb',...images};
    // Commit before a reentrant/throwing sink. Only these owned copies transfer.
    lastWall=wall;state.sequence=frame.sequence;
    postMessage({type:'eyes',id:job.requestId,snapshot},[images.left.buffer,images.right.buffer]);
   }catch(error){enabled=false;report(error);}
  },
  endJob,report,
  dispose(){if(disposed)return;disposed=true;enabled=false;endJob();},
  get state(){return {enabled,disposed,jobId:job?.id??null,sampleSequence,trialSequence};},
 };
}
