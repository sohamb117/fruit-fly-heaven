import {sensoryRates} from './body-world.js';
import {validateColorMapping} from './color-vision.js';
import {bancBodyRate} from './banc-ground-sense.js';
import {validateTegulaSensoryManifest,createTegulaInputMapper} from './banc-tegula.js';

const clamp=(n,lo=0,hi=100)=>Math.max(lo,Math.min(hi,Number.isFinite(n)?n:0));
const mean=a=>a.length?a.reduce((s,n)=>s+n,0)/a.length:0;
export const SENSORY_GAINS=Object.freeze({lightBaseHz:2,lightHz:18,contrastHz:40,adaptSeconds:.15,
  supportHz:8,jointAngleHz:8,jointSpeedHz:3,bodySpeedHz:2,yawHz:4,antennaAngleHz:35,antennaSpeedHz:2,airSpeedHz:2,tiltHz:10,touchHz:70,vibrationHz:4,impactHz:8});

export function validateSensoryManifest(manifest,neuronCount){
  if(manifest.schema_version!==1||manifest.neuron_count!==neuronCount)throw new Error('Sensory annotations do not match this connectome');
  const v=manifest.vision;
  if(!Number.isInteger(v.width)||!Number.isInteger(v.height)||v.width<16||v.height<8||v.width>1024||v.height>1024||!v.receptors.length)throw new Error('Unsupported retinal dimensions');
  const seen=new Set();
  const check=i=>{if(!Number.isInteger(i)||i<0||i>=neuronCount||seen.has(i))throw new Error('Invalid or duplicate sensory neuron');seen.add(i);};
  for(const r of v.receptors){check(r.index);if(!['left','right'].includes(r.side)||!(r.u>=0&&r.u<=1&&r.v>=0&&r.v<=1))throw new Error('Invalid visual column');}
  for(const channel of manifest.channels){if(!channel.indices.length)throw new Error('Empty body-sense channel');channel.indices.forEach(check);}
  const bodyIds=new Set(manifest.channels.flatMap(c=>c.indices)),transducerIds=new Set();
  for(const s of manifest.body_transducers||[]){
    if(!bodyIds.has(s.index)||transducerIds.has(s.index)||!['load','touch','position','velocity','vibration','rotation','wing_strain'].includes(s.kind))throw new Error('Invalid body transducer');
    if(!['rotation','wing_strain'].includes(s.kind)&&(!Number.isInteger(s.leg)||s.leg<0||s.leg>5))throw new Error('Invalid sensory leg');
    transducerIds.add(s.index);
  }
  validateTegulaSensoryManifest(manifest);
}

export function bodyInputRates(feedback,enabled=true){
  const rates={},g=SENSORY_GAINS;
  for(const [i,side]of ['left','right'].entries()){
    const legs=feedback?.legs?.slice(i*3,i*3+3)||[],antenna=feedback?.antennae?.[i];
    const jointSpeed=mean(legs.map(l=>clamp(l.speed,0,30))),support=mean(legs.map(l=>clamp(l.support,0,1)));
    rates['self_motion_'+side]=enabled?clamp(g.supportHz*support+g.jointAngleHz*mean(legs.map(l=>clamp(l.angle,0,2)))+g.jointSpeedHz*jointSpeed+g.bodySpeedHz*clamp(feedback?.speed,0,30)+g.yawHz*Math.abs(feedback?.yaw||0)):0;
    rates['antenna_'+side]=enabled?clamp(g.antennaAngleHz*Math.abs(antenna?.angle||0)+g.antennaSpeedHz*clamp(antenna?.speed,0,20)+g.airSpeedHz*clamp(feedback?.speed,0,30)+g.tiltHz*clamp(feedback?.tilt,0,Math.PI)):0;
    rates['touch_'+side]=enabled?g.touchHz*clamp(feedback?.touch?.[i],0,1):0;
    rates['vibration_'+side]=enabled?clamp(g.vibrationHz*jointSpeed*support+g.impactHz*clamp(feedback?.impact,0,20)):0;
  }
  return rates;
}

export class SensoryEncoder{
  constructor(manifest,groups,environment,{visualMapping=null,colorMapping=null,tasteMapper=null}={}){
    validateSensoryManifest(manifest,manifest.neuron_count);
    this.manifest=manifest;this.groups=groups;this.environment=environment;
    this.tasteMapper=tasteMapper;
    this.tegula=manifest.tegula_model===undefined?null:createTegulaInputMapper(manifest.tegula_model);
    this.bodyTransducers=new Map((manifest.body_transducers||[]).map(s=>[s.index,s]));
    // Exclusions prevent an unsupported modality from borrowing a broad
    // channel's fallback if console artifacts are combined across versions.
    // They remove only added host current, never recurrent neural activity.
    this.bodyExclusions=new Set((manifest.body_transducer_exclusions||[]).map(s=>s.index));
    this.foodCount=groups.odor_left.length+groups.odor_right.length+groups.sweet.length;
    if(colorMapping)validateColorMapping(colorMapping,manifest.neuron_count);
    this.indices=Uint32Array.from([...groups.odor_left,...groups.odor_right,...groups.sweet,...manifest.vision.receptors.map(r=>r.index),...manifest.channels.flatMap(c=>c.indices),...(visualMapping?.cells.map(c=>c.index)||[]),...(colorMapping?.cells.map(c=>c.index)||[])]);
    this.projectionCount=visualMapping?.cells.length||0;
    this.colorCount=colorMapping?.cells.length||0;
    if(this.indices.some(i=>i>=manifest.neuron_count))throw new Error('Visual projection outside connectome');
    if(new Set(this.indices).size!==this.indices.length)throw new Error('Sensory inputs overlap');
    this.ratesHz=new Float32Array(this.indices.length);
    this.adapted=new Float32Array(manifest.vision.width*manifest.vision.height*2);
    this.lightDrive=new Float32Array(this.adapted.length);this.lastSequence=-1;this.lastFrameTime=null;this.lastKey=null;
  }
  update(pose,frame,{odor=true,taste=true,vision=true,luminance=true,bodySense=true,graded=null,color=null}={}){
    // Validate opt-in feedback before caching the update key: a failed sample
    // may be corrected and retried, and never borrows aggregate body rates.
    const tegula=this.tegula?.rates(pose.feedback,bodySense,pose.bodyTime);
    const key=[pose.bodyTime,frame?.sequence,odor,taste,vision,luminance,bodySense,graded?.serial,color?.serial].join('|');
    if(key===this.lastKey)return null;
    this.lastKey=key;
    const g=SENSORY_GAINS,v=this.manifest.vision,n=v.width*v.height;
    const food=sensoryRates(pose,this.environment,{odor,taste});let offset=0;
    const contactTaste=this.tasteMapper&&'legFoodContact'in(pose.feedback||{});
    for(const [i,group]of [this.groups.odor_left,this.groups.odor_right,this.groups.sweet].entries()){
      if(i===2&&contactTaste){
        let total=0;for(const index of group){const rate=this.tasteMapper.rate(index,pose.feedback,taste);this.ratesHz[offset++]=rate;total+=rate;}
        food[2]=total/Math.max(1,group.length);
      }else{this.ratesHz.fill(food[i],offset,offset+group.length);offset+=group.length;}
    }
    const validFrame=frame&&frame.pixels?.length===2*n&&Number.isFinite(frame.bodyTime)&&Number.isInteger(frame.sequence)&&frame.sequence>=0;
    if(vision&&validFrame&&frame.sequence!==this.lastSequence){
      const first=this.lastFrameTime===null,dt=first?0:Math.max(0,frame.bodyTime-this.lastFrameTime),alpha=1-Math.exp(-dt/g.adaptSeconds);
      let change=0;
      for(let i=0;i<2*n;i++){
        const light=frame.pixels[i]/255;
        if(first)this.adapted[i]=light;
        const contrast=first?0:light-this.adapted[i];change+=Math.abs(contrast);
        this.lightDrive[i]=clamp(g.lightBaseHz+g.lightHz*light+g.contrastHz*contrast,0,80);
        this.adapted[i]+=alpha*contrast;
      }
      this.contrast=change/(2*n);this.lastSequence=frame.sequence;this.lastFrameTime=frame.bodyTime;
    }
    if(!vision){this.lastFrameTime=null;this.lastSequence=-1;this.contrast=0;}
    const sums=[0,0],counts=[0,0];
    for(const r of v.receptors){
      const side=r.side==='left'?0:1,x=Math.round(r.u*(v.width-1)),y=Math.round(r.v*(v.height-1));
      const hz=vision&&luminance&&validFrame?this.lightDrive[side*n+y*v.width+x]:0;
      this.ratesHz[offset++]=hz;sums[side]+=hz;counts[side]++;
    }
    const body=bodyInputRates(pose.feedback,bodySense);
    if(tegula)Object.assign(body,tegula);
    const nativeLegs=pose.feedback?.legs?.some(leg=>Number.isFinite(leg.loadBodyWeights));
    for(const c of this.manifest.channels){
      if(tegula&&(c.key==='tegula_left'||c.key==='tegula_right')){
        this.ratesHz.fill(body[c.key],offset,offset+c.indices.length);offset+=c.indices.length;continue;
      }
      if(this.bodyExclusions.size||(this.bodyTransducers.size&&nativeLegs)){
        let sum=0;
        for(const index of c.indices){const sensor=nativeLegs&&this.bodyTransducers.get(index),rate=this.bodyExclusions.has(index)?0:sensor?bancBodyRate(sensor,pose.feedback,bodySense):body[c.key];this.ratesHz[offset++]=rate;sum+=rate;}
        body[c.key]=sum/c.indices.length;
      }else{this.ratesHz.fill(body[c.key],offset,offset+c.indices.length);offset+=c.indices.length;}
    }
    if(this.projectionCount){
      if(vision&&graded?.summary.ready){
        if(graded.ratesHz?.length!==this.projectionCount||!graded.ratesHz.every(n=>Number.isFinite(n)&&n>=0))throw new Error('Invalid graded visual drive');
        this.ratesHz.set(graded.ratesHz,offset);
      }else this.ratesHz.fill(0,offset,offset+this.projectionCount);
      offset+=this.projectionCount;
    }
    if(this.colorCount){
      if(vision&&color?.summary.ready){
        if(color.ratesHz?.length!==this.colorCount||!color.ratesHz.every(n=>Number.isFinite(n)&&n>=0&&n<=200))throw new Error('Invalid color visual drive');
        this.ratesHz.set(color.ratesHz,offset);
      }else this.ratesHz.fill(0,offset);
    }
    this.sample={food,taste:contactTaste?{source:'native organ contact',coverage:this.tasteMapper.coverage}:null,vision:{enabled:vision,ready:!!validFrame,leftHz:sums[0]/counts[0],rightHz:sums[1]/counts[1],contrast:this.contrast||0,
      frameBodyTime:validFrame?frame.bodyTime:null,sequence:validFrame?frame.sequence:null,graded:graded?.summary||null,color:color?.summary||null},
      body:{enabled:bodySense,rates:body,support:mean((pose.feedback?.legs||[]).map(l=>l.support)),jointSpeed:mean((pose.feedback?.legs||[]).map(l=>l.speed)),
        speed:pose.feedback?.speed||0,yaw:pose.feedback?.yaw||0,tilt:pose.feedback?.tilt||0},bodyTime:pose.bodyTime};
    return {indices:this.indices,ratesHz:this.ratesHz,sample:this.sample};
  }
}
