import {RETINAL_DEFAULTS,RETINAL_AXES} from './retinal-sensor.js';

/** Compact engineered motion transduction, not a fitted T4/T5 physiology.
 * All retinal pixels enter a coarse-to-fine Lucas–Kanade field. The last
 * refinement operates at the original resolution. ON/OFF temporal contrast
 * separately weights signed camera-horizontal/vertical motion at each mapped
 * cell, so local expansion is retained rather than collapsed into eye means.
 *
 * This module's ONLY injection boundary is annotated BANC T4/T5 cells. Do not
 * also inject its images into L1/L2/L3 or run the full graded FlyVis network.
 * The supplied representative-point retinotopy is a prior, not measured RFs.
 */
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
export const COMPACT_DIRECTION_HYPOTHESIS=Object.freeze({a:'right',b:'left',c:'up',d:'down'});
const DIRECTIONS=Object.freeze({right:[1,0],left:[-1,0],up:[0,1],down:[0,-1]});
export const COMPACT_VISION_DEFAULTS=Object.freeze({width:256,height:128,pyramidLevels:3,iterations:3,
  windowRadius:3,regularization:.0001,contrastScale:.08,gainHzPerRad:20,maxRateHz:60,
  maxAngularSpeed:40,maxGapSeconds:.1});

export function filterCompactVisionMapping(mapping){
  if(!mapping||!Array.isArray(mapping.cells)||!Number.isInteger(mapping.neuron_count)||mapping.neuron_count<=0)throw new Error('Compact vision requires a connectome mapping');
  const cells=mapping.cells.filter(c=>/^T[45][abcd]$/.test(c.type)).map(c=>({...c})),seen=new Set();
  if(!cells.length)throw new Error('No annotated T4/T5 cells for compact vision');
  for(const c of cells){
    if(!Number.isInteger(c.index)||c.index<0||c.index>=mapping.neuron_count||seen.has(c.index)||!['left','right'].includes(c.side)||!Number.isFinite(c.u)||!Number.isFinite(c.v)||c.u<0||c.u>1||c.v<0||c.v>1)throw new Error('Invalid/duplicate compact visual cell');
    seen.add(c.index);
  }
  return {...mapping,cells,compact_boundary:'Annotated T4/T5 only; original indices and coordinate priors retained',
    retinotopy_assumption:'BANC representative-point y/z quantiles and right-eye mirroring are not measured receptive fields'};
}
export function validateCompactVisionSettings(value={}){
  const s={...COMPACT_VISION_DEFAULTS,...value};
  for(const k of ['width','height'])if(!Number.isInteger(s[k])||s[k]<8||s[k]>1024)throw new Error(`Invalid compact vision ${k}`);
  for(const [k,min,max]of [['pyramidLevels',1,4],['iterations',1,5],['windowRadius',1,8]])if(!Number.isInteger(s[k])||s[k]<min||s[k]>max)throw new Error(`Invalid compact vision ${k}`);
  for(const k of ['regularization','contrastScale','gainHzPerRad','maxRateHz','maxAngularSpeed','maxGapSeconds'])if(!Number.isFinite(s[k])||s[k]<=0)throw new Error(`Invalid compact vision ${k}`);
  return Object.freeze(Object.fromEntries(Object.keys(COMPACT_VISION_DEFAULTS).map(k=>[k,s[k]])));
}
function sample(a,w,h,x,y){
  x=clamp(x,0,w-1);y=clamp(y,0,h-1);
  const x0=Math.floor(x),y0=Math.floor(y),x1=Math.min(w-1,x0+1),y1=Math.min(h-1,y0+1),fx=x-x0,fy=y-y0;
  return (a[y0*w+x0]*(1-fx)+a[y0*w+x1]*fx)*(1-fy)+(a[y1*w+x0]*(1-fx)+a[y1*w+x1]*fx)*fy;
}
function pyramid(a,w,h,levels){
  const result=[{a,w,h}];
  while(result.length<levels&&w>=16&&h>=16){
    const nw=Math.ceil(w/2),nh=Math.ceil(h/2),b=new Float32Array(nw*nh);
    for(let y=0;y<nh;y++)for(let x=0;x<nw;x++){
      const x1=Math.min(w-1,x*2+1),y1=Math.min(h-1,y*2+1);
      b[y*nw+x]=(a[y*2*w+x*2]+a[y*2*w+x1]+a[y1*w+x*2]+a[y1*w+x1])*.25;
    }
    a=b;w=nw;h=nh;result.push({a,w,h});
  }
  return result;
}
function integral(a,w,h){
  const out=new Float64Array((w+1)*(h+1)),stride=w+1;
  for(let y=0;y<h;y++){
    let sum=0;for(let x=0;x<w;x++){sum+=a[y*w+x];out[(y+1)*stride+x+1]=out[y*stride+x+1]+sum;}
  }
  return out;
}
const rect=(a,stride,x0,y0,x1,y1)=>a[y1*stride+x1]-a[y0*stride+x1]-a[y1*stride+x0]+a[y0*stride+x0];
function motion(previous,current,s){
  let oldFlow=null;
  for(let level=current.length-1;level>=0;level--){
    const {a:now,w,h}=current[level],before=previous[level].a,n=w*h,
      horizontal=new Float32Array(n),vertical=new Float32Array(n),terms=Array.from({length:5},()=>new Float32Array(n));
    if(oldFlow)for(let y=0;y<h;y++)for(let x=0;x<w;x++){
      const i=y*w+x;
      horizontal[i]=sample(oldFlow.horizontal,oldFlow.w,oldFlow.h,x*.5,y*.5)*2;
      vertical[i]=sample(oldFlow.vertical,oldFlow.w,oldFlow.h,x*.5,y*.5)*2;
    }
    for(let iteration=0;iteration<s.iterations;iteration++){
      for(let y=0;y<h;y++)for(let x=0;x<w;x++){
        const i=y*w+x,px=x-horizontal[i],py=y-vertical[i],
          gx=(sample(before,w,h,px+1,py)-sample(before,w,h,px-1,py))*.5,
          gy=(sample(before,w,h,px,py+1)-sample(before,w,h,px,py-1))*.5,
          dt=now[i]-sample(before,w,h,px,py);
        terms[0][i]=gx*gx;terms[1][i]=gy*gy;terms[2][i]=gx*gy;terms[3][i]=gx*dt;terms[4][i]=gy*dt;
      }
      const sums=terms.map(a=>integral(a,w,h)),stride=w+1,r=s.windowRadius;
      for(let y=0;y<h;y++)for(let x=0;x<w;x++){
        const i=y*w+x,x0=Math.max(0,x-r),y0=Math.max(0,y-r),x1=Math.min(w,x+r+1),y1=Math.min(h,y+r+1),area=(x1-x0)*(y1-y0),ridge=s.regularization*area,
          xx=rect(sums[0],stride,x0,y0,x1,y1)+ridge,yy=rect(sums[1],stride,x0,y0,x1,y1)+ridge,
          xy=rect(sums[2],stride,x0,y0,x1,y1),xt=rect(sums[3],stride,x0,y0,x1,y1),yt=rect(sums[4],stride,x0,y0,x1,y1),det=xx*yy-xy*xy;
        horizontal[i]+=clamp((-yy*xt+xy*yt)/det,-2,2);
        vertical[i]+=clamp((xy*xt-xx*yt)/det,-2,2);
      }
    }
    oldFlow={horizontal,vertical,w,h};
  }
  return oldFlow;
}
function polarity(before,now,w,h,r,scale){
  const n=w*h,pos=new Float32Array(n),neg=new Float32Array(n),on=new Float32Array(n),off=new Float32Array(n);
  for(let i=0;i<n;i++){const d=now[i]-before[i];pos[i]=Math.max(0,d);neg[i]=Math.max(0,-d);}
  const a=integral(pos,w,h),b=integral(neg,w,h),stride=w+1;
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){
    const x0=Math.max(0,x-r),y0=Math.max(0,y-r),x1=Math.min(w,x+r+1),y1=Math.min(h,y+r+1),den=(x1-x0)*(y1-y0)*scale;
    on[y*w+x]=clamp(rect(a,stride,x0,y0,x1,y1)/den,0,1);off[y*w+x]=clamp(rect(b,stride,x0,y0,x1,y1)/den,0,1);
  }
  return {on,off};
}

export function createCompactVision({mapping:source,directionHypothesis=COMPACT_DIRECTION_HYPOTHESIS,...options}={}){
  const mapping=filterCompactVisionMapping(source),settings=validateCompactVisionSettings({maxRateHz:source.max_rate_hz??COMPACT_VISION_DEFAULTS.maxRateHz,...options}),
    directions={...directionHypothesis},indices=Uint32Array.from(mapping.cells,c=>c.index),ratesHz=new Float32Array(indices.length);
  for(const subtype of ['a','b','c','d'])if(!DIRECTIONS[directions[subtype]])throw new Error('Invalid compact camera-direction hypothesis');
  Object.freeze(directions);
  const assumptions=Object.freeze({motion:RETINAL_AXES.motion,
    subtypeDirections:directions,subtypeStatus:'Explicit camera-axis hypothesis; NOT validated biological a/b/c/d tuning in either eye',
    retinotopy:mapping.retinotopy_assumption,
    physiology:'Engineered pyramidal image motion with ON/OFF temporal-contrast weighting; neither FlyVis weights nor fitted receptor/T4/T5 physiology',
    boundary:'Only mapped T4/T5 current; no added L1/L2/L3, T2/T3 or color drive'});
  let previous=null,lastTime=null,lastSequence=null;
  const result={mapping,settings,indices,ratesHz,serial:0,fields:null,summary:{ready:false,assumptions},
    reset(){previous=null;lastTime=null;lastSequence=null;ratesHz.fill(0);this.fields=null;this.serial++;this.summary={ready:false,assumptions};return this;},
    update(frame){
      if(frame===null||frame===undefined)return this.reset();
      const {width:w,height:h,bodyTime,sequence,pixels}=frame,n=w*h;
      if(w!==settings.width||h!==settings.height||!(pixels instanceof Uint8Array)||pixels.length!==2*n||!Number.isFinite(bodyTime)||bodyTime<0||!Number.isInteger(sequence)||sequence<0)throw new Error('Invalid compact retinal frame');
      const hfov=frame.horizontalFovDegrees??RETINAL_DEFAULTS.horizontalFovDegrees,vfov=frame.verticalFovDegrees??RETINAL_DEFAULTS.verticalFovDegrees;
      if(!Number.isFinite(hfov)||!Number.isFinite(vfov)||hfov<=0||vfov<=0||hfov>=180||vfov>=180)throw new Error('Invalid compact retinal FOV');
      if(lastSequence===sequence){if(lastTime!==bodyTime)throw new Error('Retinal sequence reused at a different simulation time');return this;}
      if(lastTime!==null&&(bodyTime<=lastTime||sequence<lastSequence))throw new Error('Compact retinal time/sequence must increase; reset between episodes');
      const eyes=[0,1].map(eye=>pyramid(Float32Array.from(pixels.subarray(eye*n,(eye+1)*n),v=>v/255),w,h,settings.pyramidLevels)),dt=lastTime===null?0:bodyTime-lastTime,
        // Native simulation timestamps can place an exact cadence a few ULPs
        // above its limit. One nanosecond tolerates roundoff, not a missed frame.
        first=previous===null||dt>settings.maxGapSeconds+1e-9;
      ratesHz.fill(0);this.serial++;
      if(first){this.fields=null;this.summary={ready:false,reason:previous===null?'first_frame':'sampling_gap',width:w,height:h,bodyTime,sequence,assumptions};}
      else{
        const hScale=hfov*Math.PI/180/(w-1)/dt,vScale=vfov*Math.PI/180/(h-1)/dt;
        this.fields=eyes.map((eye,i)=>{
          const field=motion(previous[i],eye,settings),polar=polarity(previous[i][0].a,eye[0].a,w,h,settings.windowRadius,settings.contrastScale);
          for(let j=0;j<n;j++){field.horizontal[j]=clamp(field.horizontal[j]*hScale,-settings.maxAngularSpeed,settings.maxAngularSpeed);field.vertical[j]=clamp(-field.vertical[j]*vScale,-settings.maxAngularSpeed,settings.maxAngularSpeed);}
          return {...field,...polar};
        });
        const sums=[{horizontal:0,vertical:0,expansion:0,on:0,off:0},{horizontal:0,vertical:0,expansion:0,on:0,off:0}],counts=[0,0],hz=[0,0];
        for(let side=0;side<2;side++)for(let y=0;y<h;y++)for(let x=0;x<w;x++){
          const i=y*w+x,f=this.fields[side],s=sums[side],u=(x/(w-1)-.5)*2,v=(.5-y/(h-1))*2;
          s.horizontal+=f.horizontal[i]/n;s.vertical+=f.vertical[i]/n;s.expansion+=(f.horizontal[i]*u+f.vertical[i]*v)/n;s.on+=f.on[i]/n;s.off+=f.off[i]/n;
        }
        for(let i=0;i<mapping.cells.length;i++){
          const c=mapping.cells[i],side=c.side==='left'?0:1,f=this.fields[side],x=c.u*(w-1),y=c.v*(h-1),d=DIRECTIONS[directions[c.type[2]]],
            direction=sample(f.horizontal,w,h,x,y)*d[0]+sample(f.vertical,w,h,x,y)*d[1],contrast=sample(c.type[1]==='4'?f.on:f.off,w,h,x,y);
          ratesHz[i]=clamp(Math.max(0,direction)*contrast*settings.gainHzPerRad,0,settings.maxRateHz);hz[side]+=ratesHz[i];counts[side]++;
        }
        this.summary={ready:true,width:w,height:h,bodyTime,sequence,dt,fullResolutionSamples:2*n,
          mappedCells:indices.length,leftHz:hz[0]/Math.max(1,counts[0]),rightHz:hz[1]/Math.max(1,counts[1]),
          eyes:sums,assumptions};
      }
      previous=eyes;lastTime=bodyTime;lastSequence=sequence;
      return this;
    }
  };
  return result;
}
