/** Deterministic, worker/Node-compatible optics in native FlyBody centimetres.
 *
 * This is a sensory renderer, independent of the preview and wall clock. Every
 * output pixel casts its own ray. Smooth floor/fruit solids approximate the
 * native heightfield/convex collision meshes; no physics state is changed.
 * Eye centres, FOV, and world-fixed surface textures are optical priors, not
 * measurements of this animal. Self-occlusion, wings, refraction and shadows
 * are intentionally absent. See RETINAL_ASSUMPTIONS for exact differences.
 */
const RAD=Math.PI/180,EPS=1e-9;
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
export const RETINAL_DEFAULTS=Object.freeze({width:256,height:128,horizontalFovDegrees:150,
  verticalFovDegrees:120,eyeYawDegrees:60,nearCm:.002,farCm:150,texturePeriodCm:.35,textureAntialias:false});
export const RETINAL_AXES=Object.freeze({world:'native centimetres; +Z up',root:'+X forward; +Y left; +Z up',
  quaternion:'[w,x,y,z], root to world',image:'u increases camera-right; v increases camera-down',
  motion:'horizontal positive camera-right; vertical positive camera-up; radians/second',
  projection:'separable azimuth/elevation angular chart; not a calibrated compound eye'});
export const RETINAL_ASSUMPTIONS=Object.freeze([
  'Root-mounted eye centres [.08,+/-.03,-.002] cm and +/-60 degree yaw are geometric priors; no measured retinal centre or head articulation.',
  'Analytic paraboloid and capped outer floor approximate the native 257x257 heightfield; the declared native floor height and cap are retained.',
  'Smooth ellipsoids and capped finite cylinders approximate native low-poly apple meshes and convex banana tube slices; supplied banana paths take priority.',
  'A cylinder at the inside face of the 64-box wall approximates its polygon, and the native ceiling is at ceilingCm+0.15.',
  'World-fixed floor, fruit, wall and ceiling textures and fixed illumination are optical priors; they do not add forces, obstacles or measured biological detail.',
  'Analytic ray differentials project a one-pixel footprint onto the hit tangent plane; sinusoid texture components are box-filtered and smoothly removed before their local frequency reaches pixel Nyquist. Geometry edges are not supersampled.',
  'No self-occlusion, wing rendering, shadows, refraction or inter-fly geometry; this renderer is for one fly.'
]);

export function validateRetinalSettings(value={}){
  const s={...RETINAL_DEFAULTS,...value};
  for(const k of ['width','height'])if(!Number.isInteger(s[k])||s[k]<8||s[k]>1024)throw new Error(`Invalid retinal ${k}`);
  for(const k of ['horizontalFovDegrees','verticalFovDegrees'])if(!Number.isFinite(s[k])||s[k]<=0||s[k]>=180)throw new Error(`Invalid retinal ${k}`);
  if(!Number.isFinite(s.eyeYawDegrees)||Math.abs(s.eyeYawDegrees)>180||!(s.nearCm>0)||!(s.farCm>s.nearCm)||!Number.isFinite(s.farCm)||!(s.texturePeriodCm>0)||!Number.isFinite(s.texturePeriodCm))throw new Error('Invalid retinal camera settings');
  if(typeof s.textureAntialias!=='boolean')throw new Error('Invalid retinal textureAntialias');
  return Object.freeze(Object.fromEntries(Object.keys(RETINAL_DEFAULTS).map(k=>[k,s[k]])));
}
const vector=(v,n,label)=>{
  if(!v||v.length!==n||!Array.from(v).every(Number.isFinite))throw new Error(`Invalid retinal ${label}`);
  return Array.from(v);
};
function rotation(quaternion){
  let [w,x,y,z]=vector(quaternion,4,'quaternion');
  const norm=Math.hypot(w,x,y,z);if(norm<1e-8)throw new Error('Invalid retinal zero quaternion');
  w/=norm;x/=norm;y/=norm;z/=norm;
  return [1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)];
}
const rotate=(m,x,y,z)=>[m[0]*x+m[1]*y+m[2]*z,m[3]*x+m[4]*y+m[5]*z,m[6]*x+m[7]*y+m[8]*z];
// Nearest strictly positive quadratic root; also works for a linear equation.
function firstRoot(a,b,c,near,far){
  if(Math.abs(a)<EPS){const t=-c/b;return t>=near&&t<far?t:Infinity;}
  const d=b*b-4*a*c;if(d<0)return Infinity;
  const s=Math.sqrt(d),t0=(-b-s)/(2*a),t1=(-b+s)/(2*a);
  const lo=Math.min(t0,t1),hi=Math.max(t0,t1);
  return lo>=near&&lo<far?lo:hi>=near&&hi<far?hi:Infinity;
}
function sphereShape(center,radii,kind){return {center,radii,kind};}
function sceneGeometry(bowl,food){
  const f=bowl?.floor;
  if(!f||!Number.isFinite(f.baseCm)||!Number.isFinite(f.radialCoefficientPerCm)||f.radialCoefficientPerCm<0||!(bowl.radiusCm>.2)||!Number.isFinite(bowl.radiusCm)||!Number.isFinite(bowl.ceilingCm))throw new Error('Retina requires native bowl/floor/ceiling geometry');
  if(f.capRadiusCm!==undefined&&(!Number.isFinite(f.capRadiusCm)||f.capRadiusCm<=0))throw new Error('Invalid retinal floor cap');
  const floor={base:f.baseCm,k:f.radialCoefficientPerCm,cap:f.capRadiusCm??Infinity,extent:f.capRadiusCm===undefined?6.6:bowl.radiusCm+1};
  if(!Array.isArray(food))throw new Error('Retina requires a native food array');
  const fruits=food.map(item=>{
    const center=vector(item.position,3,'food position'),r=item.radiusCm;
    if(!(r>0)||!Number.isFinite(r)||!['apple','banana'].includes(item.kind))throw new Error('Invalid retinal fruit');
    if(item.kind==='apple')return {center,bound:r,shapes:[sphereShape(center,[r,r,.94*r],1)]};
    let points;
    if(item.path?.length>=2)points=item.path.map(p=>vector(p,3,'banana path'));
    else{
      const length=item.lengthCm,angle=item.rotation??0;
      if(!(length>0)||!Number.isFinite(length)||!Number.isFinite(angle))throw new Error('Invalid retinal banana length/rotation');
      points=Array.from({length:31},(_,i)=>{
        const t=i/30,x=(t-.5)*length,y=.9*(4*(t-.5)**2-1);
        return [center[0]+Math.cos(angle)*x-Math.sin(angle)*y,center[1]+Math.sin(angle)*x+Math.cos(angle)*y,center[2]];
      });
    }
    const shapes=[];let bound=0;
    for(const p of points)bound=Math.max(bound,Math.hypot(...p.map((v,i)=>v-center[i]))+Math.max(r,.25));
    for(let i=1;i<points.length;i++){
      const a=points[i-1],b=points[i],d=b.map((v,j)=>v-a[j]),length=Math.hypot(...d);
      if(length>EPS)shapes.push({a,axis:d.map(v=>v/length),length,r,kind:2});
    }
    for(const p of [points[0],points.at(-1)])shapes.push(sphereShape(p,[.25,.25,.25],3));
    return {center,bound,shapes};
  });
  return {floor,fruits,wallRadius:bowl.radiusCm-.15,ceiling:bowl.ceilingCm+.15};
}
function ellipsoid(o,d,s,near,far,hit){
  const x=(o[0]-s.center[0])/s.radii[0],y=(o[1]-s.center[1])/s.radii[1],z=(o[2]-s.center[2])/s.radii[2],
    dx=d[0]/s.radii[0],dy=d[1]/s.radii[1],dz=d[2]/s.radii[2];
  const t=firstRoot(dx*dx+dy*dy+dz*dz,2*(x*dx+y*dy+z*dz),x*x+y*y+z*z-1,near,far);
  if(t===Infinity)return far;
  hit[0]=(x+dx*t)/s.radii[0];hit[1]=(y+dy*t)/s.radii[1];hit[2]=(z+dz*t)/s.radii[2];
  return t;
}
function cylinder(o,d,s,near,far,hit){
  const px=o[0]-s.a[0],py=o[1]-s.a[1],pz=o[2]-s.a[2],u=s.axis,
    along=px*u[0]+py*u[1]+pz*u[2],speed=d[0]*u[0]+d[1]*u[1]+d[2]*u[2];
  const a=1-speed*speed,b=2*(px*d[0]+py*d[1]+pz*d[2]-along*speed),c=px*px+py*py+pz*pz-along*along-s.r*s.r;
  let best=far;
  const disc=b*b-4*a*c;
  if(a>EPS&&disc>=0)for(const t of [(-b-Math.sqrt(disc))/(2*a),(-b+Math.sqrt(disc))/(2*a)]){
    const h=along+t*speed;
    if(t>=near&&t<best&&h>=0&&h<=s.length){best=t;hit[0]=px+t*d[0]-h*u[0];hit[1]=py+t*d[1]-h*u[1];hit[2]=pz+t*d[2]-h*u[2];}
  }
  if(Math.abs(speed)>EPS)for(const h of [0,s.length]){
    const t=(h-along)/speed;if(t<near||t>=best)continue;
    const x=px+t*d[0]-h*u[0],y=py+t*d[1]-h*u[1],z=pz+t*d[2]-h*u[2];
    if(x*x+y*y+z*z<=s.r*s.r){best=t;const sign=h===0?-1:1;hit[0]=sign*u[0];hit[1]=sign*u[1];hit[2]=sign*u[2];}
  }
  return best;
}
function trace(o,d,scene,settings,out,normal,scratch){
  const near=settings.nearCm;let best=settings.farCm,kind=0;
  const f=scene.floor,tryFloor=t=>{
    if(!(t>=near&&t<best))return;
    const x=o[0]+t*d[0],y=o[1]+t*d[1],r2=x*x+y*y;
    if(Math.abs(x)>f.extent||Math.abs(y)>f.extent||r2>f.cap*f.cap+1e-8)return;
    best=t;kind=4;normal[0]=-2*f.k*x;normal[1]=-2*f.k*y;normal[2]=1;
  };
  const a=-f.k*(d[0]*d[0]+d[1]*d[1]),b=d[2]-2*f.k*(o[0]*d[0]+o[1]*d[1]),c=o[2]-f.base-f.k*(o[0]*o[0]+o[1]*o[1]);
  if(Math.abs(a)<EPS)tryFloor(-c/b);
  else{const disc=b*b-4*a*c;if(disc>=0){const root=Math.sqrt(disc);tryFloor((-b-root)/(2*a));tryFloor((-b+root)/(2*a));}}
  if(Number.isFinite(f.cap)&&Math.abs(d[2])>EPS){
    const t=(f.base+f.k*f.cap*f.cap-o[2])/d[2],x=o[0]+t*d[0],y=o[1]+t*d[1];
    if(t>=near&&t<best&&x*x+y*y>=f.cap*f.cap&&Math.abs(x)<=f.extent&&Math.abs(y)<=f.extent){best=t;kind=4;normal[0]=normal[1]=0;normal[2]=1;}
  }
  const wall=firstRoot(d[0]*d[0]+d[1]*d[1],2*(o[0]*d[0]+o[1]*d[1]),o[0]*o[0]+o[1]*o[1]-scene.wallRadius**2,near,best);
  if(wall<best&&o[2]+wall*d[2]>=-1&&o[2]+wall*d[2]<=scene.ceiling){best=wall;kind=5;normal[0]=-(o[0]+wall*d[0]);normal[1]=-(o[1]+wall*d[1]);normal[2]=0;}
  const ceiling=(scene.ceiling-o[2])/d[2];
  if(ceiling>=near&&ceiling<best){best=ceiling;kind=6;normal[0]=normal[1]=0;normal[2]=-1;}
  for(const fruit of scene.fruits){
    const x=o[0]-fruit.center[0],y=o[1]-fruit.center[1],z=o[2]-fruit.center[2];
    if(x*x+y*y+z*z>fruit.bound*fruit.bound&&firstRoot(1,2*(x*d[0]+y*d[1]+z*d[2]),x*x+y*y+z*z-fruit.bound*fruit.bound,near,best)===Infinity)continue;
    for(const s of fruit.shapes){
      const t=s.radii?ellipsoid(o,d,s,near,best,scratch):cylinder(o,d,s,near,best,scratch);
      if(t<best){best=t;kind=s.kind;normal[0]=scratch[0];normal[1]=scratch[1];normal[2]=scratch[2];}
    }
  }
  out[0]=best;out[1]=kind;
}
// Static surface detail has no simulation-time argument and cannot create
// apparent motion while the body and environment remain fixed.
function colorAt(kind,x,y,z,nx,ny,nz,period,out){
  let r,g,b,texture;
  if(kind===4){
    const coarse=Math.sin(x/period*6.283)*Math.sin(y/period*6.283),fine=Math.sin((x*1.71+y*.63)/period*15.2)*Math.cos((y*1.37-x*.43)/period*13.7);
    texture=.78+.17*coarse+.07*fine;[r,g,b]=[130,142,132];
  }else if(kind===1){texture=.82+.13*Math.sin(19*x+11*y+7*z)+.05*Math.sin(51*y-23*z);[r,g,b]=[189,64,43];}
  else if(kind===2){texture=.87+.10*Math.sin(21*x-16*y+11*z)+.03*Math.cos(59*y+43*z);[r,g,b]=[217,176,56];}
  else if(kind===3){texture=.85+.12*Math.sin(43*x+29*y+17*z);[r,g,b]=[88,70,34];}
  else if(kind===5){texture=.77+.12*Math.sin(Math.atan2(y,x)*29)*Math.cos(z*3.7)+.06*Math.sin(z*13.1+x*2);[r,g,b]=[151,163,170];}
  else if(kind===6){texture=.82+.12*Math.sin(x*1.3)*Math.cos(y*1.7);[r,g,b]=[172,182,185];}
  else{out[0]=167;out[1]=184;out[2]=199;return;}
  const length=Math.hypot(nx,ny,nz)||1,light=.67+.33*Math.max(0,(.25*nx-.35*ny+.9027735*nz)/length),v=texture*light;
  out[0]=clamp(Math.round(r*v),0,255);out[1]=clamp(Math.round(g*v),0,255);out[2]=clamp(Math.round(b*v),0,255);
}

/** Footprint response to a sinusoid with radian phase change dx/dy per pixel.
 * Exact box averaging for a locally linear phase, plus a smooth low-pass
 * transition from half-Nyquist to Nyquist. The hard zero above Nyquist avoids
 * the distant moire that a box/sinc filter's repeated side lobes would retain.
 */
export function textureFootprintGain(dx,dy){
  const frequency=Math.max(Math.abs(dx),Math.abs(dy));
  if(!Number.isFinite(frequency)||frequency>=Math.PI)return 0;
  const t=clamp((frequency-Math.PI*.5)/(Math.PI*.5),0,1),rolloff=1-t*t*(3-2*t),
    sx=Math.abs(dx)<1e-7?1:Math.sin(dx*.5)/(dx*.5),sy=Math.abs(dy)<1e-7?1:Math.sin(dy*.5)/(dy*.5);
  return rolloff*sx*sy;
}
/** dP/dpixel = t[dD/dpixel - D (N.dD/dpixel)/(N.D)]. Normal scale cancels.
 * No neighbouring ray is intersected; all derivatives use the existing hit.
 */
export function surfacePixelFootprint(distance,direction,normal,du,dv,out=new Float64Array(6)){
  const nd=normal[0]*direction[0]+normal[1]*direction[1]+normal[2]*direction[2];
  const den=Math.abs(nd)<1e-9?(nd<0?-1e-9:1e-9):nd,
    u=(normal[0]*du[0]+normal[1]*du[1]+normal[2]*du[2])/den,
    v=(normal[0]*dv[0]+normal[1]*dv[1]+normal[2]*dv[2])/den;
  for(let k=0;k<3;k++){out[k]=distance*(du[k]-direction[k]*u);out[k+3]=distance*(dv[k]-direction[k]*v);}
  return out;
}
function filteredWave(phase,kx,ky,kz,footprint,cosine=false){
  const gain=textureFootprintGain(kx*footprint[0]+ky*footprint[1]+kz*footprint[2],kx*footprint[3]+ky*footprint[4]+kz*footprint[5]);
  return gain===0?0:gain*(cosine?Math.cos(phase):Math.sin(phase));
}
function filteredColorAt(kind,x,y,z,nx,ny,nz,period,footprint,out){
  let r,g,b,texture;
  if(kind===4){
    const k=6.283/period,ax=1.71*15.2/period,ay=.63*15.2/period,bx=-.43*13.7/period,by=1.37*13.7/period,
      coarse=.5*(filteredWave(k*(x-y),k,-k,0,footprint,true)-filteredWave(k*(x+y),k,k,0,footprint,true)),
      fine=.5*(filteredWave((ax+bx)*x+(ay+by)*y,ax+bx,ay+by,0,footprint)+filteredWave((ax-bx)*x+(ay-by)*y,ax-bx,ay-by,0,footprint));
    texture=.78+.17*coarse+.07*fine;[r,g,b]=[130,142,132];
  }else if(kind===1){texture=.82+.13*filteredWave(19*x+11*y+7*z,19,11,7,footprint)+.05*filteredWave(51*y-23*z,0,51,-23,footprint);[r,g,b]=[189,64,43];}
  else if(kind===2){texture=.87+.10*filteredWave(21*x-16*y+11*z,21,-16,11,footprint)+.03*filteredWave(59*y+43*z,0,59,43,footprint,true);[r,g,b]=[217,176,56];}
  else if(kind===3){texture=.85+.12*filteredWave(43*x+29*y+17*z,43,29,17,footprint);[r,g,b]=[88,70,34];}
  else if(kind===5){
    const theta=29*Math.atan2(y,x),r2=Math.max(1e-12,x*x+y*y),kx=-29*y/r2,ky=29*x/r2;
    texture=.77+.06*(filteredWave(theta+z*3.7,kx,ky,3.7,footprint)+filteredWave(theta-z*3.7,kx,ky,-3.7,footprint))+.06*filteredWave(z*13.1+x*2,2,0,13.1,footprint);[r,g,b]=[151,163,170];
  }else if(kind===6){texture=.82+.06*(filteredWave(x*1.3+y*1.7,1.3,1.7,0,footprint)+filteredWave(x*1.3-y*1.7,1.3,-1.7,0,footprint));[r,g,b]=[172,182,185];}
  else{out[0]=167;out[1]=184;out[2]=199;return;}
  const length=Math.hypot(nx,ny,nz)||1,light=.67+.33*Math.max(0,(.25*nx-.35*ny+.9027735*nz)/length),v=texture*light;
  out[0]=clamp(Math.round(r*v),0,255);out[1]=clamp(Math.round(g*v),0,255);out[2]=clamp(Math.round(b*v),0,255);
}

export function createRetinalSensor(options={}){
  const settings=validateRetinalSettings(options),{width,height}=settings,n=width*height,
    rays=new Float64Array(2*n*3),rayDu=new Float64Array(2*n*3),rayDv=new Float64Array(2*n*3),
    offsets=[[.08,.03,-.002],[.08,-.03,-.002]],hStep=settings.horizontalFovDegrees*RAD/(width-1),vStep=settings.verticalFovDegrees*RAD/(height-1);
  for(let eye=0;eye<2;eye++)for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    const yaw=(eye===0?1:-1)*settings.eyeYawDegrees*RAD-(x/(width-1)-.5)*settings.horizontalFovDegrees*RAD,
      elevation=(.5-y/(height-1))*settings.verticalFovDegrees*RAD,i=(eye*n+y*width+x)*3;
    rays[i]=Math.cos(elevation)*Math.cos(yaw);rays[i+1]=Math.cos(elevation)*Math.sin(yaw);rays[i+2]=Math.sin(elevation);
    rayDu[i]=Math.cos(elevation)*Math.sin(yaw)*hStep;rayDu[i+1]=-Math.cos(elevation)*Math.cos(yaw)*hStep;
    rayDv[i]=Math.sin(elevation)*Math.cos(yaw)*vStep;rayDv[i+1]=Math.sin(elevation)*Math.sin(yaw)*vStep;rayDv[i+2]=-Math.cos(elevation)*vStep;
  }
  let sequence=0;
  return {settings,axes:RETINAL_AXES,assumptions:RETINAL_ASSUMPTIONS,
    render({position,quaternion,bodyTime,bowl,food}){
      const p=vector(position,3,'root position'),m=rotation(quaternion);
      if(!Number.isFinite(bodyTime)||bodyTime<0)throw new Error('Invalid retinal bodyTime');
      const scene=sceneGeometry(bowl,food),pixels=new Uint8Array(2*n),rgb=new Uint8Array(6*n),
        hitCounts=new Uint32Array(7),normal=[0,0,1],scratch=[0,0,0],hit=[0,0],color=[0,0,0],d=[0,0,0],du=[0,0,0],dv=[0,0,0],footprint=new Float64Array(6),eyes=[];
      for(let eye=0;eye<2;eye++){
        const o=rotate(m,...offsets[eye]).map((v,i)=>v+p[i]),forward=rotate(m,Math.cos(settings.eyeYawDegrees*RAD),(eye===0?1:-1)*Math.sin(settings.eyeYawDegrees*RAD),0);
        eyes.push({side:eye===0?'left':'right',originCm:o,forwardWorld:forward});
        for(let j=0;j<n;j++){
          const i=eye*n+j,k=i*3,x=rays[k],y=rays[k+1],z=rays[k+2];
          d[0]=m[0]*x+m[1]*y+m[2]*z;d[1]=m[3]*x+m[4]*y+m[5]*z;d[2]=m[6]*x+m[7]*y+m[8]*z;
          trace(o,d,scene,settings,hit,normal,scratch);hitCounts[hit[1]]++;
          if(settings.textureAntialias&&hit[1]!==0){
            for(let row=0;row<3;row++){du[row]=m[row*3]*rayDu[k]+m[row*3+1]*rayDu[k+1]+m[row*3+2]*rayDu[k+2];dv[row]=m[row*3]*rayDv[k]+m[row*3+1]*rayDv[k+1]+m[row*3+2]*rayDv[k+2];}
            surfacePixelFootprint(hit[0],d,normal,du,dv,footprint);
            filteredColorAt(hit[1],o[0]+hit[0]*d[0],o[1]+hit[0]*d[1],o[2]+hit[0]*d[2],...normal,settings.texturePeriodCm,footprint,color);
          }else colorAt(hit[1],o[0]+hit[0]*d[0],o[1]+hit[0]*d[1],o[2]+hit[0]*d[2],...normal,settings.texturePeriodCm,color);
          rgb[k]=color[0];rgb[k+1]=color[1];rgb[k+2]=color[2];pixels[i]=Math.round(.2126*color[0]+.7152*color[1]+.0722*color[2]);
        }
      }
      return {width,height,pixels,rgb,sequence:sequence++,bodyTime,eyes,
        horizontalFovDegrees:settings.horizontalFovDegrees,verticalFovDegrees:settings.verticalFovDegrees,
        summary:{width,height,rays:2*n,textureAntialias:settings.textureAntialias,hitCounts:Array.from(hitCounts),axes:RETINAL_AXES,assumptions:RETINAL_ASSUMPTIONS}};
    },
    reset(){sequence=0;}
  };
}
