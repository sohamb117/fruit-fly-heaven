import test from 'node:test';
import assert from 'node:assert/strict';
import {createRetinalSensor,validateRetinalSettings,textureFootprintGain,surfacePixelFootprint} from '../training/retinal-sensor.js';
import {createCompactVision,filterCompactVisionMapping} from '../training/compact-vision.js';

const bowl={radiusCm:6.5,floor:{baseCm:.15,radialCoefficientPerCm:.037},ceilingCm:6};
const pose={position:[0,0,2.5],quaternion:[1,0,0,0],bodyTime:0,bowl,food:[]};
const types=['T4a','T4b','T4c','T4d','T5a','T5b','T5c','T5d'];
function makeMapping(){
  const cells=[];
  for(const side of ['left','right'])for(const type of types)for(const u of [.2,.35,.5,.65,.8])for(const v of [.25,.5,.75])cells.push({index:cells.length,type,side,u,v,root_id:String(cells.length)});
  cells.push({index:cells.length,type:'T2',side:'left',u:.5,v:.5});
  return {schema_version:1,neuron_count:cells.length+1,cells,max_rate_hz:60};
}
const mapping=makeMapping(),mean=a=>a.reduce((s,v)=>s+v,0)/a.length;
const texture=(x,y)=>128+36*Math.sin(.23*x+.15*y)+31*Math.cos(.10*x-.33*y)+23*Math.sin(.51*x+.49*y);
function image(fn,{width=64,height=32,sequence=0,bodyTime=sequence*.02,right=fn}={}){
  const n=width*height,pixels=new Uint8Array(2*n);
  for(let side=0;side<2;side++)for(let y=0;y<height;y++)for(let x=0;x<width;x++)pixels[side*n+y*width+x]=Math.round(Math.max(0,Math.min(255,(side?right:fn)(x,y))));
  return {width,height,pixels,sequence,bodyTime,horizontalFovDegrees:150,verticalFovDegrees:120};
}
const meanType=(vision,type,side='left')=>mean(Array.from(vision.ratesHz).filter((_,i)=>vision.mapping.cells[i].type===type&&vision.mapping.cells[i].side===side));
const innerMean=(a,w,h,select=()=>true)=>{
  const values=[];for(let y=5;y<h-5;y++)for(let x=6;x<w-6;x++)if(select(x,y))values.push(a[y*w+x]);return mean(values);
};

test('native centimetres, root quaternion and true 256x128 independent eye rays',()=>{
  const a=createRetinalSensor(),b=createRetinalSensor(),frame=a.render(pose),copy=b.render(pose);
  assert.equal(frame.width,256);assert.equal(frame.height,128);assert.equal(frame.pixels.length,65536);assert.equal(frame.rgb.length,196608);
  assert.equal(frame.summary.rays,65536);assert.deepEqual(frame.pixels,copy.pixels);assert.deepEqual(frame.rgb,copy.rgb);
  assert.deepEqual(frame.eyes[0].originCm,[.08,.03,2.498]);assert.deepEqual(frame.eyes[1].originCm,[.08,-.03,2.498]);
  assert.notDeepEqual(frame.pixels.subarray(0,32768),frame.pixels.subarray(32768));
  // Not an 8x nearest-neighbour enlargement of the old 32x16 image.
  let detail=0;for(let y=75;y<120;y++)for(let x=20;x<230;x++)if(frame.pixels[y*256+x]!==frame.pixels[y*256+x+1])detail++;
  assert.ok(detail>2000,`only ${detail} high-resolution differences`);
  const turned=a.render({...pose,quaternion:[Math.SQRT1_2,0,0,Math.SQRT1_2]});
  assert.ok(Math.abs(turned.eyes[0].originCm[0]+.03)<1e-12);assert.ok(Math.abs(turned.eyes[0].originCm[1]-.08)<1e-12);
  assert.notDeepEqual(frame.pixels,turned.pixels);assert.ok(frame.summary.hitCounts[4]>0);
});

test('native food, height, and capped spacious floor change the actual retinal image',()=>{
  const sensor=createRetinalSensor({width:64,height:32}),empty=sensor.render(pose),
    food=[{kind:'apple',position:[1,1,2.5],radiusCm:.65}],apple=sensor.render({...pose,food}),
    moved=sensor.render({...pose,food:[{...food[0],position:[-1,1,2.5]}]}),
    high=sensor.render({...pose,position:[0,0,3.5]}),
    spacious=sensor.render({...pose,position:[8,0,3],bowl:{radiusCm:50,floor:{baseCm:.15,radialCoefficientPerCm:.037,capRadiusCm:6.5},ceilingCm:50}});
  assert.ok(apple.summary.hitCounts[1]>0);assert.notDeepEqual(empty.pixels,apple.pixels);assert.notDeepEqual(apple.pixels,moved.pixels);assert.notDeepEqual(empty.pixels,high.pixels);
  assert.ok(spacious.summary.hitCounts[4]>0);
  const banana=sensor.render({...pose,food:[{kind:'banana',position:[1,1,2.5],radiusCm:.2,lengthCm:2,rotation:.2}]});
  assert.ok(banana.summary.hitCounts[2]+banana.summary.hitCounts[3]>0);
  assert.throws(()=>sensor.render({...pose,quaternion:[0,0,0,0]}),/quaternion/);
  assert.throws(()=>validateRetinalSettings({width:0}),/width/);
});

test('texture antialiasing preserves geometry and ray count while filtering unresolved detail',()=>{
  const settings={width:64,height:32},scene={...pose,food:[
    {kind:'apple',position:[1,1,2.5],radiusCm:.65},
    {kind:'banana',position:[1,-1,2.5],radiusCm:.2,lengthCm:2,rotation:.2}]},
    disabled=createRetinalSensor({...settings,textureAntialias:false}).render(scene),
    enabled=createRetinalSensor({...settings,textureAntialias:true}).render(scene);
  assert.equal(disabled.summary.textureAntialias,false);assert.equal(enabled.summary.textureAntialias,true);
  assert.equal(enabled.summary.rays,2*settings.width*settings.height);assert.equal(enabled.summary.rays,disabled.summary.rays);
  assert.deepEqual(enabled.summary.hitCounts,disabled.summary.hitCounts);assert.deepEqual(enabled.eyes,disabled.eyes);
  assert.ok(enabled.summary.hitCounts[1]>0&&enabled.summary.hitCounts[2]>0&&enabled.summary.hitCounts[4]>0);
  assert.notDeepEqual(enabled.rgb,disabled.rgb);
  assert.deepEqual(createRetinalSensor({...settings,textureAntialias:true}).render(scene).rgb,enabled.rgb);
  assert.throws(()=>validateRetinalSettings({textureAntialias:'true'}),/textureAntialias/);
});

test('texture footprint cutoff preserves resolved frequencies and suppresses local Nyquist aliasing',()=>{
  assert.equal(textureFootprintGain(0,0),1);assert.ok(textureFootprintGain(.01,.01)>.9999);
  assert.equal(textureFootprintGain(Math.PI,0),0);assert.equal(textureFootprintGain(0,Math.PI*7),0);
  let previous=1;
  for(let i=0;i<=100;i++){
    const gain=textureFootprintGain(Math.PI*i/100,0);assert.ok(gain>=0&&gain<=previous+1e-12);previous=gain;
  }
  assert.equal(textureFootprintGain(.8,-1.2),textureFootprintGain(-.8,1.2));
});

test('surface pixel footprints match finite ray-plane derivatives and expand at grazing angles',()=>{
  const normal=[.2,-.1,1],direction=[.6,0,-.8],du=[0,.01,0],dv=[.008,0,.006],origin=[0,0,3],
    dot=(a,b)=>a.reduce((sum,x,i)=>sum+x*b[i],0),distance=-dot(normal,origin)/dot(normal,direction),
    footprint=surfacePixelFootprint(distance,direction,normal,du,dv),
    hit=ray=>{const t=-dot(normal,origin)/dot(normal,ray);return ray.map((x,i)=>origin[i]+t*x);};
  for(const [axis,derivative]of [du,dv].entries()){
    const epsilon=1e-4,a=hit(direction.map((x,i)=>x+epsilon*derivative[i])),b=hit(direction.map((x,i)=>x-epsilon*derivative[i]));
    for(let i=0;i<3;i++)assert.ok(Math.abs((a[i]-b[i])/(2*epsilon)-footprint[axis*3+i])<1e-7);
  }
  const front=surfacePixelFootprint(3,[0,0,-1],[0,0,1],[.01,0,0],[0,.01,0]),
    grazing=surfacePixelFootprint(30,[Math.sqrt(.99),0,-.1],[0,0,1],[.001,0,.01],[0,.01,0]);
  assert.ok(Math.hypot(...grazing.slice(0,3))>Math.hypot(...front.slice(0,3))*50);
});

test('one injection boundary retains only individually mapped T4/T5 cells',()=>{
  const filtered=filterCompactVisionMapping(mapping),vision=createCompactVision({mapping,width:64,height:32});
  assert.equal(filtered.cells.length,240);assert.equal(vision.indices.length,240);assert.ok(filtered.cells.every(c=>/^T[45][abcd]$/.test(c.type)));
  assert.deepEqual(Array.from(vision.indices),filtered.cells.map(c=>c.index));assert.equal(new Set(vision.indices).size,240);
  assert.equal(vision.summary.ready,false);assert.equal(vision.summary.assumptions.subtypeStatus.includes('NOT validated'),true);
  assert.throws(()=>filterCompactVisionMapping({...mapping,cells:[mapping.cells[0],mapping.cells[0]]}),/duplicate/);
});

test('signed camera horizontal and vertical motion reverse independently',()=>{
  for(const [dx,dy]of [[1,0],[-1,0],[0,-1],[0,1]]){
    const vision=createCompactVision({mapping,width:64,height:32});
    vision.update(image(texture));assert.equal(vision.summary.ready,false);assert.ok(vision.ratesHz.every(x=>x===0));
    vision.update(image((x,y)=>texture(x-dx,y-dy),{sequence:1}));
    const f=vision.fields[0],hx=innerMean(f.horizontal,64,32),vy=innerMean(f.vertical,64,32);
    assert.equal(vision.summary.ready,true);
    if(dx){assert.ok(hx*dx>1.2,`dx=${dx}, horizontal=${hx}`);assert.ok(Math.abs(vy)<.2);}
    if(dy){assert.ok(vy*(-dy)>2.3,`dy=${dy}, vertical=${vy}`);assert.ok(Math.abs(hx)<.2);}
    const preferred=dx>0?'a':dx<0?'b':dy<0?'c':'d',opposite={a:'b',b:'a',c:'d',d:'c'}[preferred];
    assert.ok(meanType(vision,'T4'+preferred)+meanType(vision,'T5'+preferred)>(meanType(vision,'T4'+opposite)+meanType(vision,'T5'+opposite))*5);
    assert.ok(vision.ratesHz.every(v=>Number.isFinite(v)&&v>=0&&v<=60));
  }
});

test('full-resolution refinement retains motion invisible to an old coarse raster',()=>{
  const vision=createCompactVision({mapping,width:256,height:128});
  const fine=(x,y)=>128+42*Math.sin(2*Math.PI*x/8)+32*Math.cos(2*Math.PI*y/8)+15*Math.sin(2*Math.PI*(x+y)/5);
  vision.update(image(fine,{width:256,height:128}));
  vision.update(image((x,y)=>fine(x-.5,y),{width:256,height:128,sequence:1}));
  assert.equal(vision.summary.fullResolutionSamples,65536);
  assert.ok(innerMean(vision.fields[0].horizontal,256,128)>.15);
  assert.ok(vision.ratesHz.some(v=>v>.1));
});

test('spatial expansion retains opposite local directions rather than an eye mean',()=>{
  const vision=createCompactVision({mapping,width:64,height:32});vision.update(image(texture));
  vision.update(image((x,y)=>texture((x-31.5)/1.035+31.5,(y-15.5)/1.035+15.5),{sequence:1}));
  const f=vision.fields[0];
  assert.ok(innerMean(f.horizontal,64,32,x=>x<22)<-.4);assert.ok(innerMean(f.horizontal,64,32,x=>x>42)>.4);
  assert.ok(innerMean(f.vertical,64,32,(_x,y)=>y<12)>.3);assert.ok(innerMean(f.vertical,64,32,(_x,y)=>y>20)<-.3);
  assert.ok(vision.summary.eyes[0].expansion>.2);
});

test('ON/OFF moving-edge channels remain distinct; opposite eye history is independent',()=>{
  for(const polarity of [1,-1]){
    const edge=t=>(x,_y)=>polarity>0?(x<t?220:35):(x<t?35:220),vision=createCompactVision({mapping,width:64,height:32});
    vision.update(image(edge(31),{right:()=>128}));vision.update(image(edge(32),{sequence:1,right:()=>128}));
    const on=meanType(vision,'T4a'),off=meanType(vision,'T5a');
    assert.ok(polarity>0?on>off+1:off>on+1,`polarity ${polarity}: ON ${on}, OFF ${off}`);
    assert.ok(vision.mapping.cells.every((c,i)=>c.side!=='right'||vision.ratesHz[i]===0));
  }
});

test('static input, causal time, immutable input bytes and episode reset',()=>{
  const vision=createCompactVision({mapping,width:64,height:32}),first=image(texture),bytes=first.pixels.slice();
  vision.update(first);vision.update(image(texture,{sequence:1}));assert.ok(vision.ratesHz.every(v=>v===0));assert.deepEqual(first.pixels,bytes);
  const serial=vision.serial;vision.update(image(texture,{sequence:1}));assert.equal(vision.serial,serial);
  assert.throws(()=>vision.update(image(texture,{sequence:2,bodyTime:0})),/time/);
  vision.update(image(texture,{sequence:2,bodyTime:1}));assert.equal(vision.summary.reason,'sampling_gap');assert.ok(vision.ratesHz.every(v=>v===0));
  vision.reset();vision.update(first);assert.equal(vision.summary.reason,'first_frame');
});
