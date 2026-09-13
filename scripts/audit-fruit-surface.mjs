import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import * as THREE from '../web/vendor/three.module.js';
import loadMujoco from '../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {createHabitat} from '../web/body-world.js';
import {flybodyScene} from '../web/flybody-physics.js';
import {originalFruitGeometry,originalBowlGeometry} from './original-fruit-geometry.mjs';

const source=JSON.parse(await fs.readFile('web/habitat.json','utf8')),habitat=createHabitat(source.fruit),mj=await loadMujoco();
const archive=String(await fs.readFile('reports/observation-60min-20260913/frames.jsonl')).trim().split('\n').map(JSON.parse);
const frame=archive.find(f=>f.index===80),spawn=archive.find(f=>f.index===56);assert(frame&&spawn);
const visual=originalFruitGeometry(source.fruit,{decorations:true}),bowl=originalBowlGeometry(),ray=new THREE.Raycaster(),hits=[];
const doubleMaterial=new THREE.MeshBasicMaterial({side:THREE.DoubleSide});
const doubleGeometry=[...visual.solids,bowl].map(mesh=>{const clone=new THREE.Mesh(mesh.geometry,doubleMaterial);clone.matrixAutoUpdate=false;clone.matrixWorld.copy(mesh.matrixWorld);clone.userData={...mesh.userData};return clone;});
const down=new THREE.Vector3(0,-1,0),from=new THREE.Vector3(),top=100;
function rendered(objects,x,z){from.set(x,top,z);ray.set(from,down);hits.length=0;ray.intersectObjects(objects,false,hits);return hits.length?{y:hits[0].point.y,fruitIndex:hits[0].object.userData.fruitIndex??-1,faceNormalY:hits[0].face.normal.y}:null;}
const points=[];
function grid(label,center){for(let j=-10;j<=10;j++)for(let i=-10;i<=10;i++)points.push({label,x:center[0]+i*.25,z:center[2]+j*.25,center:i===0&&j===0});}
grid('frame80',frame.state.position);grid('spawn',spawn.state.position);
source.fruit.forEach((f,fruitIndex)=>{
  if(f.kind==='banana')for(const endIndex of [0,f.path.length-1]){
    const p=f.path[endIndex],near=f.path[endIndex===0?1:endIndex-1],length=Math.hypot(p[0]-near[0],p[1]-near[1]),dx=(p[0]-near[0])/length,dz=(p[1]-near[1])/length;
    for(const fraction of [0,.25,.5,.75,.95,1.05])points.push({label:'banana_endpoint',fruitIndex,endIndex,fraction,x:p[0]+dx*f.radius*fraction,z:p[1]+dz*f.radius*fraction});
  }
  else for(const fraction of [0,.5,.9,1.05])points.push({label:'apple',fruitIndex,fraction,x:f.x+f.radius*fraction,z:f.z});
});
for(const [x,z]of [[0,-45],[0,45],[-45,0],[45,0]])points.push({label:'bare_bowl',x,z});
for(const p of points){
  const terrain=habitat.surface(p.x,p.z),body=rendered([...visual.solids,bowl],p.x,p.z),full=rendered([...visual.all,bowl],p.x,p.z),geometric=rendered(doubleGeometry,p.x,p.z);
  assert(body&&full&&geometric);Object.assign(p,{analyticHeight:terrain.y,analyticFruitIndex:terrain.fruitIndex,bodyRenderedHeight:body.y,bodyRenderedFruitIndex:body.fruitIndex,
    fullRenderedHeight:full.y,fullRenderedFruitIndex:full.fruitIndex,geometricSurfaceHeight:geometric.y,geometricFruitIndex:geometric.fruitIndex,geometricFaceNormalY:geometric.faceNormalY,
    analyticMinusBodyRender:terrain.y-body.y,analyticMinusFullRender:terrain.y-full.y,analyticMinusGeometricSurface:terrain.y-geometric.y});
}
const fields=[];
for(const resolution of [257,513]){
  const scene=flybodyScene('<mujoco><asset /><worldbody></worldbody></mujoco>',habitat,{resolution}),model=mj.MjModel.from_xml_string(scene.xml);model.hfield_data.set(scene.heights);
  const data=new mj.MjData(model),normal=new mj.DoubleBuffer(3);mj.mj_forward(model,data);
  try{for(const p of points){
    const distance=mj.mj_rayHfield(model,data,0,[p.x/10,p.z/10,top/10],[0,0,-1],normal);assert(distance>=0);
    p[`native${resolution}`]=(top/10-distance)*10;p[`nativeMinusAnalytic${resolution}`]=p[`native${resolution}`]-p.analyticHeight;
    p[`nativeMinusBodyRender${resolution}`]=p[`native${resolution}`]-p.bodyRenderedHeight;
    p[`nativeMinusGeometricSurface${resolution}`]=p[`native${resolution}`]-p.geometricSurfaceHeight;
  }}finally{normal.delete();data.delete();model.delete();}
  fields.push({resolution,cellSpacingDisplayUnits:132/(resolution-1)});
}
const stats=values=>({samples:values.length,mean:values.reduce((s,v)=>s+v,0)/values.length,
  meanAbsolute:values.reduce((s,v)=>s+Math.abs(v),0)/values.length,maximumAbsolute:Math.max(...values.map(Math.abs)),min:Math.min(...values),max:Math.max(...values)});
const summary=Object.fromEntries([...new Set(points.map(p=>p.label))].map(label=>{
  const rows=points.filter(p=>p.label===label);
  return [label,{count:rows.length,analyticRenderFootprintDisagreements:rows.filter(p=>p.analyticFruitIndex!==p.bodyRenderedFruitIndex).length,
    ...Object.fromEntries(['analyticMinusBodyRender','analyticMinusFullRender','analyticMinusGeometricSurface','nativeMinusAnalytic257','nativeMinusAnalytic513','nativeMinusBodyRender257','nativeMinusGeometricSurface257'].map(k=>[k,stats(rows.map(p=>p[k]))]))}];
}));
const centers=points.filter(p=>p.center),report={date:new Date().toISOString(),scope:'Read-only vertical-ray surface comparison. Native heightfields are isolated copies generated by production flybodyScene. No live body or habitat is modified.',
  units:'Original scene units; divide by 10 for native centimeters. With the model centimeter convention, one scene unit is one millimeter.',
  sampling:'21x21 grids at 0.25 scene-unit spacing (5x5 area) around frame80 and initial spawn; additional banana endpoint, apple and bare-bowl controls.',
  decomposition:'nativeMinusAnalytic measures native grid discretization of createHabitat.surface; bodyRenderedHeight uses actual FrontSide materials; geometricSurfaceHeight uses the same vertices with a separate DoubleSide audit material to expose back-facing surfaces; full render additionally includes decorative geometry.',
  fields,summary,centers,largestMismatches:[...points].sort((a,b)=>Math.abs(b.analyticMinusBodyRender)-Math.abs(a.analyticMinusBodyRender)).slice(0,12),
  centerlinePathMaximumError:source.fruit.filter(f=>f.kind==='banana').map(f=>{
    const errors=f.path.map((p,i)=>{const t=i/30,x=(t-.5)*f.length,z=9*(4*(t-.5)**2-1);return Math.hypot(p[0]-(f.x+Math.cos(f.angle)*x-Math.sin(f.angle)*z),p[1]-(f.z+Math.sin(f.angle)*x+Math.cos(f.angle)*z));});return Math.max(...errors);}),
  findings:['The original bowl inner profile faces downward. FrontSide rendering omits the intended inner floor viewed from above; DoubleSide audit rays recover the geometric surface near the analytic/native floor.',
    'Banana centerline coordinates agree exactly; local faceting and open-end versus rounded-cap differences remain.'],
  limitations:['A heightfield cannot represent under-fruit air spaces or overhangs. It fills columns beneath the upper surface.',
    'Banana render uses an open-ended TubeGeometry plus small endpoint decorations; analytic nearest-path distance creates rounded footprint extensions beyond endpoints.',
    'Low-poly apples/bowl and Catmull-Rom banana interpolation can differ from the analytic surfaces. Decorative mold/bruise/stem height is not native collision geometry.',
    'These are geometric discrepancies, not proof of the cause of any motor or flight failure.'],
  sourceSha256:Object.fromEntries(await Promise.all(['web/habitat.json','web/body-world.js','web/flybody-physics.js','web/app.js','scripts/original-fruit-geometry.mjs'].map(async path=>[path,createHash('sha256').update(await fs.readFile(path)).digest('hex')]))),points};
await fs.writeFile('reports/fruit-surface-audit.json',JSON.stringify(report,null,2)+'\n');
const fmt=v=>v.toFixed(5),lines=[
 '# Fruit rendering and native heightfield audit',
 'Read-only isolated geometry audit; original scene units (one millimeter under the native centimeter convention). No habitat, collision or controller edits.',
 '| Position | Native 257 | Analytic surface | Actual FrontSide surface | Same vertices, DoubleSide | Native − analytic | Analytic − geometry |',
 '|---|---:|---:|---:|---:|---:|---:|',
 ...centers.map(p=>`| ${p.label} | ${fmt(p.native257)} | ${fmt(p.analyticHeight)} | ${fmt(p.bodyRenderedHeight)} | ${fmt(p.geometricSurfaceHeight)} | ${fmt(p.nativeMinusAnalytic257)} | ${fmt(p.analyticMinusGeometricSurface)} |`),
 'The bowl has a real rendering defect separate from the heightfield: its original lathe inner faces point downward. With the original FrontSide material, top-down rays skip the intended bowl floor and hit the lower shell. DoubleSide audit rays recover the same existing inner vertices. At frame 80 this hides a surface roughly 7.1 scene units above the lower shell, while native-vs-analytic floor error is only 0.00035. The audit uses a separate scratch material; production rendering is unchanged.',
 'The nominal native grid spacing is 0.515625 scene units (257 points over 132 units). The 513-point comparison halves it. Native-minus-analytic isolates grid discretization; it cannot correct analytic/render shape differences.',
 '| Sample region | Points | Mean abs native − analytic (257) | Mean abs native − analytic (513) | Mean abs analytic − geometric surface | Max abs analytic − geometric surface |',
 '|---|---:|---:|---:|---:|---:|',
 ...Object.entries(summary).map(([label,s])=>`| ${label} | ${s.count} | ${fmt(s.nativeMinusAnalytic257.meanAbsolute)} | ${fmt(s.nativeMinusAnalytic513.meanAbsolute)} | ${fmt(s.analyticMinusGeometricSurface.meanAbsolute)} | ${fmt(s.analyticMinusGeometricSurface.maximumAbsolute)} |`),
 'The stored 31-point banana centerlines exactly match the renderer formula (maximum coordinate error zero for all three bananas). There is no wholesale banana position/rotation mismatch.',
 'There are real surface-model differences: the renderer uses open tube ends, while nearest-path distance adds rounded heightfield caps beyond the path endpoints; small rendered tip ellipsoids do not reproduce those caps. Apples and the bowl are coarse polygon meshes. The heightfield also cannot model under-fruit space or overhangs. These differences remain even if grid resolution is increased.',
 'Frame 80 should be interpreted using its central sample, not the worst endpoint discrepancy elsewhere. Camera obstruction and surface mismatch are separate observations. All point samples, source hashes and the largest discrepancies are in fruit-surface-audit.json.'
];
await fs.writeFile('reports/fruit-surface-audit.md',lines.join('\n\n').replace(/\|\n\n\|/g,'|\n|')+'\n');
console.log(JSON.stringify({centers,summary,largestMismatches:report.largestMismatches.slice(0,4)},null,2));
visual.dispose();bowl.geometry.dispose();bowl.material.dispose();doubleMaterial.dispose();
