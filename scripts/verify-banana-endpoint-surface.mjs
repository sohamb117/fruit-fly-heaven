import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import * as THREE from '../web/vendor/three.module.js';
import loadMujoco from '../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {createHabitat} from '../web/body-world.js';
import {createHabitat as beforeHabitat} from '../reports/observation-60min-20260913/source-snapshots/body-world-before-endpoint-fix.js';
import {flybodyScene} from '../web/flybody-physics.js';
import {originalFruitGeometry,originalBowlGeometry} from './original-fruit-geometry.mjs';

const {fruit}=JSON.parse(await fs.readFile('web/habitat.json','utf8'));
const before=beforeHabitat(fruit),after=createHabitat(fruit),visual=originalFruitGeometry(fruit,{decorations:true}),bowl=originalBowlGeometry({reverse:true});
const ray=new THREE.Raycaster(),mj=await loadMujoco(),points=[];
const frames=String(await fs.readFile('reports/observation-60min-20260913/frames.jsonl')).trim().split('\n').map(JSON.parse);
for(const [label,index] of [['frame150',150],['spawn',56]]){
  const frame=frames.find(f=>f.index===index);assert(frame);
  points.push({label,frame:index,x:frame.state.position[0],z:frame.state.position[2],physicalRootCm:frame.state.physicalRoot});
}
fruit.forEach((f,fruitIndex)=>{
  if(f.kind==='banana')for(const end of [0,f.path.length-1]){
    const p=f.path[end],near=f.path[end===0?1:end-1],length=Math.hypot(p[0]-near[0],p[1]-near[1]),dx=(p[0]-near[0])/length,dz=(p[1]-near[1])/length;
    for(const distance of [-2,1.5,3.5,4,5])points.push({label:distance<0?'tube_interior':distance<2.5?'tip':'removed_cap',fruitIndex,end,distance,x:p[0]+dx*distance,z:p[1]+dz*distance});
  }else for(const fraction of [0,.5,.9])points.push({label:'apple',fruitIndex,x:f.x+fraction*f.radius,z:f.z});
});
for(const p of points){
  ray.set(new THREE.Vector3(p.x,100,p.z),new THREE.Vector3(0,-1,0));
  const hit=ray.intersectObjects([...visual.all,bowl],false)[0];assert(hit);
  Object.assign(p,{before:before.surface(p.x,p.z),after:after.surface(p.x,p.z),renderHeight:hit.point.y,renderFruitIndex:hit.object.userData.fruitIndex??-1});
  const spacing=132/256,gx=Math.floor((p.x+66)/spacing),gz=Math.floor((p.z+66)/spacing);
  p.afterGridVertexFruitIndices=[...new Set([0,1].flatMap(dx=>[0,1].map(dz=>after.surface(-66+(gx+dx)*spacing,-66+(gz+dz)*spacing).fruitIndex)))];
}
const xml='<mujoco><asset /><worldbody><body name="probe"><freejoint/><geom name="probe" type="sphere" size=".015" mass=".001"/></body></worldbody></mujoco>';
const models=[];
for(const [label,habitat]of [['before',before],['after',after]]){
  const scene=flybodyScene(xml,habitat),model=mj.MjModel.from_xml_string(scene.xml);model.hfield_data.set(scene.heights);
  const data=new mj.MjData(model),normal=new mj.DoubleBuffer(3);mj.mj_forward(model,data);models.push({label,model,data});
  try{for(const p of points){const distance=mj.mj_rayHfield(model,data,0,[p.x/10,p.z/10,10],[0,0,-1],normal);assert(distance>=0);p[`${label}NativeHeight`]=(10-distance)*10;}}
  finally{normal.delete();}
}
const contact=(record,p,height)=>{const {model,data}=record;data.qpos.set([p.x/10,p.z/10,height/10+.015-.001,1,0,0,0]);data.qvel.fill(0);mj.mj_forward(model,data);return data.ncon;};
const checks=[];
for(const p of points.filter(p=>p.label==='removed_cap'&&p.before.fruitIndex>=0&&p.after.fruitIndex<0&&p.afterGridVertexFruitIndices.every(i=>i<0)&&p.beforeNativeHeight-p.afterNativeHeight>1)){
  const oldAtOld=contact(models[0],p,p.beforeNativeHeight),newAtOld=contact(models[1],p,p.beforeNativeHeight),newAtFloor=contact(models[1],p,p.afterNativeHeight);
  assert(oldAtOld>0&&newAtOld===0&&newAtFloor>0);
  assert.equal(p.renderFruitIndex,-1);assert.ok(Math.abs(p.afterNativeHeight-p.renderHeight)<.03,JSON.stringify(p));
  checks.push({x:p.x,z:p.z,fruitIndex:p.fruitIndex,end:p.end,distance:p.distance,oldAtOld,newAtOld,newAtFloor,removedHeight:p.beforeNativeHeight-p.afterNativeHeight});
}
assert(checks.length>=12);
for(const p of points.filter(p=>['frame150','spawn','apple','tube_interior'].includes(p.label)))assert.deepEqual(p.after,p.before);
const frame150=points.find(p=>p.label==='frame150');assert.equal(frame150.before.fruitIndex,-1);assert.equal(frame150.afterNativeHeight,frame150.beforeNativeHeight);
const denseGeometry={sampled:0,changed:0,maximumVisibleHeightAboveCorrected:0};
for(const f of fruit.filter(f=>f.kind==='banana')){
  const old=beforeHabitat([f]),corrected=createHabitat([f]),geometry=originalFruitGeometry([f],{decorations:true});
  for(const end of [0,f.path.length-1]){
    const p=f.path[end],near=f.path[end===0?1:end-1],length=Math.hypot(p[0]-near[0],p[1]-near[1]),dx=(p[0]-near[0])/length,dz=(p[1]-near[1])/length;
    for(let axial=-2;axial<=6;axial+=.25)for(let lateral=-6;lateral<=6;lateral+=.25){
      const x=p[0]+dx*axial-dz*lateral,z=p[1]+dz*axial+dx*lateral,y=corrected.surface(x,z).y;denseGeometry.sampled++;
      if(Math.abs(old.surface(x,z).y-y)<1e-8)continue;denseGeometry.changed++;
      ray.set(new THREE.Vector3(x,100,z),new THREE.Vector3(0,-1,0));const hit=ray.intersectObjects([...geometry.all,bowl],false)[0];assert(hit);
      denseGeometry.maximumVisibleHeightAboveCorrected=Math.max(denseGeometry.maximumVisibleHeightAboveCorrected,hit.point.y-y);
    }
  }
  geometry.dispose();
}
assert.ok(denseGeometry.maximumVisibleHeightAboveCorrected<.03,JSON.stringify(denseGeometry));
const neighboringFruitBoundaryControls=points.filter(p=>p.label==='removed_cap'&&p.after.fruitIndex<0&&p.afterGridVertexFruitIndices.some(i=>i>=0));
const maximumNeighborBoundaryError=Math.max(0,...neighboringFruitBoundaryControls.map(p=>Math.abs(p.afterNativeHeight-p.renderHeight)));
const report={date:new Date().toISOString(),scope:'Only createHabitat banana endpoint footprint changed. Original visual fruit geometry and the active observer are unchanged.',
  interpretation:'Frame150 root projects onto bare bowl beside an interior banana segment, not an endpoint cap. Ground contact points were not recorded, so this correction is not asserted to explain its supported posture.',
  implementation:'Retain raw nearest-segment projection. A projection beyond the first/last endpoint uses the existing visible radius-2.5 tip sphere instead of a full-radius rounded tube cap. Other nearest interior segments retain the tube radius.',
  limitations:['The native 257-point heightfield still approximates abrupt boundaries over one grid cell (0.515625 scene units). Controls adjacent to another fruit footprint are reported separately from bare-bowl contact tests.','Existing tube interpolation and low-poly sphere differences remain; no under-fruit voids or overhangs can be represented by a heightfield.','No motor or behavior parameters changed and no simulation reload was performed.'],
  frame150,points,contacts:checks,denseGeometry,neighboringFruitBoundaryControls,maximumNeighborBoundaryError,passed:true,
  sourceSha256:Object.fromEntries(await Promise.all(['web/body-world.js','web/habitat.json','reports/observation-60min-20260913/source-snapshots/body-world-before-endpoint-fix.js'].map(async path=>[path,createHash('sha256').update(await fs.readFile(path)).digest('hex')]))) };
await fs.writeFile('reports/banana-endpoint-surface-validation.json',JSON.stringify(report,null,2)+'\n');
const removed=points.filter(p=>p.label==='removed_cap'&&p.after.fruitIndex<0&&p.afterGridVertexFruitIndices.every(i=>i<0)),maxAfter=Math.max(...removed.map(p=>Math.abs(p.afterNativeHeight-p.renderHeight)));
await fs.writeFile('reports/banana-endpoint-surface-validation.md',`# Banana endpoint surface correction\n\n${report.scope}\n\n${report.interpretation}\n\nFrame 150 scene coordinates: (${frame150.x}, ${frame150.z}); native floor before/after ${frame150.beforeNativeHeight}, rendered floor ${frame150.renderHeight}.\n\n${report.implementation}\n\n${checks.length} isolated native sphere probes contacted the former phantom cap before the fix, had zero contacts at that old height after the fix, and contacted the actual lower bowl after the fix. Maximum corrected native/render height error across bare-bowl removed-cap controls: ${maxAfter} scene units. Spawn, apples, inner tube points and frame 150 retain exactly the same analytic surfaces.\n\nA separate ${denseGeometry.sampled}-point mesh-ray grid over all six endpoints found ${denseGeometry.changed} changed positions. The largest visible height above the corrected analytic surface was ${denseGeometry.maximumVisibleHeightAboveCorrected} scene units; the correction does not discard visible tube geometry. Endpoint-plane roundoff is covered by a focused regression test.\n\n${neighboringFruitBoundaryControls.length} other controls lie in grid cells crossing the adjacent apple footprint: their native/render height error remains as large as ${maximumNeighborBoundaryError} scene units. This is a separate unresolved heightfield boundary artifact, and those controls are not included in the bare-bowl accuracy claim.\n\n${report.limitations.join(' ')}\n`);
console.log(JSON.stringify({passed:true,contactChecks:checks.length,frame150,maxCorrectedNativeRenderError:maxAfter,denseGeometry,sourceSha256:report.sourceSha256},null,2));
models.forEach(({data,model})=>{data.delete();model.delete();});visual.dispose();bowl.geometry.dispose();bowl.material.dispose();
