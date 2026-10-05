import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import * as THREE from '../web/vendor/three.module.js';
import loadMujoco from '../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {createHabitat} from '../reports/flybody-onset-causal/sources/body-world.js';
import {originalFruitGeometry,originalBowlGeometry} from './original-fruit-geometry.mjs';

const dir='reports/flybody-onset-causal',capture=JSON.parse(await fs.readFile(`${dir}/capture.json`,'utf8'));
const mechanics=JSON.parse(await fs.readFile(`${dir}/mechanics.json`,'utf8')),baseline=mechanics.cases.find(c=>c.mode==='baseline');
assert(mechanics.baselineGate.passed&&capture.capturePassed);
const habitat=createHabitat(capture.scene.fruit),visual=originalFruitGeometry(capture.scene.fruit,{decorations:true}),bowl=originalBowlGeometry({reverse:true});
const objects=[...visual.solids,bowl],ray=new THREE.Raycaster(),from=new THREE.Vector3(),point=new THREE.Vector3(),closest=new THREE.Vector3();
const triangles=objects.map(mesh=>{
  const g=mesh.geometry,p=g.attributes.position,index=g.index,rows=[];
  for(let i=0;i<(index?.count??p.count);i+=3){
    const vertices=[0,1,2].map(k=>new THREE.Vector3().fromBufferAttribute(p,index?index.getX(i+k):i+k).applyMatrix4(mesh.matrixWorld));
    rows.push(new THREE.Triangle(...vertices));
  }
  return {fruitIndex:mesh.userData.fruitIndex??-1,triangles:rows};
});
function nearestVisible(position){
  point.fromArray(position);let distance=Infinity,result;
  for(const object of triangles)for(const triangle of object.triangles){triangle.closestPointToPoint(point,closest);const d=closest.distanceTo(point);
    if(d<distance){distance=d;result={distance:d,fruitIndex:object.fruitIndex,point:closest.toArray()};}}
  return result;
}
function vertical(meshes,x,z,start=100,direction=-1){
  from.set(x,start,z);ray.set(from,new THREE.Vector3(0,direction,0));
  return ray.intersectObjects(meshes,false).map(h=>({height:h.point.y,fruitIndex:h.object.userData.fruitIndex??-1}));
}
const selected=[];
const first=baseline.trace.find(t=>t.contacts.some(c=>c.category==='wing_environment'&&c.forceContactFrame[0]>0));assert(first);
first.contacts.filter(c=>c.category==='wing_environment'&&c.forceContactFrame[0]>0).forEach((c,i)=>selected.push({...c,event:`first_impact_${i+1}`,time:first.time}));
selected.push({...baseline.peaks.wingEnvironmentNormalForce,event:'maximum_wing_normal_force'});
for(const event of ['angularSpeed','fluidRootForce']){
  const peak=baseline.peaks[event];
  peak.contacts.filter(c=>c.category==='wing_environment'&&c.forceContactFrame[0]>0).forEach((c,i)=>selected.push({...c,event:`${event}_peak_contact_${i+1}`,time:peak.time}));
}
const mj=await loadMujoco(),model=mj.MjModel.from_xml_string(capture.scene.xml);model.hfield_data.set(capture.scene.heights);
const data=new mj.MjData(model),normal=new mj.DoubleBuffer(3);data.qpos.set(capture.initial.native.qpos);mj.mj_forward(model,data);
const rows=[];
try{for(const c of selected){
  assert(c.geoms[0]===0&&c.normal?.length===3);
  // MuJoCo position is the midpoint of the two nearest surface points.
  // Its normal points from geom0 (heightfield) to geom1 (wing).
  const nativeEnvironmentPoint=c.position.map((v,i)=>v-.5*c.distance*c.normal[i]);
  const [x,z,y]=nativeEnvironmentPoint.map(v=>v*10),scenePosition=[x,y,z],analytic=habitat.surface(x,z);
  const nativeDistance=mj.mj_rayHfield(model,data,0,[x/10,z/10,10],[0,0,-1],normal);assert(nativeDistance>=0);
  const nativeHeight=(10-nativeDistance)*10,nativeNormal=Array.from(normal.GetView());
  const tops=vertical(objects,x,z),renderTop=tops[0],undersides=vertical(visual.solids,x,z,-100,1);
  const fruitIntervals=visual.solids.flatMap(mesh=>{
    const upper=vertical([mesh],x,z)[0],lower=vertical([mesh],x,z,-100,1)[0];
    return upper&&lower?[{fruitIndex:mesh.userData.fruitIndex,lower:lower.height,upper:upper.height}]:[];
  });
  const nearest=nearestVisible(scenePosition),spacing=132/256,gx=Math.floor((x+66)/spacing),gz=Math.floor((z+66)/spacing);
  const gridVertices=[0,1].flatMap(dx=>[0,1].map(dz=>{const vx=-66+(gx+dx)*spacing,vz=-66+(gz+dz)*spacing;return {x:vx,z:vz,...habitat.surface(vx,vz)};}));
  const boundary=gridVertices.some(v=>v.fruitIndex!==gridVertices[0].fruitIndex),belowFruit=fruitIntervals.filter(f=>y<f.lower-.1);
  const insideFruit=fruitIntervals.filter(f=>y>=f.lower-.1&&y<=f.upper+.1);
  const gridAlignment=[(x+66)/spacing,(z+66)/spacing].map(v=>Math.abs(v-Math.round(v)));
  const normalDotTop=c.normal.reduce((s,v,i)=>s+v*nativeNormal[i],0);
  let classification;
  if(nearest.distance<.03&&nearest.fruitIndex===-1)classification='visible_bowl_surface';
  else if(belowFruit.length&&insideFruit.length===0)classification='invisible_fill_below_visible_fruit';
  else if(boundary&&insideFruit.length===0&&nearest.distance>.3)classification='heightfield_footprint_cliff_in_visible_air';
  else if(analytic.fruitIndex>=0&&nearest.distance<.3)classification='near_visible_fruit_surface_with_approximation_gap';
  else classification='requires_individual_geometry_review';
  const axisNormal=Math.max(Math.abs(c.normal[0]),Math.abs(c.normal[1]))>1-1e-6&&Math.abs(c.normal[2])<1e-5;
  const internalGridSide=axisNormal&&Math.min(...gridAlignment)<1e-6&&!boundary;
  rows.push({...c,wingName:mechanics.wingCollisionGeoms.find(g=>c.geoms.includes(g.id))?.name,nativeEnvironmentPoint,scenePosition,
    analytic,nativeHeight,nativeTopNormal:nativeNormal,normalDotTop,environmentPointBelowNativeTop:nativeHeight-y,
    renderTop,undersides,fruitIntervals,nearestVisible:nearest,gridVertices,gridAlignment,internalGridSide,
    classification,nativeMinusRenderedTop:nativeHeight-renderTop.height,
    nearestFruitVerticalGap:belowFruit.length?Math.min(...belowFruit.map(f=>f.lower-y)):null});
}}finally{normal.delete();data.delete();model.delete();visual.dispose();bowl.geometry.dispose();bowl.material.dispose();}
const hashes=Object.fromEntries(await Promise.all([`${dir}/capture.json`,`${dir}/mechanics.json`,`${dir}/sources/body-world.js`,`${dir}/sources/app.js`,'scripts/original-fruit-geometry.mjs'].map(async path=>[path,createHash('sha256').update(await fs.readFile(path)).digest('hex')])));
const report={date:new Date().toISOString(),scope:'Read-only classification of selected recorded wing-heightfield contacts; original captured physics scene is compiled only for static rays. No replay, controller, collision or UI changes.',
  units:'Native coordinates are centimeters in [x,y,z-up]. Scene coordinates are millimeters in [x,y-up,z]. Forces retain recorded g cm/s^2 units.',
  environmentPoint:'position minus 0.5 * signedDistance * normal; contact.position is the surface-point midpoint and normal points from ground geom0 toward wing geom1.',
  timing:mechanics.sampleTiming,selection:'Both contacts at first positive wing impact, maximum wing normal force, and positive wing contacts at peak angular speed and peak fluid root force. This is not an impulse-weighted census of the complete trajectory.',
  findings:['The first wing impact occurs after the first overturn; this audit does not identify the cause of that overturn.',
    'Selected contacts include near-top banana contacts, actual bowl contact, and later invisible heightfield fill/cliff contacts.',
    'An almost horizontal first-contact normal aligns with an internal heightfield grid plane, although neighboring height samples all belong to the same banana. This is a separate prism-seam concern, not a visible fruit silhouette.',
    'The largest normal-force contact lies on the real bowl. Large amplification from a real collision is not by itself evidence of a collision bug.',
    'The recorded behavior still fails: the fly overturns before these impacts. No behavioral success or correction has been demonstrated by this geometry audit.'],
  nextMechanicsBoundary:'In a separate matched replay, compare surface-consistent fruit collision volumes while retaining legitimate wing-bowl contacts and explicitly checking internal heightfield seam contacts. Independently investigate wing actuation and fluid coupling before first impact. A terrain correction alone is not established as a cure for the initial overturn.',
  limitations:['Scene contact midpoints alone cannot determine impulse causality. The independent matched mechanics replay is the source for intervention effects.',
    'Ray and closest-triangle checks use original main fruit surfaces and corrected original bowl winding. Decorative spots, mold and stems are excluded from nearest-solid distances; they do not explain the selected large gaps.',
    'Banana TubeGeometry has open end rings, so vertical intervals describe the rendered cross-section rather than asserting a closed global volume. Selected under-fruit cases are away from endpoints.',
    'Some contact normals can reflect edges/features. Internal-grid alignment is evidence of a collision representation artifact, not an assertion that changing terrain alone cures the unstable wing drive.'],
  sourceSha256:hashes,rows};
await fs.writeFile(`${dir}/contact-geometry.json`,JSON.stringify(report,null,2)+'\n');
const fmt=n=>Number(n).toFixed(5);
const lines=['# Recorded wing-contact geometry audit',report.scope,report.selection,
  '| Event | Time (s) | Normal force | Contact surface height | Rendered top | Nearest visible surface distance | Classification |',
  '|---|---:|---:|---:|---:|---:|---|',...rows.map(r=>`| ${r.event} | ${fmt(r.time)} | ${fmt(r.forceContactFrame[0])} | ${fmt(r.scenePosition[1])} | ${fmt(r.renderTop.height)} | ${fmt(r.nearestVisible.distance)} | ${r.classification} |`),
  'Heights and distances in the table are scene millimeters; force is g cm/s². Surface positions reconstruct the heightfield-side point from the recorded contact midpoint and penetration distance.',
  ...report.findings,report.nextMechanicsBoundary,...report.limitations];
await fs.writeFile(`${dir}/contact-geometry.md`,lines.join('\n\n').replace(/\|\n\n\|/g,'|\n|')+'\n');
console.log(JSON.stringify(rows.map(r=>({event:r.event,time:r.time,force:r.forceContactFrame[0],classification:r.classification,point:r.scenePosition,
  nativeHeight:r.nativeHeight,renderTop:r.renderTop,nearest:r.nearestVisible.distance,fruitIntervals:r.fruitIntervals,internalGridSide:r.internalGridSide,gridAlignment:r.gridAlignment,normalDotTop:r.normalDotTop})),null,2));
