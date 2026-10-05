import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import * as THREE from '../web/vendor/three.module.js';
import loadMujoco from '../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {createHabitat} from '../web/body-world.js';
import {createFlybodyHabitatCollision} from '../web/flybody-habitat-collision.js';
import {originalFruitGeometry,originalBowlGeometry} from './original-fruit-geometry.mjs';

const {fruit}=JSON.parse(await fs.readFile('web/habitat.json','utf8')),habitat=createHabitat(fruit);
const failure=JSON.parse(await fs.readFile('reports/flybody-onset-causal/contact-geometry.json','utf8'));
const helper=createFlybodyHabitatCollision(habitat),mj=await loadMujoco(),compileStart=performance.now();
// Deliberately reproduce FlyBody's inherited mesh scale, to catch double scaling.
const xml=`<mujoco><default><mesh scale=".1 .1 .1"/></default><asset>${helper.assets}</asset><worldbody>${helper.geoms}<body name="probe"><freejoint/><geom name="probe" type="sphere" size=".001" mass=".001"/></body></worldbody></mujoco>`;
const model=mj.MjModel.from_xml_string(xml);model.hfield_data.set(helper.heights);const compileMs=performance.now()-compileStart,data=new mj.MjData(model);
const geomFruit=new Map(Object.entries(helper.fruitGeomNames).map(([name,index])=>[mj.mj_name2id(model,5,name),index]));
assert.equal(model.nbody,2);for(const geom of geomFruit.keys())assert.equal(model.geom_bodyid[geom],0);
const probe=(position)=>{
  data.qpos.set([...position,1,0,0,0]);data.qvel.fill(0);mj.mj_forward(model,data);
  const contacts=data.ncon?data.contact:null,rows=[];
  try{for(let i=0;i<data.ncon;i++){const c=contacts.get(i);try{
    const geoms=Array.from(c.geom);rows.push({geoms,fruitIndices:geoms.filter(g=>geomFruit.has(g)).map(g=>geomFruit.get(g)),distance:c.dist});
  }finally{c.delete();}}}finally{contacts?.delete();}
  return rows;
};
const failures=failure.rows.map(row=>{
  const contacts=probe(row.nativeEnvironmentPoint),expectedContact=row.classification==='visible_bowl_surface';
  assert.equal(contacts.length>0,expectedContact,row.event);
  return {event:row.event,positionCm:row.nativeEnvironmentPoint,previousClassification:row.classification,expectedContact,contacts};
});
assert.equal(failures.length,17);

const visual=originalFruitGeometry(fruit,{decorations:true}),bowl=originalBowlGeometry({reverse:true}),ray=new THREE.Raycaster(),positive=[];
const samples=[];
fruit.forEach((f,index)=>{
  if(f.kind==='banana'){
    for(const pathIndex of [2,7,15,23,28])samples.push({kind:'banana',fruitIndex:index,x:f.path[pathIndex][0],z:f.path[pathIndex][1],meshes:[visual.solids[index]]});
    for(const end of [0,f.path.length-1]){
      const p=f.path[end],near=f.path[end===0?1:end-1],length=Math.hypot(p[0]-near[0],p[1]-near[1]);
      samples.push({kind:'tip',fruitIndex:index,x:p[0]+1.5*(p[0]-near[0])/length,z:p[1]+1.5*(p[1]-near[1])/length,
        meshes:visual.all.filter(m=>m.userData.fruitIndex===index)});
    }
  }else for(const fraction of [0,.4,.8])samples.push({kind:'apple',fruitIndex:index,x:f.x+f.radius*fraction,z:f.z,meshes:[visual.solids[index]]});
});
for(const sample of samples)for(const direction of [-1,1]){
  ray.set(new THREE.Vector3(sample.x,direction===-1?100:-100,sample.z),new THREE.Vector3(0,direction,0));
  const hit=ray.intersectObjects(sample.meshes,false)[0];assert(hit);
  const position=[hit.point.x/10,hit.point.z/10,hit.point.y/10],contacts=probe(position);
  assert.ok(contacts.some(c=>c.fruitIndices.includes(sample.fruitIndex)),JSON.stringify({sample:{...sample,meshes:undefined},direction,position,contacts}));
  positive.push({kind:sample.kind,fruitIndex:sample.fruitIndex,side:direction===-1?'upper':'lower',positionCm:position,contacts});
}
for(const [x,z]of [[0,0],[-45,0],[45,0],[0,-45],[0,45]]){
  const position=[x/10,z/10,(1.5+.0037*(x*x+z*z))/10],contacts=probe(position);
  assert.ok(contacts.some(c=>c.geoms.includes(0)));positive.push({kind:'bowl',positionCm:position,contacts});
}

// Native mesh rays independently compare the generated triangles with the
// original renderer, including empty air at former footprint cliffs.
const normal=new mj.DoubleBuffer(3),raySamples=[];
for(const sample of [...samples,...failure.rows.map(r=>({kind:r.event,x:r.scenePosition[0],z:r.scenePosition[2]}))]){
  // Exact sphere meridians can miss shared triangle edges in native mesh rays.
  // Offset both native/render rays equally; the contact probes above are exact.
  const x=sample.x+.0000137,z=sample.z+.0000211;
  ray.set(new THREE.Vector3(x,100,z),new THREE.Vector3(0,-1,0));const visible=ray.intersectObjects([...visual.solids,...visual.all.filter(m=>{
    const fi=m.userData.fruitIndex,f=fruit[fi];return f.kind==='banana'&&m!==visual.solids[fi]&&m.scale.x===2.5;
  }),bowl],false)[0];assert(visible);
  const origin=[x/10,z/10,10],direction=[0,0,-1];let best=mj.mj_rayHfield(model,data,0,origin,direction,normal),bestGeom=0;
  for(const geom of geomFruit.keys()){const d=mj.mj_rayMesh(model,data,geom,origin,direction,normal);if(d>=0&&(best<0||d<best)){best=d;bestGeom=geom;}}
  const nativeHeight=(10-best)*10,error=nativeHeight-visible.point.y;
  assert.ok(Math.abs(error)<.015,JSON.stringify({sample:{...sample,meshes:undefined},error,nativeHeight,visibleHeight:visible.point.y}));
  raySamples.push({kind:sample.kind,x,z,nativeHeight,visibleHeight:visible.point.y,error,nativeFruitIndex:geomFruit.get(bestGeom)??-1});
}
normal.delete();data.delete();model.delete();visual.dispose();bowl.geometry.dispose();bowl.material.dispose();

// Inserting static world geoms must preserve every native body and joint ID.
const flyXml=await fs.readFile('models/flybody-mujoco.xml','utf8'),base=mj.MjModel.from_xml_string(flyXml);
const compiled=mj.MjModel.from_xml_string(flyXml.replace('<asset />',`<asset>${helper.assets}</asset>`).replace('<worldbody>',`<worldbody>${helper.geoms}`));
const idParity={bodyCount:base.nbody,jointCount:base.njnt,sameBodyCount:base.nbody===compiled.nbody,sameJointCount:base.njnt===compiled.njnt};
assert(idParity.sameBodyCount&&idParity.sameJointCount);
assert.deepEqual(Array.from(base.body_parentid),Array.from(compiled.body_parentid));assert.deepEqual(Array.from(base.jnt_bodyid),Array.from(compiled.jnt_bodyid));
assert.deepEqual(Array.from(base.jnt_qposadr),Array.from(compiled.jnt_qposadr));base.delete();compiled.delete();
const sourceSha256=Object.fromEntries(await Promise.all(['web/flybody-habitat-collision.js','web/habitat.json','scripts/original-fruit-geometry.mjs'].map(async path=>[path,createHash('sha256').update(await fs.readFile(path)).digest('hex')])));
const report={date:new Date().toISOString(),passed:true,scope:'Static native collision geometry verification. No behavioral/controller success claim.',
  representation:helper.representation,stats:helper.stats,compileMs,probeRadiusMm:.01,idParity,
  failures,positive,raySamples,rayOffsetMm:[.0000137,.0000211],maxRayErrorMm:Math.max(...raySamples.map(r=>Math.abs(r.error))),limitations:helper.limitations,sourceSha256};
await fs.writeFile('reports/flybody-habitat-collision-validation.json',JSON.stringify(report,null,2)+'\n');
await fs.writeFile('reports/flybody-habitat-collision-validation.md',`# Fruit collision solid validation\n\n${helper.representation}\n\nAll 17 recorded problem locations were tested using a 0.01 mm radius native sphere. Sixteen locations in visible air now have no contacts; the genuine bowl location retains contact. ${positive.length} additional native probes retain legitimate contacts on banana interiors, tips, apples and bowl, including fruit undersides.\n\n${raySamples.length} native ray comparisons against original rendered solid meshes pass; maximum height discrepancy ${report.maxRayErrorMm} mm. Native body and joint IDs remain unchanged (${idParity.bodyCount} bodies, ${idParity.jointCount} joints).\n\nThis validates static geometry only. Closed-loop behavior and performance require separate checks. ${helper.limitations.join(' ')}\n`);
console.log(JSON.stringify({passed:true,problemLocations:failures.length,positiveContacts:positive.length,nativeRays:raySamples.length,maxRayErrorMm:report.maxRayErrorMm,idParity,compileMs,sourceSha256},null,2));
