import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import * as THREE from '../web/vendor/three.module.js';
import loadMujoco from '../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {createHabitat} from '../web/body-world.js';
import {createFlybodyHabitatCollision} from '../web/flybody-habitat-collision.js';

const bananas=JSON.parse(await fs.readFile('web/habitat.json','utf8')).fruit.filter(f=>f.kind==='banana'),mj=await loadMujoco();
const sampleOffsets=[-.4,-.2,-.05,-.01,-.002,0,.002,.01,.05,.2,.4],rows=[];
const point=(f,t)=>{const x=(t-.5)*f.length,z=9*(4*(t-.5)**2-1);return new THREE.Vector3(f.x+Math.cos(f.angle)*x-Math.sin(f.angle)*z,f.y,f.z+Math.sin(f.angle)*x+Math.cos(f.angle)*z);};
for(const [bananaIndex,f]of bananas.entries())for(const kind of ['sphere','capsule']){
  const helper=createFlybodyHabitatCollision(createHabitat([f])),radius=.002;
  const geometry=kind==='sphere'?`type="sphere" size="${radius}"`:`type="capsule" size="${radius} .08"`;
  const xml=`<mujoco><option timestep=".00005" gravity="0 0 -981"/><default><geom solref=".002 1" solimp=".95 .99 .01"/></default><asset>${helper.assets}</asset><worldbody>${helper.geoms}<body name="probe"><freejoint/><geom name="probe" ${geometry} mass=".000001"/></body></worldbody></mujoco>`;
  const model=mj.MjModel.from_xml_string(xml);model.hfield_data.set(helper.heights);const data=new mj.MjData(model),force=new mj.DoubleBuffer(6),bodies=Int32Array.from(model.geom_bodyid);
  const curve=new THREE.CatmullRomCurve3(Array.from({length:31},(_,i)=>point(f,i/30)));
  const tube=new THREE.Mesh(new THREE.TubeGeometry(curve,60,f.radius,12,false),new THREE.MeshBasicMaterial()),ray=new THREE.Raycaster();tube.updateMatrixWorld(true);
  try{for(const seam of [6,12,18,24,30,36,42,48,54])for(const offset of sampleOffsets){
    const t=(seam+offset)/60,p=curve.getPointAt(t),tangent=curve.getTangentAt(t);
    ray.set(new THREE.Vector3(p.x,100,p.z),new THREE.Vector3(0,-1,0));const hit=ray.intersectObject(tube,false)[0];assert(hit);
    const axis=new THREE.Vector3(-tangent.z,tangent.x,0).normalize(),q=new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0,0,1),axis);
    const center=hit.point.clone().multiplyScalar(.1).addScaledVector(hit.face.normal,radius-.00002);
    data.qpos.set([center.x,center.z,center.y,q.w,q.x,q.y,q.z]);
    data.qvel.fill(0);mj.mj_forward(model,data);
    const contacts=data.ncon?data.contact:null;let normalForce=0,minNormalZ=1,maxTangentialNormal=0,count=0;
    try{for(let i=0;i<data.ncon;i++){const c=contacts.get(i);try{
      const g=c.geom;if((bodies[g[0]]===0)===(bodies[g[1]]===0))continue;
      mj.mj_contactForce(model,data,i,force);const magnitude=force.GetView()[0];if(magnitude<=1e-10)continue;
      const sign=bodies[g[0]]===0?1:-1,nz=sign*c.frame[2];
      normalForce+=magnitude;minNormalZ=Math.min(minNormalZ,nz);maxTangentialNormal=Math.max(maxTangentialNormal,Math.hypot(c.frame[0],c.frame[1]));count++;
    }finally{c.delete();}}}finally{contacts?.delete();}
    assert(count>0,JSON.stringify({bananaIndex,kind,seam,offset,ncon:data.ncon,reason:'lost surface contact'}));
    assert(minNormalZ>.75,JSON.stringify({bananaIndex,kind,seam,offset,minNormalZ,reason:'nearly horizontal seam normal'}));
    rows.push({bananaIndex,kind,seam,offset,count,normalForce,minNormalZ,maxTangentialNormal,normalImpulseEstimate:normalForce*.00005});
  }}finally{force.delete();data.delete();model.delete();tube.geometry.dispose();tube.material.dispose();}
}
const groups=[];
for(const bananaIndex of [0,1,2])for(const kind of ['sphere','capsule'])for(const seam of [6,12,18,24,30,36,42,48,54]){
  const selected=rows.filter(r=>r.bananaIndex===bananaIndex&&r.kind===kind&&r.seam===seam),reference=selected.filter(r=>Math.abs(r.offset)===.4).reduce((s,r)=>s+r.normalForce,0)/2;
  const peakRatio=Math.max(...selected.map(r=>r.normalForce))/reference;
  assert(peakRatio<3,JSON.stringify({bananaIndex,kind,seam,peakRatio}));groups.push({bananaIndex,kind,seam,peakRatio});
}
const report={date:new Date().toISOString(),passed:true,scope:'Controlled exterior contact sampling across compound banana slice boundaries, using a sphere and thin transverse capsule. No fly/controller modification.',
  method:'At each prescribed pose the stationary probe penetrates the visible top by 0.00002 cm and native mj_forward computes contact normals/forces. Force times 50 us is reported only as a one-step impulse estimate, not an integrated free-flight trajectory.',
  probe:{radiusCm:.002,capsuleHalfLengthCm:.08,massG:.000001},samples:rows.length,minimumUpwardNormal:Math.min(...rows.map(r=>r.minNormalZ)),
  maximumBoundaryForceRatio:Math.max(...groups.map(g=>g.peakRatio)),maximumNormalForce:Math.max(...rows.map(r=>r.normalForce)),
  limitations:['This tests the exterior top at nine boundaries per banana, not every contact direction or deeply interpenetrating fly mesh.','Compound pieces can generate multiple legitimate contacts at a shared boundary; this diagnostic rules out the prior horizontal-normal spike in the sampled exterior sweep only.'],groups,rows};
await fs.writeFile('reports/banana-collision-seams.json',JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({...report,groups:undefined,rows:undefined},null,2));
