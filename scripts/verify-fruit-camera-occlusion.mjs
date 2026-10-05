import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import * as THREE from '../web/vendor/three.module.js';
import {createFruitCameraOcclusion} from '../web/fruit-camera-occlusion.js';

const habitat=JSON.parse(await fs.readFile('web/habitat.json','utf8'));
const frame=String(await fs.readFile('reports/observation-60min-20260913/frames.jsonl')).trim().split('\n').map(JSON.parse).find(f=>f.index===80);
assert(frame,'Recorded frame 80 is required');
const guard=createFruitCameraOcclusion(),solids=[];
// Match app.js createFruit body geometry exactly; omit decorative children.
for(const f of habitat.fruit){
  let mesh;
  if(f.kind==='banana'){
    const points=Array.from({length:31},(_,i)=>{const t=i/30,x=(t-.5)*f.length,z=9*(4*(t-.5)**2-1);
      return new THREE.Vector3(f.x+Math.cos(f.angle)*x-Math.sin(f.angle)*z,f.y,f.z+Math.sin(f.angle)*x+Math.cos(f.angle)*z);});
    mesh=new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(points),60,f.radius,12,false),new THREE.MeshStandardMaterial());
  }else{
    mesh=new THREE.Mesh(new THREE.SphereGeometry(1,12,8),new THREE.MeshStandardMaterial());
    mesh.position.set(f.x,f.y,f.z);mesh.scale.set(f.radius,f.radius*.94,f.radius);mesh.visible=false;
  }
  guard.addSolid(mesh);solids.push(mesh);
}
const [x,y,z]=frame.state.position,target=new THREE.Vector3(x,y+1,z),request={target,azimuth:.63,elevation:.79,distance:25,following:true};
const original={target:target.toArray(),azimuth:request.azimuth,elevation:request.elevation,distance:request.distance},result=guard.resolve(request);
assert(result.occluded&&result.adjusted&&result.resolved,JSON.stringify(result));
assert.equal(guard.resolve({...request,elevation:result.elevation}).occluded,false);
assert.deepEqual({target:target.toArray(),azimuth:request.azimuth,elevation:request.elevation,distance:request.distance},original);
for(let k=0;k<100;k++)guard.resolve(request);
const start=performance.now(),runs=1000;
for(let k=0;k<runs;k++)guard.resolve(request);
const millisecondsPerResolve=(performance.now()-start)/runs;
const report={date:new Date().toISOString(),scope:'Geometry-only camera test, with exact original fruit body tessellation. No body simulation or state changes.',
  fixture:{frame:80,version:frame.version,recordedPosition:frame.state.position,
    cameraAssumption:'Original follow parameters azimuth 0.63, elevation 0.79, distance 25, with settled look=(fly.x,fly.y+1,fly.z). Actual interpolated camera state was not recorded in frame80.'},
  original,result,raisedDegrees:(result.elevation-original.elevation)*180/Math.PI,solidCount:guard.solidCount,
  noTargetOrAzimuthOrDistanceMutation:true,originalAppleMeshesRemainHidden:solids.slice(3).every(s=>s.visible===false),
  originalMaterialsUnmodified:solids.every(s=>s.material.side===THREE.FrontSide&&s.material.opacity===1),
  sharedLoadTiming:{runs,millisecondsPerResolve,note:'Node geometry timing under shared workstation load; not an end-to-end UI performance benchmark.'},
  sourceSha256:Object.fromEntries(await Promise.all(['web/habitat.json','web/fruit-camera-occlusion.js'].map(async path=>[path,createHash('sha256').update(await fs.readFile(path)).digest('hex')]))),passed:true};
await fs.writeFile('reports/fruit-camera-occlusion.json',JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));
solids.forEach(s=>{s.geometry.dispose();s.material.dispose();});
