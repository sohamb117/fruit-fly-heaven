import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import * as THREE from '../web/vendor/three.module.js';
import {originalBowlGeometry} from './original-fruit-geometry.mjs';

const audit=JSON.parse(await fs.readFile('reports/fruit-surface-audit.json','utf8'));
const original=originalBowlGeometry(),reversed=originalBowlGeometry({reverse:true}),twoSided=new THREE.Mesh(original.geometry,new THREE.MeshBasicMaterial({side:THREE.DoubleSide}));
twoSided.updateMatrixWorld(true);const ray=new THREE.Raycaster();
function cast(mesh,origin,direction){ray.set(new THREE.Vector3(...origin),new THREE.Vector3(...direction));const h=ray.intersectObject(mesh,false)[0];return h?{point:h.point.toArray(),normal:h.face.normal.toArray(),distance:h.distance}:null;}
const vertices=mesh=>{const p=mesh.geometry.attributes.position;return Array.from({length:p.count},(_,i)=>[p.getX(i),p.getY(i),p.getZ(i)].join(',')).sort();};
assert.deepEqual(vertices(original),vertices(reversed),'Reversal must retain every original vertex');
const samples=audit.points.filter(p=>['frame80','bare_bowl'].includes(p.label)&&p.analyticFruitIndex===-1),rows=[];
for(const p of samples){
  const before=cast(original,[p.x,100,p.z],[0,-1,0]),after=cast(reversed,[p.x,100,p.z],[0,-1,0]),geometric=cast(twoSided,[p.x,100,p.z],[0,-1,0]);
  assert(before&&after&&geometric);assert(after.normal[1]>0);assert(Math.abs(after.point[1]-geometric.point[1])<.001);
  assert(Math.abs(after.point[1]-p.native257)<.012);
  rows.push({x:p.x,z:p.z,label:p.label,center:p.center??false,originalHeight:before.point[1],correctedHeight:after.point[1],nativeHeight:p.native257,
    correctedMinusNative:after.point[1]-p.native257,correctedMinusOriginalGeometry:after.point[1]-geometric.point[1],correctedNormalY:after.normal[1]});
}
const exterior=[];
for(const [x,z]of [[0,0],[20,0],[45,0],[60,0],[-45,0],[0,45]]){
  const after=cast(reversed,[x,-20,z],[0,1,0]),geometric=cast(twoSided,[x,-20,z],[0,1,0]);
  assert(after&&geometric);assert(after.normal[1]<0);assert(Math.abs(after.point[1]-geometric.point[1])<.001);
  exterior.push({origin:[x,-20,z],corrected:after,geometric});
}
for(const origin of [[80,10,0],[-80,10,0],[0,10,80],[0,10,-80]]){
  const direction=origin[0]?[ -Math.sign(origin[0]),0,0]:[0,0,-Math.sign(origin[2])],after=cast(reversed,origin,direction),geometric=cast(twoSided,origin,direction);
  assert(after&&geometric);assert(Math.abs(after.distance-geometric.distance)<.001);exterior.push({origin,corrected:after,geometric});
}
const report={date:new Date().toISOString(),scope:'Viewer-only winding verification using the existing bowl profile, tessellation and FrontSide material; no production file or physics changes.',
  exactAppChange:'new THREE.LatheGeometry(points,128) -> new THREE.LatheGeometry(points.reverse(),128)',
  sameVertexMultiset:true,sameTriangleCount:original.geometry.index.count===reversed.geometry.index.count,materialSide:reversed.material.side,
  innerFloorSamples:rows.length,maximumCorrectedNativeHeightError:Math.max(...rows.map(r=>Math.abs(r.correctedMinusNative))),
  maximumProfileReversalTriangulationDifference:Math.max(...rows.map(r=>Math.abs(r.correctedMinusOriginalGeometry))),
  meanCorrectedNativeHeightError:rows.reduce((s,r)=>s+Math.abs(r.correctedMinusNative),0)/rows.length,
  frame80:rows.find(r=>r.center),exteriorChecks:exterior,passed:true,rows};
await fs.writeFile('reports/bowl-winding-validation.json',JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({...report,rows:undefined},null,2));
original.geometry.dispose();original.material.dispose();reversed.geometry.dispose();reversed.material.dispose();twoSided.material.dispose();
