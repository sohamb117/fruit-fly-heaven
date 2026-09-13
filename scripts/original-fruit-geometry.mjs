// Audit fixture: geometry/tessellation copied from app.js createFruit, without
// rendering, fly meshes or physics. Includes decoration only when requested.
import * as THREE from '../web/vendor/three.module.js';
export function originalFruitGeometry(fruit,{decorations=false}={}){
  const sphere=new THREE.SphereGeometry(1,12,8),cylinder=new THREE.CylinderGeometry(1,.85,1,5),material=new THREE.MeshBasicMaterial();
  const scene=new THREE.Scene(),solids=[],all=[];
  const ellipsoid=(group,x,y,z,sx,sy,sz)=>{const m=new THREE.Mesh(sphere,material);m.position.set(x,y,z);m.scale.set(sx,sy,sz);group.add(m);return m;};
  const point=(f,t)=>{const x=(t-.5)*f.length,z=9*(4*(t-.5)**2-1);return new THREE.Vector3(f.x+Math.cos(f.angle)*x-Math.sin(f.angle)*z,f.y,f.z+Math.sin(f.angle)*x+Math.cos(f.angle)*z);};
  fruit.forEach((f,index)=>{
    let seed=900+index;const random=()=>{seed=(Math.imul(1664525,seed)+1013904223)>>>0;return seed/4294967296;};
    const group=new THREE.Group();scene.add(group);let body;
    if(f.kind==='banana'){
      const curve=new THREE.CatmullRomCurve3(Array.from({length:31},(_,i)=>point(f,i/30)));
      body=new THREE.Mesh(new THREE.TubeGeometry(curve,60,f.radius,12,false),material);group.add(body);
      if(decorations){
        for(const t of [0,1]){const p=point(f,t);ellipsoid(group,p.x,p.y,p.z,2.5,2.5,2.5);}
        for(let i=0;i<95;i++){
          const p=point(f,.03+random()*.94),angle=random()*Math.PI*2,r=f.radius*.99,tangent=curve.getTangent((i+.5)/95);
          const radial=new THREE.Vector3(-tangent.z,0,tangent.x).multiplyScalar(Math.cos(angle)*r),size=.18+random()*.58;
          ellipsoid(group,p.x+radial.x,p.y+Math.sin(angle)*r,p.z+radial.z,size,size*.4,size*.75);
        }
      }
    }else{
      body=ellipsoid(group,f.x,f.y,f.z,f.radius,f.radius*.94,f.radius);
      if(decorations){
        const top=f.y+f.radius*.91;ellipsoid(group,f.x,top,f.z,2.7,.9,2.7);
        const start=new THREE.Vector3(f.x,top,f.z),end=new THREE.Vector3(f.x+1.2,top+3,f.z+.5),vector=end.clone().sub(start),stem=new THREE.Mesh(cylinder,material);
        stem.scale.set(.7,vector.length(),.7);stem.position.copy(start).add(end).multiplyScalar(.5);stem.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),vector.normalize());group.add(stem);
        for(let i=0;i<85;i++){
          const angle=random()*Math.PI*2,u=.03+random()*.94,radial=Math.sqrt(1-u*u),r=f.radius*1.002;
          const x=f.x+r*radial*Math.cos(angle),z=f.z+r*radial*Math.sin(angle),y=f.y+r*u*.94,size=i<25?.5+random()*1.2:.1+random()*.4;
          ellipsoid(group,x,y,z,size,.23,size*.75);
        }
        ellipsoid(group,f.x-f.radius*.6,f.y+f.radius*.75,f.z,3.1,.5,4.5).rotation.z=.55;
      }
    }
    body.userData.fruitIndex=index;solids.push(body);
    for(const child of group.children){child.userData.fruitIndex=index;all.push(child);}
  });
  scene.updateMatrixWorld(true);
  return {scene,solids,all,dispose(){const geometries=new Set(all.map(m=>m.geometry));geometries.add(sphere);geometries.add(cylinder);geometries.forEach(g=>g.dispose());material.dispose();}};
}

export function originalBowlGeometry({reverse=false}={}){
  const points=[];for(let r=0;r<=65;r+=2.5)points.push(new THREE.Vector2(r,1.5+.0037*r*r));
  points.push(new THREE.Vector2(66,17),new THREE.Vector2(66.5,15),new THREE.Vector2(62,9),new THREE.Vector2(51,1),new THREE.Vector2(38,-.4),new THREE.Vector2(0,-.4));
  const bowl=new THREE.Mesh(new THREE.LatheGeometry(reverse?points.reverse():points,128),new THREE.MeshBasicMaterial());bowl.updateMatrixWorld(true);return bowl;
}
