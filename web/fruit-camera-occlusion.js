import * as THREE from './vendor/three.module.js';

// Viewer geometry only. Register the original banana body / apple skin Mesh,
// never its Group: fruit decoration and fly meshes must not enter this list.
// Apples are rendered through instancing, but their original Mesh retains the
// exact geometry and transform even while visible=false.
export function createFruitCameraOcclusion({maximumElevation=1.47,searchStep=Math.PI/45,targetClearance=.1,refineSteps=4}={}){
  if(![maximumElevation,searchStep,targetClearance].every(Number.isFinite)||searchStep<=0||targetClearance<0||
    maximumElevation<=0||maximumElevation>=Math.PI/2||!Number.isInteger(refineSteps)||refineSteps<0||refineSteps>8)
    throw new Error('Invalid fruit camera occlusion settings');
  const solids=[],ray=new THREE.Ray(),position=new THREE.Vector3(),direction=new THREE.Vector3(),origin=new THREE.Vector3(),end=new THREE.Vector3();
  const a=new THREE.Vector3(),b=new THREE.Vector3(),c=new THREE.Vector3(),hit=new THREE.Vector3();
  const orbit=(target,azimuth,elevation,distance)=>position.set(
    target.x+distance*Math.cos(elevation)*Math.sin(azimuth),target.y+distance*Math.sin(elevation),
    target.z+distance*Math.cos(elevation)*Math.cos(azimuth));
  function blocked(target,azimuth,elevation,distance){
    orbit(target,azimuth,elevation,distance);
    for(const solid of solids){
      origin.copy(target).applyMatrix4(solid.inverse);end.copy(position).applyMatrix4(solid.inverse);
      direction.subVectors(end,origin);const length=direction.length(),scale=length/distance;
      ray.set(origin,direction.multiplyScalar(1/length));
      if(!ray.intersectsBox(solid.bounds))continue;
      // Small contiguous triangle blocks retain the exact original surface;
      // they avoid scanning every banana triangle on every camera frame.
      for(const block of solid.blocks){
        if(!ray.intersectsBox(block.bounds))continue;
        for(let triangle=block.start;triangle<block.end;triangle++){
          a.fromBufferAttribute(solid.positions,solid.index?solid.index.getX(triangle*3):triangle*3);
          b.fromBufferAttribute(solid.positions,solid.index?solid.index.getX(triangle*3+1):triangle*3+1);
          c.fromBufferAttribute(solid.positions,solid.index?solid.index.getX(triangle*3+2):triangle*3+2);
          // Double-sided intersection is equivalent to checking both segment
          // directions with the original opaque FrontSide fruit materials.
          if(ray.intersectTriangle(a,b,c,false,hit)){
            const along=origin.distanceTo(hit);
            if(along>=targetClearance*scale&&along<=length-1e-5*scale)return true;
          }
        }
      }
    }
    return false;
  }
  return {
    addSolid(mesh){
      if(!mesh?.isMesh||mesh.isInstancedMesh||!mesh.geometry?.isBufferGeometry)throw new Error('Register an original fruit solid Mesh, not a Group or instance batch');
      if(!solids.some(s=>s.mesh===mesh)){
        const positions=mesh.geometry.attributes.position,index=mesh.geometry.index,count=(index?.count??positions.count)/3;
        const bounds=new THREE.Box3(),blocks=[];
        for(let start=0;start<count;start+=48){
          const block={start,end:Math.min(start+48,count),bounds:new THREE.Box3()};
          for(let i=start*3;i<block.end*3;i++)block.bounds.expandByPoint(a.fromBufferAttribute(positions,index?index.getX(i):i));
          bounds.union(block.bounds);blocks.push(block);
        }
        solids.push({mesh,positions,index,blocks,bounds,inverse:new THREE.Matrix4()});
      }
      return mesh;
    },
    get solidCount(){return solids.length;},
    resolve({target,azimuth,elevation,distance,following=true}){
      const result={elevation,adjusted:false,occluded:false,resolved:false,checks:0};
      if(!following||!solids.length)return result;
      if(!target||![target.x,target.y,target.z,azimuth,elevation,distance].every(Number.isFinite)||distance<=targetClearance)
        return {...result,invalid:true};
      for(const solid of solids){solid.mesh.updateWorldMatrix(true,false);solid.inverse.copy(solid.mesh.matrixWorld).invert();}
      const test=angle=>{result.checks++;return blocked(target,azimuth,angle,distance);};
      if(!test(elevation)){result.resolved=true;return result;}
      result.occluded=true;
      // Bound work and angle. Search upward only; an unresolved obstruction
      // leaves the requested view unchanged and is explicitly reported.
      let low=elevation;
      for(let k=1;k<=24&&low<maximumElevation;k++){
        let high=Math.min(maximumElevation,elevation+k*searchStep);
        if(!test(high)){
          for(let j=0;j<refineSteps;j++){
            const mid=(low+high)/2;if(test(mid))low=mid;else high=mid;
          }
          result.elevation=high;result.adjusted=true;result.resolved=true;return result;
        }
        low=high;
      }
      return result;
    }
  };
}
