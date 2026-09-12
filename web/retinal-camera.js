import * as THREE from './vendor/three.module.js';

// Render the same habitat geometry from two head-mounted cameras. The observer
// camera, overlays and trails are excluded. This is a coarse achromatic eye model.
export function createRetinalCamera(renderer,scene,{width=32,height=16,eye_yaw_degrees=60,vertical_fov_degrees=120}={}){
  const target=new THREE.WebGLRenderTarget(width,height,{depthBuffer:true,stencilBuffer:false});
  target.texture.colorSpace=THREE.SRGBColorSpace;
  const camera=new THREE.PerspectiveCamera(vertical_fov_degrees,width/height,.35,500);
  const rgba=new Uint8Array(width*height*4),position=new THREE.Vector3(),direction=new THREE.Vector3();
  const up=new THREE.Vector3(),forward=new THREE.Vector3(),right=new THREE.Vector3();let sequence=0;
  return {
    capture(f,mesh,exclude=[]){
      const pixels=new Uint8Array(width*height*2),before=renderer.getRenderTarget(),shadow=renderer.shadowMap.autoUpdate;
      const visibility=exclude.map(o=>o.visible);
      try{
        for(const o of exclude)o.visible=false;
        renderer.shadowMap.autoUpdate=false;
        up.set(0,1,0).applyQuaternion(mesh.quaternion);forward.set(1,0,0).applyQuaternion(mesh.quaternion);right.set(0,0,1).applyQuaternion(mesh.quaternion);
        for(const [i,side]of [-1,1].entries()){
          position.set(1.15,1.02,.32*side).applyMatrix4(mesh.matrixWorld);
          const yaw=side*eye_yaw_degrees*Math.PI/180;
          direction.copy(forward).multiplyScalar(Math.cos(yaw)).addScaledVector(right,Math.sin(yaw));
          camera.position.copy(position);camera.up.copy(up);camera.lookAt(position.clone().add(direction));
          renderer.setRenderTarget(target);renderer.render(scene,camera);renderer.readRenderTargetPixels(target,0,0,width,height,rgba);
          for(let y=0;y<height;y++)for(let x=0;x<width;x++){
            const p=((height-1-y)*width+x)*4;
            pixels[i*width*height+y*width+x]=Math.round(.2126*rgba[p]+.7152*rgba[p+1]+.0722*rgba[p+2]);
          }
        }
      }finally{renderer.setRenderTarget(before);renderer.shadowMap.autoUpdate=shadow;exclude.forEach((o,i)=>o.visible=visibility[i]);}
      return {id:f.id,sequence:++sequence,bodyTime:f.bodyTime,pixels};
    },
    dispose(){target.dispose();}
  };
}
