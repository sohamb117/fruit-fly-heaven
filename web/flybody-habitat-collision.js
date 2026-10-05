import {validateMaintainedScene,maintainedFloorHeightScene} from './flight-scene-profile.js';
import * as THREE from './vendor/three.module.js';

const TAU=2*Math.PI,RADIAL_SEGMENTS=12,TUBE_SEGMENTS=60;
const number=value=>Number(value).toPrecision(12);
const nativePoint=([x,y,z])=>[x/10,z/10,y/10];
const pointOnBanana=(f,t)=>{
  const x=(t-.5)*f.length,z=9*(4*(t-.5)**2-1);
  return new THREE.Vector3(f.x+Math.cos(f.angle)*x-Math.sin(f.angle)*z,f.y,f.z+Math.sin(f.angle)*x+Math.cos(f.angle)*z);
};

/** Static solids matching the original habitat renderer. All geoms belong to
 * worldbody, preserving native body/joint IDs. Only the bowl uses a heightfield;
 * fruit is a compound of convex mesh volumes, so air below fruit stays empty.
 * These are collision approximations, not changes to the rendered fruit.
 */
export function createFlybodyHabitatCollision(habitat,{resolution=257,extent=6.6}={}){
  if(!Number.isInteger(resolution)||resolution<3||!(extent>0)||!Number.isFinite(habitat.ceiling))throw new Error('Invalid habitat collision dimensions');
  const scene=validateMaintainedScene(habitat.maintainedScene);
  if(scene&&(resolution!==257||extent!==6.6||habitat.ceiling!==scene.ceilingCm*10))throw new Error('Spacious curriculum requires its original central grid and declared ceiling');
  const hasFruit=Array.isArray(habitat.fruit),fruit=hasFruit?habitat.fruit:[];
  const floor=scene?(x,z)=>maintainedFloorHeightScene(x,z,scene):hasFruit?(x,z)=>1.5+.0037*(x*x+z*z):(x,z)=>habitat.surface(x,z).y;
  const heights=new Float32Array(resolution*resolution);let maximum=0;
  for(let row=0;row<resolution;row++)for(let col=0;col<resolution;col++){
    const h=floor((col/(resolution-1)*2-1)*extent*10,(row/(resolution-1)*2-1)*extent*10)/10;
    if(!Number.isFinite(h)||h<0)throw new Error('Habitat heightfield requires finite nonnegative heights');
    heights[row*resolution+col]=h;maximum=Math.max(maximum,h);
  }
  maximum=Math.max(maximum,1e-6);
  // Keep original normalization and sample spacing for central grid parity.
  if(scene)maximum=(1.5+.0037*((extent*10)**2+(extent*10)**2))/10;
  for(let i=0;i<heights.length;i++)heights[i]/=maximum;
  const assets=[`<hfield name="habitat" nrow="${resolution}" ncol="${resolution}" size="${extent} ${extent} ${maximum} .1"/>`];
  const contact='friction=".6 .005 .0001" condim="3" contype="1" conaffinity="1"';
  const geoms=[`<geom name="ground" type="hfield" hfield="habitat" ${contact}/>`];
  geoms.push(`<geom name="ceiling" type="plane" pos="0 0 ${habitat.ceiling/10+.15}" quat="0 1 0 0" size="${scene?scene.radiusCm+1:7} ${scene?scene.radiusCm+1:7} .1"/>`);
  for(let k=0;k<64;k++){
    const a=TAU*k/64,half=TAU/128;
    const radius=scene?scene.radiusCm-.1:6.4,center=scene?(scene.ceilingCm+.15-1)/2:3,halfHeight=scene?(scene.ceilingCm+.15+1)/2:4;
    geoms.push(`<geom name="wall${k}" type="box" pos="${radius*Math.cos(a)} ${radius*Math.sin(a)} ${center}" euler="0 0 ${a}" size=".05 ${radius*Math.tan(half)} ${halfHeight}"/>`);
  }
  if(scene){
    // Four non-overlapping slabs surround the unchanged-size central square.
    // Their top meets its capped edge; no enormous/coarser heightfield is used.
    const outer=scene.radiusCm+1,inner=extent,top=maintainedFloorHeightScene(inner*10,0,scene)/10;
    const center=(outer+inner)/2,width=(outer-inner)/2,z=(top-1)/2,h=(top+1)/2;
    const slabs=[[center,0,width,outer],[-center,0,width,outer],[0,center,inner,width],[0,-center,inner,width]];
    slabs.forEach(([x,y,sx,sy],i)=>geoms.push(`<geom name="curriculum_apron${i}" type="box" pos="${x} ${y} ${z}" size="${sx} ${sy} ${h}" ${contact}/>`));
  }
  const fruitGeomNames={},stats={bananas:0,bananaSlices:0,bananaTips:0,apples:0,fruitGeoms:0,vertices:0,triangles:0};
  function mesh(name,fruitIndex,vertices,faces,origin){
    // Swapping scene Y/Z reverses handedness; reverse triangle winding too.
    const nativeVertices=vertices.flatMap(nativePoint),nativeFaces=faces.flatMap(([a,b,c])=>[a,c,b]);
    // The FlyBody XML has a default mesh scale of .1; these coordinates are
    // already native centimeters, so explicitly override that inherited scale.
    assets.push(`<mesh name="${name}" scale="1 1 1" vertex="${nativeVertices.map(number).join(' ')}" face="${nativeFaces.join(' ')}"/>`);
    geoms.push(`<geom name="${name}" type="mesh" mesh="${name}" pos="${nativePoint(origin).map(number).join(' ')}" ${contact}/>`);
    fruitGeomNames[name]=fruitIndex;stats.fruitGeoms++;stats.vertices+=vertices.length;stats.triangles+=faces.length;
  }
  const sphere=new THREE.SphereGeometry(1,12,8),spherePosition=sphere.attributes.position;
  const sphereFaces=Array.from({length:sphere.index.count/3},(_,i)=>[sphere.index.getX(i*3),sphere.index.getX(i*3+1),sphere.index.getX(i*3+2)]);
  const addSphere=(name,index,center,scale)=>mesh(name,index,Array.from({length:spherePosition.count},(_,i)=>
    [spherePosition.getX(i)*scale[0],spherePosition.getY(i)*scale[1],spherePosition.getZ(i)*scale[2]]),sphereFaces,center);
  try{fruit.forEach((f,index)=>{
    if(f.kind==='apple'){
      addSphere(`habitat_fruit_${index}_apple`,index,[f.x,f.y,f.z],[f.radius,.94*f.radius,f.radius]);stats.apples++;return;
    }
    if(f.kind!=='banana')throw new Error(`Unsupported fruit collision kind: ${f.kind}`);
    const curve=new THREE.CatmullRomCurve3(Array.from({length:31},(_,i)=>pointOnBanana(f,i/30)));
    const tube=new THREE.TubeGeometry(curve,TUBE_SEGMENTS,f.radius,RADIAL_SEGMENTS,false),p=tube.attributes.position;
    try{for(let segment=0;segment<TUBE_SEGMENTS;segment++){
      const vertices=[];
      for(const ring of [segment,segment+1])for(let side=0;side<RADIAL_SEGMENTS;side++){
        const i=ring*(RADIAL_SEGMENTS+1)+side;vertices.push([p.getX(i)-f.x,p.getY(i)-f.y,p.getZ(i)-f.z]);
      }
      const faces=[];
      for(let side=0;side<RADIAL_SEGMENTS;side++){
        const next=(side+1)%RADIAL_SEGMENTS;
        // TubeGeometry's original side winding, followed by planar end caps.
        faces.push([side,RADIAL_SEGMENTS+side,next],[RADIAL_SEGMENTS+side,RADIAL_SEGMENTS+next,next]);
      }
      for(let side=1;side<RADIAL_SEGMENTS-1;side++)faces.push([0,side,side+1],[RADIAL_SEGMENTS,RADIAL_SEGMENTS+side+1,RADIAL_SEGMENTS+side]);
      mesh(`habitat_fruit_${index}_banana_${segment}`,index,vertices,faces,[f.x,f.y,f.z]);stats.bananaSlices++;
    }}finally{tube.dispose();}
    for(const end of [0,1]){addSphere(`habitat_fruit_${index}_tip_${end}`,index,pointOnBanana(f,end).toArray(),[2.5,2.5,2.5]);stats.bananaTips++;}
    stats.bananas++;
  });}finally{sphere.dispose();}
  return {assets:assets.join(''),geoms:geoms.join(''),heights,fruitGeomNames,groundName:'ground',stats,
    representation:scene?'Original fruit solids, capped central bowl heightfield and four flat outer apron slabs; explicit spacious curriculum.':'Original 60x12 banana tube rings as convex slices; original 12x8 sphere mesh for apples/tips; bowl-only heightfield.',
    limitations:['Adjacent banana slices form a compound convex approximation; end caps close the original open tube rings.','Decorative spots, mold and stems are not load-bearing collision solids.','Bowl heightfield retains its existing tessellation; this helper does not correct unstable wing actuation.']};
}
