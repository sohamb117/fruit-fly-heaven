// Register the existing procedural mouth to native joint anchors and the two
// actual contact ellipsoids. This changes rendering only, never body dynamics.
const SCALE=10,swap=[0,2,1];
export function createMouthLandmarks(model,metadata){
  const mouthBodies=new Set(metadata.mouth_bodies),geoms=[];
  for(let id=0;id<model.ngeom;id++)if(mouthBodies.has(model.geom_bodyid[id])){
    if(model.geom_type[id]!==4)throw new Error('Unsupported native mouth contact geometry');
    geoms.push({id,size:Array.from(model.geom_size.slice(id*3,id*3+3))});
  }
  return {joints:['rostrum','haustellum'].map(name=>metadata.joints.find(j=>j.name===name).id),bodies:[...mouthBodies],geoms};
}

export function sampleMouthPose(data,landmarks,root,rotation){
  const local=p=>{
    const d=p.map((v,i)=>v-root[i]),q=[0,1,2].map(i=>rotation[i]*d[0]+rotation[3+i]*d[1]+rotation[6+i]*d[2]);
    return [q[0]*SCALE+.13,q[2]*SCALE+.91,q[1]*SCALE];
  };
  const anchors=landmarks.joints.map(id=>local(Array.from(data.xanchor.slice(id*3,id*3+3))));
  anchors.push(local([0,1,2].map(axis=>landmarks.bodies.reduce((sum,id)=>sum+data.xpos[id*3+axis],0)/landmarks.bodies.length)));
  const ellipsoids=landmarks.geoms.map(({id,size})=>{
    const r=data.geom_xmat.subarray(id*9,id*9+9),relative=[];
    for(let row=0;row<3;row++)for(let col=0;col<3;col++)relative[row*3+col]=[0,1,2].reduce((sum,k)=>sum+rotation[k*3+row]*r[k*3+col],0);
    // Reflect both the frame and its axes so the rendering rotation remains
    // proper; swap matching radii to preserve the exact physical ellipsoid.
    return {id,center:local(Array.from(data.geom_xpos.slice(id*3,id*3+3))),
      rotation:swap.flatMap(row=>swap.map(col=>relative[row*3+col])),size:swap.map(i=>size[i]*SCALE)};
  });
  return {anchors,ellipsoids};
}
