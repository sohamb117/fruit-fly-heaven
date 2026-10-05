// Register the original wing ellipsoids to the native aerodynamic geometry.
// This returns render transforms only. It does not alter joints or forces.
const SCALE=10,OFFSET=[.13,.91,0],SWAP=[0,2,1];
const column=(matrix,col)=>[matrix[col],matrix[3+col],matrix[6+col]];
const dot=(a,b)=>a.reduce((sum,value,i)=>sum+value*b[i],0);
const multiply=(matrix,vector)=>[0,1,2].map(row=>dot(matrix.slice(row*3,row*3+3),vector));

export function createWingLandmarks(model,metadata){
  const fluidStride=model.geom_fluid.length/model.ngeom;
  if(!Number.isInteger(fluidStride)||fluidStride<1)throw new Error('Missing native aerodynamic geometry');
  return ['left','right'].map((side,index)=>{
    const joint=metadata.joints.find(j=>j.name===`wing_yaw_${side}`||j.name===`walker/wing_yaw_${side}`);
    if(!joint)throw new Error(`Missing native ${side} wing joint`);
    const body=model.jnt_bodyid[joint.id],candidates=[];
    for(let id=0;id<model.ngeom;id++)if(model.geom_bodyid[id]===body&&model.geom_fluid[id*fluidStride]>0)candidates.push(id);
    if(candidates.length!==1)throw new Error(`Expected one aerodynamic geometry for ${side} wing`);
    const geom=candidates[0];
    if(model.geom_type[geom]!==4)throw new Error('Unsupported native wing geometry');
    return {side,body,geom,size:Array.from(model.geom_size.slice(geom*3,geom*3+3)),
      // These are the existing original UI wing child-mesh centers, not an
      // anatomical hinge. Retaining them preserves its mesh and material.
      originalCenter:[-.85,0,.22*(index===0?-1:1)]};
  });
}

export function sampleWingPose(data,landmarks,root,rootRotation){
  const localVector=vector=>SWAP.map(axis=>
    rootRotation[axis]*vector[0]+rootRotation[3+axis]*vector[1]+rootRotation[6+axis]*vector[2]);
  const localPoint=point=>localVector(point.map((value,i)=>value-root[i])).map((value,i)=>value*SCALE+OFFSET[i]);
  return landmarks.map(({side,body,geom,size,originalCenter})=>{
    const nativeCenter=Array.from(data.geom_xpos.slice(geom*3,geom*3+3));
    const nativeAnchor=Array.from(data.xpos.slice(body*3,body*3+3));
    const nativeRotation=Array.from(data.geom_xmat.slice(geom*9,geom*9+9));
    const span=column(nativeRotation,2),normal=column(nativeRotation,0),chord=column(nativeRotation,1);
    const distalSign=dot(span,nativeCenter.map((value,i)=>value-nativeAnchor[i]))>=0?1:-1;
    // Native geom axes are [thickness,chord,span]; original mesh axes are
    // [span,thickness,chord]. swapYZ is a reflection. The paired axis signs
    // below produce a proper rotation and make the mesh's -X distal on both
    // sides, rather than attempting to mirror noncommuting Euler angles.
    const x=localVector(span).map(value=>-distalSign*value);
    const y=localVector(normal).map(value=>distalSign*value);
    const z=localVector(chord);
    const rotation=[0,1,2].flatMap(row=>[x[row],y[row],z[row]]);
    const center=localPoint(nativeCenter),offset=multiply(rotation,originalCenter);
    const position=center.map((value,i)=>value-offset[i]);
    return {side,body,geom,position,rotation,center,anchor:localPoint(nativeAnchor),
      distalTip:center.map((value,i)=>value-x[i]*size[2]*SCALE),
      nativeHalfExtents:[size[2]*SCALE,size[0]*SCALE,size[1]*SCALE],
      originalCenter};
  });
}
