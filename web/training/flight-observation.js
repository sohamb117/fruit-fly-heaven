// Training-only observation: no changes to native controls or dynamics.
const FOOT_COUNT=6;
const SUPPORT_EPSILON=1e-5; // Fraction of body weight, to ignore solver noise.
const footLoadScratch=new WeakMap();
const kinematicRoots=new WeakMap();

// mj_step integrates qpos/qvel after computing derived fields. Refresh only
// kinematics before measuring COM: mj_forward would also recompute dynamics.
// MuJoCo's subtree COM and linear velocity use world coordinates and include
// every articulated mass, unlike the translating free-joint/thorax origin.
// https://mujoco.readthedocs.io/en/stable/computation/#forward-dynamics
export function measureFlightKinematics(body){
  const {mj,model,data}=body;
  let root=kinematicRoots.get(body);
  if(root===undefined){
    // The exported FlyBody starts with its free root joint. Selecting that
    // subtree excludes habitat bodies even if they acquire nonzero mass.
    root=model.jnt_bodyid[0];
    const mass=model.body_subtreemass[root],expectedMass=body.metadata.mass_g;
    if(model.jnt_type[0]!==0||!(root>0)||!(mass>0)||!Number.isFinite(expectedMass)||
      Math.abs(mass-expectedMass)>expectedMass*1e-8)
      throw new Error('Flight observation requires the complete FlyBody free-root mass');
    kinematicRoots.set(body,root);
  }
  mj.mj_kinematics(model,data);
  mj.mj_comPos(model,data);
  mj.mj_comVel(model,data);
  mj.mj_subtreeVel(model,data);
  const offset=root*3,com=data.subtree_com,velocity=data.subtree_linvel;
  const position=[com[offset],com[offset+1],com[offset+2]];
  return {position,height:position[2],verticalSpeed:velocity[offset+2],
    speedCmPerSecond:Math.hypot(velocity[offset],velocity[offset+1],velocity[offset+2])};
}

// MuJoCo stores contact-frame axes as rows in world coordinates. Its force
// acts on geom2; geom1 receives the equal and opposite reaction. All three
// translational components matter on slopes and when friction carries load.
// https://mujoco.readthedocs.io/en/stable/computation/#contact
export function upwardContactForce(frame,force,bodyIsGeom2){
  const sign=bodyIsGeom2?1:-1;
  return Math.max(0,sign*(frame[2]*force[0]+frame[5]*force[1]+frame[8]*force[2]));
}

export function measureFlightObservation(body){
  const data=body.data,ncon=data.ncon;
  // data.contact creates an owned Embind vector. Flight is the common case;
  // do not allocate a vector, native handles, or scratch storage without it.
  if(!ncon)return {environmentContacts:0,nonFootEnvironmentContacts:0,footSupportCount:0,footSupportFraction:0};

  const weight=body.metadata.mass_g*981;
  if(!(weight>0&&Number.isFinite(weight)))throw new RangeError('Flight observation requires a positive finite body mass');
  const minimumLoad=weight*SUPPORT_EPSILON,minimumLoadSquared=minimumLoad*minimumLoad;
  let loads=footLoadScratch.get(body);
  if(!loads){loads=new Float64Array(FOOT_COUNT);footLoadScratch.set(body,loads);}
  loads.fill(0);
  let environmentContacts=0,nonFootEnvironmentContacts=0;
  const contacts=data.contact;
  try{
    for(let i=0;i<ncon;i++){
      const contact=contacts.get(i);
      try{
        const geoms=contact.geom,body0=body.geomBodyIds[geoms[0]],body1=body.geomBodyIds[geoms[1]];
        const static0=body0===0,static1=body1===0;
        if(static0===static1)continue;

        body.mj.mj_contactForce(body.model,data,i,body.contactForce);
        // Acquire after the native call; never retain a view across allocation.
        const force=body.contactForce.GetView();
        const forceSquared=force[0]*force[0]+force[1]*force[1]+force[2]*force[2];
        // Contacts inside the solver margin may transmit force while still
        // geometrically separated. Unloaded proximity alone is not contact.
        if(!(contact.dist<=0||forceSquared>minimumLoadSquared))continue;
        environmentContacts++;

        // This lookup labels only each tarsus and its descendants. The broader
        // body_to_leg map includes tibiae and cannot identify a safe landing.
        const foot=body.tasteBodyToLeg[static0?body1:body0];
        if(Number.isInteger(foot)&&foot>=0&&foot<FOOT_COUNT){
          loads[foot]+=upwardContactForce(contact.frame,force,static0);
        }else nonFootEnvironmentContacts++;
      }finally{contact.delete();}
    }
  }finally{contacts.delete();}

  let footSupportCount=0,totalSupport=0;
  for(let foot=0;foot<FOOT_COUNT;foot++)if(loads[foot]>minimumLoad){footSupportCount++;totalSupport+=loads[foot];}
  return {environmentContacts,nonFootEnvironmentContacts,footSupportCount,footSupportFraction:totalSupport/weight};
}
