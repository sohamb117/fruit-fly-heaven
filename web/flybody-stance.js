// Initialize the native articulated feet on the actual terrain. This only sets
// the starting posture; it never advances a gait or changes the root in flight.
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
function solve3(a,b){
 const m=a.map((r,i)=>[...r,b[i]]);
 for(let k=0;k<3;k++){
  let p=k;for(let i=k+1;i<3;i++)if(Math.abs(m[i][k])>Math.abs(m[p][k]))p=i;
  [m[k],m[p]]=[m[p],m[k]];const d=m[k][k];if(Math.abs(d)<1e-12)return [0,0,0];
  for(let j=k;j<4;j++)m[k][j]/=d;
  for(let i=0;i<3;i++)if(i!==k){const f=m[i][k];for(let j=k;j<4;j++)m[i][j]-=f*m[k][j];}
 }
 return m.map(r=>r[3]);
}
export function initializeStance(mj,model,data,metadata,surface){
 const epsilon=1e-4,byName=new Map(metadata.joints.map(j=>[j.name,j]));
 const geomBodies=Int32Array.from(model.geom_bodyid);
 for(const [leg,site] of metadata.feet.entries()){
  const side=leg<3?'left':'right',segment=`T${leg%3+1}`;
  const joints=['coxa_abduct','coxa_twist','coxa','femur_twist','femur','tibia','tarsus'].map(name=>byName.get(`${name}_${segment}_${side}`));
  mj.mj_kinematics(model,data);const target=Array.from(data.site_xpos.slice(site*3,site*3+3));target[2]=surface(target[0],target[1])+.0065;
  for(let iteration=0;iteration<20;iteration++){
   mj.mj_kinematics(model,data);const p=Array.from(data.site_xpos.slice(site*3,site*3+3)),error=target.map((v,k)=>v-p[k]);
   if(Math.hypot(...error)<.0001)break;
   const columns=joints.map(j=>{
    const q=data.qpos[j.qpos];data.qpos[j.qpos]=q+epsilon;mj.mj_kinematics(model,data);
    const column=Array.from(data.site_xpos.slice(site*3,site*3+3),(v,k)=>(v-p[k])/epsilon);data.qpos[j.qpos]=q;return column;
   });
   const normal=Array.from({length:3},(_,i)=>Array.from({length:3},(_,j)=>columns.reduce((s,c)=>s+c[i]*c[j],i===j?1e-5:0)));
   const correction=solve3(normal,error);
   joints.forEach((j,k)=>{const dq=columns[k].reduce((s,v,i)=>s+v*correction[i],0);data.qpos[j.qpos]=clamp(data.qpos[j.qpos]+clamp(dq,-.08,.08),...j.range);});
  }
 }
 // A claw site is the centre of a capsule, not its contact point. Its offset
 // depends on the joint posture and the local triangle normal. Refine against
 // MuJoCo's actual contacts instead of assuming one clearance for all feet.
 for(const [leg,site] of metadata.feet.entries()){
  const body=metadata.claw_bodies[leg],geom=model.body_geomadr[body];
  // On a flat heightfield, broad-phase contacts start at the claw margin;
  // targeting zero reported distance can alternate between absent/contact.
  const contactDepth=model.geom_margin[geom]+.00002;
  const side=leg<3?'left':'right',segment=`T${leg%3+1}`;
  const joints=['coxa_abduct','coxa_twist','coxa','femur_twist','femur','tibia','tarsus'].map(name=>byName.get(`${name}_${segment}_${side}`));
   for(let iteration=0;iteration<40;iteration++){
    mj.mj_fwdPosition(model,data);
    let distance=Infinity,normalZ=1;
    const contacts=data.ncon?data.contact:null;
    try{for(let i=0;i<data.ncon;i++){
     const c=contacts.get(i);
     try{
     const pair=c.geom,g0=pair[0],g1=pair[1];
     if(!((geomBodies[g0]===0&&g1===geom)||(geomBodies[g1]===0&&g0===geom)))continue;
     if(c.dist<distance){distance=c.dist;normalZ=Math.abs(c.frame[2]);}
     }finally{c.delete();}
    }}finally{contacts?.delete();}
    if(!Number.isFinite(distance))distance=.00025;
    if(Math.abs(distance+contactDepth)<.000005)break;
    const p=Array.from(data.site_xpos.slice(site*3,site*3+3)),error=[0,0,-(distance+contactDepth)/Math.max(.2,normalZ)];
    const columns=joints.map(j=>{
     const q=data.qpos[j.qpos];data.qpos[j.qpos]=q+epsilon;mj.mj_kinematics(model,data);
     const column=Array.from(data.site_xpos.slice(site*3,site*3+3),(v,k)=>(v-p[k])/epsilon);data.qpos[j.qpos]=q;return column;
    });
    const normal=Array.from({length:3},(_,i)=>Array.from({length:3},(_,j)=>columns.reduce((s,c)=>s+c[i]*c[j],i===j?1e-5:0)));
    const correction=solve3(normal,error);
    joints.forEach((j,k)=>{const dq=columns[k].reduce((s,v,i)=>s+v*correction[i],0);data.qpos[j.qpos]=clamp(data.qpos[j.qpos]+clamp(dq,-.04,.04),...j.range);});
   }
 }
 mj.mj_kinematics(model,data);
 const rest=Float64Array.from(data.qpos);
 for(const a of metadata.actuators)if(a.joint!==null&&!a.name.startsWith('wing_')){
  const target=rest[byName.get(a.name).qpos];data.ctrl[a.id]=target;
  const act=model.actuator_actadr[a.id];if(act>=0)data.act[act]=target;
 }
 return rest;
}
