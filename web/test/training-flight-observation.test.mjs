import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {measureFlightKinematics,measureFlightObservation,upwardContactForce} from '../training/flight-observation.js';

const UP=[0,0,1,1,0,0,0,1,0];
const DOWN=[0,0,-1,1,0,0,0,-1,0];
const HORIZONTAL=[1,0,0,0,1,0,0,0,1];
const NO_SUPPORT={environmentContacts:0,nonFootEnvironmentContacts:0,footSupportCount:0,footSupportFraction:0};

function fixture(rows,{throwOnForce,throwOnGet,throwOnFrame,throwOnView}={}){
  const released={contacts:[],vectors:0,forceCalls:[]};
  const values=new Float64Array(6);
  const contactForce={GetView(){if(throwOnView)throw new Error('view failed');return values;}};
  const data={ncon:rows.length,get contact(){
    return {get(i){
      if(i===throwOnGet)throw new Error('get failed');
      return {geom:rows[i].geom,dist:rows[i].dist??0,
        get frame(){if(i===throwOnFrame)throw new Error('frame failed');return rows[i].frame??UP;},
        delete(){released.contacts.push(i);}};
    },delete(){released.vectors++;}};
  }};
  const body={data,metadata:{mass_g:1/981},model:{},contactForce,
    // Two geoms may belong to the same foot; wing, thorax and tibia do not.
    geomBodyIds:[0,1,2,3,4,5,1,0],tasteBodyToLeg:[-1,0,1,-1,-1,-1],
    mj:{mj_contactForce(model,passedData,i,buffer){
      assert.equal(model,body.model);assert.equal(passedData,data);assert.equal(buffer,contactForce);
      released.forceCalls.push(i);
      if(i===throwOnForce)throw new Error('force failed');
      values.fill(0);values.set(rows[i].force??[0,0,0]);
    }}};
  return {body,released};
}

test('no native contacts avoid all Embind wrappers and contact scratch access',()=>{
  const body={data:{ncon:0,get contact(){throw new Error('allocated contact wrapper');}},
    get metadata(){throw new Error('accessed unnecessary mass');}};
  assert.deepEqual(measureFlightObservation(body),NO_SUPPORT);
});

test('grounded tarsal feet carry body weight and several contacts on one foot count once',()=>{
  const {body,released}=fixture([
    {geom:[0,1],force:[.2,0,0]},
    {geom:[0,6],force:[.3,0,0]},
    {geom:[0,2],force:[.5,0,0]},
  ]);
  assert.deepEqual(measureFlightObservation(body),{
    environmentContacts:3,nonFootEnvironmentContacts:0,footSupportCount:2,footSupportFraction:1,
  });
  assert.deepEqual(released.contacts,[0,1,2]);assert.equal(released.vectors,1);
});

test('mixed foot, wing, thorax and tibia collisions retain non-foot crash evidence',()=>{
  const {body}=fixture([
    {geom:[0,1],force:[.4,0,0]},
    {geom:[0,2],force:[.6,0,0]},
    {geom:[0,3],force:[10,0,0]},
    {geom:[0,4],force:[20,0,0]},
    {geom:[0,5],force:[30,0,0]},
  ]);
  assert.deepEqual(measureFlightObservation(body),{
    environmentContacts:5,nonFootEnvironmentContacts:3,footSupportCount:2,footSupportFraction:1,
  });
});

test('self contacts and static pairs neither support the fly nor call native force extraction',()=>{
  const {body,released}=fixture([{geom:[1,2],force:[100,0,0]},{geom:[0,7],force:[100,0,0]}]);
  assert.deepEqual(measureFlightObservation(body),NO_SUPPORT);
  assert.deepEqual(released.forceCalls,[]);assert.deepEqual(released.contacts,[0,1]);assert.equal(released.vectors,1);
});

test('contact frame and geom ordering include tangential force but reject horizontal and downward loads',()=>{
  assert.equal(upwardContactForce(UP,[2,0,0],true),2);
  assert.equal(upwardContactForce(UP,[2,0,0],false),0);
  assert.equal(upwardContactForce(DOWN,[2,0,0],false),2);
  assert.equal(upwardContactForce(HORIZONTAL,[8,0,0],true),0);
  assert.equal(upwardContactForce(HORIZONTAL,[8,0,.5],true),.5);
  assert.equal(upwardContactForce([1,0,0,0,0,1,0,-1,0],[8,.5,0],true),.5);
  const {body}=fixture([
    {geom:[1,0],frame:UP,force:[2,0,0]},
    {geom:[0,2],frame:HORIZONTAL,force:[8,0,0]},
  ]);
  assert.deepEqual(measureFlightObservation(body),{
    environmentContacts:2,nonFootEnvironmentContacts:0,footSupportCount:0,footSupportFraction:0,
  });
});

test('reversing geom order preserves upward foot support with the corresponding contact frame',()=>{
  const forward=fixture([{geom:[0,1],frame:UP,force:[1,0,0]}]);
  const reverse=fixture([{geom:[1,0],frame:DOWN,force:[1,0,0]}]);
  assert.deepEqual(measureFlightObservation(forward.body),measureFlightObservation(reverse.body));
  assert.equal(measureFlightObservation(reverse.body).footSupportFraction,1);
});

test('force-bearing positive separation counts; unloaded proximity and numerical force noise do not',()=>{
  const {body}=fixture([
    {geom:[0,1],dist:.001,force:[1,0,0]},
    {geom:[0,3],dist:.001,force:[2,0,0]},
    {geom:[0,2],dist:.001,force:[0,0,0]},
    {geom:[0,2],dist:.001,force:[1e-8,0,0]},
    {geom:[0,2],dist:0,force:[0,0,0]},
  ]);
  assert.deepEqual(measureFlightObservation(body),{
    environmentContacts:3,nonFootEnvironmentContacts:1,footSupportCount:1,footSupportFraction:1,
  });
});

test('observation scratch resets between frames and independent bodies',()=>{
  const rows=[{geom:[0,1],force:[1,0,0]}],first=fixture(rows),second=fixture([{geom:[0,2],force:[2,0,0]}]);
  assert.equal(measureFlightObservation(first.body).footSupportFraction,1);
  assert.equal(measureFlightObservation(second.body).footSupportFraction,2);
  rows[0].force=[0,0,0];
  assert.deepEqual(measureFlightObservation(first.body),{
    environmentContacts:1,nonFootEnvironmentContacts:0,footSupportCount:0,footSupportFraction:0,
  });
});

test('Embind element and vector handles are released when measurement fails',async t=>{
  for(const [option,message] of [['throwOnForce','force failed'],['throwOnFrame','frame failed'],['throwOnView','view failed']]){
    await t.test(option,()=>{
      const {body,released}=fixture([{geom:[0,1],force:[1,0,0]}],{[option]:option==='throwOnView'?true:0});
      assert.throws(()=>measureFlightObservation(body),new RegExp(message));
      assert.deepEqual(released.contacts,[0]);assert.equal(released.vectors,1);
    });
  }
  await t.test('vector released when element construction fails',()=>{
    const {body,released}=fixture([{geom:[0,1],force:[1,0,0]}],{throwOnGet:0});
    assert.throws(()=>measureFlightObservation(body),/get failed/);
    assert.deepEqual(released.contacts,[]);assert.equal(released.vectors,1);
  });
});

let nativeAssets;
async function nativeBody(){
  nativeAssets??=Promise.all([
    import('../../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js').then(m=>m.default()),
    readFile(new URL('../../models/flybody-mujoco.xml',import.meta.url),'utf8'),
    readFile(new URL('../../models/flybody-mujoco.json',import.meta.url),'utf8').then(JSON.parse),
  ]);
  const [mj,xml,metadata]=await nativeAssets,model=mj.MjModel.from_xml_string(xml),data=new mj.MjData(model);
  for(const joint of metadata.joints)data.qpos[joint.qpos]=joint.neutral;
  data.qpos[2]=2;
  return {mj,model,data,metadata,dispose(){data.delete();model.delete();}};
}
function massWeightedPosition(body){
  const {mj,model,data}=body;
  mj.mj_kinematics(model,data);
  const center=[0,0,0];let total=0;
  for(let id=1;id<model.nbody;id++){
    const mass=model.body_mass[id];total+=mass;
    for(let axis=0;axis<3;axis++)center[axis]+=mass*data.xipos[id*3+axis];
  }
  return center.map(value=>value/total);
}

test('native COM observation includes articulated recoil and refreshes post-integration kinematics',async()=>{
  const body=await nativeBody(),{mj,model,data,metadata}=body;
  try{
    const wing=metadata.joints.find(j=>j.name==='wing_roll_left');
    data.qvel[wing.dof]=1000;
    mj.mj_forward(model,data);
    // Change root position after forward: the helper must not return stale COM.
    data.qpos[2]+=1;
    const result=measureFlightKinematics(body),expectedPosition=massWeightedPosition(body);
    result.position.forEach((value,axis)=>assert(Math.abs(value-expectedPosition[axis])<1e-12));
    assert.equal(result.height,result.position[2]);assert.equal(data.qvel[2],0);
    assert(Math.abs(result.verticalSpeed)>.1,'moving wing mass changes COM velocity despite a stationary root');

    // Independently differentiate the mass-weighted position along the only
    // moving hinge coordinate, rather than re-reading subtree_linvel.
    const angle=data.qpos[wing.qpos],epsilon=1e-7;
    data.qpos[wing.qpos]=angle+1000*epsilon;const after=massWeightedPosition(body);
    data.qpos[wing.qpos]=angle-1000*epsilon;const before=massWeightedPosition(body);
    data.qpos[wing.qpos]=angle;
    const expectedVelocity=after.map((value,axis)=>(value-before[axis])/(2*epsilon));
    assert(Math.abs(result.verticalSpeed-expectedVelocity[2])<1e-6);
    assert(Math.abs(result.speedCmPerSecond-Math.hypot(...expectedVelocity))<1e-6);
  }finally{body.dispose();}
});

test('native COM observation preserves integration state and the subsequent trajectory exactly',async()=>{
  const observed=await nativeBody(),control=await nativeBody();
  try{
    for(const body of [observed,control]){
      body.data.qvel[3]=.2;
      body.mj.mj_forward(body.model,body.data);
    }
    const fields=['qpos','qvel','act','ctrl','qacc_warmstart','qfrc_applied','xfrc_applied'];
    for(let step=0;step<200;step++){
      for(const body of [observed,control])body.mj.mj_step(body.model,body.data);
      const before=Object.fromEntries(fields.map(field=>[field,Array.from(observed.data[field])]));
      const time=observed.data.time;
      measureFlightKinematics(observed);
      assert.equal(observed.data.time,time);
      for(const field of fields){
        assert.deepEqual(Array.from(observed.data[field]),before[field],`observation preserves ${field}`);
        assert.deepEqual(Array.from(observed.data[field]),Array.from(control.data[field]),`trajectory preserves ${field}`);
      }
    }
  }finally{observed.dispose();control.dispose();}
});

test('COM helper rejects an incomplete or mismatched body subtree before calling native kinematics',()=>{
  const body={metadata:{mass_g:1},model:{jnt_bodyid:[1],jnt_type:[0],body_subtreemass:[1,.5]},data:{},mj:{}};
  assert.throws(()=>measureFlightKinematics(body),/complete FlyBody free-root mass/);
});
