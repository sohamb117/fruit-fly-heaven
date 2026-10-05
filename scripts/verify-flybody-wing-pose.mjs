// Native Python fixtures -> MuJoCo WASM -> actual THREE parent/child transforms.
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import loadMujoco from '../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {Object3D,Matrix4,Vector3} from '../web/vendor/three.core.js';
import {createWingLandmarks,sampleWingPose} from '../web/flybody-wing-pose.js';

const [mj,xml,metadata,fixture]=await Promise.all([loadMujoco(),fs.readFile('models/flybody-mujoco.xml','utf8'),
  fs.readFile('models/flybody-mujoco.json','utf8').then(JSON.parse),fs.readFile('reports/flybody-wing-landmarks.json','utf8').then(JSON.parse)]);
assert.equal(createHash('sha256').update(xml).digest('hex'),fixture.xml_sha256);
const report={scope:'Rendering-only registration against independent native MuJoCo pose fixtures. No integration, forces, or controller changes.',
  xmlSha256:fixture.xml_sha256,poses:fixture.cases.length,models:[],maximumError:0};
const world=point=>new Vector3(point[0]*10,point[2]*10,point[1]*10);
const matrix=rotation=>new Matrix4().set(rotation[0],rotation[1],rotation[2],0,rotation[3],rotation[4],rotation[5],0,rotation[6],rotation[7],rotation[8],0,0,0,0,1);
for(const shifted of [false,true]){
  // Habitat inserts geoms before the fly. Native geom indices therefore must
  // be discovered from the runtime model, rather than copied from the file.
  const source=shifted?xml.replace('<worldbody>','<worldbody><geom name="registration_marker" type="sphere" size=".001" pos="100 0 0" contype="0" conaffinity="0"/>'):xml;
  const model=mj.MjModel.from_xml_string(source),data=new mj.MjData(model),landmarks=createWingLandmarks(model,metadata);
  let maximum=0,unchangedState=true;
  for(const entry of fixture.cases){
    data.qpos.set(entry.qpos);mj.mj_forward(model,data);
    const before=Array.from(data.qpos),root=new Object3D();
    const [px,py,pz,w,x,y,z]=entry.qpos;
    root.quaternion.set(-x,-z,-y,w);
    root.position.set(px*10,pz*10,py*10).sub(new Vector3(.13,.91,0).applyQuaternion(root.quaternion));
    const poses=sampleWingPose(data,landmarks,[px,py,pz],entry.root_rotation);
    for(let i=0;i<2;i++){
      const pose=poses[i],expected=entry.wings[i],hinge=new Object3D(),original=new Object3D();
      hinge.position.fromArray(pose.position);hinge.quaternion.setFromRotationMatrix(matrix(pose.rotation));
      original.position.fromArray(pose.originalCenter);root.add(hinge);hinge.add(original);root.updateMatrixWorld(true);
      assert(Math.abs(matrix(pose.rotation).determinant()-1)<1e-12,'Wing registration must be a proper rotation');
      const checks=[
        [original.getWorldPosition(new Vector3()),world(expected.world_center_cm)],
        [root.localToWorld(new Vector3().fromArray(pose.anchor)),world(expected.world_anchor_cm)],
        [root.localToWorld(new Vector3().fromArray(pose.distalTip)),world(expected.world_distal_tip_cm)],
        [original.localToWorld(new Vector3(-pose.nativeHalfExtents[0],0,0)),world(expected.world_distal_tip_cm)],
      ];
      for(const [actual,target] of checks)maximum=Math.max(maximum,actual.distanceTo(target));
      // With the ORIGINAL mesh radius retained, its direction is still exact;
      // the known 0.04 scene-unit outline difference is a stylistic scale.
      const tip=original.localToWorld(new Vector3(-1.18,0,0));
      const direction=tip.sub(original.getWorldPosition(new Vector3())).normalize();
      const expectedDirection=world(expected.world_distal_span_axis).normalize();
      maximum=Math.max(maximum,direction.distanceTo(expectedDirection));
      root.remove(hinge);
    }
    unchangedState&&=before.every((value,index)=>value===data.qpos[index]);
  }
  assert(maximum<1e-9,`Native wing landmark registration error ${maximum}`);
  assert(unchangedState,'Rendering helper changed physical state');
  report.maximumError=Math.max(report.maximumError,maximum);
  report.models.push({extraWorldGeometry:shifted,wingGeomIds:landmarks.map(l=>l.geom),maximumSceneUnitError:maximum,physicalStateUnchanged:unchangedState});
  data.delete();model.delete();
}
report.restingAngles=fixture.cases[0].wings.map(wing=>({side:wing.side,nativeDegrees:wing.native_span_elevation_degrees,oldRendererDegrees:wing.original_euler_span_elevation_degrees}));
report.passed=true;
await fs.writeFile('reports/flybody-wing-pose-verification.json',JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));
