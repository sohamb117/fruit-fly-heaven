// Two live simulated seconds of the integrated reference wing renderer.
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'/tmp/fruit-fly-browser-tests/node_modules/playwright/index.mjs');
const browser=await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,args:['--enable-unsafe-webgpu']});
const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
const report={scope:'Rendering-only check of native wing registration in one live published-controller reference fly using the original UI. Own isolated browser; no scripted physics advancement or production edits.',
  date:new Date().toISOString(),browser:browser.version(),errors,sourceHashes:{},samples:[]};
page.on('pageerror',e=>errors.push(e.message));
page.on('response',async response=>{
  const path=new URL(response.url()).pathname;
  if(['/app.js','/flybody-reference-world.js','/flybody-wing-pose.js'].includes(path)){
    try{report.sourceHashes[path]=createHash('sha256').update(await response.body()).digest('hex');}catch{}
  }
});
async function read(){
  return page.evaluate(async()=>{
    const {Matrix4,Vector3,Quaternion}=await import('/vendor/three.core.js');
    const h=window.heaven,w=h.bodyWorld,f=h.state.flies[0],c=w.controller,d=c.data;
    const rootMatrix=new Matrix4().compose(new Vector3().fromArray(f.physicsPosition),new Quaternion().fromArray(f.physicsQuaternion),new Vector3(1,1,1));
    const wings=f.physicsWings.map((pose,i)=>{
      const r=pose.rotation,rotation=new Matrix4().set(r[0],r[1],r[2],0,r[3],r[4],r[5],0,r[6],r[7],r[8],0,0,0,0,1);
      const parentMatrix=new Matrix4().compose(new Vector3().fromArray(pose.position),new Quaternion().setFromRotationMatrix(rotation),new Vector3(1,1,1));
      const center=new Vector3().fromArray(pose.originalCenter).applyMatrix4(parentMatrix).applyMatrix4(rootMatrix);
      const native=d.geom_xpos.slice(pose.geom*3,pose.geom*3+3),o=w.renderOrigin;
      const expected=new Vector3(o[0]+native[0]*10,o[1]+native[2]*10,o[2]+native[1]*10);
      const landmark=w.wingLandmarks[i],stride=c.model.geom_fluid.length/c.model.ngeom;
      const joint=w.metadata.joints.find(j=>j.name===`walker/wing_yaw_${pose.side}`);
      return {side:pose.side,body:pose.body,geom:pose.geom,nativeCenterWorld:expected.toArray(),registeredCenterWorld:center.toArray(),
        centerError:center.distanceTo(expected),determinant:rotation.determinant(),
        nativeBodyMatches:pose.body===c.model.geom_bodyid[pose.geom]&&pose.body===c.model.jnt_bodyid[joint.id],
        aerodynamicGeometry:c.model.geom_fluid[pose.geom*stride]>0,landmarkMatches:landmark.geom===pose.geom,
        finite:[...pose.position,...pose.rotation].every(Number.isFinite)};
    });
    return {bodyTime:w.time,neuralTimeMs:h.state.time_ms,population:h.state.flies.length,reference:w.reference,
      paused:h.state.paused,complete:w.complete,referenceErrorCm:c.sample().referenceError,
      externallyAppliedForce:c.sample().maximumAppliedForce,renderCalls:h.renderer.info.render.calls,wings};
  });
}
try{
  await page.addInitScript(()=>{localStorage.setItem('fruit-fly-population','1');localStorage.setItem('fruit-fly-flight','true');});
  await page.goto('http://127.0.0.1:7842/?controller=flybody-reference&follow=1');
  await page.waitForFunction(()=>window.heaven?.ready&&window.heaven.bodyWorld.time>=.03,null,{timeout:180000});
  for(const id of ['habitat','subject','controls','optics','cortex','circuits','model'])assert.equal(await page.locator('#window-'+id).count(),1);
  report.samples.push(await read());
  await page.screenshot({path:'reports/flybody-reference-wing-ui-start.png',timeout:15000});
  const start=performance.now();
  for(const target of [.5,1,1.5,2]){
    await page.waitForFunction(t=>window.heaven.bodyWorld.time>=t||window.heaven.state.paused||window.heaven.bodyWorld.complete,target,{timeout:180000});
    const sample=await read();report.samples.push(sample);
    assert(sample.bodyTime>=target&&!sample.paused&&!sample.complete);
    console.log(JSON.stringify({bodyTime:sample.bodyTime,centerError:Math.max(...sample.wings.map(w=>w.centerError))}));
  }
  report.observationWallSeconds=(performance.now()-start)/1000;
  await page.screenshot({path:'reports/flybody-reference-wing-ui-end.png',timeout:15000});
  for(const sample of report.samples){
    assert.equal(sample.population,1);assert.equal(sample.reference,true);assert(sample.renderCalls>0);
    assert.equal(sample.externallyAppliedForce,0);assert(sample.referenceErrorCm<.08);
    for(const wing of sample.wings){
      assert(wing.centerError<1e-9&&Math.abs(wing.determinant-1)<1e-12);
      assert(wing.nativeBodyMatches&&wing.aerodynamicGeometry&&wing.landmarkMatches&&wing.finite);
    }
  }
  assert.deepEqual(errors,[]);assert.equal((await page.locator('#error').textContent()).trim(),'');
  report.passed=true;
}catch(error){report.passed=false;report.failure=error.stack;process.exitCode=1;}
finally{
  await browser.close();
  await fs.writeFile('reports/flybody-reference-wing-ui.json',JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify({...report,samples:report.samples.length},null,2));
}
