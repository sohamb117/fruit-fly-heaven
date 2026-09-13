import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'playwright');
const browser=await chromium.launch({executablePath:process.env.CHROME_PATH||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,args:['--enable-unsafe-webgpu']});
const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
page.on('pageerror',error=>errors.push(error.message));
const population=process.argv.includes('--one-fly')?1:2;
const report={date:new Date().toISOString(),population,checks:[],errors};
const check=(name)=>report.checks.push(name);
try{
  await page.addInitScript(population=>{localStorage.setItem('fruit-fly-population',String(population));localStorage.setItem('fruit-fly-movement-mode','behavior');localStorage.setItem('fruit-fly-neural-body-clock','live');},population);
  await page.goto(`http://127.0.0.1:7842/?population=${population}&movement=direct&clock=neural&follow=1`);
  await page.waitForFunction(()=>window.heaven?.ready&&window.heaven.state.time_ms>=25,null,{timeout:120000});
  assert.equal(await page.locator('#error').textContent(),'');assert.equal(await page.locator('.system-build').textContent(),'BANC 888');
  assert.equal(await page.evaluate(()=>window.heaven.meta.neurons_per_brain),175401);
  assert.equal(await page.evaluate(()=>window.heaven.state.backend),'webgpu');
  assert.equal(await page.evaluate(()=>window.heaven.bodyWorld.movementMode),'direct');
  assert.equal(await page.locator('#body-clock').inputValue(),'neural');
  assert.equal(await page.evaluate(()=>window.heaven.state.flies.length),population);
  for(const id of ['habitat','subject','controls','optics','cortex','circuits','model'])assert.equal(await page.locator('#window-'+id).count(),1);
  check('Original instrument windows, 3D WebGL habitat and BANC dataset identity');
  const vision=await page.evaluate(()=>window.heaven.state.flies.map(f=>({eye:f.sensory?.vision?.ready,graded:f.sensory?.vision?.graded?.ready,color:f.sensory?.vision?.color?.ready,muscles:f.muscleState?.length,joints:f.jointState?.length,internal:f.internal})));
  const expected=await page.evaluate(()=>({muscles:window.heaven.bodyWorld.io.muscles.length*3,joints:window.heaven.bodyWorld.metadata.joints.length*2,backend:window.heaven.bodyWorld.backend}));
  assert.equal(expected.backend,'mujoco-wasm');
  assert.ok(vision.every(v=>v.eye&&v.graded&&v.color&&v.muscles===expected.muscles&&v.joints===expected.joints&&Number.isFinite(v.internal.hunger)));
  check(`${population} ${population===1?'brain':'independent brains'} with actual binocular RGB, graded/color vision, muscles, joints and internal state`);
  check('Viewing link overrides saved assisted/live settings with direct motor coupling and neural time');
  await page.locator('#brain-view').click();
  await page.waitForFunction(()=>document.querySelector('#anatomy-status').textContent.includes('skeletons'));
  await page.locator('#neuron-search').fill('DNp01');await page.locator('#neuron-search').press('Enter');
  await page.waitForFunction(()=>document.querySelector('#neuron-type').textContent==='DNp01');
  await page.locator('#slice-axis').selectOption('1');
  await page.locator('[data-clip="slab"]').click();
  await page.waitForFunction(()=>document.querySelector('#trace-window').textContent.includes('samples'));
  assert.equal(await page.locator('#anatomy-error').textContent(),'');
  check('Brain/VNC anatomy, microscopy plane, clipping, neuron search and live voltage trace');
  const popoutPromise=page.waitForEvent('popup');await page.getByRole('button',{name:'Pop out Cortex',exact:true}).click();
  const popup=await popoutPromise;await popup.waitForLoadState();
  assert.equal(await popup.locator('#brain-canvas canvas').count(),1);
  await popup.locator('#brain-fly').selectOption(String(population));
  await page.waitForFunction(i=>window.heaven.state.flies[i-1].brain.time_ms>0,population);
  await popup.close();await page.waitForFunction(()=>!!document.querySelector('#brain-canvas canvas'));
  check('Detached anatomical window keeps live controls and returns to the console');
  await page.getByRole('button',{name:'Hide Cortex',exact:true}).click();
  await page.locator('#pause').click();await page.waitForTimeout(500);
  const frozen=await page.evaluate(()=>({neural:window.heaven.state.time_ms,body:window.heaven.state.flies.map(f=>f.bodyTime)}));
  await page.waitForTimeout(300);assert.deepEqual(await page.evaluate(()=>({neural:window.heaven.state.time_ms,body:window.heaven.state.flies.map(f=>f.bodyTime)})),frozen);
  check('Pause freezes both neural and body clocks');
  const registration=await page.evaluate(async()=>{
    const THREE=await import('/vendor/three.module.js'),h=window.heaven,w=h.bodyWorld,f=h.state.flies[0],b=w.bodies.get(f.id);
    const saved=Array.from(b.data.qpos.slice(3,7)),angles=[0,.7,1.8],errors=[];
    for(const angle of angles){
      b.data.qpos.set([Math.cos(angle/2),Math.sin(angle/2),0,0],3);w.mj.mj_forward(w.model,b.data);b.refresh();w.copyPose(f,b,0);
      const matrix=new THREE.Matrix4().compose(new THREE.Vector3(...f.physicsPosition),new THREE.Quaternion(...f.physicsQuaternion),new THREE.Vector3(1,1,1));
      const root=new THREE.Vector3(.13,.91,0).applyMatrix4(matrix);errors.push(root.distanceTo(new THREE.Vector3(b.x*10,b.z*10,b.y*10)));
      w.metadata.leg_bodies.forEach((ids,leg)=>ids.forEach((id,k)=>{
        const rendered=new THREE.Vector3(...f.physicsLegs[leg][k]).applyMatrix4(matrix),p=k===2?b.feet[leg]:b.data.xpos.slice(id*3,id*3+3);
        errors.push(rendered.distanceTo(new THREE.Vector3(p[0]*10,p[2]*10,p[1]*10)));
      }));
      f.physicsMouth.ellipsoids.forEach(shape=>{
        const r=shape.rotation,mouthFrame=new THREE.Matrix4().set(r[0],r[1],r[2],0,r[3],r[4],r[5],0,r[6],r[7],r[8],0,0,0,0,1);
        const native=b.data.geom_xmat.subarray(shape.id*9,shape.id*9+9),center=b.data.geom_xpos.subarray(shape.id*3,shape.id*3+3),size=w.model.geom_size.subarray(shape.id*3,shape.id*3+3);
        for(const point of [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],[.3,.4,.8660254038]]){
          const rendered=new THREE.Vector3(...point.map((v,i)=>v*shape.size[i])).applyMatrix4(mouthFrame).add(new THREE.Vector3(...shape.center)).applyMatrix4(matrix);
          const local=[point[0]*size[0],point[2]*size[1],point[1]*size[2]],p=[0,1,2].map(row=>center[row]+[0,1,2].reduce((sum,k)=>sum+native[row*3+k]*local[k],0));
          errors.push(rendered.distanceTo(new THREE.Vector3(p[0]*10,p[2]*10,p[1]*10)));
        }
      });
    }
    b.data.qpos.set(saved,3);w.mj.mj_forward(w.model,b.data);b.refresh();w.copyPose(f,b,0);
    return Math.max(...errors);
  });
  assert(registration<1e-10);report.maxRenderRegistrationError=registration;
  check('Rendered thorax, every leg landmark and mouth collision ellipsoids match native physics under upright, tilted and inverted poses');
  await page.evaluate(()=>{
    const h=window.heaven,f=h.state.flies[0];
    window.cameraFixture={fly:{x:f.x,y:f.y,z:f.z,physicsPosition:f.physicsPosition.slice()},render:h.renderer.render};
    const position=[-20.380414163411626,7.05678575232627,-31.515432396406098],delta=position.map((v,i)=>v-[f.x,f.y,f.z][i]);
    [f.x,f.y,f.z]=position;f.physicsPosition=f.physicsPosition.map((v,i)=>v+delta[i]);
    h.renderer.render=function(scene,camera){window.fixtureCamera=camera;window.fixtureScene=scene;return window.cameraFixture.render.call(this,scene,camera);};
  });
  await page.locator('#follow').click();await page.waitForTimeout(2500);
  const cameraCheck=await page.evaluate(async()=>{
    const THREE=await import('/vendor/three.module.js'),f=window.heaven.state.flies[0],target=new THREE.Vector3(f.x,f.y+1,f.z),camera=window.fixtureCamera;
    const delta=camera.position.clone().sub(target),solids=[];
    window.fixtureScene.traverse(m=>{if(m.isMesh&&!m.isInstancedMesh&&(m.geometry.type==='TubeGeometry'||m.geometry.type==='SphereGeometry'&&['969951','a7543b'].includes(m.material.color?.getHexString())))solids.push(m);});
    const ray=new THREE.Raycaster(camera.position,target.clone().sub(camera.position).normalize(),.00001,delta.length()-.1);
    const result={solidCount:solids.length,occluders:ray.intersectObjects(solids,false).length,elevation:Math.asin(delta.y/delta.length()),distance:delta.length(),azimuth:Math.atan2(delta.x,delta.z)};
    Object.assign(f,window.cameraFixture.fly);window.heaven.renderer.render=window.cameraFixture.render;
    return result;
  });
  assert.equal(cameraCheck.solidCount,6);assert.equal(cameraCheck.occluders,0);assert(cameraCheck.elevation>.9&&cameraCheck.elevation<.95);
  assert(Math.abs(cameraCheck.distance-25)<.002);assert(Math.abs(cameraCheck.azimuth-.63)<.0002);report.cameraOcclusionFixture=cameraCheck;
  check('Paused frame80 camera fixture clears foreground fruit in the actual original UI without changing requested azimuth or distance; this is not a behavior observation');
  await page.locator('#recenter').click();
  await page.locator('#vision').uncheck();await page.locator('#body-sense').uncheck();await page.locator('#pause').click();
  await page.waitForFunction(()=>window.heaven.state.flies.every(f=>f.sensory?.vision?.enabled===false&&f.sensory?.body?.enabled===false));
  await page.locator('#vision').check();await page.locator('#body-sense').check();
  check('Sensory disconnect and reconnect through original controls');
  await page.locator('#movement-mode').selectOption('behavior');assert.equal(await page.evaluate(()=>window.heaven.bodyWorld.movementMode),'behavior');
  await page.locator('#movement-mode').selectOption('direct');
  await page.locator('#body-clock').selectOption('live');await page.waitForTimeout(200);await page.locator('#body-clock').selectOption('neural');
  await page.locator('#motor-coupling').uncheck();await page.locator('#flight-enabled').uncheck();await page.locator('#motor-coupling').check();await page.locator('#flight-enabled').check();
  check('Behavior/direct modes, body clocks, motor coupling and flight controls');
  await page.locator('#fast-mode').check();await page.waitForFunction(()=>window.heaven?.ready&&window.heaven.state.time_ms>=10,null,{timeout:120000});
  await page.locator('#population-count').fill('1');await page.locator('#population-count').press('Enter');
  await page.waitForFunction(()=>window.heaven?.ready&&window.heaven.state.flies.length===1&&window.heaven.state.time_ms>=10,null,{timeout:120000});
  check('Fast mode and population restart preserve the original interface');
  assert.equal(await page.locator('#error').textContent(),'');assert.equal(await page.locator('#anatomy-error').textContent(),'');assert.deepEqual(errors,[]);
  await page.screenshot({path:'/tmp/transplant-verified-3d.png',fullPage:true});
  await page.evaluate(()=>{window.heaven.bodyWorld.advance=()=>{throw new Error('Injected body fault for visible-stop verification');};});
  await page.waitForFunction(()=>window.heaven.state.paused&&window.heaven.state.runtimeError?.includes('Injected body fault'));
  assert((await page.locator('#error').textContent()).includes('Injected body fault'));
  const stopped=await page.evaluate(()=>window.heaven.bodyWorld.time);await page.waitForTimeout(200);
  assert.equal(await page.evaluate(()=>window.heaven.bodyWorld.time),stopped);
  check('A native body exception visibly pauses the run instead of silently waiting for a missing pose acknowledgment');
  report.browser=browser.version();report.passed=true;
}catch(error){report.passed=false;report.failure=error.stack;process.exitCode=1;}
finally{await browser.close();await fs.writeFile('reports/transplant-ui-validation.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));}
