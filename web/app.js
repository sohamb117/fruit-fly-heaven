import * as THREE from './vendor/three.module.js';
import {createAnatomicalViewer} from './brain-view.js';
import {SIMULATION_MODES} from './simulation-modes.js';
import {BodyWorld,BODY_LABELS,MOTOR_DECODER,isAirborne,applyNeuralOutput} from './body-world.js';
import {createRetinalCamera} from './retinal-camera.js';
import {createCircuitInspector} from './circuit-inspector.js';

import {uiElement as $,isUIVisible,onUIFrame} from './ui-elements.js';
import {mountConsole} from './console-shell.js';
const consoleUI=mountConsole();
const countFormat=n=>Math.round(n).toLocaleString();
function showError(message){for(const id of ['error','anatomy-error']){$(id).hidden=false;$(id).textContent=message;}}
let meta, snapshot=null, selected=1, stopped=false, following=false;
let renderer, scene, camera, flyMeshes=[], targets=[], selectedRing,renderBatches=[];
let workers=[],readyWorkers=0,totalWorkers=0,latestActivation=null,brainView=false,startedWall=0,pausedWall=0,pauseStarted=0;
let anatomyViewer=null,selectedNeuron=0;
let habitatData,groupsData,motorOutputsData,fastMode=false,workerGeneration=0,populationSize=100;
let sensoryInputsData,retinalCamera,retinaCursor=0,lastEyeDraw='';
let circuitProbeData,visualModelData,visualProjectionData,colorMappingData;
const circuitInspector=createCircuitInspector($('circuit-inspector'));
let bodyWorld,bodyClock='neural',movementMode='behavior',flightEnabled=true,motorCoupling=true,lastMotionNeuralMs=0,lastPoseSent=0,lastMotionPanel=0;
let flightTrails,trailPositions,lastTrailTime=0;
const poseUp=new THREE.Vector3(),poseForward=new THREE.Vector3(),poseSide=new THREE.Vector3(),poseMatrix=new THREE.Matrix4();
let azimuth=.63,elevation=.79,distance=178;
const look=new THREE.Vector3(0,8,0), goalLook=look.clone();
const host=$('scene');
const sphere=new THREE.SphereGeometry(1,12,8);
const cylinder=new THREE.CylinderGeometry(1,.85,1,5);
const materials={};
function material(color,roughness=.85){return new THREE.MeshStandardMaterial({color,roughness});}
function ellipsoid(parent,mat,x,y,z,sx,sy,sz){const m=new THREE.Mesh(sphere,mat);m.position.set(x,y,z);m.scale.set(sx,sy,sz);parent.add(m);m.castShadow=true;m.receiveShadow=true;return m;}
function seeded(seed){let s=seed;return()=>{s=(Math.imul(1664525,s)+1013904223)>>>0;return s/4294967296;};}
function curvePoint(f,t){const x=(t-.5)*f.length,z=9*(4*(t-.5)**2-1);return new THREE.Vector3(f.x+Math.cos(f.angle)*x-Math.sin(f.angle)*z,f.y,f.z+Math.sin(f.angle)*x+Math.cos(f.angle)*z);}
function lineBetween(parent,a,b,mat,radius=.08){const vector=b.clone().sub(a);const mesh=new THREE.Mesh(cylinder,mat);mesh.scale.set(radius,vector.length(),radius);mesh.position.copy(a).add(b).multiplyScalar(.5);mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),vector.normalize());parent.add(mesh);return mesh;}

function createFruit(f,index){
  const random=seeded(900+index),group=new THREE.Group();scene.add(group);
  const bruise=material('#5e4a2d'),mold=material('#a4a68b');
  if(f.kind==='banana'){
    const points=Array.from({length:31},(_,i)=>curvePoint(f,i/30));
    const curve=new THREE.CatmullRomCurve3(points);
    const body=new THREE.Mesh(new THREE.TubeGeometry(curve,60,f.radius,12,false),material(index===2?'#b69946':'#c4aa58'));
    body.castShadow=body.receiveShadow=true;group.add(body);
    for(const t of [0,1]){const p=curvePoint(f,t);ellipsoid(group,bruise,p.x,p.y,p.z,2.5,2.5,2.5);}
    for(let i=0;i<95;i++){
      const p=curvePoint(f,.03+random()*.94),a=random()*Math.PI*2,r=f.radius*.99;
      const tangent=curve.getTangent((i+.5)/95);const radial=new THREE.Vector3(-tangent.z,0,tangent.x).multiplyScalar(Math.cos(a)*r);
      const size=.18+random()*.58;
      ellipsoid(group,bruise,p.x+radial.x,p.y+Math.sin(a)*r,p.z+radial.z,size,size*.4,size*.75);
    }
  }else{
    const skin=material(index===5?'#969951':'#a7543b');
    ellipsoid(group,skin,f.x,f.y,f.z,f.radius,f.radius*.94,f.radius);
    const top=f.y+f.radius*.91;
    ellipsoid(group,bruise,f.x,top,f.z,2.7,.9,2.7);
    const stem=lineBetween(group,new THREE.Vector3(f.x,top,f.z),new THREE.Vector3(f.x+1.2,top+3,f.z+.5),material('#635839'),.7);
    for(let i=0;i<85;i++){
      const a=random()*Math.PI*2,u=.03+random()*.94;
      const radial=Math.sqrt(1-u*u),r=f.radius*1.002;
      const x=f.x+r*radial*Math.cos(a),z=f.z+r*radial*Math.sin(a),y=f.y+r*u*.94;
      const size=i<25?.5+random()*1.2:.1+random()*.4;
      ellipsoid(group,i<25?mold:bruise,x,y,z,size,.23,size*.75);
    }
    const patch=ellipsoid(group,bruise,f.x-f.radius*.6,f.y+f.radius*.75,f.z,3.1,.5,4.5);patch.rotation.z=.55;
  }
}

function createFly(id){
  const group=new THREE.Group();group.userData.flyId=id;
  const bodyMat=materials.body,headMat=materials.head;
  ellipsoid(group,bodyMat,-.58,.79,0,.72,.46,.45);
  ellipsoid(group,headMat,.13,.91,0,.49,.54,.46);
  ellipsoid(group,headMat,.76,.88,0,.40,.40,.40);
  const wings=[],antennae=[];
  for(const side of [-1,1]){
    ellipsoid(group,materials.eye,.88,.98,.28*side,.22,.25,.15);
    const hinge=new THREE.Group();hinge.position.set(.05,1.2,.2*side);group.add(hinge);
    ellipsoid(hinge,materials.wing,-.85,0,.22*side,1.18,.025,.38);
    const blur=ellipsoid(hinge,materials.wingBlur,-.65,0,.25*side,1.35,.43,.5);
    blur.userData.motionVisible=false;wings.push({hinge,blur,side});
    const antenna=new THREE.Group();antenna.position.set(1,.91,.18*side);group.add(antenna);
    lineBetween(antenna,new THREE.Vector3(),new THREE.Vector3(.37,.16,.15*side),headMat,.045);antennae.push({hinge:antenna,side});
  }
  for(let i=0;i<4;i++){
    const stripe=ellipsoid(group,headMat,-.26-i*.24,.795,0,.055,.445-i*.035,.442-i*.034);stripe.scale.y*=.99;
  }
  const legs=[];
  for(const side of [-1,1])for(let k=0;k<3;k++){
    const leg=new THREE.Group();leg.position.set(.35-k*.49,.66,.28*side);group.add(leg);
    const end=new THREE.Vector3(.45-k*.4,-.15,.69*side);
    lineBetween(leg,new THREE.Vector3(),end,bodyMat,.056);
    lineBetween(leg,end,new THREE.Vector3(.62-k*.53,-.67,1.05*side),bodyMat,.039);
    legs.push({node:leg,phase:(k%2)*Math.PI+(side>0?Math.PI:0),side,front:k===0});
  }
  const proboscis=ellipsoid(group,bodyMat,1.08,.43,0,.10,.30,.10);
  group.userData.legs=legs;group.userData.wings=wings;group.userData.antennae=antennae;group.userData.proboscis=proboscis;group.userData.trail=[];
  // Larger invisible picking target; visible anatomy stays at fly scale.
  const hit=new THREE.Mesh(new THREE.SphereGeometry(2.0,8,6),new THREE.MeshBasicMaterial({visible:false}));
  hit.userData.flyId=id;group.add(hit);targets.push(hit);
  group.traverse(object=>{object.userData.flyId=id;});
  scene.add(group);flyMeshes.push(group);
}

function initScene(){
  renderer=new THREE.WebGLRenderer({antialias:true,alpha:false});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio,1.5));renderer.shadowMap.enabled=true;
  renderer.shadowMap.type=THREE.PCFShadowMap;renderer.setClearColor('#e8e7dd');
  renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.2;
  host.appendChild(renderer.domElement);scene=new THREE.Scene();
  scene.fog=new THREE.Fog('#e8e7dd',260,500);
  camera=new THREE.PerspectiveCamera(38,1,.1,700);
  scene.add(new THREE.HemisphereLight('#fff7df','#9b9c81',2.6));
  const sun=new THREE.DirectionalLight('#fff5dc',3.2);sun.position.set(-70,130,70);sun.castShadow=true;
  sun.shadow.mapSize.set(2048,2048);sun.shadow.camera.left=-85;sun.shadow.camera.right=85;sun.shadow.camera.top=85;sun.shadow.camera.bottom=-85;sun.shadow.normalBias=.1;scene.add(sun);
  const plane=new THREE.Mesh(new THREE.PlaneGeometry(1000,1000),material('#dbdaca'));plane.rotation.x=-Math.PI/2;plane.position.y=-1;plane.receiveShadow=true;scene.add(plane);
  const points=[];for(let r=0;r<=65;r+=2.5)points.push(new THREE.Vector2(r,1.5+.0037*r*r));
  points.push(new THREE.Vector2(66,17),new THREE.Vector2(66.5,15),new THREE.Vector2(62,9),new THREE.Vector2(51,1),new THREE.Vector2(38,-.4),new THREE.Vector2(0,-.4));
  const bowl=new THREE.Mesh(new THREE.LatheGeometry(points,128),material('#f1eddd',.33));bowl.receiveShadow=true;bowl.castShadow=true;scene.add(bowl);
  const rim=new THREE.Mesh(new THREE.TorusGeometry(65.3,.8,12,128),material('#879788',.4));rim.rotation.x=Math.PI/2;rim.position.y=17;scene.add(rim);
  meta.fruit.forEach(createFruit);
  materials.body=material('#805b2d');materials.head=material('#493d2b');materials.eye=material('#a83d28',.38);
  materials.wing=new THREE.MeshStandardMaterial({color:'#e0dfcf',transparent:true,opacity:.55,roughness:.4,side:THREE.DoubleSide,depthWrite:false});
  materials.wingBlur=new THREE.MeshBasicMaterial({color:'#f6f1d9',transparent:true,opacity:.13,depthWrite:false});
  // Cache body geometry for the available range; only the active population
  // receives brain instances, draw instances, and picking targets.
  for(let i=1;i<=habitatData.flies.length;i++)createFly(i);
  // Batch repeated fly/fruit geometry into a few draw calls rather than thousands.
  const batches=new Map();
  scene.traverse(object=>{if(object.isMesh&&(object.geometry===sphere||object.geometry===cylinder)&&object.material.visible){const key=object.geometry.uuid+object.material.uuid;if(!batches.has(key))batches.set(key,[]);batches.get(key).push(object);}});
  for(const originals of batches.values()){
    const batch=new THREE.InstancedMesh(originals[0].geometry,originals[0].material,originals.length);
    batch.instanceMatrix.setUsage(THREE.DynamicDrawUsage);batch.castShadow=batch.receiveShadow=true;batch.frustumCulled=false;
    scene.add(batch);for(const object of originals)object.visible=false;renderBatches.push({batch,originals});
  }
  selectedRing=new THREE.Mesh(new THREE.RingGeometry(2.0,2.2,48),new THREE.MeshBasicMaterial({color:'#77924d',side:THREE.DoubleSide,transparent:true,opacity:.9,depthTest:false}));
  selectedRing.rotation.x=-Math.PI/2;scene.add(selectedRing);
  trailPositions=new Float32Array(habitatData.flies.length*24*6);
  const trailGeometry=new THREE.BufferGeometry();trailGeometry.setAttribute('position',new THREE.BufferAttribute(trailPositions,3).setUsage(THREE.DynamicDrawUsage));trailGeometry.setDrawRange(0,0);
  flightTrails=new THREE.LineSegments(trailGeometry,new THREE.LineBasicMaterial({color:'#667e6a',transparent:true,opacity:.3,depthWrite:false}));flightTrails.frustumCulled=false;scene.add(flightTrails);
  retinalCamera=createRetinalCamera(renderer,scene,sensoryInputsData.vision);
  const observer=new ResizeObserver(()=>{const w=host.clientWidth,h=host.clientHeight;if(!w||!h)return;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();if(!following)distance=Math.max(178,205/camera.aspect);});observer.observe(host);
  let pointer=null,dragged=false;
  host.addEventListener('pointerdown',e=>{pointer={x:e.clientX,y:e.clientY};dragged=false;host.setPointerCapture(e.pointerId);});
  host.addEventListener('pointermove',e=>{if(!pointer)return;const dx=e.clientX-pointer.x,dy=e.clientY-pointer.y;if(Math.abs(dx)+Math.abs(dy)>2)dragged=true;azimuth-=dx*.006;elevation=Math.max(.25,Math.min(1.47,elevation+dy*.006));pointer={x:e.clientX,y:e.clientY};});
  host.addEventListener('pointerup',e=>{if(!dragged){const rect=host.getBoundingClientRect();const pos=new THREE.Vector2((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1);const ray=new THREE.Raycaster();ray.setFromCamera(pos,camera);const hits=ray.intersectObjects(targets.filter(target=>target.userData.flyId<=populationSize));if(hits.length)selectFly(hits[0].object.userData.flyId);}pointer=null;});
  host.addEventListener('pointercancel',()=>pointer=null);
  host.addEventListener('wheel',e=>{e.preventDefault();distance=Math.max(16,Math.min(250,distance*Math.exp(e.deltaY*.001)));},{passive:false});
  onUIFrame(render);
}

let previousRender=0;
function render(now){
  const gap=(now-previousRender)/1000,dt=gap>0&&gap<.25?gap:0;previousRender=now;
  const neuralDelta=snapshot?Math.max(0,snapshot.time_ms-lastMotionNeuralMs)/1000:0;
  lastMotionNeuralMs=snapshot?.time_ms||0;
  if(!isUIVisible())return;
  if(bodyWorld&&snapshot&&!snapshot.paused&&readyWorkers===totalWorkers){
    bodyWorld.advance(bodyClock==='live'?dt:neuralDelta);
    if(now-lastPoseSent>50){
      const poses=bodyWorld.poses();
      workers.forEach((worker,n)=>worker.postMessage({type:'poses',poses:poses.filter((f,i)=>i%totalWorkers===n)}));lastPoseSent=now;
    }
  }
  if(snapshot&&now-lastMotionPanel>150){updateMotionPanel();lastMotionPanel=now;}
  if(snapshot){
    snapshot.flies.forEach((f,i)=>{
      const mesh=flyMeshes[i],air=isAirborne(f),a=f.actuators,walking=f.motion==='walking'||f.motion==='turning';
      mesh.position.set(f.x,f.y+.14+(walking?Math.abs(Math.sin(f.gait))*.055:0),f.z);
      if(air)mesh.rotation.set(f.bank,-f.heading,f.pitch,'YXZ');
      else{
        poseUp.fromArray(f.normal);poseForward.set(Math.cos(f.heading),0,Math.sin(f.heading));
        poseForward.addScaledVector(poseUp,-poseForward.dot(poseUp)).normalize();
        poseSide.crossVectors(poseForward,poseUp).normalize();poseMatrix.makeBasis(poseForward,poseUp,poseSide);mesh.quaternion.setFromRotationMatrix(poseMatrix);
      }
      for(const [j,leg]of mesh.userData.legs.entries()){
        leg.node.rotation.set(...f.joints.legs[j]);
      }
      for(const [j,wing]of mesh.userData.wings.entries()){
        const power=wing.side<0?a.wingLeft:a.wingRight;
        wing.hinge.rotation.set(...f.joints.wings[j]);
        wing.blur.userData.motionVisible=power>.08;
      }
      for(const [j,antenna]of mesh.userData.antennae.entries())antenna.hinge.rotation.y=f.joints.antennae[j];
      mesh.userData.proboscis.scale.y=f.joints.proboscis;
      if(bodyWorld.time-lastTrailTime>.06){
        const trail=mesh.userData.trail;
        if(air)trail.push({x:f.x,y:f.y+.7,z:f.z,time:bodyWorld.time});
        while(trail.length>24||(trail.length&&bodyWorld.time-trail[0].time>1.5))trail.shift();
      }
    });
    if(bodyWorld.time-lastTrailTime>.06)lastTrailTime=bodyWorld.time;
    const f=snapshot.flies[selected-1];selectedRing.position.set(f.x,f.y+.3,f.z);
    selectedRing.rotation.set(-Math.PI/2,0,0);selectedRing.scale.setScalar(isAirborne(f)?1.3:1);
    if(following)goalLook.set(f.x,f.y+1,f.z);
  }
  look.lerp(goalLook,Math.min(1,dt*5));
  camera.position.set(look.x+distance*Math.cos(elevation)*Math.sin(azimuth),look.y+distance*Math.sin(elevation),look.z+distance*Math.cos(elevation)*Math.cos(azimuth));camera.lookAt(look);
  scene.updateMatrixWorld(true);
  for(const{batch,originals}of renderBatches){
    let count=0;
    for(const object of originals){
      if(object.userData.flyId>populationSize||object.userData.motionVisible===false)continue;
      batch.setMatrixAt(count++,object.matrixWorld);
    }
    batch.count=count;batch.instanceMatrix.needsUpdate=true;
  }
  let trailCount=0;
  for(const mesh of flyMeshes.slice(0,populationSize)){
    const trail=mesh.userData.trail;
    for(let i=1;i<trail.length;i++)for(const p of [trail[i-1],trail[i]]){trailPositions[trailCount++]=p.x;trailPositions[trailCount++]=p.y;trailPositions[trailCount++]=p.z;}
  }
  flightTrails.geometry.setDrawRange(0,trailCount/3);flightTrails.geometry.attributes.position.needsUpdate=true;flightTrails.visible=$('flight-trails').checked;
  if(snapshot&&!snapshot.paused&&readyWorkers===totalWorkers&&$('vision').checked){
    // One fly (two small renders) per animation frame bounds GPU/readback work.
    // Every fly is sampled at body time, independent of which one is inspected.
    for(let checked=0;checked<populationSize;checked++){
      const i=retinaCursor++%populationSize,f=snapshot.flies[i];
      if(f.eyeBodyTime!==undefined&&f.bodyTime-f.eyeBodyTime+1e-9<sensoryInputsData.vision.frame_interval_body_seconds)continue;
      try{
        const frame=retinalCamera.capture(f,flyMeshes[i],[selectedRing,flightTrails]);f.eyeBodyTime=f.bodyTime;
        workers[i%totalWorkers].postMessage({type:'eyes',frames:[frame]});
      }catch(error){control({paused:true});showError('Cannot sample visual input: '+error.message);}
      break;
    }
  }
  if(consoleUI.isOpen('habitat')&&!host.ownerDocument.hidden)renderer.render(scene,camera);
}

function selectFly(id){if(!Number.isInteger(id)||id<1||id>populationSize)return;selected=id;snapshot.flies[id-1].circuit=null;$('fly-id').value=String(id);$('brain-fly').value=String(id);latestActivation=null;snapshot.selected_spikes=[];anatomyViewer?.selectFly(id);$('matrix-label').textContent=`Reading Fly ${String(id).padStart(3,'0')} membrane voltages`;$('activation-matrix').getContext('2d').clearRect(0,0,512,512);for(const w of workers)w.postMessage({type:'control',selected});if(snapshot)updatePanel();}
function setBrainView(value){value?consoleUI.open('cortex'):consoleUI.hide('cortex');}
consoleUI.panels.cortex.node.addEventListener('windowvisibility',event=>{brainView=event.detail.visible;anatomyViewer?.setActive(brainView);});
function updatePanel(){
  const f=snapshot.flies[selected-1],b=f.brain;
  updateMotionPanel();
  circuitInspector.update(f.circuit,selected);
  const motor=b.motor||{};
  const outputs={...motor,walk:b.walk_hz,wing:((motor.wing_left||0)+(motor.wing_right||0))/2};
  for(const [label,key,bar]of[['feeding','proboscis','feed-bar'],['left','turn_left','left-bar'],['right','turn_right','right-bar'],['behavior-walk','walk','behavior-walk-bar'],['walking','forward','walk-bar'],['wing-output','wing','wing-bar'],['jump-output','takeoff','jump-bar'],['landing-output','landing','landing-bar'],['groom-output','groom','groom-bar']]){
    const hz=outputs[key]||0;$(label).textContent=hz.toFixed(1)+' Hz';$(bar).style.width=Math.min(100,hz)+'%';
  }
  for(const channel of motorOutputsData.channels)$('motor-rate-'+channel.key).textContent=(motor[channel.key]||0).toFixed(1)+' Hz';
  $('individual-spikes').textContent=countFormat(b.spikes)+' spikes';$('individual-active').textContent=countFormat(b.active_ever)+' neurons fired';
  $('sim-time').textContent=(snapshot.time_ms/1000).toFixed(2)+' s';
  $('speed').textContent=snapshot.speed?snapshot.speed.toFixed(4)+'× real time':'Measuring';
  $('total-spikes').textContent=countFormat(snapshot.total_spikes);
  $('pause').textContent=snapshot.paused?'Resume habitat':'Pause habitat';
  $('connection').textContent=readyWorkers<totalWorkers?`Loading brains · ${readyWorkers}/${totalWorkers} workers`:snapshot.paused?'Habitat paused':`${populationSize} WASM ${populationSize===1?'brain':'brains'} · ${totalWorkers} ${totalWorkers===1?'worker':'workers'}`;$('live-dot').classList.add('online');
  $('slow-note').textContent=snapshot.speed?`1 neural second ≈ ${Math.round(1/snapshot.speed)} seconds of computation`:'The clock advances with neural computation.';
  drawRaster(snapshot.selected_spikes);
}
function updateMotionPanel(){
  const f=snapshot.flies[selected-1];
  $('behavior').textContent=BODY_LABELS[f.motion];$('contact').textContent=isAirborne(f)?'Airborne':f.contact?'On fruit':'On bowl';
  $('body-time').textContent=bodyWorld.time.toFixed(1)+' s';
  $('body-speed').textContent=(f.velocity/2.6).toFixed(2)+' lengths/s';
  $('body-heading').textContent=(((f.heading*180/Math.PI)%360+360)%360).toFixed(1)+'°';
  $('body-height').textContent=(f.altitude/2.6).toFixed(1)+' lengths';
  $('body-position').textContent=`${(f.x/2.6).toFixed(2)}, ${(f.y/2.6).toFixed(2)}, ${(f.z/2.6).toFixed(2)}`;
  $('airborne-count').textContent=snapshot.flies.filter(isAirborne).length+' airborne';
  $('follow-flight').disabled=!snapshot.flies.some(isAirborne);
  $('landings-count').textContent=bodyWorld.landings+(bodyWorld.landings===1?' landing':' landings');
  $('body-clock-note').textContent=bodyClock==='live'?'Live actuation holds the latest neural outputs on a separate body clock.':'Bodies advance with computed neural time. Use Live to watch the latest outputs drive movement at wall-clock speed.';
  $('movement-note').textContent=movementMode==='behavior'?'Behavior control: walking and steering activity drive movement and prepare flight; feeding slows preparation. A modeled flight program supplies sustained wings and landing using body feedback.':'Direct motor mapping: each annotated channel drives its own actuator, with a 2 Hz threshold. Steering alone turns the body in place; flight needs sustained wing-neuron output.';
  const a=bodyWorld.readCommands(f.brain,f),drives=[['forward',a.forward],['reverse',a.reverse],['turn',Math.abs(a.turn)],['wing power',a.wing],['jump',a.jump],['landing',a.landing],['groom',a.groom],['proboscis',a.proboscis],['antennae',Math.max(a.antennaLeft,a.antennaRight)]].filter(([,value])=>value>.01).map(([name])=>name);
  $('command-speed').textContent=((MOTOR_DECODER.walkSpeed*a.forward-MOTOR_DECODER.reverseSpeed*a.reverse)/2.6).toFixed(2)+' lengths/s';
  $('command-turn').textContent=(MOTOR_DECODER.yawRate*a.turn*180/Math.PI).toFixed(1)+'°/s';
  const program=f.flightProgram;
  $('flight-program').textContent=!flightEnabled?'Flight off':!motorCoupling?'Disconnected':movementMode==='direct'?'Direct neurons':program.phase==='grounded'?(program.recovery>0?'Resting after flight':`Preparing · ${(program.preparation*100).toFixed(0)}%`):({takeoff:'Takeoff',cruise:'Sustained flight',landing:'Landing'}[program.phase]);
  $('command-wing').textContent=(a.wing*100).toFixed(0)+'%';
  const controller=movementMode==='behavior'?'Behavior mapping':'Direct motor mapping';
  $('actuator-status').textContent=!motorCoupling?'Motor outputs disconnected · '+(snapshot.paused?'habitat paused':'brain still computing'):drives.length?controller+': '+drives.join(' · '):controller+' · no active drive';
  $('sampled-senses').textContent=f.senses?`Latest brain inputs · odor ${((f.senses[0]+f.senses[1])/2).toFixed(0)} Hz · sugar ${f.senses[2].toFixed(0)} Hz`:'Waiting for sensory feedback';
  updateSensoryPanel(f);
}
function updateSensoryPanel(f){
  const s=f.sensory;
  const g=s?.vision.graded,c=s?.vision.color;
  $('color-model-status').textContent=!$('vision').checked?'Color inputs disconnected':c?.ready?`RGB → receptor mapping · WASM · ${c.meanDriveHz.toFixed(1)} Hz mean color input`:'Waiting for RGB input';
  for(const side of ['left','right'])$('color-'+side).textContent=(c?.[side]||[0,0,0,0]).map(v=>v.toFixed(4)).join(' · ');
  $('vision-model-status').textContent=!$('vision').checked?'Visual inputs disconnected':g?.ready?`Graded vision running · ${(g.neuralTimeMs/1000).toFixed(2)} neural s · ${g.meanDriveHz.toFixed(1)} Hz mean bridge drive`:'Loading graded visual model';
  for(const [j,side]of ['left','right'].entries())for(const path of ['T4','T5']){
    const values=['a','b','c','d'].map(suffix=>(g?.activity[path+suffix]?.[j]||0).toFixed(3));
    $('graded-'+side+'-'+path).textContent=values.join(' · ');
  }
  $('sensory-fly').textContent=`What Fly ${String(f.id).padStart(3,'0')} sees and senses`;
  $('eye-status').textContent=!$('vision').checked?'Vision disconnected':s?.vision.ready?`Supplied to brain · frame ${s.vision.sequence} · body ${s.vision.frameBodyTime.toFixed(2)} s`:'Waiting for rendered eyes';
  for(const side of ['left','right'])$('eye-rate-'+side).textContent=(s?.vision[side+'Hz']||0).toFixed(1)+' Hz';
  $('eye-change').textContent=((s?.vision.contrast||0)*100).toFixed(1)+'%';
  $('body-sense-status').textContent=!$('body-sense').checked?'Body sense disconnected':s?'Posture and motion → ascending-input proxy':'Waiting for body sample';
  $('sense-support').textContent=((s?.body.support||0)*100).toFixed(0)+'%';
  $('sense-joints').textContent=(s?.body.jointSpeed||0).toFixed(2)+' rad/s';
  $('sense-tilt').textContent=((s?.body.tilt||0)*180/Math.PI).toFixed(1)+'°';
  $('sense-yaw').textContent=((s?.body.yaw||0)*180/Math.PI).toFixed(1)+'°/s';
  for(const c of sensoryInputsData.channels)$('sense-rate-'+c.key).textContent=(s?.body.rates[c.key]||0).toFixed(1)+' Hz';
  const frame=f.sensoryFrame,key=[f.id,frame?.sequence,$('vision').checked].join(':');
  if(key!==lastEyeDraw){
    lastEyeDraw=key;
    for(const [i,side]of ['left','right'].entries()){
      const canvas=$('eye-'+side),ctx=canvas.getContext('2d'),image=ctx.createImageData(32,16);
      for(let p=0;p<512;p++){const at=i*512+p,n=frame?.pixels[at]??0,base=at*3;image.data.set(frame?.rgb?[frame.rgb[base],frame.rgb[base+1],frame.rgb[base+2],255]:[n,n,n,255],p*4);}
      ctx.putImageData(image,0,0);
    }
  }
}
function drawRaster(events){
  const canvas=$('raster'),width=canvas.clientWidth,height=canvas.clientHeight,dpr=window.devicePixelRatio||1;
  canvas.width=width*dpr;canvas.height=height*dpr;const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);ctx.clearRect(0,0,width,height);
  ctx.strokeStyle='#1a302d';ctx.lineWidth=1;
  for(let i=1;i<4;i++){ctx.beginPath();ctx.moveTo(0,height*i/4);ctx.lineTo(width,height*i/4);ctx.stroke();}
  const max=events.length?events[events.length-1][0]:snapshot.time_ms,min=events.length?events[0][0]:Math.max(0,max-100);
  ctx.fillStyle='#8bf7b3';for(const[t,id]of events){const x=4+(t-min)/Math.max(.1,max-min)*(width-8),y=4+id/meta.neurons_per_brain*(height-8);ctx.fillRect(x,y,1.3,2.8);}
  $('raster-start').textContent=min.toFixed(1)+' ms';$('raster-end').textContent=max.toFixed(1)+' ms';
}
async function control(data){
  if('paused'in data && data.paused!==snapshot.paused){
    if(data.paused)pauseStarted=startedWall?performance.now():0;
    else if(pauseStarted){pausedWall+=performance.now()-pauseStarted;pauseStarted=0;}
    snapshot.paused=data.paused;
  }
  for(const w of workers)w.postMessage({type:'control',...data});updatePanel();
}
function drawActivation(){
  if(!latestActivation)return;
  const canvas=$('activation-matrix'),width=512,height=Math.ceil(meta.neurons_per_brain/width);
  canvas.width=width;canvas.height=height;const ctx=canvas.getContext('2d'),pixels=ctx.createImageData(width,height);
  for(let i=0;i<latestActivation.length;i++){
    const v=latestActivation[i],p=i*4,a=Math.min(1,Math.max(0,(v+52)/7));
    if(v<-52){pixels.data[p]=32;pixels.data[p+1]=56;pixels.data[p+2]=82;}
    else{pixels.data[p]=24+Math.round(a*221);pixels.data[p+1]=36+Math.round(a*186);pixels.data[p+2]=32+Math.round(a*89);}
    pixels.data[p+3]=255;
  }
  ctx.putImageData(pixels,0,0);$('matrix-label').textContent=`Fly ${String(selected).padStart(3,'0')} · all ${countFormat(meta.neurons_per_brain)} membrane voltages`;
}
function launchWorkers(groups){
  const generation=++workerGeneration,mode=fastMode?'fast':'reference';
  totalWorkers=Math.min(populationSize,4,Math.max(1,(navigator.hardwareConcurrency||4)-2));
  for(let n=0;n<totalWorkers;n++){
    const worker=new Worker('/wasm-world-worker.js',{type:'module'});workers.push(worker);
    worker.onmessage=({data})=>{
      if(generation!==workerGeneration)return;
      if(data.type==='error'){control({paused:true});showError(data.message);return;}
      if(data.type==='ready'){readyWorkers++;if(readyWorkers===totalWorkers){startedWall=performance.now();if(snapshot.paused)pauseStarted=startedWall;}}
      if(data.type==='update'){
        for(const f of data.flies)applyNeuralOutput(snapshot.flies[f.id-1],f);
        snapshot.time_ms=Math.min(...snapshot.flies.map(f=>f.brain.time_ms));
        const elapsed=(pauseStarted||performance.now())-startedWall-pausedWall;
        snapshot.speed=startedWall&&elapsed>0?snapshot.time_ms/elapsed:0;
        snapshot.total_spikes=snapshot.flies.reduce((sum,f)=>sum+f.brain.spikes,0);
        if(data.selectedId===selected){if(data.circuit)snapshot.flies[selected-1].circuit=data.circuit;latestActivation=data.activation;snapshot.selected_spikes=data.spikes;drawActivation();try{anatomyViewer?.update({flyId:selected,activation:data.activation,lastSpikeMs:data.lastSpikeMs,neuralTimeMs:snapshot.flies[selected-1].brain.time_ms,recording:data.trace});}catch(error){showError('Cannot display neural state: '+error.message);}}
      }
      updatePanel();
    };
    worker.onerror=event=>{if(generation===workerGeneration){control({paused:true});showError(event.message);}};
    worker.postMessage({type:'init',flies:bodyWorld.poses().filter((f,i)=>i%totalWorkers===n),fruit:meta.fruit,groups,motorOutputs:motorOutputsData,sensoryInputs:sensoryInputsData,circuitProbe:circuitProbeData,visualModel:visualModelData,visualMapping:visualProjectionData,colorMapping:colorMappingData,selectedNeuron,mode,selected,paused:snapshot.paused,odor:$('odor').checked,taste:$('taste').checked,vision:$('vision').checked,bodySense:$('body-sense').checked});
  }
}

function freshSnapshot(paused=false){
  const zero=()=>({time_ms:0,spikes:0,active_ever:0,odor_left_hz:0,odor_right_hz:0,sweet_hz:0,walk_hz:0,left_hz:0,right_hz:0,feed_hz:0,antenna_hz:0,motor:{}});
  const flies=habitatData.flies.slice(0,populationSize).map(f=>({...f,brain:zero()}));
  bodyWorld=new BodyWorld(habitatData.fruit,flies,{flightEnabled,motorCoupling,movementMode});lastMotionNeuralMs=lastPoseSent=lastTrailTime=0;
  retinaCursor=0;lastEyeDraw='';
  for(const mesh of flyMeshes)mesh.userData.trail=[];
  return {flies,time_ms:0,speed:0,total_spikes:0,selected_spikes:[],paused};
}
function populationValue(value){
  const number=Number(value);
  return value===null||String(value).trim()===''||!Number.isFinite(number)?populationSize:Math.max(1,Math.min(habitatData.flies.length,Math.round(number)));
}
function previewPopulation(value){
  const count=populationValue(value);
  $('population-size').value=String(count);$('population-count').value=String(count);
  $('population-unit').textContent=count===1?'fly':'flies';
  $('population-size').setAttribute('aria-valuetext',`${count} ${count===1?'fly':'flies'}`);
}
function syncPopulationView(){
  meta.flies=populationSize;selected=Math.min(selected,populationSize);
  previewPopulation(populationSize);
  for(const id of ['fly-id','brain-fly']){
    const picker=$(id);picker.replaceChildren();
    for(let i=1;i<=populationSize;i++)picker.add(new Option(String(i).padStart(3,'0'),i));
    picker.value=String(selected);
  }
  const flies=`${populationSize} ${populationSize===1?'fly':'flies'}`;
  $('population').textContent=flies;
  $('population-summary').textContent=`${populationSize} independent neural ${populationSize===1?'state':'states'}. A bottomless bowl of bananas and apples.`;
  $('model-population').textContent=`Actual connectome wiring · ${populationSize} independent WASM ${populationSize===1?'brain':'brains'} · modeled bodies`;
  host.setAttribute('aria-label',`A three-dimensional bowl of rotting bananas and apples with ${flies}. Movement commands come from annotated brain outputs; body mechanics and muscle actuation are modeled.`);
  for(const mesh of flyMeshes)mesh.visible=mesh.userData.flyId<=populationSize;
}
function changePopulation(value){
  const next=populationValue(value);previewPopulation(next);
  if(next===populationSize)return;
  populationSize=next;
  try{localStorage.setItem('fruit-fly-population',String(populationSize));}catch{}
  restartPopulation();
}
function showPrecision(){
  const mode=SIMULATION_MODES[fastMode?'fast':'reference'];
  $('fast-mode').checked=fastMode;$('precision-label').textContent=mode.label;$('precision-description').textContent=mode.description;
  if(anatomyViewer)anatomyViewer.setTimeStep(mode.dtMs);
  else $('trace-window').textContent=`Samples every ${mode.dtMs} ms of neural time`;
}
function restartPopulation(){
  ++workerGeneration;
  for(const worker of workers)worker.terminate();
  workers=[];readyWorkers=0;startedWall=pausedWall=pauseStarted=0;
  snapshot=freshSnapshot(snapshot.paused);latestActivation=null;
  syncPopulationView();
  for(const id of ['error','anatomy-error']){$(id).hidden=true;$(id).textContent='';}
  const matrix=$('activation-matrix');matrix.getContext('2d').clearRect(0,0,matrix.width,matrix.height);
  $('matrix-label').textContent='Waiting for new neural state';
  anatomyViewer?.selectFly(selected);showPrecision();launchWorkers(groupsData);updatePanel();
}

try{
  const responses=await Promise.all([fetch('/connectome/metadata.json'),fetch('/habitat.json'),fetch('/connectome/groups.json'),fetch('/motor-outputs.json'),fetch('/sensory-inputs.json'),fetch('/circuit-probe.json'),fetch('/visual-model.json'),fetch('/visual-projections.json'),fetch('/color-inputs.json')]);
  if(responses.some(r=>!r.ok))throw new Error('Cannot load connectome or habitat');
  meta=await responses[0].json();const habitat=await responses[1].json(),groups=await responses[2].json();meta.fruit=habitat.fruit;meta.flies=habitat.flies.length;
  habitatData=habitat;groupsData=groups;motorOutputsData=await responses[3].json();
  colorMappingData=await responses[8].json();
  sensoryInputsData=await responses[4].json();circuitProbeData=await responses[5].json();
  const modelText=await responses[6].text();visualModelData=JSON.parse(modelText);visualProjectionData=await responses[7].json();
  const modelHash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(modelText))),n=>n.toString(16).padStart(2,'0')).join('');
  if(modelHash!==visualProjectionData.model_sha256||visualProjectionData.ids_sha256!==meta.prepared_sha256['ids.bin'])throw new Error('Graded vision data do not match this model/connectome');
  for(const file of ['ids.bin','indptr.bin','targets.bin','weights.bin'])if(circuitProbeData.source_sha256['data/prepared/'+file]!==meta.prepared_sha256[file])throw new Error('Circuit probes do not match the loaded graph: '+file);
  for(const file of ['ids.bin','indptr.bin','targets.bin','weights.bin'])if(sensoryInputsData.source_sha256['data/prepared/'+file]!==meta.prepared_sha256[file])throw new Error('Sensory annotations do not match the loaded graph: '+file);
  $('sensory-mapping-summary').textContent=`${countFormat(sensoryInputsData.vision.receptors.length)} R1–6 photoreceptors receive spatial visual input. ${countFormat(sensoryInputsData.vision.unmapped_root_ids.length)} R1–6 cells lack a usable column assignment and receive no added visual drive. Eight body-sense channels use existing input neurons; their tuning is a modeled boundary approximation.`;
  for(const c of sensoryInputsData.channels){
    const row=document.createElement('div');row.className='sense-channel';
    const label=document.createElement('span');label.textContent=c.label;
    const value=document.createElement('strong');value.id='sense-rate-'+c.key;value.textContent='0.0 Hz';
    row.append(label,value);$('body-sense-rates').appendChild(row);
  }
  if(motorOutputsData.source_sha256['data/prepared/ids.bin']!==meta.prepared_sha256['ids.bin'])throw new Error('Motor output annotations do not match the loaded neuron IDs');
  for(const channel of motorOutputsData.channels){
    const card=document.createElement('div');card.className='motor-channel';
    const title=document.createElement('strong');title.textContent=channel.label;
    const rate=document.createElement('span');rate.id='motor-rate-'+channel.key;rate.textContent='0.0 Hz';
    const cells=document.createElement('span');cells.textContent=[...new Set(channel.cells.map(c=>c.type))].join(', ')+` · ${channel.indices.length} neurons`;
    const mapping=document.createElement('p');mapping.textContent=channel.decoder;
    const source=document.createElement('a');source.href=channel.source;source.target='_blank';source.rel='noreferrer';source.textContent='Evidence ↗';
    card.append(title,rate,cells,mapping,source);$('motor-channels').appendChild(card);
  }
  populationSize=habitatData.flies.length;
  try{fastMode=localStorage.getItem('fruit-fly-fast-mode')==='true';}catch{}
  try{populationSize=populationValue(localStorage.getItem('fruit-fly-population'));}catch{}
  try{bodyClock=localStorage.getItem('fruit-fly-neural-body-clock')==='live'?'live':'neural';flightEnabled=localStorage.getItem('fruit-fly-flight')!=='false';}catch{}
  try{movementMode=localStorage.getItem('fruit-fly-movement-mode')==='direct'?'direct':'behavior';}catch{}
  $('body-clock').value=bodyClock;$('flight-enabled').checked=flightEnabled;$('movement-mode').value=movementMode;
  $('movement-mode').addEventListener('change',()=>{movementMode=$('movement-mode').value;bodyWorld.setMovementMode(movementMode);try{localStorage.setItem('fruit-fly-movement-mode',movementMode);}catch{}updateMotionPanel();});
  $('body-clock').addEventListener('change',()=>{bodyClock=$('body-clock').value;try{localStorage.setItem('fruit-fly-neural-body-clock',bodyClock);}catch{}updateMotionPanel();});
  $('motor-coupling').addEventListener('change',()=>{motorCoupling=$('motor-coupling').checked;bodyWorld.setMotorCoupling(motorCoupling);updateMotionPanel();});
  $('flight-enabled').addEventListener('change',()=>{flightEnabled=$('flight-enabled').checked;bodyWorld.setFlightEnabled(flightEnabled);try{localStorage.setItem('fruit-fly-flight',String(flightEnabled));}catch{}});
  snapshot=freshSnapshot();syncPopulationView();showPrecision();$('fast-mode').disabled=false;
  for(const id of ['population-size','population-count']){$(id).max=String(habitatData.flies.length);$(id).disabled=false;}
  $('population-max').textContent=String(habitatData.flies.length);
  $('population-size').addEventListener('input',()=>previewPopulation($('population-size').value));
  $('population-size').addEventListener('change',()=>changePopulation($('population-size').value));
  $('population-count').addEventListener('change',()=>changePopulation($('population-count').value));
  $('population-count').addEventListener('keydown',event=>{if(event.key==='Enter'){event.preventDefault();changePopulation($('population-count').value);$('population-count').blur();}});
  $('fast-mode').addEventListener('change',()=>{
    fastMode=$('fast-mode').checked;
    try{localStorage.setItem('fruit-fly-fast-mode',String(fastMode));}catch{}
    restartPopulation();
  });
  $('neuron-count').textContent=countFormat(meta.neurons_per_brain);
  selectedNeuron=groups.steer_left[0];initScene();syncPopulationView();launchWorkers(groups);updatePanel();
  $('fly-id').addEventListener('change',()=>selectFly(Number($('fly-id').value)));
  $('brain-fly').addEventListener('change',()=>selectFly(Number($('brain-fly').value)));
  $('pause').addEventListener('click',()=>control({paused:!snapshot?.paused}).catch(console.error));
  $('odor').addEventListener('change',()=>control({odor:$('odor').checked}).catch(console.error));
  $('taste').addEventListener('change',()=>control({taste:$('taste').checked}).catch(console.error));
  $('vision').addEventListener('change',()=>{for(const f of snapshot.flies)delete f.eyeBodyTime;control({vision:$('vision').checked}).catch(console.error);});
  $('body-sense').addEventListener('change',()=>control({bodySense:$('body-sense').checked}).catch(console.error));
  $('follow').addEventListener('click',()=>{following=true;distance=25;});
  $('follow-flight').addEventListener('click',()=>{const flier=snapshot.flies.find(isAirborne);if(flier){selectFly(flier.id);following=true;distance=32;}});
  $('recenter').addEventListener('click',()=>{following=false;distance=Math.max(178,205/camera.aspect);elevation=.79;goalLook.set(0,8,0);});
  $('brain-view').addEventListener('click',()=>setBrainView(true));
  $('back-to-bowl').addEventListener('click',()=>setBrainView(false));
  brainView=consoleUI.isOpen('cortex');
  createAnatomicalViewer({initialNeuron:selectedNeuron,onNeuronSelect:index=>{selectedNeuron=index;for(const w of workers)w.postMessage({type:'control',selectedNeuron:index});}}).then(viewer=>{anatomyViewer=viewer;showPrecision();viewer.selectFly(selected);viewer.setActive(brainView);}).catch(error=>{$('anatomy-error').hidden=false;$('anatomy-error').textContent='Cannot load anatomical viewer: '+error.message;console.error(error);});
  // Small inspectable surface for local QA; no fabricated or replayed brain data.
  window.heaven={get state(){return snapshot;},get meta(){return meta;},get renderer(){return renderer;},selectFly};
}catch(error){$('error').hidden=false;$('error').textContent='Unable to start: '+error.message;console.error(error);}
