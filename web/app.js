import * as THREE from './vendor/three.module.js';

const $=id=>document.getElementById(id);
const countFormat=n=>Math.round(n).toLocaleString();
let meta, snapshot=null, selected=1, stopped=false, following=false;
let renderer, scene, camera, flyMeshes=[], targets=[], selectedRing,renderBatches=[];
let workers=[],readyWorkers=0,totalWorkers=0,latestActivation=null,brainView=false,startedWall=0;
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
  for(const side of [-1,1]){
    ellipsoid(group,materials.eye,.88,.98,.28*side,.22,.25,.15);
    const wing=ellipsoid(group,materials.wing,-.85,1.18,.36*side,1.09,.025,.35);
    wing.rotation.y=-.18*side;
    lineBetween(group,new THREE.Vector3(1,.91,.18*side),new THREE.Vector3(1.37,1.07,.33*side),headMat,.045);
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
    legs.push({node:leg,phase:k*Math.PI*.65+(side>0?Math.PI:0)});
  }
  const proboscis=ellipsoid(group,bodyMat,1.08,.43,0,.10,.30,.10);
  group.userData.legs=legs;group.userData.proboscis=proboscis;
  // Larger invisible picking target; visible anatomy stays at fly scale.
  const hit=new THREE.Mesh(new THREE.SphereGeometry(2.0,8,6),new THREE.MeshBasicMaterial({visible:false}));
  hit.userData.flyId=id;group.add(hit);targets.push(hit);
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
  for(let i=1;i<=meta.flies;i++)createFly(i);
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
  const observer=new ResizeObserver(()=>{const w=host.clientWidth,h=host.clientHeight;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();if(!following)distance=Math.max(178,205/camera.aspect);});observer.observe(host);
  let pointer=null,dragged=false;
  host.addEventListener('pointerdown',e=>{pointer={x:e.clientX,y:e.clientY};dragged=false;host.setPointerCapture(e.pointerId);});
  host.addEventListener('pointermove',e=>{if(!pointer)return;const dx=e.clientX-pointer.x,dy=e.clientY-pointer.y;if(Math.abs(dx)+Math.abs(dy)>2)dragged=true;azimuth-=dx*.006;elevation=Math.max(.25,Math.min(1.47,elevation+dy*.006));pointer={x:e.clientX,y:e.clientY};});
  host.addEventListener('pointerup',e=>{if(!dragged){const rect=host.getBoundingClientRect();const pos=new THREE.Vector2((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1);const ray=new THREE.Raycaster();ray.setFromCamera(pos,camera);const hits=ray.intersectObjects(targets);if(hits.length)selectFly(hits[0].object.userData.flyId);}pointer=null;});
  host.addEventListener('pointercancel',()=>pointer=null);
  host.addEventListener('wheel',e=>{e.preventDefault();distance=Math.max(16,Math.min(250,distance*Math.exp(e.deltaY*.001)));},{passive:false});
  requestAnimationFrame(render);
}

let previousRender=0;
function render(now){
  if(document.hidden||brainView){requestAnimationFrame(render);return;}
  const dt=Math.min(.1,(now-previousRender)/1000||.016);previousRender=now;
  if(snapshot){
    snapshot.flies.forEach((f,i)=>{
      const mesh=flyMeshes[i];mesh.position.set(f.x,f.y+.12,f.z);mesh.rotation.y=-f.heading;
      const phase=snapshot.time_ms*.01*Math.min(f.velocity,2);
      for(const leg of mesh.userData.legs)leg.node.rotation.z=Math.sin(phase+leg.phase)*Math.min(.18,f.velocity*.09);
      mesh.userData.proboscis.scale.y=.3*(1+Math.min(1,f.brain.feed_hz/30));
    });
    const f=snapshot.flies[selected-1];selectedRing.position.set(f.x,f.y+.3,f.z);
    if(following)goalLook.set(f.x,f.y+1,f.z);
  }
  look.lerp(goalLook,Math.min(1,dt*5));
  camera.position.set(look.x+distance*Math.cos(elevation)*Math.sin(azimuth),look.y+distance*Math.sin(elevation),look.z+distance*Math.cos(elevation)*Math.cos(azimuth));camera.lookAt(look);
  scene.updateMatrixWorld(true);
  for(const{batch,originals}of renderBatches){originals.forEach((object,i)=>batch.setMatrixAt(i,object.matrixWorld));batch.instanceMatrix.needsUpdate=true;}
  renderer.render(scene,camera);requestAnimationFrame(render);
}

function selectFly(id){selected=id;$('fly-id').value=String(id);latestActivation=null;snapshot.selected_spikes=[];$('matrix-label').textContent=`Reading Fly ${String(id).padStart(3,'0')} membrane voltages`;$('activation-matrix').getContext('2d').clearRect(0,0,512,512);for(const w of workers)w.postMessage({type:'control',selected});if(snapshot)updatePanel();}
function updatePanel(){
  const f=snapshot.flies[selected-1],b=f.brain;
  $('behavior').textContent=b.feed_hz>2?'Feeding signals active':f.velocity>.1?'Motor signals active':'Neural activity';
  $('contact').textContent=f.contact?'On fruit':'On bowl';
  for(const [label,key,bar]of[['feeding','feed_hz','feed-bar'],['left','left_hz','left-bar'],['right','right_hz','right-bar'],['walking','walk_hz','walk-bar']]){
    $(label).textContent=b[key].toFixed(1)+' Hz';$(bar).style.width=Math.min(100,b[key])+'%';
  }
  $('individual-spikes').textContent=countFormat(b.spikes)+' spikes';$('individual-active').textContent=countFormat(b.active_ever)+' neurons fired';
  $('sim-time').textContent=(snapshot.time_ms/1000).toFixed(2)+' s';
  $('speed').textContent=snapshot.speed?snapshot.speed.toFixed(4)+'× real time':'Measuring';
  $('total-spikes').textContent=countFormat(snapshot.total_spikes);
  $('pause').textContent=snapshot.paused?'Resume brains':'Pause brains';
  $('connection').textContent=readyWorkers<totalWorkers?`Loading brains · ${readyWorkers}/${totalWorkers} workers`:snapshot.paused?'WASM brains paused':`100 WASM brains · ${totalWorkers} workers`;$('live-dot').classList.add('online');
  $('slow-note').textContent=snapshot.speed?`1 neural second ≈ ${Math.round(1/snapshot.speed)} seconds of computation`:'The clock advances with neural computation.';
  drawRaster(snapshot.selected_spikes);
}
function drawRaster(events){
  const canvas=$('raster'),width=canvas.clientWidth,height=canvas.clientHeight,dpr=window.devicePixelRatio||1;
  canvas.width=width*dpr;canvas.height=height*dpr;const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);ctx.clearRect(0,0,width,height);
  ctx.strokeStyle='#dde2d4';ctx.lineWidth=1;
  for(let i=1;i<4;i++){ctx.beginPath();ctx.moveTo(0,height*i/4);ctx.lineTo(width,height*i/4);ctx.stroke();}
  const max=events.length?events[events.length-1][0]:snapshot.time_ms,min=events.length?events[0][0]:Math.max(0,max-100);
  ctx.fillStyle='#6e8950';for(const[t,id]of events){const x=4+(t-min)/Math.max(.1,max-min)*(width-8),y=4+id/meta.neurons_per_brain*(height-8);ctx.fillRect(x,y,1.3,2.8);}
  $('raster-start').textContent=min.toFixed(1)+' ms';$('raster-end').textContent=max.toFixed(1)+' ms';
}
async function control(data){if('paused'in data)snapshot.paused=data.paused;for(const w of workers)w.postMessage({type:'control',...data});updatePanel();}
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
  totalWorkers=Math.min(4,Math.max(1,(navigator.hardwareConcurrency||4)-2));
  for(let n=0;n<totalWorkers;n++){
    const worker=new Worker('/wasm-world-worker.js',{type:'module'});workers.push(worker);
    worker.onmessage=({data})=>{
      if(data.type==='error'){$('error').hidden=false;$('error').textContent=data.message;return;}
      if(data.type==='ready'){readyWorkers++;if(readyWorkers===totalWorkers)startedWall=performance.now();}
      if(data.type==='update'){
        for(const f of data.flies)snapshot.flies[f.id-1]=f;
        snapshot.time_ms=Math.min(...snapshot.flies.map(f=>f.brain.time_ms));
        snapshot.speed=startedWall?snapshot.time_ms/(performance.now()-startedWall):0;
        snapshot.total_spikes=snapshot.flies.reduce((sum,f)=>sum+f.brain.spikes,0);
        if(data.selectedId===selected){latestActivation=data.activation;snapshot.selected_spikes=data.spikes;drawActivation();}
      }
      updatePanel();
    };
    worker.onerror=event=>{$('error').hidden=false;$('error').textContent=event.message;};
    worker.postMessage({type:'init',flies:snapshot.flies.filter((f,i)=>i%totalWorkers===n),fruit:meta.fruit,groups});
  }
}

try{
  const responses=await Promise.all([fetch('/connectome/metadata.json'),fetch('/habitat.json'),fetch('/connectome/groups.json')]);
  if(responses.some(r=>!r.ok))throw new Error('Cannot load connectome or habitat');
  meta=await responses[0].json();const habitat=await responses[1].json(),groups=await responses[2].json();meta.fruit=habitat.fruit;meta.flies=habitat.flies.length;
  const zero=()=>({time_ms:0,spikes:0,active_ever:0,odor_left_hz:0,odor_right_hz:0,sweet_hz:0,walk_hz:0,left_hz:0,right_hz:0,feed_hz:0,antenna_hz:0});
  snapshot={flies:habitat.flies.map(f=>({...f,velocity:0,brain:zero()})),time_ms:0,speed:0,total_spikes:0,selected_spikes:[],paused:false};
  for(let i=1;i<=meta.flies;i++){$('fly-id').add(new Option(String(i).padStart(3,'0'),i));}
  $('population').textContent=meta.flies+' flies';$('neuron-count').textContent=countFormat(meta.neurons_per_brain);
  initScene();launchWorkers(groups);updatePanel();
  $('fly-id').addEventListener('change',()=>selectFly(Number($('fly-id').value)));
  $('pause').addEventListener('click',()=>control({paused:!snapshot?.paused}).catch(console.error));
  $('odor').addEventListener('change',()=>control({odor:$('odor').checked}).catch(console.error));
  $('taste').addEventListener('change',()=>control({taste:$('taste').checked}).catch(console.error));
  $('follow').addEventListener('click',()=>{following=true;distance=25;});
  $('recenter').addEventListener('click',()=>{following=false;distance=Math.max(178,205/camera.aspect);elevation=.79;goalLook.set(0,8,0);});
  $('brain-view').addEventListener('click',()=>{brainView=!brainView;$('brain-panel').hidden=!brainView;$('brain-view').textContent=brainView?'Show bowl':'Show brain matrix';drawActivation();});
  // Small inspectable surface for local QA; no fabricated or replayed brain data.
  window.heaven={get state(){return snapshot;},get meta(){return meta;},get renderer(){return renderer;},selectFly};
}catch(error){$('error').hidden=false;$('error').textContent='Unable to start: '+error.message;console.error(error);}
