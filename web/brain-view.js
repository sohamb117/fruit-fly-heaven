import * as THREE from './vendor/three.module.js';
import {createBrainViewModule} from '/view-engine/index.js';
import {uiElement as $,uiQueryAll,onUIFrame} from './ui-elements.js';
const fmt=n=>Math.round(n).toLocaleString();

export async function createAnatomicalViewer({onNeuronSelect,initialNeuron=0,base='/anatomy/'}){
  const runtime=await createBrainViewModule();
  async function get(name,type){const r=await fetch(base+name);if(!r.ok)throw new Error(`Anatomy ${name}: HTTP ${r.status}`);return type?new type(await r.arrayBuffer()):r.json();}
  const metadata=await get('metadata.json');
  $('brain-canvas').setAttribute('aria-label',`Measured ${metadata.dataset} anatomy colored by simulated membrane voltage`);
  $('em-slice').setAttribute('aria-label',`Measured ${metadata.dataset} electron microscopy with simulated activity overlays`);
  const caption=$('em-slice').parentElement.querySelector('.scan-caption');
  if(caption)caption.textContent=`Static scan · ${metadata.volume.spacing.map(v=>v.toFixed(3)).join(' × ')} µm voxels. Colored points: simulated activity in the slab. Scroll to zoom · click to inspect.`;
  const [positions,owners,branchPositions,branchOwners,branchEdges,surfacePositions,surfaceTriangles,volumeValues,neurons]=await Promise.all([
    get('positions.bin',Float32Array),get('neuron-indices.bin',Uint32Array),get('skeleton-positions.bin',Float32Array),get('skeleton-neurons.bin',Uint32Array),get('skeleton-edges.bin',Uint32Array),get('surface-positions.bin',Float32Array),get('surface-triangles.bin',Uint32Array),get('em-volume.bin',Uint8Array),get('neurons.json'),
  ]);
  const pointEngine=runtime.createGeometry({positions,neuronIndices:owners,neuronCount:metadata.neuronCount});
  const branchEngine=runtime.createGeometry({positions:branchPositions,neuronIndices:branchOwners,neuronCount:metadata.neuronCount});
  const volume=runtime.createVolume({values:volumeValues,dimensions:metadata.volume.dimensions});
  const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
  for(let i=0;i<positions.length;i++){lo[i%3]=Math.min(lo[i%3],positions[i]);hi[i%3]=Math.max(hi[i%3],positions[i]);}
  const center=lo.map((x,k)=>(x+hi[k])/2),extent=lo.map((x,k)=>hi[k]-x);
  const byNeuron=new Int32Array(metadata.neuronCount).fill(-1);owners.forEach((id,i)=>byNeuron[id]=i);
  let active=false,fly=1,timeMs=0,selectedNeuron=initialNeuron,axis=2,cutMode='cutaway',offset=0,halfThickness=6;
  let voltage=null,lastSpike=null,pointColors=new Float32Array(positions.length).fill(.2),trace=[],lastTraceTime=-1,traceDtMs=.1;
  let scanZoom=1,scanRect=null,slicePoints=new Uint32Array(),contours=new Float32Array(),sliceDirty=true;
  let zoom=1,azimuth=.13,elevation=.12,dirty=true,lastRender=0,lastFrameWall=0,frames=0;
  const host=$('brain-canvas'),renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});
  renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.setClearColor(0,0);host.appendChild(renderer.domElement);
  const scene=new THREE.Scene(),root=new THREE.Group();root.position.set(-center[0],center[1],-center[2]);root.scale.y=-1;scene.add(root);
  const camera=new THREE.OrthographicCamera(-500,500,300,-300,1,5000);
  const uniforms={uNormal:{value:new THREE.Vector3(0,0,1)},uOffset:{value:0},uHalf:{value:6},uMode:{value:1},uSelected:{value:selectedNeuron}};
  const clipping='float d=dot(vRaw,uNormal)-uOffset;if((uMode==1&&d>0.0)||(uMode==2&&abs(d)>uHalf))discard;';
  const declarations='uniform vec3 uNormal;uniform float uOffset;uniform float uHalf;uniform int uMode;uniform float uSelected;varying vec3 vRaw;';
  function activityMaterial(points){return new THREE.ShaderMaterial({uniforms,transparent:true,depthWrite:false,vertexColors:true,blending:THREE.AdditiveBlending,
    vertexShader:`attribute float neuron;varying vec3 vColor;varying float vNeuron;varying vec3 vRaw;void main(){vRaw=position;vColor=color;vNeuron=neuron;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);gl_PointSize=${points?'3.2':'1.0'};}`,
    fragmentShader:`${declarations}varying vec3 vColor;varying float vNeuron;void main(){${clipping}float a=${points?'pow(max(0.0,1.0-length(gl_PointCoord-0.5)*2.0),1.5)':'1.0'};float selected=abs(vNeuron-uSelected)<0.5?1.0:0.0;vec3 c=mix(vColor,vec3(1.0,.87,.52),selected);gl_FragColor=vec4(c,a*(${points?'.20+.5':'.008+.065'}*c.r*c.r+selected*.6));}`});}
  const pointGeometry=new THREE.BufferGeometry();pointGeometry.setAttribute('position',new THREE.BufferAttribute(positions,3));pointGeometry.setAttribute('color',new THREE.BufferAttribute(pointColors,3).setUsage(THREE.DynamicDrawUsage));pointGeometry.setAttribute('neuron',new THREE.BufferAttribute(Float32Array.from(owners),1));
  const points=new THREE.Points(pointGeometry,activityMaterial(true));root.add(points);
  const branchGeometry=new THREE.BufferGeometry();branchGeometry.setAttribute('position',new THREE.BufferAttribute(branchPositions,3));branchGeometry.setAttribute('neuron',new THREE.BufferAttribute(Float32Array.from(branchOwners),1));branchGeometry.setAttribute('color',new THREE.BufferAttribute(new Float32Array(branchPositions.length).fill(.18),3).setUsage(THREE.DynamicDrawUsage));branchGeometry.setIndex(new THREE.BufferAttribute(branchEdges,1));
  const branches=new THREE.LineSegments(branchGeometry,activityMaterial(false));root.add(branches);
  const surfaceGeometry=new THREE.BufferGeometry();surfaceGeometry.setAttribute('position',new THREE.BufferAttribute(surfacePositions,3));surfaceGeometry.setIndex(new THREE.BufferAttribute(surfaceTriangles,1));surfaceGeometry.computeVertexNormals();
  const surface=new THREE.Mesh(surfaceGeometry,new THREE.ShaderMaterial({uniforms,transparent:true,depthWrite:false,side:THREE.DoubleSide,vertexShader:'varying vec3 vRaw;varying vec3 vNormal;void main(){vRaw=position;vNormal=normalize(normalMatrix*normal);gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}',fragmentShader:`${declarations}varying vec3 vNormal;void main(){${clipping}float rim=pow(1.0-abs(normalize(vNormal).z),2.0);gl_FragColor=vec4(.39,.67,.64,.035+rim*.16);}`}));root.add(surface);
  const contourGeometry=new THREE.BufferGeometry(),contourLine=new THREE.LineSegments(contourGeometry,new THREE.LineBasicMaterial({color:'#9dcfbd',transparent:true,opacity:.65}));root.add(contourLine);
  const planeGeometry=new THREE.BufferGeometry(),planeOutline=new THREE.LineLoop(planeGeometry,new THREE.LineBasicMaterial({color:'#bbba83',transparent:true,opacity:.2}));root.add(planeOutline);
  const marker=new THREE.Mesh(new THREE.SphereGeometry(3,12,8),new THREE.MeshBasicMaterial({color:'#ffe5a8',transparent:true,opacity:.9,depthTest:false}));root.add(marker);

  function axes(){return [0,1,2].filter(i=>i!==axis);}
  function updateCamera(){const w=host.clientWidth||600,h=host.clientHeight||400,aspect=w/h,span=Math.max(extent[1]*1.22,extent[0]/aspect*1.15)/zoom;camera.left=-span*aspect/2;camera.right=span*aspect/2;camera.top=span/2;camera.bottom=-span/2;camera.updateProjectionMatrix();camera.position.set(1800*Math.sin(azimuth)*Math.cos(elevation),1800*Math.sin(elevation),1800*Math.cos(azimuth)*Math.cos(elevation));camera.lookAt(0,0,0);camera.updateMatrixWorld();root.updateMatrixWorld(true);dirty=true;}
  new ResizeObserver(()=>{if(host.clientWidth&&host.clientHeight){renderer.setSize(host.clientWidth,host.clientHeight);updateCamera();sliceDirty=true;drawTrace();}}).observe(host);
  function updatePlane(){
    axis=Number($('slice-axis').value);offset=lo[axis]+Number($('slice-position').value)/1000*extent[axis];halfThickness=Number($('slice-thickness').value)/2;
    uniforms.uNormal.value.set(0,0,0).setComponent(axis,1);uniforms.uOffset.value=offset;uniforms.uHalf.value=halfThickness;uniforms.uMode.value={all:0,cutaway:1,slab:2}[cutMode];
    $('slice-value').textContent=offset.toFixed(1)+' µm';$('thickness-value').textContent=(halfThickness*2).toFixed(0)+' µm';$('scan-axis').textContent=['YZ','XZ','XY'][axis]+' · '+offset.toFixed(1)+' µm';
    const normal=uniforms.uNormal.value.toArray();slicePoints=pointEngine.filter({normal,offset,halfThickness,mode:'slab'});
    contours=runtime.sliceMesh({positions:surfacePositions,triangles:surfaceTriangles,normal,offset});contourGeometry.setAttribute('position',new THREE.BufferAttribute(contours,3));contourGeometry.computeBoundingSphere();
    const [a,b]=axes(),corners=[];for(const [u,v]of [[0,0],[1,0],[1,1],[0,1]]){const p=[...center];p[axis]=offset;p[a]=u?hi[a]:lo[a];p[b]=v?hi[b]:lo[b];corners.push(...p);}planeGeometry.setAttribute('position',new THREE.Float32BufferAttribute(corners,3));planeGeometry.computeBoundingSphere();
    planeOutline.visible=cutMode!=='all';contourLine.visible=cutMode!=='all';sliceDirty=true;dirty=true;
  }
  function scanMapping(){const [a,b]=axes(),ratio=extent[b]/Math.max(extent[a],1e-6),w=Math.max(1,Math.min(640,Math.floor(4096/ratio))),h=Math.min(4096,Math.max(150,Math.round(w*ratio))),cx=(lo[a]+hi[a])/2,cy=(lo[b]+hi[b])/2;return {a,b,w,h,left:cx-extent[a]*.53/scanZoom,top:cy-extent[b]*.53/scanZoom,spanX:extent[a]*1.06/scanZoom,spanY:extent[b]*1.06/scanZoom};}
  let scanBase=null;
  function drawSlice(resample=false){
    if(!active)return;
    const canvas=$('em-slice'),ctx=canvas.getContext('2d'),r=scanMapping();scanRect=r;
    if(resample||!scanBase){
      canvas.width=r.w;canvas.height=r.h;
      const world=[...center];world[axis]=offset;world[r.a]=r.left;world[r.b]=r.top;
      const origin=world.map((x,k)=>(x-metadata.volume.origin[k])/metadata.volume.spacing[k]);
      const u=[0,0,0],v=[0,0,0];u[r.a]=r.spanX/(r.w-1)/metadata.volume.spacing[r.a];v[r.b]=r.spanY/(r.h-1)/metadata.volume.spacing[r.b];
      const pixels=volume.samplePlane({origin,u,v,width:r.w,height:r.h});scanBase=ctx.createImageData(r.w,r.h);
      for(let i=0;i<pixels.length;i++){const p=i*4,c=pixels[i];scanBase.data[p]=c||11;scanBase.data[p+1]=c||20;scanBase.data[p+2]=c||24;scanBase.data[p+3]=255;}
    }
    ctx.putImageData(scanBase,0,0);
    ctx.strokeStyle='rgba(130,220,190,.5)';ctx.lineWidth=1;
    ctx.beginPath();for(let i=0;i<contours.length;i+=6){ctx.moveTo((contours[i+r.a]-r.left)/r.spanX*r.w,(contours[i+r.b]-r.top)/r.spanY*r.h);ctx.lineTo((contours[i+3+r.a]-r.left)/r.spanX*r.w,(contours[i+3+r.b]-r.top)/r.spanY*r.h);}ctx.stroke();
    for(const index of slicePoints){const p=index*3,activity=pointColors[p];if(activity<.36)continue;ctx.fillStyle=`rgba(255,208,112,${Math.min(.8,activity)})`;ctx.fillRect((positions[p+r.a]-r.left)/r.spanX*r.w,(positions[p+r.b]-r.top)/r.spanY*r.h,2,2);}
    const selectedPoint=byNeuron[selectedNeuron];if(selectedPoint>=0&&Math.abs(positions[selectedPoint*3+axis]-offset)<=halfThickness){const x=(positions[selectedPoint*3+r.a]-r.left)/r.spanX*r.w,y=(positions[selectedPoint*3+r.b]-r.top)/r.spanY*r.h;ctx.strokeStyle='#ffe7ab';ctx.lineWidth=2;ctx.beginPath();ctx.arc(x,y,7,0,Math.PI*2);ctx.stroke();}
    ctx.fillStyle='#fff0cb';ctx.font='12px monospace';ctx.fillText(`${fmt(slicePoints.length)} neuron anchors in slab`,12,r.h-12);
  }
  function drawTrace(){
    const canvas=$('neuron-trace'),w=canvas.clientWidth||600,h=canvas.clientHeight||145,dpr=Math.min(devicePixelRatio,2);canvas.width=w*dpr;canvas.height=h*dpr;const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);ctx.font='9px monospace';
    const minV=Math.min(-60,...trace.map(p=>p[1])),maxV=Math.max(-43,...trace.map(p=>p[1]));
    const y=v=>12+(maxV-v)/(maxV-minV)*(h-32),minT=trace.length?trace[0][0]:0,maxT=trace.length?trace.at(-1)[0]:1,x=t=>37+(t-minT)/Math.max(.1,maxT-minT)*(w-45);
    for(const v of [-60,-52,-45]){ctx.strokeStyle=v===-45?'#665b3d':'#263c3b';ctx.setLineDash(v===-45?[3,4]:[]);ctx.beginPath();ctx.moveTo(34,y(v));ctx.lineTo(w,y(v));ctx.stroke();ctx.fillStyle='#7f9a91';ctx.fillText(String(v),0,y(v)+3);}ctx.setLineDash([]);
    if(trace.length){ctx.strokeStyle='#dbc892';ctx.lineWidth=1.5;ctx.beginPath();trace.forEach((p,i)=>i?ctx.lineTo(x(p[0]),y(p[1])):ctx.moveTo(x(p[0]),y(p[1])));ctx.stroke();ctx.strokeStyle='#f7ac6a';for(let i=1;i<trace.length;i++)if(trace[i][2]>trace[i-1][2]){ctx.beginPath();ctx.moveTo(x(trace[i][0]),2);ctx.lineTo(x(trace[i][0]),11);ctx.stroke();}ctx.fillStyle='#91a59b';ctx.fillText(minT.toFixed(1)+' ms',37,h-2);ctx.textAlign='right';ctx.fillText(maxT.toFixed(1)+' ms',w-2,h-2);ctx.textAlign='left';}
    else{ctx.fillStyle='#91a59b';ctx.fillText('Recording begins with the next neural step…',40,h/2);}
    $('trace-window').textContent=trace.length?`${trace.length} samples · ${traceDtMs} ms spacing`:`Samples every ${traceDtMs} ms of neural time`;
  }
  function selectNeuron(index,{moveSlice=true}={}){
    selectedNeuron=index;trace=[];lastTraceTime=-1;uniforms.uSelected.value=index;
    $('neuron-type').textContent=neurons.labels[index];$('neuron-root').textContent=neurons.ids[index];$('neuron-root').href=(metadata.cellUrl||'https://codex.flywire.ai/app/cell_details?root_id=')+neurons.ids[index];
    if(metadata.dataset.startsWith('BANC')){
      // Use BANC's published precomputed layers; FlyWire Codex v783 does not
      // resolve BANC root IDs. Keep IDs as strings in the external viewer state.
      const p=byNeuron[index],position=p>=0?Array.from(positions.slice(p*3,p*3+3),(v,k)=>v/[.004,.004,.045][k]):[125097.5,122589.5,2827.5];
      const state={title:'BANC v888 · '+neurons.labels[index],dimensions:{x:[4e-9,'m'],y:[4e-9,'m'],z:[45e-9,'m']},position,
        crossSectionScale:15,projectionScale:302229,layout:'xy-3d',layers:[
          {name:'BANC EM',type:'image',source:'precomputed://'+metadata.sources.em},
          {name:'BANC neuron',type:'segmentation',source:'precomputed://https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/neuron_meshes',segments:[neurons.ids[index]]}]};
      $('neuron-root').href='https://neuroglancer-demo.appspot.com/#!'+encodeURIComponent(JSON.stringify(state));
    }
    const p=byNeuron[index];marker.visible=p>=0;if(p>=0){marker.position.fromArray(positions,p*3);if(moveSlice){$('slice-position').value=String(Math.round((positions[p*3+axis]-lo[axis])/extent[axis]*1000));updatePlane();}}
    $('neuron-reading').textContent=voltage?voltage[index].toFixed(2)+' mV':'Waiting for neural state';onNeuronSelect(index);drawTrace();dirty=true;sliceDirty=true;
  }
  function paintActivity(){
    if(!voltage)return;
    pointColors=pointEngine.updateActivity({voltage,lastSpikeMs:lastSpike,timeMs});pointGeometry.attributes.color.array.set(pointColors);pointGeometry.attributes.color.needsUpdate=true;
    branchGeometry.attributes.color.array.set(branchEngine.updateActivity({voltage,lastSpikeMs:lastSpike,timeMs}));branchGeometry.attributes.color.needsUpdate=true;dirty=true;
  }
  let pointer=null;
  host.addEventListener('pointerdown',e=>{pointer={x:e.clientX,y:e.clientY,startX:e.clientX,startY:e.clientY,dragged:false};host.setPointerCapture(e.pointerId);});
  host.addEventListener('pointermove',e=>{if(!pointer)return;const dx=e.clientX-pointer.x,dy=e.clientY-pointer.y;pointer.dragged ||=Math.hypot(e.clientX-pointer.startX,e.clientY-pointer.startY)>3;if(e.shiftKey){$('slice-position').value=String(Math.max(0,Math.min(1000,Number($('slice-position').value)-dy*3)));updatePlane();}else{azimuth-=dx*.007;elevation=Math.max(-1.4,Math.min(1.4,elevation+dy*.007));updateCamera();}pointer.x=e.clientX;pointer.y=e.clientY;});
  host.addEventListener('pointerup',e=>{if(pointer&&!pointer.dragged){updateCamera();const r=host.getBoundingClientRect(),matrix=new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix,camera.matrixWorldInverse).multiply(root.matrixWorld);const query={matrix:Float32Array.from(matrix.elements),x:(e.clientX-r.left)/r.width*2-1,y:1-(e.clientY-r.top)/r.height*2,width:r.width,height:r.height,radius:8,normal:uniforms.uNormal.value.toArray(),offset,halfThickness,mode:cutMode};const index=points.visible?pointEngine.pick(query):-1;if(index>=0)selectNeuron(owners[index],{moveSlice:false});else if(branches.visible){const branch=branchEngine.pick(query);if(branch>=0)selectNeuron(branchOwners[branch],{moveSlice:false});}}pointer=null;});
  host.addEventListener('pointercancel',()=>pointer=null);
  host.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.65,Math.min(8,zoom*Math.exp(-e.deltaY*.001)));updateCamera();},{passive:false});
  $('em-slice').addEventListener('wheel',e=>{e.preventDefault();scanZoom=Math.max(1,Math.min(8,scanZoom*Math.exp(-e.deltaY*.001)));sliceDirty=true;},{passive:false});
  $('em-slice').addEventListener('click',e=>{
    const canvas=$('em-slice'),box=canvas.getBoundingClientRect(),r=scanRect;if(!r)return;
    const scale=Math.min(box.width/r.w,box.height/r.h),displayW=r.w*scale,displayH=r.h*scale,l=box.left+(box.width-displayW)/2,t=box.top+(box.height-displayH)/2;
    const x=(e.clientX-l)/displayW*2-1,y=1-(e.clientY-t)/displayH*2;if(Math.abs(x)>1||Math.abs(y)>1)return;
    const m=new Float32Array(16);m[r.a*4]=2/r.spanX;m[r.b*4+1]=-2/r.spanY;m[12]=-1-2*r.left/r.spanX;m[13]=1+2*r.top/r.spanY;m[15]=1;
    const index=pointEngine.pick({matrix:m,x,y,width:displayW,height:displayH,radius:10,normal:uniforms.uNormal.value.toArray(),offset,halfThickness,mode:'slab'});if(index>=0)selectNeuron(owners[index],{moveSlice:false});
  });
  for(const element of [$('slice-position'),$('slice-thickness')])element.addEventListener('input',updatePlane);
  $('slice-axis').addEventListener('change',()=>{scanZoom=1;updatePlane();});
  for(const button of uiQueryAll('[data-clip]'))button.addEventListener('click',()=>{cutMode=button.dataset.clip;for(const b of uiQueryAll('[data-clip]'))b.setAttribute('aria-pressed',String(b===button));updatePlane();});
  for(const [id,object]of [['show-surface',surface],['show-skeletons',branches],['show-anchors',points]])$(id).addEventListener('change',()=>{object.visible=$(id).checked;dirty=true;});
  $('neuron-search-form').addEventListener('submit',e=>{e.preventDefault();const query=$('neuron-search').value.trim().toLowerCase();let i=neurons.ids.indexOf(query);if(i<0)i=neurons.labels.findIndex(label=>label.toLowerCase()===query);if(i<0&&query.length>1)i=neurons.labels.findIndex(label=>label.toLowerCase().includes(query));$('neuron-search-status').textContent=i<0?'No matching neuron in this model.':'';if(i>=0)selectNeuron(i);});
  const labels=[...new Set(metadata.skeletonRanges.map(r=>neurons.labels[r.index]))].sort();for(const label of labels){const option=document.createElement('option');option.value=label;$('neuron-options').appendChild(option);}
  $('anatomy-coverage').textContent=`${fmt(metadata.mappedNeurons)} of ${fmt(metadata.neuronCount)} neurons have measured anchors; ${metadata.missingIndices.length} missing locations are omitted. ${fmt(metadata.skeletonCount)} full branching skeletons are shown. The simulation retains every source connection.`;
  $('anatomy-status').textContent=`${fmt(metadata.mappedNeurons)} anchors · ${metadata.skeletonCount} skeletons`;
  updatePlane();selectNeuron(initialNeuron,{moveSlice:false});
  function render(now){
    if(active&&!host.ownerDocument.hidden){
      if(sliceDirty){drawSlice(true);sliceDirty=false;}
      if(dirty&&now-lastRender>33){renderer.render(scene,camera);lastRender=now;dirty=false;frames++;}
    }
  }
  onUIFrame(render);
  return {
    setTimeStep(dtMs){traceDtMs=dtMs;trace=[];lastTraceTime=-1;drawTrace();},
    setActive(value){active=value;if(value){renderer.setSize(host.clientWidth,host.clientHeight);updateCamera();paintActivity();sliceDirty=true;drawTrace();}},
    selectFly(id){fly=id;voltage=null;lastSpike=null;trace=[];lastTraceTime=-1;pointColors.fill(.2);pointGeometry.attributes.color.array.fill(.2);pointGeometry.attributes.color.needsUpdate=true;branchGeometry.attributes.color.array.fill(.18);branchGeometry.attributes.color.needsUpdate=true;$('anatomy-clock').textContent=`Reading Fly ${String(fly).padStart(3,'0')}`;$('neuron-reading').textContent='Waiting for neural state';drawTrace();dirty=true;sliceDirty=true;},
    update({flyId,activation,lastSpikeMs,neuralTimeMs,recording}){
      if(flyId!==fly)return;voltage=activation;lastSpike=lastSpikeMs;timeMs=neuralTimeMs;
      const start=performance.now();
      if(active){paintActivity();drawSlice();}
      if(recording?.neuronIndex===selectedNeuron){traceDtMs=recording.dtMs??traceDtMs;for(const sample of recording.samples)if(sample[0]>lastTraceTime){trace.push(sample);lastTraceTime=sample[0];}if(trace.length>800)trace.splice(0,trace.length-800);drawTrace();}
      $('neuron-reading').textContent=`${voltage[selectedNeuron].toFixed(2)} mV · Fly ${String(fly).padStart(3,'0')}`;$('anatomy-clock').textContent=`${timeMs.toFixed(1)} ms neural time`;
      lastFrameWall=performance.now()-start;dirty=true;
    },
    get diagnostics(){return {selectedNeuron,fly,timeMs,frames,activityUpdateMs:lastFrameWall,heapBytes:runtime.allocatedHeapBytes,scanPixels:scanBase?.data.length/4,traceSamples:trace.length};},
  };
}
