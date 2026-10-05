import * as THREE from '../vendor/three.module.js';

export function validateBrainSample(value,config){
  const n=value?.sampleCount,indices=value?.indices;
  if(value?.schemaVersion!==1||value.dataset!=='BANC v888'||value.modelFingerprint!==config?.modelFingerprint||
    !/^[a-f0-9]{64}$/.test(value.preparedIdsSha256)||!Number.isSafeInteger(value.neuronCount)||value.neuronCount<1||
    !Number.isSafeInteger(n)||n<1||n>4096||!Array.isArray(indices)||indices.length!==n||new Set(indices).size!==n||
    indices.some(index=>!Number.isSafeInteger(index)||index<0||index>=value.neuronCount)||
    !Array.isArray(value.positions)||value.positions.length!==n*3||!value.positions.every(Number.isFinite)||
    !Array.isArray(value.ids)||value.ids.length!==n||!value.ids.every(id=>typeof id==='string'&&/^[1-9][0-9]{0,39}$/.test(id))||
    !Array.isArray(value.labels)||value.labels.length!==n||!value.labels.every(label=>typeof label==='string'&&label.length<=200))
    throw new Error('Brain positions do not match this run');
  return value;
}

/** Measured point anchors, colored only by received activity. No simulation. */
export class TrainingBrainPreview{
  constructor(container,{onStatus=()=>{},onSelect=()=>{}}={}){
    Object.assign(this,{container,onStatus,onSelect});this.active=false;this.sample=null;this.snapshot=null;this.jobId=null;this.timer=null;this.lastDraw=-Infinity;
    this.yaw=.13;this.elevation=.12;this.zoom=1;this.selected=-1;this.drag=null;this.disposed=false;
    container.tabIndex=0;
    this.down=e=>{this.drag={x:e.clientX,y:e.clientY,startX:e.clientX,startY:e.clientY};container.setPointerCapture(e.pointerId);};
    this.move=e=>{if(!this.drag)return;this.yaw-=(e.clientX-this.drag.x)*.007;this.elevation=Math.max(-1.4,Math.min(1.4,this.elevation+(e.clientY-this.drag.y)*.007));this.drag.x=e.clientX;this.drag.y=e.clientY;this.requestDraw();};
    this.up=e=>{if(this.drag&&Math.hypot(e.clientX-this.drag.startX,e.clientY-this.drag.startY)<4)this.pick(e.clientX,e.clientY);this.drag=null;};
    this.wheel=e=>{e.preventDefault();this.zoom=Math.max(.5,Math.min(8,this.zoom*Math.exp(-e.deltaY*.001)));this.requestDraw();};
    this.key=e=>{if(e.key==='ArrowLeft')this.yaw-=.12;else if(e.key==='ArrowRight')this.yaw+=.12;else if(e.key==='ArrowUp')this.elevation=Math.min(1.4,this.elevation+.1);else if(e.key==='ArrowDown')this.elevation=Math.max(-1.4,this.elevation-.1);else if(e.key==='+'||e.key==='=')this.zoom=Math.min(8,this.zoom*1.15);else if(e.key==='-')this.zoom=Math.max(.5,this.zoom/1.15);else return;e.preventDefault();this.requestDraw();};
    for(const [event,fn]of [['pointerdown',this.down],['pointermove',this.move],['pointerup',this.up],['pointercancel',this.up],['wheel',this.wheel],['keydown',this.key]])container.addEventListener(event,fn,{passive:false});
  }
  setSample(sample){this.sample=sample;this.onStatus({hasFrame:true,count:sample.sampleCount});this.requestDraw();}
  setActive(active){this.active=!!active;if(!active){clearTimeout(this.timer);this.timer=null;}else this.requestDraw();}
  setJob(jobId,instance){if(jobId===this.jobId&&instance===this.instance)return;this.jobId=jobId;this.instance=instance;this.snapshot=null;this.selected=-1;this.colorsDirty=true;this.onSelect(null);this.requestDraw();}
  setSnapshot(snapshot){
    const n=this.sample?.sampleCount;
    if(!n||snapshot?.jobId!==this.jobId||!Number.isFinite(snapshot.neuralTimeMs)||snapshot.neuralTimeMs<0||
      (this.snapshot&&snapshot.neuralTimeMs<this.snapshot.neuralTimeMs)||
      !snapshot.indices||snapshot.indices.length!==n||Array.from(snapshot.indices).some((index,i)=>index!==this.sample.indices[i])||
      ['voltage','rates','lastSpikeMs'].some(key=>!snapshot[key]||snapshot[key].length!==n||!Array.from(snapshot[key]).every(Number.isFinite)))return false;
    this.snapshot=snapshot;this.colorsDirty=true;this.updateSelection();this.requestDraw();return true;
  }
  recenter(){this.yaw=.13;this.elevation=.12;this.zoom=1;this.requestDraw();}
  requestDraw(){
    if(!this.active||!this.sample||this.disposed||document.hidden||this.timer!==null)return;
    this.timer=setTimeout(()=>{this.timer=null;if(!this.active||document.hidden)return;try{this.ensureRenderer();this.draw();this.lastDraw=performance.now();}catch{this.onStatus({error:true});}},Math.max(0,500-(performance.now()-this.lastDraw)));
  }
  ensureRenderer(){
    if(this.renderer)return;
    this.renderer=new THREE.WebGLRenderer({antialias:false,alpha:false,powerPreference:'low-power'});this.renderer.setPixelRatio(1);this.renderer.setSize(480,270,false);this.container.replaceChildren(this.renderer.domElement);
    this.scene=new THREE.Scene();this.scene.background=new THREE.Color('#102017');this.camera=new THREE.OrthographicCamera(-1,1,1,-1,.01,10000);
    const positions=new Float32Array(this.sample.positions),lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
    for(let i=0;i<positions.length;i++){lo[i%3]=Math.min(lo[i%3],positions[i]);hi[i%3]=Math.max(hi[i%3],positions[i]);}
    const center=lo.map((v,i)=>(v+hi[i])/2);this.extent=Math.max(...hi.map((v,i)=>v-lo[i]),1);
    for(let i=0;i<positions.length;i++)positions[i]=(positions[i]-center[i%3])*(i%3===1?-1:1);
    this.positions=positions;this.colors=new Float32Array(positions.length);this.geometry=new THREE.BufferGeometry();
    this.geometry.setAttribute('position',new THREE.BufferAttribute(positions,3));this.geometry.setAttribute('color',new THREE.BufferAttribute(this.colors,3).setUsage(THREE.DynamicDrawUsage));
    this.material=new THREE.PointsMaterial({size:2.5,sizeAttenuation:false,vertexColors:true});this.points=new THREE.Points(this.geometry,this.material);this.scene.add(this.points);
    this.colorsDirty=true;
  }
  updateCamera(){
    const span=this.extent*.6/this.zoom,aspect=16/9,c=this.camera;c.left=-span*aspect;c.right=span*aspect;c.top=span;c.bottom=-span;c.updateProjectionMatrix();
    const distance=this.extent*3;c.position.set(distance*Math.sin(this.yaw)*Math.cos(this.elevation),distance*Math.sin(this.elevation),distance*Math.cos(this.yaw)*Math.cos(this.elevation));c.lookAt(0,0,0);c.updateMatrixWorld();
  }
  draw(){
    if(this.colorsDirty){
      for(let i=0;i<this.sample.sampleCount;i++){
        const level=this.snapshot?Math.max(0,Math.min(1,(this.snapshot.voltage[i]+75)/40)):0;
        const spike=this.snapshot&&this.snapshot.lastSpikeMs[i]>=0&&this.snapshot.neuralTimeMs-this.snapshot.lastSpikeMs[i]<=5;
        this.colors.set(i===this.selected?[1,.93,.62]:spike?[1,.51,.23]:[.18+.66*level,.31+.51*level,.25+.3*level],i*3);
      }
      this.geometry.attributes.color.needsUpdate=true;this.colorsDirty=false;
    }
    this.updateCamera();this.renderer.render(this.scene,this.camera);
    this.onStatus({hasFrame:true,count:this.sample.sampleCount,hasActivity:!!this.snapshot,timeMs:this.snapshot?.neuralTimeMs});
  }
  pick(x,y){
    if(!this.renderer)return;this.updateCamera();const outer=this.container.getBoundingClientRect(),scale=Math.min(outer.width/480,outer.height/270);
    const box={width:480*scale,height:270*scale,left:outer.left+(outer.width-480*scale)/2,top:outer.top+(outer.height-270*scale)/2};
    if(x<box.left||x>box.left+box.width||y<box.top||y>box.top+box.height)return;
    const point=new THREE.Vector3();let chosen=-1,best=100;
    for(let i=0;i<this.sample.sampleCount;i++){point.fromArray(this.positions,i*3).project(this.camera);if(Math.abs(point.z)>1)continue;
      const dx=box.left+(point.x+1)*box.width/2-x,dy=box.top+(1-point.y)*box.height/2-y,d=dx*dx+dy*dy;if(d<best){best=d;chosen=i;}}
    this.selected=chosen;this.colorsDirty=true;this.updateSelection();this.requestDraw();
  }
  updateSelection(){const i=this.selected;this.onSelect(i<0?null:{id:this.sample.ids[i],label:this.sample.labels[i],voltage:this.snapshot?.voltage[i],rate:this.snapshot?.rates[i]});}
  dispose(){this.disposed=true;clearTimeout(this.timer);for(const [event,fn]of [['pointerdown',this.down],['pointermove',this.move],['pointerup',this.up],['pointercancel',this.up],['wheel',this.wheel],['keydown',this.key]])this.container.removeEventListener(event,fn);this.geometry?.dispose();this.material?.dispose();this.renderer?.dispose();this.container.replaceChildren();}
}
