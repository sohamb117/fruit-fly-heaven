import * as THREE from '../vendor/three.module.js';

export const PREVIEW_QUALITIES={off:{width:480,height:270,hz:0},low:{width:320,height:180,hz:3},balanced:{width:480,height:270,hz:6},high:{width:640,height:360,hz:10}};
const finite=(x,n)=>x&&x.length===n&&Array.from(x).every(Number.isFinite);
const up=new THREE.Vector3(0,1,0);
const matrix=new THREE.Matrix4();
function rotation(mesh,values){
  if(!finite(values,9))return false;
  matrix.set(values[0],values[1],values[2],0,values[3],values[4],values[5],0,values[6],values[7],values[8],0,0,0,0,1);
  mesh.quaternion.setFromRotationMatrix(matrix);return true;
}
function quat(mesh,values){mesh.quaternion.set(values[1],values[2],values[3],values[0]).normalize();}

/** Read-only native pose viewer. No simulation, policy or interpolated motion.
 * All supplied world coordinates are centimeters with Z up. The only assumed
 * geometry is the simple body skin; articulated limbs/wings/mouth use native
 * landmarks. No WebGL context exists before the first real native frame.
 */
export class NativeFlyPreview{
  constructor(container,{onStatus=()=>{}}={}){
    this.container=container;this.onStatus=onStatus;this.quality=PREVIEW_QUALITIES.balanced;this.qualityName='balanced';
    this.frame=null;this.renderer=null;this.lastDraw=-Infinity;this.timer=null;this.dirty=false;this.disposed=false;
    this.yaw=-1.1;this.elevation=.6;this.distance=.9;this.target=new THREE.Vector3();this.drag=null;this.environmentKey='';
    this.container.tabIndex=0;
    this.container.setAttribute('aria-label','Native three-dimensional fly preview. Drag or use arrow keys to orbit, plus and minus to zoom.');
    this.pointerDown=e=>{if(!this.frame)return;this.drag={x:e.clientX,y:e.clientY};this.container.setPointerCapture(e.pointerId);};
    this.pointerMove=e=>{if(!this.drag)return;this.yaw-=(e.clientX-this.drag.x)*.008;this.elevation=Math.max(.06,Math.min(1.5,this.elevation+(e.clientY-this.drag.y)*.006));this.drag={x:e.clientX,y:e.clientY};this.requestDraw();};
    this.pointerUp=()=>{this.drag=null;};
    this.wheel=e=>{if(!this.frame)return;e.preventDefault();this.distance=Math.max(.28,Math.min(14,this.distance*Math.exp(e.deltaY*.001)));this.requestDraw();};
    this.key=e=>{
      if(!this.frame)return;
      if(e.key==='ArrowLeft')this.yaw-=.12;else if(e.key==='ArrowRight')this.yaw+=.12;
      else if(e.key==='ArrowUp')this.elevation=Math.min(1.5,this.elevation+.1);else if(e.key==='ArrowDown')this.elevation=Math.max(.06,this.elevation-.1);
      else if(e.key==='+'||e.key==='=')this.distance=Math.max(.28,this.distance/1.15);else if(e.key==='-')this.distance=Math.min(14,this.distance*1.15);else return;
      e.preventDefault();this.requestDraw();
    };
    this.visibility=()=>{if(!document.hidden)this.requestDraw();};
    container.addEventListener('pointerdown',this.pointerDown);container.addEventListener('pointermove',this.pointerMove);
    container.addEventListener('pointerup',this.pointerUp);container.addEventListener('pointercancel',this.pointerUp);
    container.addEventListener('wheel',this.wheel,{passive:false});container.addEventListener('keydown',this.key);
    document.addEventListener('visibilitychange',this.visibility);
  }
  setQuality(name){
    this.qualityName=PREVIEW_QUALITIES[name]?name:'balanced';this.quality=PREVIEW_QUALITIES[this.qualityName];
    if(this.renderer){this.renderer.setSize(this.quality.width,this.quality.height,false);this.renderer.domElement.hidden=!this.quality.hz;}
    if(!this.quality.hz){clearTimeout(this.timer);this.timer=null;this.onStatus({off:true,hasFrame:!!this.frame,text:'Preview off · training controls are unchanged'});}
    else if(!this.frame)this.onStatus({hasFrame:false,text:'Preview idle · no pose animation'});
    else this.requestDraw();
  }
  setFrame(frame){
    if(!finite(frame?.position,3)||!finite(frame?.quaternion,4)){this.onStatus({error:true,text:'Invalid native pose · retaining the last valid frame'});return;}
    this.frame=frame;this.dirty=true;this.requestDraw();
  }
  recenter(){this.yaw=-1.1;this.elevation=.6;this.distance=.9;this.requestDraw();}
  requestDraw(){
    if(this.disposed||!this.frame||!this.quality.hz||document.hidden)return;
    this.dirty=true;if(this.timer!==null)return;
    this.timer=setTimeout(()=>{this.timer=null;if(document.hidden||!this.dirty||!this.quality.hz)return;
      try{this.ensureRenderer();this.draw();this.lastDraw=performance.now();this.dirty=false;}
      catch(error){this.onStatus({error:true,text:`3D preview unavailable: ${error.message}`});}
    },Math.max(0,1000/this.quality.hz-(performance.now()-this.lastDraw)));
  }
  ensureRenderer(){
    if(this.renderer)return;
    this.renderer=new THREE.WebGLRenderer({antialias:false,alpha:false,powerPreference:'low-power'});
    this.renderer.setPixelRatio(1);this.renderer.setSize(this.quality.width,this.quality.height,false);
    this.renderer.outputColorSpace=THREE.SRGBColorSpace;this.container.replaceChildren(this.renderer.domElement);
    this.scene=new THREE.Scene();this.scene.background=new THREE.Color('#102017');
    this.camera=new THREE.PerspectiveCamera(43,16/9,.005,40);this.camera.up.set(0,0,1);
    this.scene.add(new THREE.AmbientLight('#c2e7c4',1.5));
    const light=new THREE.DirectionalLight('#fff0c4',2.4);light.position.set(2,-3,5);this.scene.add(light);
    this.environment=new THREE.Group();this.scene.add(this.environment);
    this.sphere=new THREE.SphereGeometry(1,12,8);this.cylinder=new THREE.CylinderGeometry(1,1,1,5);
    this.materials={body:new THREE.MeshLambertMaterial({color:'#b59057'}),head:new THREE.MeshLambertMaterial({color:'#8b7246'}),
      eye:new THREE.MeshLambertMaterial({color:'#883d30'}),leg:new THREE.MeshLambertMaterial({color:'#b89a69'}),
      contact:new THREE.MeshBasicMaterial({color:'#a0efb6'}),wing:new THREE.MeshLambertMaterial({color:'#b3d0b3',transparent:true,opacity:.48,side:THREE.DoubleSide,depthWrite:false}),
      mouth:new THREE.MeshLambertMaterial({color:'#c79c65'})};
    this.skin=new THREE.Group();this.scene.add(this.skin);
    const addSkin=(size,position,material=this.materials.body)=>{const mesh=new THREE.Mesh(this.sphere,material);mesh.scale.fromArray(size);mesh.position.fromArray(position);this.skin.add(mesh);return mesh;};
    // The thorax collision ellipsoid from the native model; frozen head/abdomen
    // receive a low-detail skin. Their movement comes only from root pose.
    const thorax=addSkin([.0437,.0437,.0551],[.00168,0,-.00262]);quat(thorax,[0,-.479,0,-.878]);
    addSkin([.076,.031,.028],[-.063,0,-.009]);
    addSkin([.035,.043,.032],[.072,0,-.004],this.materials.head);
    addSkin([.023,.015,.025],[.08,.030,-.002],this.materials.eye);
    addSkin([.023,.015,.025],[.08,-.030,-.002],this.materials.eye);
    this.legSegments=Array.from({length:12},()=>{const mesh=new THREE.Mesh(this.cylinder,this.materials.leg);mesh.visible=false;this.scene.add(mesh);return mesh;});
    this.feet=Array.from({length:6},()=>{const mesh=new THREE.Mesh(this.sphere,this.materials.leg);mesh.scale.setScalar(.005);mesh.visible=false;this.scene.add(mesh);return mesh;});
    this.wings=Array.from({length:2},()=>{const mesh=new THREE.Mesh(this.sphere,this.materials.wing);mesh.visible=false;this.scene.add(mesh);return mesh;});
    this.mouth=[];
  }
  segment(mesh,a,b,radius=.003){
    if(!finite(a,3)||!finite(b,3)){mesh.visible=false;return;}
    const av=new THREE.Vector3().fromArray(a),bv=new THREE.Vector3().fromArray(b),direction=bv.clone().sub(av),length=direction.length();
    mesh.visible=length>1e-8;if(!mesh.visible)return;
    mesh.position.copy(av.add(bv).multiplyScalar(.5));mesh.scale.set(radius,length,radius);mesh.quaternion.setFromUnitVectors(up,direction.normalize());
  }
  updateEnvironment(frame){
    const food=(frame.food||[]).map(f=>({kind:f.kind,position:f.position,radiusCm:f.radiusCm,lengthCm:f.lengthCm,rotation:f.rotation}));
    const key=JSON.stringify([frame.bowl,food]);if(key===this.environmentKey)return;this.environmentKey=key;
    this.environment.traverse(o=>{o.geometry?.dispose();if(o.material)for(const m of Array.isArray(o.material)?o.material:[o.material])m.dispose();});this.environment.clear();
    if(frame.bowl?.floor){
      const radius=frame.bowl.radiusCm,base=frame.bowl.floor.baseCm,coefficient=frame.bowl.floor.radialCoefficientPerCm;
      if([radius,base,coefficient].every(Number.isFinite)&&radius>0){
        const positions=[],indices=[],rings=20,sectors=64;
        for(let ring=0;ring<=rings;ring++)for(let sector=0;sector<=sectors;sector++){
          const r=radius*ring/rings,a=2*Math.PI*sector/sectors;positions.push(r*Math.cos(a),r*Math.sin(a),base+coefficient*r*r);
        }
        for(let r=0;r<rings;r++)for(let s=0;s<sectors;s++){const a=r*(sectors+1)+s,b=a+sectors+1;indices.push(a,b,a+1,a+1,b,b+1);}
        const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));geometry.setIndex(indices);geometry.computeVertexNormals();
        this.environment.add(new THREE.Mesh(geometry,new THREE.MeshLambertMaterial({color:'#283b25',side:THREE.DoubleSide})));
      }
    }
    for(const f of food){
      if(!finite(f.position,3)||!Number.isFinite(f.radiusCm)||f.radiusCm<=0)continue;
      const material=new THREE.MeshLambertMaterial({color:f.kind==='banana'?'#b8a143':'#9a5540'});let geometry;
      if(f.kind==='banana'&&Number.isFinite(f.lengthCm)){
        const angle=Number.isFinite(f.rotation)?f.rotation:0,c=Math.cos(angle),s=Math.sin(angle);
        const points=Array.from({length:31},(_,i)=>{const t=i/30,x=(t-.5)*f.lengthCm,y=.9*(4*(t-.5)**2-1);return new THREE.Vector3(c*x-s*y,s*x+c*y,0);});
        geometry=new THREE.TubeGeometry(new THREE.CatmullRomCurve3(points),30,f.radiusCm,8,false);
        for(const point of [points[0],points.at(-1)]){const cap=new THREE.Mesh(new THREE.SphereGeometry(.25,8,6),material.clone());cap.position.copy(point).add(new THREE.Vector3().fromArray(f.position));this.environment.add(cap);}
      }else{geometry=new THREE.SphereGeometry(f.radiusCm,12,8);geometry.scale(1,1,.94);}
      const mesh=new THREE.Mesh(geometry,material);mesh.position.fromArray(f.position);this.environment.add(mesh);
    }
  }
  draw(){
    const frame=this.frame;this.updateEnvironment(frame);this.skin.position.fromArray(frame.position);quat(this.skin,frame.quaternion);
    for(let leg=0;leg<6;leg++){
      const points=frame.legs?.[leg];for(let part=0;part<2;part++)this.segment(this.legSegments[leg*2+part],points?.[part],points?.[part+1]);
      const foot=this.feet[leg],p=frame.feet?.[leg];foot.visible=!!finite(p,3);if(foot.visible)foot.position.fromArray(p);
      foot.material=frame.contacts?.legs?.[leg]?this.materials.contact:this.materials.leg;
    }
    for(let i=0;i<2;i++){
      const mesh=this.wings[i],wing=frame.wings?.[i];mesh.visible=!!(finite(wing?.position,3)&&finite(wing?.size,3)&&finite(wing?.rotation,9));
      if(mesh.visible){mesh.position.fromArray(wing.position);mesh.scale.fromArray(wing.size);rotation(mesh,wing.rotation);}
    }
    const ellipsoids=frame.mouth?.ellipsoids||[];
    while(this.mouth.length<ellipsoids.length){const mesh=new THREE.Mesh(this.sphere,this.materials.mouth);this.scene.add(mesh);this.mouth.push(mesh);}
    this.mouth.forEach((mesh,i)=>{const p=ellipsoids[i];mesh.visible=!!(finite(p?.position,3)&&finite(p?.size,3)&&finite(p?.rotation,9));if(mesh.visible){mesh.position.fromArray(p.position);mesh.scale.fromArray(p.size);rotation(mesh,p.rotation);}});
    this.target.fromArray(frame.position);this.target.z-=.015;
    const horizontal=this.distance*Math.cos(this.elevation);
    this.camera.position.set(this.target.x+horizontal*Math.cos(this.yaw),this.target.y+horizontal*Math.sin(this.yaw),this.target.z+this.distance*Math.sin(this.elevation));
    this.camera.lookAt(this.target);this.renderer.render(this.scene,this.camera);
    this.onStatus({hasFrame:true,off:false,width:this.quality.width,height:this.quality.height,hz:this.quality.hz,
      text:`${this.quality.width} × ${this.quality.height} · up to ${this.quality.hz} fps · drag to orbit`});
  }
  dispose(){
    this.disposed=true;clearTimeout(this.timer);document.removeEventListener('visibilitychange',this.visibility);
    for(const [event,listener]of [['pointerdown',this.pointerDown],['pointermove',this.pointerMove],['pointerup',this.pointerUp],['pointercancel',this.pointerUp],['wheel',this.wheel],['keydown',this.key]])this.container.removeEventListener(event,listener);
    if(this.scene){const geometries=new Set(),materials=new Set();this.scene.traverse(o=>{if(o.geometry)geometries.add(o.geometry);if(o.material)for(const m of Array.isArray(o.material)?o.material:[o.material])materials.add(m);});geometries.forEach(g=>g.dispose());materials.forEach(m=>m.dispose());}
    this.renderer?.dispose();this.container.replaceChildren();
  }
}
