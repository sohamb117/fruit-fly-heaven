const $=id=>document.getElementById(id);
let worker,generation=0,ready=false,paused=false,latest=null,muscles=[],motors=[],totalWall=0,backend='';
const names=['localization','approach','landing','probing','feeding','takeoff','flight'];
const switches=['odor','vision','taste','proprioception','gaps','coupling','flight'];
names.forEach(name=>{const node=document.createElement('div');node.className='stage';node.id=`stage-${name}`;node.textContent=name[0].toUpperCase()+name.slice(1);const time=document.createElement('span');time.textContent='Not observed';node.append(time);$('stages').append(node);});
function controls(){return Object.fromEntries(switches.map(name=>[name,$(name).checked]));}
function start(){
  worker?.terminate();ready=false;latest=null;window.bancReady=null;window.bancSnapshot=null;totalWall=0;paused=false;$('pause').textContent='Pause';$('status').className='status';$('status').textContent='Loading BANC v888…';$('fallback').textContent='';$('events').replaceChildren();$('muscles').replaceChildren();
  for(const name of names){$(`stage-${name}`).className='stage';$(`stage-${name}`).lastChild.textContent='Not observed';}
  const current=++generation;worker=new Worker('./banc/worker.js',{type:'module'});
  worker.onerror=event=>{$('status').textContent=event.message;$('status').className='status error';};
  worker.onmessage=({data})=>{
    if(current!==generation)return;
    if(data.type==='progress')$('status').textContent=data.message;
    if(data.type==='fallback')$('fallback').textContent=`Using WASM fallback: ${data.message}`;
    if(data.type==='error'){$('status').textContent=data.message;$('status').className='status error';ready=false;}
    if(data.type==='ready'){
      ready=true;backend=data.backend;muscles=data.muscles;motors=data.motors;
      $('backend-active').textContent=`${backend==='webgpu'?'WebGPU neural compute':'WASM neural compute'} + WASM muscles`;
      $('count-summary').textContent=`${data.manifest.neuron_count.toLocaleString()} cells · ${data.manifest.chemical_edges.toLocaleString()} chemical edges · ${data.manifest.mapped_motor_neurons} mapped motor neurons`;
      $('coverage').textContent=`${data.manifest.mapped_motor_neurons} of ${data.manifest.motor_neurons} motor neurons map to ${muscles.length} muscle groups. ${data.manifest.unknown_transmitter_edges_zeroed.toLocaleString()} connections retain their topology but have zero conductance because transmitter identity is unknown. ${data.manifest.electrical_directed_edges/2} literature-based electrical pairs are added separately.`;
      $('body-fit').textContent=`${data.body.active_joints.length} joint coordinates retained; ${data.body.frozen_joints.length} teacher joints frozen. Root position and yaw move; root pitch and roll are fixed. Held-out foot-position error: ${(data.body.fit.foot_position_rmse_cm*10).toFixed(3)} mm.`;
      $('muscles').replaceChildren();
      muscles.forEach((m,i)=>{const row=document.createElement('div');row.className='muscle';const label=document.createElement('span');label.textContent=`${m.joint} · ${m.target}`;label.title=m.root_ids.join(', ');const rate=document.createElement('span');rate.id=`rate-${i}`;rate.textContent='0 Hz';const bar=document.createElement('div');bar.className='bar';const fill=document.createElement('i');fill.id=`force-${i}`;bar.append(fill);row.append(label,rate,bar);$('muscles').append(row);});
      $('status').textContent='Running on neural time. Behavior stages are observations.';
      window.bancReady={backend,adapter:data.adapter,manifest:data.manifest};
    }
    if(data.type==='update'){
      latest=data;totalWall+=data.wallMs;
      $('neural-time').textContent=(data.timeMs/1000).toFixed(3);
      $('speed').textContent=`${(data.timeMs/totalWall).toFixed(2)}×`;
      $('height').textContent=(data.body.z*10).toFixed(2);$('body-speed').textContent=(Math.hypot(data.body.vx,data.body.vy)*10).toFixed(2);
      const b=data.body,values={energy:b.internal.energy,hunger:b.internal.hunger,crop:b.internal.crop,insulin:b.internal.insulin,akh:b.internal.akh};
      for(const [name,value] of Object.entries(values)){$(`${name}-value`).textContent=`${Math.round(value*100)}%`;$(`${name}-bar`).style.width=`${value*100}%`;}
      const rateMap=new Map(motors.map((m,i)=>[m.index,data.state[i*8+4]]));
      muscles.forEach((m,i)=>{$(`rate-${i}`).textContent=`${(m.indices.reduce((s,id)=>s+(rateMap.get(id)||0),0)/m.indices.length).toFixed(1)} Hz`;$(`force-${i}`).style.width=`${Math.min(100,b.muscleState[i*3+2]*100)}%`;});
      for(const event of b.events){const node=$(`stage-${event.stage}`);node.classList.add('observed');node.lastChild.textContent=`${event.time.toFixed(3)} s`;node.title=event.evidence;}
      $('events').replaceChildren(...b.events.map(e=>{const p=document.createElement('p');p.textContent=`${e.time.toFixed(3)} s · ${e.evidence}`;return p;}));
      window.bancSnapshot=data.body;
    }
  };
  worker.postMessage({type:'init',backend:$('backend').value,energy:Number($('initial-energy').value)/100,initialCondition:$('initial-condition').value,control:controls()});
}
$('restart').onclick=start;
$('pause').onclick=()=>{paused=!paused;$('pause').textContent=paused?'Resume':'Pause';worker?.postMessage({type:'control',paused});};
$('step').onclick=()=>{paused=true;$('pause').textContent='Resume';worker?.postMessage({type:'step'});};
$('initial-energy').oninput=()=>$('initial-energy-value').textContent=`${$('initial-energy').value}%`;
switches.forEach(name=>$(name).onchange=()=>worker?.postMessage({type:'control',control:controls()}));

const canvas=$('world'),context=canvas.getContext('2d');
function draw(){
  const rect=canvas.getBoundingClientRect(),ratio=Math.min(devicePixelRatio,2);if(canvas.width!==Math.round(rect.width*ratio)||canvas.height!==Math.round(rect.height*ratio)){canvas.width=Math.round(rect.width*ratio);canvas.height=Math.round(rect.height*ratio);}
  context.setTransform(ratio,0,0,ratio,0,0);const w=rect.width,h=rect.height;context.fillStyle='#16251f';context.fillRect(0,0,w,h);
  const scale=Math.min(w/4.7,(h-110)/2.4),ox=w*.48,oy=(h-70)*.53;
  context.strokeStyle='#24382c';context.lineWidth=1;
  for(let x=-5;x<=5;x+=.5){context.beginPath();context.moveTo(ox+x*scale,40);context.lineTo(ox+x*scale,h-85);context.stroke();}
  for(let y=-2;y<=2;y+=.5){context.beginPath();context.moveTo(15,oy+y*scale);context.lineTo(w-15,oy+y*scale);context.stroke();}
  const b=latest?.body,food=b?.food||{x:.6,y:0,radius:.35,remaining:1},fx=ox+food.x*scale,fy=oy-food.y*scale;
  const gradient=context.createRadialGradient(fx,fy,0,fx,fy,scale*1.3);gradient.addColorStop(0,'#8d984a40');gradient.addColorStop(1,'#8d984a00');context.fillStyle=gradient;context.beginPath();context.arc(fx,fy,scale*1.3,0,Math.PI*2);context.fill();
  context.fillStyle='#a9a15c';context.beginPath();context.ellipse(fx,fy,food.radius*scale,food.radius*scale,0,0,Math.PI*2);context.fill();
  context.fillStyle='#dce3bb';context.font='11px system-ui';context.fillText(`FOOD  ${(food.remaining*100).toFixed(1)}%`,fx-food.radius*scale,fy+food.radius*scale+20);
  const x=ox+(b?.x??-1)*scale,y=oy-(b?.y??0)*scale;context.save();context.translate(x,y);context.rotate(-(b?.heading??0));
  context.strokeStyle='#b6c899';context.lineWidth=2;
  const feet=b?.feet||[[.08,.09,0],[.08,-.09,0],[0,.11,0],[0,-.11,0],[-.09,.1,0],[-.09,-.1,0]];
  for(const p of feet){context.beginPath();context.moveTo(0,0);context.lineTo(p[0]*scale,-p[1]*scale);context.stroke();context.beginPath();context.arc(p[0]*scale,-p[1]*scale,2,0,7);context.fill();}
  for(const side of [-1,1]){context.fillStyle='#dce7bb40';context.beginPath();context.ellipse(-.035*scale,side*.075*scale,.115*scale,.04*scale,side*.5,0,7);context.fill();}
  context.fillStyle='#d9e6ab';context.beginPath();context.ellipse(-.03*scale,0,.085*scale,.041*scale,0,0,7);context.fill();context.fillStyle='#eef0d0';context.beginPath();context.arc(.065*scale,0,.035*scale,0,7);context.fill();
  if(b?.proboscis>.05){context.strokeStyle='#e3ac81';context.beginPath();context.moveTo(.08*scale,0);context.lineTo((.1+b.proboscis*.06)*scale,0);context.stroke();}context.restore();
  context.fillStyle='#90a98e';context.font='10px system-ui';context.fillText(`${b?.airborne?'AIRBORNE':'SUPPORTED'} · ${b?.mouthContact?'MOUTH CONTACT':'NO MOUTH CONTACT'}`,20,h-90);
  context.strokeStyle='#3a4c3c';context.beginPath();context.moveTo(20,h-35);context.lineTo(w-20,h-35);context.stroke();
  context.fillStyle='#566341';context.fillRect(fx-food.radius*scale,h-35-food.height*scale,food.radius*scale*2,food.height*scale);
  context.fillStyle='#d9e6ab';context.beginPath();context.ellipse(x,h-35-(b?.z??.17)*scale,.085*scale,.032*scale,0,0,7);context.fill();
  context.fillStyle='#90a98e';context.fillText('SIDE VIEW · DISTANCES IN mm',20,h-10);context.fillText('5 mm',w-62,h-10);
  requestAnimationFrame(draw);
}
draw();start();
