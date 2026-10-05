// Diagnostic observer only. A replay must match its uninstrumented physics
// digest before these native-step contacts are used as causal evidence.
import {measureFlightObservation,measureFlightKinematics} from '../web/training/flight-observation.js';

export function createFlightHistoryCapture(){
 let source=null,original=null,names=null,nativeSteps=0;
 const blocks=[],contacts=[];
 function captureContacts(body){
  nativeSteps++;
  if(!body.data.ncon)return;
  const vector=body.data.contact;
  try{
   for(let i=0;i<body.data.ncon;i++){
    const contact=vector.get(i);
    try{
     const ids=Array.from(contact.geom),bodies=ids.map(id=>body.geomBodyIds[id]);
     if((bodies[0]===0)===(bodies[1]===0))continue;
     original.mj_contactForce(body.model,body.data,i,body.contactForce);
     const force=Array.from(body.contactForce.GetView());
     if(!(contact.dist<=0||Math.hypot(...force.slice(0,3))>body.metadata.mass_g*981e-5))continue;
     if(contacts.length>=100000)throw new Error('Flight contact capture limit exceeded');
     contacts.push({timeSeconds:body.data.time,solverTimeSeconds:body.data.time-body.model.opt.timestep,
      geoms:ids.map(id=>names[id]),geomIds:ids,bodies,distance:contact.dist,forceContactFrame:force,
      position:Array.from(contact.pos),rootPosition:Array.from(body.data.qpos.slice(0,3)),
      angularSpeed:Math.hypot(...body.data.qvel.slice(3,6))});
    }finally{contact.delete();}
   }
  }finally{vector.delete();}
 }
 function sample({body,world}){
  const k=measureFlightKinematics(body),support=measureFlightObservation(body);
  const up=1-2*(body.quaternion[1]**2+body.quaternion[2]**2),angularSpeed=Math.hypot(...body.data.qvel.slice(3,6));
  const externalForce=[body.data.xfrc_applied,body.data.qfrc_applied].some(a=>a.some(v=>v!==0));
  blocks.push({timeSeconds:body.time,observation:{finite:[body.time,...k.position,k.verticalSpeed,k.speedCmPerSecond,up,angularSpeed,...body.quaternion].every(Number.isFinite),
   up,angularSpeed,...k,radius:Math.hypot(...k.position.slice(0,2)),ceiling:world.habitat.ceiling/10,
   wingPower:body.wingPower,...support,externalForce}});
 }
 return {
  start(context){
   if(source)throw new Error('Flight capture already started');
   source=context.body;original=source.mj;
   names=Array.from({length:source.model.ngeom},(_,id)=>original.mj_id2name(source.model,5,id));
   source.mj=new Proxy(original,{get(target,key){
    if(key!=='mj_step')return Reflect.get(target,key);
    return (model,data)=>{const result=original.mj_step(model,data);if(data===source.data)captureContacts(source);return result;};
   }});
   sample(context);
  },
  sample,
  finish(){
   if(source)source.mj=original;
   return {schemaVersion:1,nativeSteps,blocks,contacts,
    scope:'Read-only native-step contact forces and 2 ms objective observations. No forward dynamics or controls are added. Retained Euler contact forces precede integrated positions by one native timestep.'};
  }
 };
}
