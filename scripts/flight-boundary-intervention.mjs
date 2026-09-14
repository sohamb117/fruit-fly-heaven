// Development-only causal intervention. No pose, velocity, force or actuator
// writes. Only the static wall/ceiling collision masks change, once at reset.
export function disableHabitatBoundaryContacts(body){
 const {mj,model}=body,changed=[];
 const expected=new Set(['ceiling',...Array.from({length:64},(_,i)=>`wall${i}`)]);
 for(let id=0;id<model.ngeom;id++){
  const name=mj.mj_id2name(model,5,id);
  if(!expected.has(name))continue;
  if(model.geom_bodyid[id]!==0)throw new Error('Boundary intervention requires static world geometry');
  changed.push({id,name,contype:model.geom_contype[id],conaffinity:model.geom_conaffinity[id]});
  expected.delete(name);
 }
 if(expected.size)throw new Error('Incomplete habitat boundary geometry');
 for(const {id}of changed){model.geom_contype[id]=0;model.geom_conaffinity[id]=0;}
 return {kind:'disabled-static-boundary-contacts',changed,
  limitations:'Diagnostic only. Ground and fruit contacts and all original behavioral bounds remain active. Does not demonstrate obstacle avoidance or validate flight outside the habitat.'};
}
