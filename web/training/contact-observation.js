// Read-only force measurement. Native contact forces are expressed in the
// contact frame and act positively on geom2; they are not all vertical loads.
export function upwardContactForce(frame,force,bodyIsGeom2){
  const sign=bodyIsGeom2?1:-1;
  return Math.max(0,sign*(frame[2]*force[0]+frame[5]*force[1]+frame[8]*force[2]));
}
export function measureFootSupport(body){
  const loads=new Float64Array(6),data=body.data,contacts=data.ncon?data.contact:null;
  const weight=body.metadata.mass_g*981,minimumLoad=weight*1e-5;
  let environmentContacts=0;
  try{
    for(let i=0;i<data.ncon;i++){
      const contact=contacts.get(i);
      try{
        const geoms=contact.geom,static0=body.geomBodyIds[geoms[0]]===0,static1=body.geomBodyIds[geoms[1]]===0;
        if(static0===static1)continue;
        body.mj.mj_contactForce(body.model,data,i,body.contactForce);
        const force=body.contactForce.GetView();
        // Soft native contacts can transmit force at positive separation inside
        // the solver margin. Geometric penetration alone is not contact load.
        if(contact.dist<=0||Math.hypot(force[0],force[1],force[2])>minimumLoad)environmentContacts++;
        const dynamicGeom=static0?geoms[1]:geoms[0],foot=body.tasteBodyToLeg[body.geomBodyIds[dynamicGeom]];
        if(foot>=0)loads[foot]+=upwardContactForce(contact.frame,force,static0);
      }finally{contact.delete();}
    }
  }finally{contacts?.delete();}
  return {footSupportCount:loads.reduce((count,load)=>count+Number(load>minimumLoad),0),footSupportFraction:loads.reduce((sum,load)=>sum+(load>minimumLoad?load:0),0)/weight,environmentContacts};
}
