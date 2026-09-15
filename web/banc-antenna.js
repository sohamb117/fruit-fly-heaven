// Passive *virtual* antenna observer. Native antenna joints/actuators are not
// changed. Organ, side and JO family are anatomical; mechanics, preferred
// directions, gains and temporal tuning below are explicit modeling priors.
export const ANTENNA_PROFILE='banc-virtual-antenna-airflow-v1';
const freeze=x=>{if(x&&typeof x==='object'){Object.values(x).forEach(freeze);Object.freeze(x);}return x;};
export const ANTENNA_PRIOR=freeze({schema:1,profile:ANTENNA_PROFILE,
 timeConstantSeconds:.015,referenceAirSpeedCmPerSecond:100,maxDeflectionRadians:.5,
 baselineRateHz:5,positionGainHzPerRadian:120,velocityGainHzPerRadPerSecond:.3,
 maxRateHz:100,orientationSeed:888});
export const ANTENNA_ANNOTATION_SOURCE=freeze({
 rawMetadata:'data/raw/banc888/meta.feather',rawMetadataSha256:'86ccf5df0c67419f8c5f43e93a7ed38d23a080e9f7fde26737290252f3780098',
 preparedIdsSha256:'dd942f6fd3bf27b155b4112c94b31a8348564264070933f7f916bc18e9dd1542',
 preparedIoSha256:'1b19cd8c91ff64d1fe2af8f3fdd09d98792ecfca229e1c102ecb08a96da1ce72',
 sensoryRowsSha256:'737762784423ad797feb5bffca34642c953075c2783da71d582297a42201934f',
 evidence:'Exact BANC v888 antenna chordotonal rows: JO-C family detailed function direction; JO-D/E/F families position. No preferred axis/sign is annotated.'});
const TYPES=freeze({'JO-C':'direction','JO-CL':'direction','JO-D':'position','JO-DA':'position',
 'JO-E':'position','JO-EDM':'position','JO-EDP':'position','JO-EV':'position','JO-EVL':'position','JO-EVM':'position',
 'JO-F':'position','JO-FVA':'position'});
const SIDES=['left','right'];
// Two transverse axes around assumed anterior/outward antenna shafts. These
// are root-frame geometry priors, not registered native sensillum directions.
const PLANES=freeze([[[-.5,Math.sqrt(.75),0],[0,0,1]],[[.5,Math.sqrt(.75),0],[0,0,1]]]);
const fail=message=>{throw new Error('Antenna airflow: '+message);};
const requireThat=(condition,message)=>{if(!condition)fail(message);};
const finite=x=>typeof x==='number'&&Number.isFinite(x);
const record=x=>x!==null&&typeof x==='object'&&!Array.isArray(x)&&!ArrayBuffer.isView(x);
const vector=(value,n,label)=>{requireThat((Array.isArray(value)||ArrayBuffer.isView(value))&&value.length===n&&Array.from(value).every(finite),'invalid '+label);return Array.from(value);};
const sha=async bytes=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');
const fnv=text=>{let h=2166136261;for(let i=0;i<text.length;i++)h=Math.imul(h^text.charCodeAt(i),16777619);return h>>>0;};
const canonical=x=>JSON.stringify(x);

export function validateAntennaConfig(config=ANTENNA_PRIOR){
 if(config?.schema===2||config?.profile===ANTENNA_FAMILY_PROFILE)return validateAntennaFamilyConfig(config);
 requireThat(record(config)&&Object.keys(config).every(key=>Object.hasOwn(ANTENNA_PRIOR,key)),'unknown config field');
 const p={...ANTENNA_PRIOR,...config};
 requireThat(p.schema===1&&p.profile===ANTENNA_PROFILE,'unsupported profile');
 for(const key of ['timeConstantSeconds','referenceAirSpeedCmPerSecond','maxDeflectionRadians','maxRateHz'])requireThat(finite(p[key])&&p[key]>0,'invalid '+key);
 requireThat(p.timeConstantSeconds<=1&&p.timeConstantSeconds>=.001&&p.referenceAirSpeedCmPerSecond<=10000&&p.maxDeflectionRadians<=1&&p.maxRateHz<=200,'prior outside supported range');
 for(const key of ['baselineRateHz','positionGainHzPerRadian','velocityGainHzPerRadPerSecond'])requireThat(finite(p[key])&&p[key]>=0&&p[key]<=10000,'invalid '+key);
 requireThat(p.baselineRateHz<=p.maxRateHz&&Number.isInteger(p.orientationSeed)&&p.orientationSeed>=0&&p.orientationSeed<=0xffffffff,'invalid baseline or orientation seed');
 return freeze(p);
}

/** Verify the actual prepared identity bytes and sensory rows. Population
 * tuning is generated after this join; no cell's preferred direction is
 * attributed to BANC. Unknown/JO-mz cells remain in replacementIndices at zero,
 * preventing an enabled observer from borrowing the legacy tilt/speed proxy.
 */
export async function createAntennaPopulation(baseModel,sensory){
 const {manifest:m,io,ids}=baseModel||{},source=ANTENNA_ANNOTATION_SOURCE;
 requireThat(m?.dataset==='BANC'&&m.materialization===888&&m.neuron_count===175401&&sensory?.schema_version===1&&sensory.neuron_count===m.neuron_count,'expected BANC v888 model and sensory identity');
 requireThat(ids instanceof BigUint64Array&&ids.length===m.neuron_count&&Array.isArray(io?.sensory),'verified prepared ids and sensory IO required');
 requireThat(m.files?.['ids.bin']?.sha256===source.preparedIdsSha256&&m.files?.['io.json']?.sha256===source.preparedIoSha256,'prepared source identity mismatch');
 requireThat(await sha(new Uint8Array(ids.buffer,ids.byteOffset,ids.byteLength))===source.preparedIdsSha256,'prepared root-ID bytes mismatch');
 requireThat(await sha(new TextEncoder().encode(canonical(io.sensory)))===source.sensoryRowsSha256,'prepared sensory rows mismatch');
 const byIndex=new Map(io.sensory.map(row=>[row.index,row]));
 requireThat(byIndex.size===io.sensory.length,'duplicate prepared sensory index');
 const excluded=new Set((sensory.body_transducer_exclusions||[]).map(row=>row.index));
 const selected=[],abstained=[],replacementIndices=[],seen=new Set();
 for(const [sideIndex,side]of SIDES.entries()){
  const channels=(sensory.channels||[]).filter(c=>c.key==='antenna_'+side);
  requireThat(channels.length===1&&Array.isArray(channels[0].indices)&&channels[0].indices.length===[249,330][sideIndex],'antenna channel coverage mismatch');
  for(const index of channels[0].indices){
   const row=byIndex.get(index);
   requireThat(Number.isInteger(index)&&!seen.has(index)&&row?.kind==='proprioception'&&row.body_part==='antenna'&&row.side===side&&!excluded.has(index),'antenna channel identity/side/exclusion mismatch');
   seen.add(index);replacementIndices.push(index);
   const cell={index,root_id:String(ids[index]),cell_type:row.cell_type,side};
   if(Object.hasOwn(TYPES,row.cell_type))selected.push({...cell,function:TYPES[row.cell_type]});
   else{requireThat(row.cell_type==='JO-mz'||row.cell_type==='JO','unsupported antenna cell family');abstained.push({...cell,reason:row.cell_type==='JO-mz'?'Mixed vibro_position annotation; vibration tuning unmodeled.':'No supported detailed modality/direction annotation.'});}
  }
 }
 requireThat(selected.length===562&&abstained.length===17&&selected.filter(c=>c.side==='left').length===239,'selected anatomical coverage mismatch');
 const otherChannels=(sensory.channels||[]).filter(c=>!/^antenna_(left|right)$/.test(c.key));
 requireThat(!otherChannels.some(c=>c.indices.some(i=>seen.has(i)))&&!(sensory.vision?.receptors||[]).some(c=>seen.has(c.index))&&!(sensory.body_transducers||[]).some(c=>seen.has(c.index)),'antenna indices overlap other input routing');
 return freeze({kind:'banc-antenna-population-v1',neuronCount:m.neuron_count,source:{...source},
  selected,abstained,replacementIndices,selection:{left:239,right:323,driven:562,zeroed:17},
  replaces:'Entire antenna_left and antenna_right added-input channels; no legacy tilt/speed fallback when enabled.'});
}

/** One small observer per fly. advance consumes simulation time only and
 * integrates the PREVIOUS held airflow up to the new measurement timestamp.
 * sample/snapshot never advance state, so preview/hidden-tab cadence is inert.
 */
export function createAntennaAirflowModel(population,config=ANTENNA_PRIOR,geometry=null){
 if(config?.schema===2||config?.profile===ANTENNA_FAMILY_PROFILE)return createAntennaFamilyAirflowModel(population,config,geometry);
 const p=validateAntennaConfig(config);
 requireThat(population?.kind==='banc-antenna-population-v1'&&population.selected?.length===562&&population.replacementIndices?.length===579&&population.source?.preparedIdsSha256===ANTENNA_ANNOTATION_SOURCE.preparedIdsSha256,'validated antenna population required');
 const groups=new Map(),offsets=new Map(population.replacementIndices.map((id,k)=>[id,k]));
 for(const cell of population.selected){const key=cell.side+'/'+cell.cell_type;if(!groups.has(key))groups.set(key,[]);groups.get(key).push(cell);}
 const cells=[];
 for(const [key,group]of groups){
  group.sort((a,b)=>fnv(p.orientationSeed+'/'+a.root_id)-fnv(p.orientationSeed+'/'+b.root_id)||a.index-b.index);
  const phase=fnv(p.orientationSeed+'/'+key)/4294967296*2*Math.PI;
  group.forEach((cell,k)=>{const angle=phase+2*Math.PI*k/group.length;cells.push({...cell,preferredDirection:[Math.cos(angle),Math.sin(angle)]});});
 }
 cells.sort((a,b)=>a.index-b.index);freeze(cells);
 const metadata=freeze({kind:ANTENNA_PROFILE,config:p,source:population.source,selection:population.selection,
  mechanics:'Independent critically damped virtual bending coordinates; normalized quadratic-drag target. Native antenna joints and actuators unchanged.',
  inputFrame:'native root frame; windRootCmPerSecond minus velocityRootCmPerSecond; uniform translational airflow at the body origin',
  bendingPlanesRoot:PLANES,orientationAssignment:{status:'exchangeable-direction-prior',anatomicalKnowledge:false,seed:p.orientationSeed,method:'FNV-1a root-ID ordering; evenly spaced directions within each annotated side/type'},
  physiology:'Requested rates, not measured connected-network firing. Position/velocity gains and all mechanical constants are uncalibrated priors.',
  limitations:['No perfect body tilt, altitude, speed magnitude or native joint target enters the receptor rate.',
   'No physical antennal inertia/force feedback, native joint motion, local omega-cross-position airflow or active antennal control.',
   'No auditory vibration model; JO-A/B and olfactory cells are not selected. JO-mz/unknown channels abstain.'],cells});
 let time=0,angles=[[0,0],[0,0]],velocities=[[0,0],[0,0]],heldAirflow=[0,0,0];
 const equilibrium=flow=>{const speed=Math.hypot(...flow),scale=p.referenceAirSpeedCmPerSecond;
  return PLANES.map(plane=>plane.map(axis=>p.maxDeflectionRadians*Math.tanh(axis.reduce((sum,a,i)=>sum+a*flow[i],0)/scale*(speed/scale))));};
 const airflow=sample=>{
  requireThat(record(sample),'missing simulation airflow sample');
  const velocity=vector(sample.velocityRootCmPerSecond,3,'root velocity cm/s'),wind=vector(sample.windRootCmPerSecond,3,'root wind cm/s');
  const value=wind.map((w,i)=>w-velocity[i]);requireThat(value.every(finite)&&Math.hypot(...value)<=1e6,'relative airflow outside finite supported range');return value;
 };
 const snapshot=()=>({schema:1,profile:ANTENNA_PROFILE,config:{...p},populationSource:population.source.preparedIoSha256,
  bodyTimeSeconds:time,anglesRadians:angles.map(a=>a.slice()),velocitiesRadiansPerSecond:velocities.map(a=>a.slice()),heldAirflowRootCmPerSecond:heldAirflow.slice()});
 function reset(bodyTimeSeconds=0){requireThat(finite(bodyTimeSeconds)&&bodyTimeSeconds>=0,'invalid reset time');time=bodyTimeSeconds;angles=[[0,0],[0,0]];velocities=[[0,0],[0,0]];heldAirflow=[0,0,0];return snapshot();}
 function advance(sample){
  const flow=airflow(sample),nextTime=sample.bodyTimeSeconds;
  requireThat(finite(nextTime)&&nextTime>=time,'nonmonotone simulation time');const dt=nextTime-time;
  requireThat(dt<=.05+1e-10,'simulation gap exceeds50ms; advance independently of preview');
  if(dt===0){heldAirflow=flow;return snapshot();}
  const targets=equilibrium(heldAirflow),w=1/p.timeConstantSeconds,decay=Math.exp(-w*dt);
  const nextAngles=angles.map((a,s)=>a.map((theta,k)=>{const y=theta-targets[s][k],c=velocities[s][k]+w*y;return targets[s][k]+(y+c*dt)*decay;}));
  const nextVelocities=velocities.map((v,s)=>v.map((speed,k)=>{const c=speed+w*(angles[s][k]-targets[s][k]);return (speed-w*c*dt)*decay;}));
  requireThat([...nextAngles.flat(),...nextVelocities.flat()].every(finite),'nonfinite virtual mechanics');
  time=nextTime;angles=nextAngles;velocities=nextVelocities;heldAirflow=flow;return snapshot();
 }
 function restore(state){
  requireThat(record(state)&&state.schema===1&&state.profile===ANTENNA_PROFILE&&canonical(state.config)===canonical(p)&&state.populationSource===population.source.preparedIoSha256,'snapshot profile/config mismatch');
  requireThat(finite(state.bodyTimeSeconds)&&state.bodyTimeSeconds>=0,'invalid snapshot time');
  requireThat(Array.isArray(state.anglesRadians)&&state.anglesRadians.length===2&&Array.isArray(state.velocitiesRadiansPerSecond)&&state.velocitiesRadiansPerSecond.length===2,'invalid snapshot sides');
  const a=state.anglesRadians.map(x=>vector(x,2,'snapshot deflection')),v=state.velocitiesRadiansPerSecond.map(x=>vector(x,2,'snapshot velocity'));
  const f=vector(state.heldAirflowRootCmPerSecond,3,'snapshot airflow');
  requireThat(Math.hypot(...f)<=1e6&&a.flat().every(x=>Math.abs(x)<=10)&&v.flat().every(x=>Math.abs(x)<=10000),'snapshot outside supported range');
  time=state.bodyTimeSeconds;angles=a;velocities=v;heldAirflow=f;return snapshot();
 }
 function sample({enabled=true}={}){
  requireThat(typeof enabled==='boolean','enabled must be boolean');if(!enabled)return null;
  const ratesHz=new Float32Array(population.replacementIndices.length),sideTotals=[0,0],sideCounts=[0,0];
  for(const cell of cells){const s=cell.side==='left'?0:1,d=cell.preferredDirection;
   const projection=p.positionGainHzPerRadian*(d[0]*angles[s][0]+d[1]*angles[s][1])+p.velocityGainHzPerRadPerSecond*(d[0]*velocities[s][0]+d[1]*velocities[s][1]);
   const rate=Math.max(0,Math.min(p.maxRateHz,p.baselineRateHz+projection));ratesHz[offsets.get(cell.index)]=rate;sideTotals[s]+=rate;sideCounts[s]++;
  }
  return {indices:Uint32Array.from(population.replacementIndices),ratesHz,state:snapshot(),
   diagnostics:{kind:ANTENNA_PROFILE,bodyTimeSeconds:time,units:'requested Hz',drivenCells:cells.length,abstainedCells:population.abstained.length,
    sideMeanRateHz:sideTotals.map((x,i)=>x/sideCounts[i]),directionStatus:'exchangeable prior; not anatomical preferred directions',nativeActuatorsChanged:false}};
 }
 return Object.freeze({metadata,advance,sample,snapshot,restore,reset});
}

// Version 1 above remains the historical exchangeable-direction baseline.
// This opt-in version uses anatomical families, not a root-ID tuning lottery.
export const ANTENNA_FAMILY_PROFILE='banc-native-antenna-family-airflow-v2';
export const ANTENNA_FAMILY_PRIOR=freeze({schema:2,profile:ANTENNA_FAMILY_PROFILE,
 timeConstantSeconds:.015,referenceAirSpeedCmPerSecond:100,maxDeflectionRadians:.5,
 baselineRateHz:5,positionGainHzPerRadian:120,velocityGainHzPerRadPerSecond:.3,maxRateHz:100});
export const ANTENNA_FAMILY_EVIDENCE=freeze({
 C:{response:'tonic anterior deflection',sign:1,source:'https://doi.org/10.1038/nature07843'},
 D:{response:'anterior deflection and vibration',sign:1,source:'https://doi.org/10.3389/fphys.2014.00179',
   limitation:'Unsigned velocity sensitivity is a mixed-response prior, not a fitted 100-200 Hz transfer function.'},
 E:{response:'tonic posterior deflection',sign:-1,source:'https://doi.org/10.1038/nature07843'},
 F:{response:'unresolved; no added airflow drive',sign:null,source:'https://doi.org/10.7554/eLife.59976',
   limitation:'The tested JO-F lines did not respond to imposed pushes, pulls or tested vibrations in immobilized flies; this is not proof of in-vivo silence.'},
 scope:'Population-level literature is transferred to annotated BANC families as an explicit prior. Subtype and individual tuning are not identified; no random individual axes are invented.',
 stateLimitation:'Flight-dependent wingbeat responses are reported in https://doi.org/10.1523/JNEUROSCI.0034-15.2015; no wingbeat-flow or behavioral-state response is invented here.'});

export function validateAntennaFamilyConfig(config=ANTENNA_FAMILY_PRIOR){
 requireThat(record(config)&&Object.keys(config).every(key=>Object.hasOwn(ANTENNA_FAMILY_PRIOR,key)),'unknown family config field');
 const p={...ANTENNA_FAMILY_PRIOR,...config};
 requireThat(p.schema===2&&p.profile===ANTENNA_FAMILY_PROFILE,'unsupported family profile');
 // Numerical scales deliberately retain v1 priors; none were fitted to reward.
 const {schema,profile,...values}=p;
 validateAntennaConfig({...ANTENNA_PRIOR,...values});
 return freeze(p);
}

const dot3=(a,b)=>a.reduce((sum,x,i)=>sum+x*b[i],0);
const cross3=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const add3=(a,b)=>a.map((x,i)=>x+b[i]);
const subtract3=(a,b)=>a.map((x,i)=>x-b[i]);
const unit3=(x,label)=>{const a=vector(x,3,label),n=Math.hypot(...a);requireThat(n>1e-10,'zero '+label);return a.map(v=>v/n);};
const rotate3=(m,v)=>[0,1,2].map(i=>dot3(m.slice(i*3,i*3+3),v));
const unrotate3=(m,v)=>[0,1,2].map(i=>m[i]*v[0]+m[i+3]*v[1]+m[i+6]*v[2]);
const multiply3=(a,b)=>[0,1,2].flatMap(i=>[0,1,2].map(j=>a[i*3]*b[j]+a[i*3+1]*b[j+3]+a[i*3+2]*b[j+6]));
function quaternionMatrix(value){
 let [w,x,y,z]=vector(value,4,'native quaternion');const norm=Math.hypot(w,x,y,z);requireThat(norm>1e-10,'zero native quaternion');
 w/=norm;x/=norm;y/=norm;z/=norm;
 return [1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)];
}
function properRotation(value,label){
 const m=vector(value,9,label),rows=[m.slice(0,3),m.slice(3,6),m.slice(6,9)];
 requireThat(rows.every(r=>Math.abs(dot3(r,r)-1)<1e-8)&&Math.abs(dot3(rows[0],rows[1]))<1e-8&&Math.abs(dot3(rows[0],rows[2]))<1e-8&&Math.abs(dot3(rows[1],rows[2]))<1e-8&&dot3(cross3(rows[0],rows[1]),rows[2])>1-1e-8,'invalid '+label);
 return m;
}
const geometryKey=g=>canonical({sourceXmlSha256:g.sourceXmlSha256,rootBody:g.rootBody,headBody:g.headBody,headRotationRoot:g.headRotationRoot,sides:g.sides});
function validateFamilyGeometry(geometry){
 requireThat(record(geometry)&&geometry.schema===1&&geometry.kind==='native-frozen-antenna-geometry-v1'&&typeof geometry.sourceXmlSha256==='string'&&/^[a-f0-9]{64}$/.test(geometry.sourceXmlSha256),'native antenna geometry required');
 requireThat(Array.isArray(geometry.sides)&&geometry.sides.length===2&&geometry.key===geometryKey(geometry),'geometry identity mismatch');
 properRotation(geometry.headRotationRoot,'head rotation');
 for(const [s,side]of geometry.sides.entries()){
  requireThat(side.side===SIDES[s]&&Number.isInteger(side.body)&&Number.isInteger(side.geom),'invalid native antenna side');
  vector(side.attachmentRootCm,3,'attachment');vector(side.receiverRootCm,3,'receiver');properRotation(side.rotationRoot,'antenna rotation');
  for(const k of ['shaftLocal','anteriorTangentLocal','hingeAxisLocal'])requireThat(Math.abs(Math.hypot(...vector(side[k],3,k))-1)<1e-8,'nonunit '+k);
  requireThat(Math.abs(dot3(side.shaftLocal,side.anteriorTangentLocal))<1e-8&&dot3(cross3(side.shaftLocal,side.anteriorTangentLocal),side.hingeAxisLocal)>1-1e-8,'inconsistent virtual hinge geometry');
 }
 return freeze(structuredClone(geometry));
}

/** Derive fixed head/receiver geometry from the LOADED native model. Sampling
 * uses current free-joint qpos/qvel rather than mj_step's lagged xpos cache.
 * Frozen a2/a3 geometry is a limitation: this is an airflow-driven virtual
 * deflection, never a claim that these missing native joints are moving. */
export function createNativeAntennaKinematics({mj,model,metadata}){
 requireThat(model&&metadata&&typeof mj?.mj_name2id==='function','native topology/name resolver required');
 const bodyEnum=mj.mjtObj?.mjOBJ_BODY?.value,jointEnum=mj.mjtObj?.mjOBJ_JOINT?.value,geomEnum=mj.mjtObj?.mjOBJ_GEOM?.value;
 requireThat([bodyEnum,jointEnum,geomEnum].every(Number.isInteger),'native object enums unavailable');
 const name=(kind,label)=>{const id=mj.mj_name2id(model,kind,label);requireThat(Number.isInteger(id)&&id>=0,'missing native '+label);return id;};
 const root=name(bodyEnum,'thorax'),head=name(bodyEnum,'head'),free=name(jointEnum,'free');
 requireThat(root<model.nbody&&head<model.nbody&&free<model.njnt&&model.body_parentid[root]===0&&model.body_parentid[head]===root&&
   model.body_jntnum[root]===1&&model.body_jntnum[head]===0&&model.jnt_bodyid[free]===root&&model.jnt_type[free]===0&&model.jnt_qposadr[free]===0&&model.jnt_dofadr[free]===0,'unsupported moving head/root topology');
 requireThat(metadata.joints?.every(j=>!/^head(?:_|$)|^antenna(?:_|$)/.test(j.name)),'head/antenna joints must be frozen in this geometry profile');
 const headRotation=quaternionMatrix(model.body_quat.slice(head*4,head*4+4)),headPosition=vector(model.body_pos.slice(head*3,head*3+3),3,'head position');
 // +Y is the native FlyBody head's anterior axis; the loaded head quaternion
 // maps it into the root. The missing a2/a3 hinge axis remains inferred.
 const anteriorRoot=rotate3(headRotation,[0,1,0]);
 const sides=SIDES.map(side=>{
  const body=name(bodyEnum,'antenna_'+side),geom=name(geomEnum,'antenna_'+side+'_collision');
  requireThat(body<model.nbody&&geom<model.ngeom&&model.body_parentid[body]===head&&model.body_jntnum[body]===0&&model.geom_bodyid[geom]===body,'unsupported antenna receiver topology');
  const rotationRoot=multiply3(headRotation,quaternionMatrix(model.body_quat.slice(body*4,body*4+4))),
    attachmentRootCm=add3(headPosition,rotate3(headRotation,vector(model.body_pos.slice(body*3,body*3+3),3,'antenna attachment'))),
    receiverLocal=vector(model.geom_pos.slice(geom*3,geom*3+3),3,'antenna receiver center'),shaftLocal=unit3(receiverLocal,'receiver shaft'),
    anteriorLocal=unrotate3(rotationRoot,anteriorRoot),anteriorTangentLocal=unit3(subtract3(anteriorLocal,shaftLocal.map(x=>x*dot3(anteriorLocal,shaftLocal))),'anterior bending tangent'),
    hingeAxisLocal=cross3(shaftLocal,anteriorTangentLocal),receiverRootCm=add3(attachmentRootCm,rotate3(rotationRoot,receiverLocal));
  return {side,body,geom,rotationRoot,attachmentRootCm,receiverRootCm,shaftLocal,anteriorTangentLocal,hingeAxisLocal};
 });
 const declaration={schema:1,kind:'native-frozen-antenna-geometry-v1',sourceXmlSha256:metadata.xml_sha256,rootBody:root,headBody:head,headRotationRoot:headRotation,sides};
 const geometry=validateFamilyGeometry({...declaration,key:geometryKey(declaration)});
 function sample(data,{bodyTimeSeconds=data?.time,windWorldCmPerSecond}={}){
  requireThat(data?.qpos?.length>=7&&data?.qvel?.length>=6&&finite(bodyTimeSeconds)&&bodyTimeSeconds>=0,'invalid native kinematic state');
  if(finite(data.time))requireThat(Math.abs(data.time-bodyTimeSeconds)<1e-8,'antenna observation time differs from native state');
  const rootPosition=vector(data.qpos.slice(0,3),3,'root position'),rotation=quaternionMatrix(data.qpos.slice(3,7)),
    velocityWorld=vector(data.qvel.slice(0,3),3,'root velocity'),omegaRoot=vector(data.qvel.slice(3,6),3,'root angular velocity'),
    wind=vector(windWorldCmPerSecond,3,'physical wind'),nativeWind=vector(model.opt?.wind??[0,0,0],3,'native physical wind');
  requireThat(wind.every((x,i)=>Math.abs(x-nativeWind[i])<1e-9),'observer wind differs from native physical wind');
  const windRoot=unrotate3(rotation,wind),velocityRoot=unrotate3(rotation,velocityWorld);
  requireThat(Math.hypot(...omegaRoot)<=1e5&&Math.hypot(...velocityRoot)<=1e6,'native velocity outside supported range');
  return {schema:2,profile:ANTENNA_FAMILY_PROFILE,geometryKey:geometry.key,bodyTimeSeconds,
   sides:geometry.sides.map(side=>{
    const receiverVelocityRoot=add3(velocityRoot,cross3(omegaRoot,side.receiverRootCm)),attachmentVelocityRoot=add3(velocityRoot,cross3(omegaRoot,side.attachmentRootCm));
    return {side:side.side,airflowLocalCmPerSecond:unrotate3(side.rotationRoot,subtract3(windRoot,receiverVelocityRoot)),
      attachmentVelocityRootCmPerSecond:attachmentVelocityRoot,receiverVelocityRootCmPerSecond:receiverVelocityRoot,
      attachmentPositionWorldCm:add3(rootPosition,rotate3(rotation,side.attachmentRootCm)),receiverPositionWorldCm:add3(rootPosition,rotate3(rotation,side.receiverRootCm))};
   })};
 }
 return Object.freeze({geometry,sample});
}

function createAntennaFamilyAirflowModel(population,config,sourceGeometry){
 const p=validateAntennaFamilyConfig(config),geometry=validateFamilyGeometry(sourceGeometry);
 requireThat(population?.kind==='banc-antenna-population-v1'&&population.selected?.length===562&&population.replacementIndices?.length===579&&population.source?.preparedIdsSha256===ANTENNA_ANNOTATION_SOURCE.preparedIdsSha256,'validated antenna population required');
 const offsets=new Map(population.replacementIndices.map((id,k)=>[id,k]));
 const cells=population.selected.filter(c=>/^JO-[CDE]/.test(c.cell_type)).map(c=>({...c,family:c.cell_type[3],sideIndex:SIDES.indexOf(c.side)})),
   abstained=[...population.abstained,...population.selected.filter(c=>/^JO-F/.test(c.cell_type)).map(c=>({...c,reason:'JO-F airflow transduction unresolved; no added current requested.'}))];
 requireThat(cells.length===388&&abstained.length===191,'family selection mismatch');
 const metadata=freeze({kind:ANTENNA_FAMILY_PROFILE,config:p,source:population.source,geometry,evidence:ANTENNA_FAMILY_EVIDENCE,
  selection:{driven:cells.length,zeroed:abstained.length,left:cells.filter(c=>c.side==='left').length,right:cells.filter(c=>c.side==='right').length},cells,abstained,
  mechanics:'One critically damped passive virtual anterior/posterior coordinate per native antenna; previous held local airflow includes rigid-body rotation at the receiver center.',
  limitations:['The capsule center approximates a receiver; the true arista center of pressure and a2/a3 hinge are absent.',
   'Native head/antenna transforms are used, but a missing movable native joint is not replaced by a measured angle.',
   'All gains, mechanical constants and D velocity weighting are inherited unfit priors; no individual preferred-axis or threshold data exist here.',
   'No wing-induced flow, auditory carrier, gravitational/inertial receiver torque, or flight-state tuning is modeled. F abstention means no added drive, not biological inactivity.']});
 let time=0,angles=[0,0],velocities=[0,0],heldAirflow=[[0,0,0],[0,0,0]];
 const state=()=>({schema:2,profile:ANTENNA_FAMILY_PROFILE,config:{...p},geometryKey:geometry.key,populationSource:population.source.preparedIoSha256,
  bodyTimeSeconds:time,anglesRadians:angles.slice(),velocitiesRadiansPerSecond:velocities.slice(),heldAirflowLocalCmPerSecond:heldAirflow.map(x=>x.slice())});
 function reset(bodyTimeSeconds=0){requireThat(finite(bodyTimeSeconds)&&bodyTimeSeconds>=0,'invalid reset time');time=bodyTimeSeconds;angles=[0,0];velocities=[0,0];heldAirflow=[[0,0,0],[0,0,0]];return state();}
 function advance(input){
  requireThat(record(input)&&input.schema===2&&input.profile===ANTENNA_FAMILY_PROFILE&&input.geometryKey===geometry.key&&Array.isArray(input.sides)&&input.sides.length===2,'native family sample/geometry mismatch');
  const flow=input.sides.map((side,s)=>{requireThat(side.side===SIDES[s],'sample side mismatch');const f=vector(side.airflowLocalCmPerSecond,3,'local airflow');requireThat(Math.hypot(...f)<=1e6,'local airflow outside supported range');return f;});
  const nextTime=input.bodyTimeSeconds,dt=nextTime-time;requireThat(finite(nextTime)&&dt>=0&&dt<=.05+1e-10,'invalid family simulation cadence');
  if(dt===0){heldAirflow=flow;return state();}
  const targets=heldAirflow.map((f,s)=>p.maxDeflectionRadians*Math.tanh(dot3(f,geometry.sides[s].anteriorTangentLocal)*Math.hypot(...f)/(p.referenceAirSpeedCmPerSecond**2))),
    w=1/p.timeConstantSeconds,decay=Math.exp(-w*dt),nextAngles=[],nextVelocities=[];
  for(let s=0;s<2;s++){const y=angles[s]-targets[s],c=velocities[s]+w*y;nextAngles.push(targets[s]+(y+c*dt)*decay);nextVelocities.push((velocities[s]-w*c*dt)*decay);}
  requireThat([...nextAngles,...nextVelocities].every(finite),'nonfinite family mechanics');
  time=nextTime;angles=nextAngles;velocities=nextVelocities;heldAirflow=flow;return state();
 }
 function restore(snapshot){
  requireThat(record(snapshot)&&snapshot.schema===2&&snapshot.profile===ANTENNA_FAMILY_PROFILE&&snapshot.geometryKey===geometry.key&&canonical(snapshot.config)===canonical(p)&&snapshot.populationSource===population.source.preparedIoSha256,'family snapshot identity mismatch');
  const a=vector(snapshot.anglesRadians,2,'snapshot angles'),v=vector(snapshot.velocitiesRadiansPerSecond,2,'snapshot velocities');
  requireThat(Array.isArray(snapshot.heldAirflowLocalCmPerSecond)&&snapshot.heldAirflowLocalCmPerSecond.length===2,'invalid snapshot airflow sides');
  const f=snapshot.heldAirflowLocalCmPerSecond.map(x=>vector(x,3,'snapshot local airflow'));
  requireThat(finite(snapshot.bodyTimeSeconds)&&snapshot.bodyTimeSeconds>=0&&a.every(x=>Math.abs(x)<=10)&&v.every(x=>Math.abs(x)<=10000)&&f.every(x=>Math.hypot(...x)<=1e6),'family snapshot outside supported range');
  time=snapshot.bodyTimeSeconds;angles=a;velocities=v;heldAirflow=f;return state();
 }
 function sample({enabled=true}={}){
  requireThat(typeof enabled==='boolean','enabled must be boolean');if(!enabled)return null;
  const ratesHz=new Float32Array(population.replacementIndices.length),totals=[0,0],counts=[0,0];
  for(const cell of cells){const s=cell.sideIndex,sign=ANTENNA_FAMILY_EVIDENCE[cell.family].sign,
    tonic=p.positionGainHzPerRadian*sign*angles[s],phasic=cell.family==='D'?p.velocityGainHzPerRadPerSecond*Math.abs(velocities[s]):0,
    rate=Math.max(0,Math.min(p.maxRateHz,p.baselineRateHz+tonic+phasic));
   ratesHz[offsets.get(cell.index)]=rate;totals[s]+=rate;counts[s]++;
  }
  return {indices:Uint32Array.from(population.replacementIndices),ratesHz,state:state(),
   diagnostics:{kind:ANTENNA_FAMILY_PROFILE,bodyTimeSeconds:time,units:'requested Hz',drivenCells:cells.length,abstainedCells:abstained.length,
    sideMeanRateHz:totals.map((x,s)=>x/counts[s]),directionStatus:'native-frame family-level prior; individual tuning unmeasured',nativeActuatorsChanged:false}};
 }
 return Object.freeze({metadata,advance,sample,snapshot:state,restore,reset});
}
