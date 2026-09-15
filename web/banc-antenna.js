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
export function createAntennaAirflowModel(population,config=ANTENNA_PRIOR){
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
