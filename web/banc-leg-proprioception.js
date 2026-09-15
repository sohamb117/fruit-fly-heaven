// Structural leg afferents. BANC establishes identities/families; the native
// hinge binding and numerical response curves are explicit, uncalibrated priors.
// No random preferred directions, perfect body state, or collision-as-vibration.
const freeze=value=>{if(value&&typeof value==='object'){Object.values(value).forEach(freeze);Object.freeze(value);}return value;};
const fail=message=>{throw new Error('Leg proprioception: '+message);};
const requireThat=(ok,message)=>{if(!ok)fail(message);};
const finite=value=>typeof value==='number'&&Number.isFinite(value);
const record=value=>value!==null&&typeof value==='object'&&!Array.isArray(value)&&!ArrayBuffer.isView(value);
const sha=async bytes=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');
const canonical=value=>JSON.stringify(value);
const clip=(value,low,high)=>Math.max(low,Math.min(high,value));
const SIDES=['left','right'],PARTS=['front_leg','middle_leg','hind_leg'];
const FUNCTIONS=freeze({claw:'position',hook:'direction',club:'vibro_tactile'});
const populations=new WeakSet(),jointDescriptors=new WeakSet();

export const LEG_PROPRIOCEPTION_PROFILE='banc-leg-proprioception-structural-v1';
export const LEG_PROPRIOCEPTION_PRIOR=freeze({schema:1,profile:LEG_PROPRIOCEPTION_PROFILE,
 polarityMode:'native-positive-prior',baselineRateHz:20,positionGainHzPerRadian:12,
 velocityGainHzPerRadPerSecond:1,clubBaselineRateHz:5,clubMotionGainHzPerRadPerSecond:1,
 maxRateHz:100,cellOverrides:[]});
export const LEG_PROPRIOCEPTION_ANNOTATION_SOURCE=freeze({
 catalog:'models/banc-leg-proprioception-v1.json',
 catalogCanonicalSha256:'baf02607a8d53d2379ce811d5c8c6cbcd52afc29be6cfa06697683a33d717d69',
 catalogFileSha256:'1ee9d68e78cf3dabee29c2f9db57ec50b3a07a1f29b671ba58d2efc54289b43e',
 rawMetadataSha256:'86ccf5df0c67419f8c5f43e93a7ed38d23a080e9f7fde26737290252f3780098',
 preparedIdsSha256:'dd942f6fd3bf27b155b4112c94b31a8348564264070933f7f916bc18e9dd1542',
 preparedIoSha256:'1b19cd8c91ff64d1fe2af8f3fdd09d98792ecfca229e1c102ecb08a96da1ce72',
 familyPhysiology:['https://doi.org/10.1016/j.neuron.2018.09.009','https://doi.org/10.7554/eLife.60299']});

export function validateLegProprioceptionConfig(config=LEG_PROPRIOCEPTION_PRIOR){
 requireThat(record(config)&&Object.keys(config).every(k=>Object.hasOwn(LEG_PROPRIOCEPTION_PRIOR,k)),'unknown config field');
 const p={...LEG_PROPRIOCEPTION_PRIOR,...config};
 requireThat(p.schema===1&&p.profile===LEG_PROPRIOCEPTION_PROFILE,'unsupported profile');
 requireThat(['native-positive-prior','require-measured'].includes(p.polarityMode),'unsupported polarity mode');
 requireThat(finite(p.maxRateHz)&&p.maxRateHz>0&&p.maxRateHz<=200,'invalid maxRateHz');
 for(const k of ['baselineRateHz','clubBaselineRateHz'])requireThat(finite(p[k])&&p[k]>=0&&p[k]<=p.maxRateHz,'invalid '+k);
 for(const k of ['positionGainHzPerRadian','velocityGainHzPerRadPerSecond','clubMotionGainHzPerRadPerSecond'])requireThat(finite(p[k])&&p[k]>=0&&p[k]<=10000,'invalid '+k);
 requireThat(Array.isArray(p.cellOverrides),'cellOverrides must be an array');
 const seen=new Set();
 p.cellOverrides=p.cellOverrides.map(value=>{
  requireThat(record(value)&&Object.keys(value).every(k=>['index','root_id','joint','axis','sign','referenceAngleRadians'].includes(k)),'invalid cell override fields');
  requireThat(Number.isInteger(value.index)&&value.index>=0&&!seen.has(value.index)&&typeof value.root_id==='string'&&/^\d+$/.test(value.root_id),'invalid or duplicate cell override identity');
  seen.add(value.index);
  requireThat(typeof value.joint==='string'&&/^tibia_T[123]_(left|right)$/.test(value.joint)&&value.axis==='native_tibia_hinge','unsupported override joint/axis');
  requireThat(value.sign===1||value.sign===-1,'override sign must be +1 or -1');
  if(Object.hasOwn(value,'referenceAngleRadians'))requireThat(finite(value.referenceAngleRadians),'invalid reference angle radians');
  return {...value};
 });
 return freeze(p);
}

/** Join an independently prepared, hash-pinned catalog to actual BANC ID bytes
 * and the complete existing self_motion routing. The new sensor replaces that
 * whole route, including abstentions, so legacy speed/tilt cannot fill gaps. */
export async function createLegProprioceptionPopulation(baseModel,sensory,catalog){
 const {manifest:m,ids}=baseModel||{},source=LEG_PROPRIOCEPTION_ANNOTATION_SOURCE;
 requireThat(m?.dataset==='BANC'&&m.materialization===888&&m.neuron_count===175401&&
  sensory?.schema_version===1&&sensory.neuron_count===m.neuron_count,'expected BANC v888 model and sensory identity');
 requireThat(ids instanceof BigUint64Array&&ids.length===m.neuron_count,'prepared root-ID array required');
 requireThat(m.files?.['ids.bin']?.sha256===source.preparedIdsSha256&&m.files?.['io.json']?.sha256===source.preparedIoSha256,'prepared source identity mismatch');
 requireThat(await sha(new Uint8Array(ids.buffer,ids.byteOffset,ids.byteLength))===source.preparedIdsSha256,'prepared root-ID bytes mismatch');
 requireThat(record(catalog)&&await sha(new TextEncoder().encode(canonical(catalog)))===source.catalogCanonicalSha256,'annotation catalog hash mismatch');
 requireThat(catalog.schema===1&&catalog.kind==='banc-leg-proprioception-annotations-v1'&&catalog.neuron_count===m.neuron_count,'unsupported annotation catalog');
 requireThat(catalog.source_files['data/raw/banc888/meta.feather']===source.rawMetadataSha256,'raw annotation source mismatch');
 const expectedChannels=SIDES.map(side=>{
  const rows=(sensory.channels||[]).filter(c=>c.key==='self_motion_'+side);
  requireThat(rows.length===1&&Array.isArray(rows[0].indices),'missing/duplicate self-motion channel');
  return {key:rows[0].key,indices:rows[0].indices};
 });
 requireThat(canonical(expectedChannels)===canonical(catalog.channels)&&
  await sha(new TextEncoder().encode(canonical(expectedChannels)))===catalog.channel_identity_sha256,'self-motion channel membership/order mismatch');
 const replacementIndices=expectedChannels.flatMap(c=>c.indices),seen=new Set(replacementIndices);
 requireThat(seen.size===1066&&catalog.cells.length===1066,'duplicate or incorrect leg population size');
 const rowsByIndex=new Map(catalog.cells.map(c=>[c.index,c]));
 requireThat(rowsByIndex.size===1066,'duplicate annotation index');
 const selected=[],abstained=[];
 for(const [s,channel]of expectedChannels.entries())for(const index of channel.indices){
  const cell=rowsByIndex.get(index);
  requireThat(Number.isInteger(index)&&index>=0&&index<ids.length&&cell?.root_id===String(ids[index])&&cell.side===SIDES[s]&&cell.channel===channel.key,'leg root/side/channel identity mismatch');
  requireThat(cell.anatomical_polarity===null,'this catalog must not claim individual polarity');
  if(cell.family){
   const part=PARTS.indexOf(cell.body_part_sensory),family=cell.family;
   requireThat(part>=0&&cell.leg===part+s*3&&cell.cell_class==='chordotonal_organ_neuron'&&
    cell.cell_sub_class===cell.body_part_sensory+'_'+family+'_chordotonal_organ_neuron'&&
    cell.peripheral_target_type==='chordotonal_organ'&&cell.cell_function_detailed===FUNCTIONS[family]&&!cell.abstention,'unsupported family/function/joint assignment');
   selected.push(structuredClone(cell));
  }else{requireThat(typeof cell.abstention==='string','missing abstention reason');abstained.push(structuredClone(cell));}
 }
 // Body transducers for these cells are the intended replacement. Other input
 // channels and optical injections may never overlap these replacement cells.
 const otherChannels=(sensory.channels||[]).filter(c=>!/^self_motion_(left|right)$/.test(c.key));
 requireThat(!otherChannels.some(c=>c.indices.some(i=>seen.has(i)))&&
  !(sensory.vision?.receptors||[]).some(c=>seen.has(c.index)),'leg input overlaps another channel');
 const result=freeze({kind:'banc-leg-proprioception-population-v1',neuronCount:m.neuron_count,
  source:{...source},selected,abstained,replacementIndices,selection:{...catalog.coverage},
  evidence:structuredClone(catalog.evidence),replaces:'Entire self_motion_left and self_motion_right added-input routes.'});
 populations.add(result);return result;
}

/** The six feedback entries use native radians and rad/s, ordered left T1-T3,
 * then right T1-T3. Zero is the native model reference, NOT a measured preferred
 * biological angle. Native positive is NOT labelled flexion or extension. */
export function createLegJointDescriptors(metadata){
 requireThat(record(metadata)&&Array.isArray(metadata.joints)&&typeof metadata.xml_sha256==='string'&&/^[a-f0-9]{64}$/.test(metadata.xml_sha256),'native joint metadata required');
 const joints=SIDES.flatMap((side,s)=>PARTS.map((part,k)=>{
  const name=`tibia_T${k+1}_${side}`,matches=metadata.joints.filter(j=>j.name===name);
  requireThat(matches.length===1,'missing/duplicate native '+name);const j=matches[0];
  requireThat(Number.isInteger(j.qpos)&&j.qpos>=7&&Number.isInteger(j.dof)&&j.dof>=6&&
   finite(j.neutral)&&Array.isArray(j.range)&&j.range.length===2&&j.range.every(finite)&&
   j.range[0]<j.range[1]&&j.neutral>=j.range[0]&&j.neutral<=j.range[1]&&
   j.range.every(v=>Math.abs(v)<=2*Math.PI),'invalid native tibia joint/radian range');
  return {leg:s*3+k,side,part,joint:name,axis:'native_tibia_hinge',qpos:j.qpos,dof:j.dof,
   referenceAngleRadians:j.neutral,rangeRadians:j.range.slice()};
 }));
 requireThat(new Set(joints.map(j=>j.qpos)).size===6&&new Set(joints.map(j=>j.dof)).size===6,'aliased native tibia coordinates');
 const result=freeze({schema:1,kind:'native-tibia-joint-descriptors-v1',sourceXmlSha256:metadata.xml_sha256,
  angleUnit:'radian',velocityUnit:'radian/second',jointTransferStatus:'family-level femur-tibia binding prior',joints});
 jointDescriptors.add(result);return result;
}

/** Pure, stateless sampling of the current native feedback. No clock, hidden
 * history, numerical differentiation, actuator write or preview dependency.
 * Per-cell overrides are declared tuning priors; they cannot become measured
 * anatomy by changing a config label. In require-measured mode this v1 catalog
 * abstains on ALL claw/hook polarity, while unsigned club motion remains. */
export function createLegProprioceptionMapper(population,config=LEG_PROPRIOCEPTION_PRIOR,descriptors){
 requireThat(populations.has(population),'validated leg population required');
 requireThat(jointDescriptors.has(descriptors),'validated native joint descriptors required');
 const p=validateLegProprioceptionConfig(config),offsets=new Map(population.replacementIndices.map((i,k)=>[i,k]));
 const byIndex=new Map(population.selected.map(c=>[c.index,c])),overrides=new Map();
 for(const override of p.cellOverrides){
  const cell=byIndex.get(override.index);
  requireThat(cell?.root_id===override.root_id&&['claw','hook'].includes(cell.family),'override must identify a mapped directional cell');
  const joint=descriptors.joints[cell.leg];
  requireThat(override.joint===joint.joint&&override.axis===joint.axis,'override joint/axis differs from annotated leg');
  if(Object.hasOwn(override,'referenceAngleRadians'))requireThat(cell.family==='claw'&&
   override.referenceAngleRadians>=joint.rangeRadians[0]&&override.referenceAngleRadians<=joint.rangeRadians[1],'reference angle requires claw cell and supported native range');
  overrides.set(cell.index,override);
 }
 const cells=population.selected.map(cell=>{
  const override=overrides.get(cell.index),joint=descriptors.joints[cell.leg],directional=cell.family!=='club';
  return {...cell,joint:joint.joint,axis:joint.axis,
   referenceAngleRadians:override?.referenceAngleRadians??joint.referenceAngleRadians,
   configuredSign:directional?(override?.sign??1):null,
   polarityStatus:directional?'unknown anatomy; configured native-coordinate prior':'not applicable; bidirectional motion',
   enabled:!directional||p.polarityMode==='native-positive-prior'};
 });
 const metadata=freeze({kind:LEG_PROPRIOCEPTION_PROFILE,config:p,source:population.source,selection:population.selection,
  nativeJoints:descriptors,evidence:population.evidence,cells,
  input:'Only per-leg signed tibiaAngle [rad] and tibiaVelocity [rad/s]. No body-speed, tilt, contact or mixed-joint fallback.',
  response:'Claw: signed angle about native reference. Hook: signed velocity about tonic baseline. Club: unsigned movement component only. Rates are rectified and capped.',
  polarity:'Every individual claw/hook anatomical sign is unknown. Native-positive and explicit per-cell signs are uncalibrated priors, not inferred anatomical labels.',
  limitations:['Numerical gains, tonic baselines, preferred angles and polarity are not fitted to physiology.',
   'No measured within-family tuning heterogeneity or population opponency is invented.',
   'No receptor adaptation, hysteresis, spike timing or high-frequency vibration model.',
   'Unregistered hair plates, strain axes and conflicting/generic annotations receive zero added input.']});
 function sample(feedback,{enabled=true}={}){
  requireThat(typeof enabled==='boolean','enabled must be boolean');if(!enabled)return null;
  requireThat(Array.isArray(feedback?.legs)&&feedback.legs.length===6,'six ordered native leg feedback entries required');
  // Validate all required fields before generating any output. Native joint
  // limits are soft physics constraints: finite overshoot is clipped for the
  // receptor, counted below, never written back into the body.
  let angleRangeClips=0;
  const legs=descriptors.joints.map((joint,k)=>{
   const leg=feedback.legs[k];requireThat(record(leg)&&finite(leg.tibiaAngle)&&finite(leg.tibiaVelocity)&&
    Math.abs(leg.tibiaAngle)<=4*Math.PI&&Math.abs(leg.tibiaVelocity)<=1e6,'missing/nonfinite or unsupported native leg '+k+' units');
   const angle=clip(leg.tibiaAngle,...joint.rangeRadians);if(angle!==leg.tibiaAngle)angleRangeClips++;
   return {angle,velocity:leg.tibiaVelocity};
  });
  const ratesHz=new Float32Array(population.replacementIndices.length),familyTotals={claw:0,hook:0,club:0},familyCounts={claw:0,hook:0,club:0};
  let rateClips=0,polarityAbstentions=0;
  for(const cell of cells){
   if(!cell.enabled){polarityAbstentions++;continue;}const leg=legs[cell.leg];
   const raw=cell.family==='claw'?p.baselineRateHz+p.positionGainHzPerRadian*cell.configuredSign*(leg.angle-cell.referenceAngleRadians):
    cell.family==='hook'?p.baselineRateHz+p.velocityGainHzPerRadPerSecond*cell.configuredSign*leg.velocity:
    p.clubBaselineRateHz+p.clubMotionGainHzPerRadPerSecond*Math.abs(leg.velocity);
   const rate=clip(raw,0,p.maxRateHz);if(rate!==raw)rateClips++;
   ratesHz[offsets.get(cell.index)]=rate;familyTotals[cell.family]+=rate;familyCounts[cell.family]++;
  }
  return {indices:Uint32Array.from(population.replacementIndices),ratesHz,
   diagnostics:{kind:LEG_PROPRIOCEPTION_PROFILE,units:'requested Hz',stateless:true,
    drivenCells:cells.length-polarityAbstentions,annotationAbstentions:population.abstained.length,
    polarityAbstentions,polarityMode:p.polarityMode,angleRangeClips,rateClips,
    familyMeanRateHz:Object.fromEntries(Object.keys(familyCounts).map(k=>[k,familyCounts[k]?familyTotals[k]/familyCounts[k]:0])),
    nativeActuatorsChanged:false}};
 }
 return Object.freeze({metadata,sample});
}
