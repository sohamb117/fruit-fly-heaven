// Muscle identity determines the channel. No food bearing, desired motion,
// sensory activity, neural writes, or root pose enters this mapping.
export const PROBOSCIS_SOURCE='https://elifesciences.org/articles/54978';
export const PROBOSCIS_PRIORS={
 source:PROBOSCIS_SOURCE,
 evidence:'McKellar et al. 2020, Table 3 and Figures 7–9: muscle activation, silencing and insertion anatomy.',
 strength:'Equal relative strength per named muscle class. Forces of distinct synergist classes add and saturate at 1; moment arms and class strengths remain unmeasured here.',
 subdivisions:'Mean within m2D (m2da/m2db), m3 (m3l/m3m), and m4 (m4a/m4b); the cited experiments do not calibrate these BANC subdivisions separately.',
 labella:'m6 extension and m7 abduction are separate channels. A body without those movable joints must report them as unrepresented, not reroute them into rostrum or haustellum.',
 unmodeledCouplings:['m9 also promotes haustellum extension','m1 also affects haustellum retraction','possible m2V haustellum flexion'],
 unsupported:'m8 has no assigned reach-joint sign in this mapping; unknown labels remain explicit.'
};

const classes={
 proboscis_m1_muscle:'m1',proboscis_m2da_muscle:'m2D',proboscis_m2db_muscle:'m2D',proboscis_m2v_muscle:'m2V',
 proboscis_m3l_muscle:'m3',proboscis_m3m_muscle:'m3',proboscis_m4a_muscle:'m4',proboscis_m4b_muscle:'m4',
 proboscis_m6_muscle:'m6',proboscis_m7_muscle:'m7',proboscis_m9_muscle:'m9'
};
const clamp=x=>Math.max(0,Math.min(1,x));

export function createBancProboscisDecoder(mappings){
 const byTarget=new Map(),unsupported=[];
 mappings.forEach((mapping,index)=>{
  if(mapping.joint!=='proboscis')return;
  const muscleClass=classes[mapping.target];
  if(!muscleClass){unsupported.push({target:mapping.target??null,indices:mapping.indices??[],rootIds:mapping.root_ids??[],reason:'No supported positioning action in McKellar 2020 mapping'});return;}
  if(!byTarget.has(mapping.target))byTarget.set(mapping.target,{target:mapping.target,muscleClass,groups:[],indices:[],rootIds:[]});
  const target=byTarget.get(mapping.target);target.groups.push(index);target.indices.push(...mapping.indices??[]);target.rootIds.push(...mapping.root_ids??[]);
 });
 const targets=Array.from(byTarget.values()),classNames=['m1','m2D','m2V','m3','m4','m6','m7','m9'];
 const classCounts=Object.fromEntries(classNames.map(name=>[name,targets.filter(t=>t.muscleClass===name).length]));
 // Reuse one small result object at body-control frequency. Callers retaining
 // historical samples should copy its values, as with the muscle state buffer.
 const result={rostrumExtend:0,rostrumRetract:0,haustellumExtend:0,haustellumRetract:0,labellarExtend:0,labellarAbduct:0,
  byTarget:Object.fromEntries(targets.map(t=>[t.target,0])),byClass:Object.fromEntries(classNames.map(name=>[name,0])),unsupported};
 return {targets,unsupported,provenance:PROBOSCIS_PRIORS,read(muscleState){
  if(!muscleState||muscleState.length!==mappings.length*3)throw new Error('Invalid proboscis muscle state');
  for(const name of classNames)result.byClass[name]=0;
  for(const target of targets){
   let force=0;
   for(const group of target.groups){const value=muscleState[group*3+2];if(!Number.isFinite(value))throw new Error('Nonfinite proboscis muscle force');force+=Math.max(0,value);}
   force/=target.groups.length;
   result.byTarget[target.target]=force;result.byClass[target.muscleClass]+=force/classCounts[target.muscleClass];
  }
  const m=result.byClass;
  result.rostrumExtend=clamp(m.m9);result.rostrumRetract=clamp(m.m1+m.m2D+m.m2V);
  result.haustellumExtend=clamp(m.m4);result.haustellumRetract=clamp(m.m3);
  result.labellarExtend=clamp(m.m6);result.labellarAbduct=clamp(m.m7);
  return result;
 }};
}
