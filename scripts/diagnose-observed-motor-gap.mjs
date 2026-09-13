// Offline snapshot analysis only. No simulator, controller or parameter writes.
import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import assert from 'node:assert/strict';
import {createBancProboscisDecoder} from '../web/banc-proboscis.js';
import {createBancTasteMapper} from '../web/banc-taste.js';

const directory='reports/observation-60min-20260913';
const options=Object.fromEntries(process.argv.slice(2).map(s=>s.replace(/^--/,'').split('=')));
const paths={frames:`${directory}/frames.jsonl`,io:'data/prepared/banc888/io.json',groups:'data/prepared/banc888/console/groups.json',
  physiology:'configs/banc-physiology.json',params:'data/prepared/banc888/params.bin',mechanics:'models/flybody-mujoco.json',
  supplement:'models/banc-taste-peg-annotations.json'};
const buffers=Object.fromEntries(await Promise.all(Object.entries(paths).map(async([k,p])=>[k,await fs.readFile(p)])));
const parse=k=>JSON.parse(buffers[k]),io=parse('io'),groups=parse('groups'),physiology=parse('physiology'),mechanics=parse('mechanics'),supplement=parse('supplement');
const all=String(buffers.frames).trim().split('\n').map(JSON.parse),limit=Number(options['last-frame']||all.at(-1).index);
const correctedVersions=['banc-direct-contact-memory-sensory-mouth-fixed','banc-direct-final-contact-memory-sensory-mouth-observation'];
const frames=all.filter(f=>f.index<=limit&&((f.index>=4&&f.index<=10)||(f.index>=56&&correctedVersions.includes(f.version))));
const lookup=new Map(io.motor_neurons.map((m,i)=>[m.index,i])),mean=a=>a.length?a.reduce((s,v)=>s+v,0)/a.length:0;
const stats=a=>a.length?{n:a.length,mean:mean(a),min:Math.min(...a),max:Math.max(...a)}:null;
const clamp=x=>Math.max(0,Math.min(1,x)),decode=createBancProboscisDecoder(io.muscles);
const mapped=io.muscles.map((m,i)=>({...m,group:i}));
const report={date:new Date().toISOString(),scope:'Offline comparison of recorded one-fly direct BANC snapshots. No replay, parameter tuning or simulator changes.',
  cutoffFrame:limit,frameFile:paths.frames,recordedBuildChangesFile:`${directory}/changes.json`,selectedFrameIndices:frames.map(f=>f.index),
  sources:Object.fromEntries(Object.entries(paths).filter(([k])=>k!=='frames').map(([k,p])=>[p,createHash('sha256').update(buffers[k]).digest('hex')])),
  selectedFramesSha256:createHash('sha256').update(frames.map(f=>JSON.stringify(f)).join('\n')).digest('hex'),
  method:{rateIndex:'state.rates is motorNeuronRates in io.motor_neurons order, not full-connectome indices.',
    excitation:'clamp(mean(rateHz of each named muscle group)/80,0,1), matching the production decoder input exactly.',
    forceLimitation:'Excitation is not force. Recorded snapshots omit muscle activation/fatigue, joint qpos/qvel and actuator controls. No force or exact deployment history is reconstructed.',
    cadence:'Motor rates already have a 50 ms exponential filter. Sparse wall-clock images do not establish presence or absence of fast neural rhythms or gait coordination.',
    comparison:'Build changes and body restarts separate cohorts. Cohort summaries are snapshot averages, not continuous-time averages or causal experiments.'},frames:[]};

for(const f of frames){
  const s=f.state;
  if(!s.rates){report.frames.push({index:f.index,version:f.version,time:s.bodyTime,missingRates:true});continue;}
  assert.equal(s.rates.length,io.motor_neurons.length);
  const muscles=mapped.map(m=>{
    const rates=m.indices.map(i=>{assert(lookup.has(i));return s.rates[lookup.get(i)];});
    assert(rates.every(Number.isFinite));return {group:m.group,joint:m.joint,target:m.target,sign:m.sign,kind:m.kind,
      meanRateHz:mean(rates),excitation:clamp(mean(rates)/80),activeUnitsOver1Hz:rates.filter(v=>v>1).length,units:rates.length};
  });
  const state=new Float32Array(muscles.length*3);muscles.forEach((m,i)=>state[i*3+2]=m.excitation);
  const channels=structuredClone(decode.read(state)),prob=muscles.filter(m=>m.joint==='proboscis'),sum=prob.reduce((s,m)=>s+m.excitation,0);
  const jointNames=[...new Set(muscles.filter(m=>m.kind==='leg').map(m=>m.joint))].sort();
  const joints=jointNames.map(joint=>{
    const positive=muscles.filter(m=>m.joint===joint&&m.kind==='leg'&&m.sign>0),negative=muscles.filter(m=>m.joint===joint&&m.kind==='leg'&&m.sign<0),
      plus=mean(positive.map(m=>m.excitation)),minus=mean(negative.map(m=>m.excitation));
    return {joint,positiveTargets:positive.map(m=>m.target),negativeTargets:negative.map(m=>m.target),positive:plus,negative:minus,
      net:plus-minus,coactivation:Math.min(plus,minus)};
  });
  const wings=['left','right'].map(side=>{
    const get=(kind,target)=>muscles.find(m=>m.joint===`${kind}_${side}`&&m.target===target)?.excitation??0;
    const dlm=get('wing_power','dorsal_longitudinal_muscle'),dvm=get('wing_power','dorsoventral_muscle'),iii1=get('wing_steer','iii1_muscle'),b1=get('wing_steer','b1_muscle');
    return {side,dlm,dvm,iii1,b1,powerExcitationMean:(dlm+dvm)/2,openingExcitationProxy:1-clamp(iii1-b1),
      belowDeploymentThresholdAtExcitationProxy:1-clamp(iii1-b1)<=mechanics.wing_actuation.deployment_before_beating};
  });
  report.frames.push({index:f.index,version:f.version,time:s.bodyTime,elapsedSeconds:f.elapsedSeconds,phase:s.task?.phase,
    observed:{proboscis:s.proboscis,pump:s.pump,wingPower:s.wingPower,mouthContact:s.mouthContact,onFood:s.onFood,airborne:s.airborne,
      rootPositionCm:s.physicalRoot.slice(0,3),energy:s.internal?.energy,ingested:s.internal?.ingested,legLoads:s.legLoads},
    proboscisExcitationChannels:channels,legacyPooledProboscisExcitation:mean(prob.map(m=>m.excitation)),
    legacyProboscisContributions:prob.map(m=>({target:m.target,excitation:m.excitation,fractionOfPool:sum?m.excitation/sum:0})),
    pumpExcitation:mean(muscles.filter(m=>m.kind==='pump').map(m=>m.excitation)),joints,wings,muscles});
}
const summarize=rows=>{
  const observed=rows.filter(r=>!r.missingRates),select=f=>observed.map(f);
  const channels=['rostrumExtend','rostrumRetract','haustellumExtend','haustellumRetract','labellarExtend','labellarAbduct'];
  const muscleSummary=mapped.map(m=>({joint:m.joint,target:m.target,sign:m.sign,kind:m.kind,indices:m.indices,rootIds:m.root_ids,
    meanRateHz:stats(select(r=>r.muscles[m.group].meanRateHz)),excitation:stats(select(r=>r.muscles[m.group].excitation))}));
  return {frames:observed.map(r=>r.index),simulationTimeRange:observed.length?[observed[0].time,observed.at(-1).time]:[],
    observed:Object.fromEntries(['proboscis','pump','wingPower','energy','ingested'].map(k=>[k,stats(select(r=>r.observed[k]))])),
    mouthContactSnapshots:observed.filter(r=>r.observed.mouthContact).length,
    channels:Object.fromEntries(channels.map(k=>[k,stats(select(r=>r.proboscisExcitationChannels[k]))])),
    legacyPooledProboscisExcitation:stats(select(r=>r.legacyPooledProboscisExcitation)),
    legacyContributors:mapped.filter(m=>m.joint==='proboscis').map(m=>({target:m.target,
      averageFraction:mean(select(r=>r.legacyProboscisContributions.find(p=>p.target===m.target).fractionOfPool))})).sort((a,b)=>b.averageFraction-a.averageFraction),
    wings:['left','right'].map(side=>({side,...Object.fromEntries(['dlm','dvm','iii1','b1','powerExcitationMean','openingExcitationProxy'].map(k=>[k,stats(select(r=>r.wings.find(w=>w.side===side)[k]))])),
      snapshotsBelowProxyDeploymentThreshold:observed.filter(r=>r.wings.find(w=>w.side===side).belowDeploymentThresholdAtExcitationProxy).length})),
    joints:observed[0]?.joints.map(j=>({joint:j.joint,positiveTargets:j.positiveTargets,negativeTargets:j.negativeTargets,
      ...Object.fromEntries(['positive','negative','net','coactivation'].map(k=>[k,stats(select(r=>r.joints.find(p=>p.joint===j.joint)[k]))])),
      bothAboveHalfSnapshots:observed.filter(r=>{const p=r.joints.find(p=>p.joint===j.joint);return p.positive>.5&&p.negative>.5;}).length})),muscles:muscleSummary};
};
report.cohorts={old4to10:summarize(report.frames.filter(r=>r.index>=4&&r.index<=10)),oldFeeding8to10:summarize(report.frames.filter(r=>r.index>=8&&r.index<=10)),
  corrected56plus:summarize(report.frames.filter(r=>r.index>=56)),correctedAfter400ms:summarize(report.frames.filter(r=>r.index>=56&&r.time>=.4))};
report.correctedRuns=Object.fromEntries(correctedVersions.map(v=>{
  const rows=report.frames.filter(r=>r.version===v&&!r.missingRates),gaps=rows.slice(1).map((r,i)=>r.time-rows[i].time),roots=rows.filter(r=>r.time>=.4).map(r=>r.observed.rootPositionCm);
  return [v,{frames:rows.map(r=>r.index),simulatedSampleGapsSeconds:stats(gaps),wallSampleGapsSeconds:stats(rows.slice(1).map((r,i)=>r.elapsedSeconds-rows[i].elapsedSeconds)),
    horizontalExtentAfter400msCm:roots.length?Math.hypot(Math.max(...roots.map(r=>r[0]))-Math.min(...roots.map(r=>r[0])),Math.max(...roots.map(r=>r[1]))-Math.min(...roots.map(r=>r[1]))):null}];
}));
const params=new Float32Array(buffers.params.buffer,buffers.params.byteOffset,buffers.params.byteLength/4),profiles=new Map();
for(const cell of io.motor_neurons){const row=Array.from(params.subarray(cell.index*16,cell.index*16+16)),key=JSON.stringify(row);profiles.set(key,(profiles.get(key)??0)+1);}
report.physiology={status:physiology.status,dtMs:physiology.dt_ms,parameterOrder:physiology.parameter_order,motorParameterGroups:Array.from(profiles,([parameters,count])=>({count,parameters:JSON.parse(parameters)})),
  decoderRateSaturationHz:80,rateFilterMs:50,muscleActivationRiseMs:15,muscleActivationFallMs:40,
  fuelReserveThreshold:.1,wingDeploymentThreshold:mechanics.wing_actuation.deployment_before_beating,wingDeploymentTauSeconds:mechanics.wing_actuation.deployment_tau_s,
  caveat:'These are configured priors. Same profile and force model do not establish cell-type-specific measured muscle recruitment or appropriate relative strengths.'};
const rawSugar=JSON.parse(execFileSync('.venv/bin/python',['-c',`
import json,numpy as np,pyarrow.feather as f
ids=np.fromfile('data/prepared/banc888/ids.bin','<u8')
sweet=json.load(open('data/prepared/banc888/console/groups.json'))['sweet']
wanted={str(ids[i]):i for i in sweet}
cols=['banc_888_id','cell_type','cell_class','body_part_sensory','side','nerve','cell_function_detailed']
rows=[{'index':wanted[r['banc_888_id']],**r} for r in f.read_table('data/raw/banc888/meta.feather',columns=cols).to_pylist() if r['banc_888_id'] in wanted]
print(json.dumps(rows))`],{encoding:'utf8'}));
const countBy=(rows,key)=>rows.reduce((out,r)=>(out[r[key]??'unannotated']=(out[r[key]??'unannotated']??0)+1,out),{});
report.taste={currentRecordedRouting:'sensoryRates: taste && f.contact ? 150 : 0; SensoryEncoder broadcasts this one value to every groups.sweet index. f.contact means on fruit, not organ-specific food contact.',
  sugarCount:groups.sweet.length,rawSource:supplement.source,rawOrgans:countBy(rawSugar,'body_part_sensory'),rawNerves:countBy(rawSugar,'nerve'),rawSides:countBy(rawSugar,'side'),
  rawClasses:countBy(rawSugar,'cell_class'),preparedSensoryCoverage:createBancTasteMapper(io.sensory,groups.sweet).coverage,
  withVerifiedPegSupplement:createBancTasteMapper(io.sensory.concat(supplement.annotations),groups.sweet).coverage,
  missingFromPrepared:supplement.annotations,
  limitation:'No taste sensory currents were recorded with these images; overbroad stimulation follows the recorded build routing and onFood state. It is a plausible source of pump recruitment, not an isolated causal proof.'};
report.findings=[
  'Recorded wingPower is a post-deployment mechanical quantity, not absence of wing motor-neuron activity. Sampled DLM/DVM excitation can remain substantial while wingPower is zero.',
  'III1-versus-B1 deployment and the 0.85 beating threshold are a candidate mechanical bottleneck. Excitation-only opening is a diagnostic proxy; actual muscle force and deployment history were not recorded.',
  'Old pooled reach mixed unsupported m8 and labellar abduction m7 with reach muscles. Corrected separation removes their ability to command rostrum/haustellum extension.',
  'After correction, recorded m4 haustellum extensor output is nearly absent; intermittent m9 rostrum output remains. Pump output is still strong, independently of physical mouth contact.',
  'Generic on-fruit taste stimulated sugar cells from every leg, wing margin and labellum. The data contain organ annotations sufficient for a compartment-specific contact interface, except explicit missing side annotations.',
  'Opposing leg excitation and sparse surface displacement are not evidence of coordinated walking. Dense simultaneous MN, muscle-force, joint-control and contact recording is still needed for motor validation.'
];
await fs.writeFile(`${directory}/motor-gap.json`,JSON.stringify(report,null,2)+'\n');
const old=report.cohorts.oldFeeding8to10,now=report.cohorts.correctedAfter400ms,fmt=x=>Number.isFinite(x)?x.toFixed(3):'n/a';
const text=[
 '# Recorded BANC motor-output diagnosis',
 `Analyzed frames ${report.selectedFrameIndices.join(', ')}; cutoff ${limit}. Read-only offline analysis. Exact input hashes and per-frame values are in motor-gap.json.`,
 'The fly is producing substantial motor output. The low displayed wing power is downstream of MN output, and strong pumping is separate from a correctly placed, extended proboscis. These snapshots do not establish that the required behavior has been solved.',
 'Excitation below means clamp(mean named-group MN rate / 80 Hz, 0, 1). It is the production decoder input, not measured muscle force. Rates are already smoothed over 50 ms; snapshots are separated by roughly 18 wall seconds and 0.3 simulated seconds. No claim about missing fast rhythms follows from this sample cadence.',
 '## Proboscis and pumping',
 '| Excitation channel | Old feeding frames 8–10 mean | Corrected after 400 ms mean |',
 '|---|---:|---:|',
 ...['rostrumExtend','rostrumRetract','haustellumExtend','haustellumRetract','labellarExtend','labellarAbduct'].map(k=>`| ${k} | ${fmt(old.channels[k].mean)} | ${fmt(now.channels[k].mean)} |`),
 `Old pooled reach contributions: ${old.legacyContributors.slice(0,3).map(r=>`${r.target} ${(100*r.averageFraction).toFixed(1)}%`).join(', ')}. The present decoder keeps m7 as labellar abduction and m8 unsupported; neither extends rostrum/haustellum. Old displayed proboscis mean ${fmt(old.observed.proboscis.mean)} therefore did not prove correctly routed reaching.`,
 `Observed pump mean: ${fmt(old.observed.pump.mean)} → ${fmt(now.observed.pump.mean)}; corrected mouth-contact snapshots: ${now.mouthContactSnapshots}/${now.frames.length}. The three active pump groups are listed by name and rate in the JSON; five groups share one pump average.`,
 '## Wing outputs',
 '| Side | DLM excitation mean | DVM mean | III1 mean | B1 mean | Opening proxy mean | Below 0.85 proxy threshold |',
 '|---|---:|---:|---:|---:|---:|---:|',
 ...now.wings.map(w=>`| ${w.side} | ${fmt(w.dlm.mean)} | ${fmt(w.dvm.mean)} | ${fmt(w.iii1.mean)} | ${fmt(w.b1.mean)} | ${fmt(w.openingExcitationProxy.mean)} | ${w.snapshotsBelowProxyDeploymentThreshold}/${now.frames.length} |`),
 `Corrected observed wingPower mean ${fmt(now.observed.wingPower.mean)}. The hinge uses 1 - clamp(III1 force - B1 force) for opening, then requires deployment >0.85 before beating. High power excitation can thus coexist with suppressed wing motion. Actual force/fatigue/deployment were not recorded, so these numbers identify a candidate gap rather than reproduce exact physical output.`,
 '## Opposing leg outputs',
 '| Joint | Positive excitation mean | Negative mean | Signed difference | Coactivation mean | Both >0.5 snapshots |',
 '|---|---:|---:|---:|---:|---:|',
 ...now.joints.map(j=>`| ${j.joint} | ${fmt(j.positive.mean)} | ${fmt(j.negative.mean)} | ${fmt(j.net.mean)} | ${fmt(j.coactivation.mean)} | ${j.bothAboveHalfSnapshots}/${now.frames.length} |`),
 'Names of each positive/negative target, per-muscle BANC IDs, rates, and all per-frame values are retained in the JSON. These are opposition proxies; different fatigue, length and velocity terms can change actual torque.',
 '## Taste and physiology gaps',
 `All ${groups.sweet.length} sugar GRNs were driven from generic on-fruit contact in these recorded builds: raw annotations identify ${report.taste.rawOrgans.labellum} labellar, ${report.taste.rawOrgans.wing_margin} wing-margin, and ${report.taste.rawOrgans.front_leg+report.taste.rawOrgans.middle_leg+report.taste.rawOrgans.hind_leg} leg cells. This also stimulates the labellar sugar pathway while mouthContact is false. The association with pumping is mechanistically plausible but has not been isolated experimentally here.`,
 `Nine labellar taste-peg cells were omitted from io.sensory by its taste-bristle-only classifier. They are restored as a separate verified annotation supplement; one has null side. Across all 532 sugar cells, ${report.taste.withVerifiedPegSupplement.unmapped.length} have missing laterality and the new mapper abstains. Nerve strings are retained in the diagnosis; they are not used to invent a side assignment.`,
 `Prepared motor profile groups: ${report.physiology.motorParameterGroups.map(p=>`${p.count} cells sharing one 16-parameter profile`).join('; ')}. The model uses common 80 Hz excitation saturation, 15/40 ms muscle activation kinetics and shared fatigue/force coefficients. Energy remains above the 0.1 force-limiting reserve in these frames; fuel depletion does not explain the low wing output. Numerical physiology and relative actuator strength remain priors, not calibrated BANC measurements.`,
 'Next discriminating evidence is dense synchronized recording of named MN rates, muscle activation/force, wing deployment, reach-joint controls/positions and organ contacts, after the compartment-specific taste fix. No motor gain, desired movement, or artificial forcing was changed for this diagnosis.'
].join('\n\n');
await fs.writeFile(`${directory}/motor-gap.md`,text.replace(/\|\n\n\|/g,'|\n|')+'\n');
console.log(JSON.stringify({cutoff:limit,selected:report.selectedFrameIndices,oldPooling:old.legacyContributors.slice(0,3),correctedChannels:now.channels,wings:now.wings,
  observed:now.observed,motorProfiles:report.physiology.motorParameterGroups,tasteCoverage:report.taste.withVerifiedPegSupplement.mapped,
  outputs:[`${directory}/motor-gap.json`,`${directory}/motor-gap.md`]},null,2));
