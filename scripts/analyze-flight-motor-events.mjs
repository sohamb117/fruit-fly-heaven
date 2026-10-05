// Offline observation audit only: does not initialize a simulator or modify inputs.
import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {isDeepStrictEqual} from 'node:util';
import {fileURLToPath} from 'node:url';

const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const TAU=2*Math.PI,WINDOW={fromTimeMs:100,timeMs:280,boundary:'(100, 280] ms'},BINS=12;
const DIGEST_FIELDS=['body.time','qpos','qvel','act','ctrl','muscleState','wingPhase','wingDeployment','wingPower','wingTarget'];
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const fail=message=>{throw new Error(message);};
const ensure=(condition,message)=>{if(!condition)fail(message);};
const near=(a,b,tolerance=1e-7)=>Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<=tolerance;
const phase=value=>((value%TAU)+TAU)%TAU;
const same=(a,b,label)=>ensure(isDeepStrictEqual(a,b),'Paired mismatch: '+label);
const hash=value=>typeof value==='string'&&/^[a-f0-9]{64}$/.test(value);
const stats=values=>values.length?{min:Math.min(...values),max:Math.max(...values),mean:values.reduce((a,b)=>a+b,0)/values.length}:null;
function phaseStats(values){
 const histogram=Array(BINS).fill(0);
 for(const value of values)histogram[Math.min(BINS-1,Math.floor(value/TAU*BINS))]++;
 const c=values.reduce((sum,value)=>sum+Math.cos(value),0),s=values.reduce((sum,value)=>sum+Math.sin(value),0);
 const resultantLength=values.length?Math.hypot(c,s)/values.length:null;
 return {events:values.length,histogram,meanRadians:resultantLength>1e-12?phase(Math.atan2(s,c)):null,resultantLength};
}

function validateRecord(record){
 ensure(record?.schemaVersion===1&&record.kind==='native-held-out-evaluation','Expected native-held-out-evaluation schemaVersion 1');
 ensure(record.complete===true&&!record.error,'Evaluation must be complete and error-free');
 const a=record.assignment,e=record.evaluation;
 ensure(a&&e&&!e.cancelled&&e.reason!=='simulation_error'&&!e.metrics?.error,'Cancelled or invalid evaluation');
 ensure(Number.isInteger(a.seed)&&a.seed>=0&&a.seed<=0xffffffff&&typeof a.stage==='string','Invalid assignment identity');
 ensure(Array.isArray(a.parameters)&&a.parameters.length>0&&a.parameters.every(Number.isFinite),'Invalid assigned parameters');
 ensure(Number.isFinite(a.durationSeconds)&&a.durationSeconds>0,'Invalid assigned duration');
 ensure(Number.isFinite(e.return)&&Number.isFinite(e.simSeconds)&&e.simSeconds>0&&Number.isInteger(e.steps)&&e.steps>0,'Invalid evaluation result');
 ensure(e.dtMs===.5&&e.bodyBlockMs===2&&near(e.simSeconds,e.steps*.002),'Unsupported timing or incomplete step count');
 ensure(typeof e.success==='boolean'&&typeof e.terminated==='boolean'&&typeof e.truncated==='boolean','Invalid result flags');
 const full=near(e.simSeconds,a.durationSeconds);
 const physicalFailure=!e.success&&e.terminated&&['outside_habitat','excessive_rotation','overturned'].includes(e.reason);
 ensure(e.simSeconds<=a.durationSeconds+1e-7&&(full||physicalFailure),'Result is neither full-horizon nor a known physical failure');
 for(const key of ['seed','stage','parameters'])same(a[key],e[key],'assignment/evaluation '+key);
 for(const key of ['configHash','modelFingerprint']){ensure(hash(record[key]),'Missing '+key);same(record[key],e[key],key);}
 ensure(['wasm','dawn-metal'].includes(record.backendLabel),'Unsupported backend label');
 same(e.backend,record.backendLabel==='dawn-metal'?'webgpu':'wasm','neural backend');
 ensure(e.bodyBackend==='mujoco-wasm','Expected native MuJoCo body');
 ensure(record.sourceHashes&&Object.keys(record.sourceHashes).length>0&&Object.values(record.sourceHashes).every(hash),'Missing source hashes');
 const d=record.physicsDigest;
 ensure(a.capturePhysicsDigest===true&&d&&hash(d.sha256),'Both evaluations must record a physics digest');
 ensure(d.rows===e.steps+1&&Number.isInteger(d.valuesPerRow)&&d.valuesPerRow>0,'Incomplete physics digest rows or layout');
 same(d.fields,DIGEST_FIELDS,'physics digest fields');
 ensure(typeof d.encoding==='string'&&d.encoding.includes('float64'),'Missing physics digest encoding');
 return record;
}

/** Analyze the confirmed evaluator schema; input records and metadata are not mutated. */
export function analyzeMotorEventPair(first,second,io,ioSha256){
 const a=validateRecord(first),b=validateRecord(second);
 const flags=[a.assignment.captureMotorEvents,b.assignment.captureMotorEvents];
 ensure(flags.includes(true)&&flags.includes(false),'Require one explicit captureMotorEvents=true and one false');
 const captured=flags[0]?a:b,control=flags[0]?b:a;
 ensure(!Object.hasOwn(control,'motorEvents'),'Capture-disabled record unexpectedly contains motor events');
 const settings=record=>Object.fromEntries(Object.entries(record.assignment).filter(([key])=>!['name','captureMotorEvents'].includes(key)));
 same(settings(a),settings(b),'assigned settings excluding name and observation flag');
 for(const key of ['configHash','modelFingerprint','sourceHashes','backendLabel','nativeWebGPU'])same(a[key],b[key],key);
 for(const key of ['return','success','terminated','truncated','reason','simSeconds','steps','parameters','seed','stage','environmentVersion','backend','bodyBackend','dtMs','bodyBlockMs','provenance'])same(a.evaluation[key],b.evaluation[key],'evaluation '+key);
 for(const key of ['initialCondition','initialObservation','finalObservation'])same(a.evaluation.metrics?.[key],b.evaluation.metrics?.[key],'evaluation metrics '+key);
 same(a.physicsDigest,b.physicsDigest,'physics digest');
 ensure(hash(ioSha256)&&captured.sourceHashes['/banc-data/io.json']===ioSha256,'IO metadata is not pinned to the captured source');
 ensure(Array.isArray(io?.muscles)&&Array.isArray(io.motor_neurons),'Invalid IO metadata');
 const neurons=new Map(io.motor_neurons.map(row=>[row.index,row]));
 const mappings=io.muscles.filter(row=>['asynchronous_wing','wing_steering_assumption'].includes(row.kind));
 const byIndex=new Map();
 for(const mapping of mappings){
  const side=mapping.joint.endsWith('_left')?'left':mapping.joint.endsWith('_right')?'right':null;
  ensure(side&&Array.isArray(mapping.indices)&&mapping.indices.length,'Invalid wing mapping');
  for(const index of mapping.indices){
   const neuron=neurons.get(index);
   ensure(neuron&&!byIndex.has(index)&&neuron.side===side&&neuron.peripheral_target_type===mapping.target,'Ambiguous wing MN mapping');
   const family=mapping.kind==='wing_steering_assumption'?'steering':mapping.target==='dorsal_longitudinal_muscle'?'DLM':mapping.target==='dorsoventral_muscle'?'DVM':null;
   ensure(family,'Unknown power-muscle target');
   byIndex.set(index,{index,rootId:neuron.root_id,cellType:neuron.cell_type,muscle:mapping.target,side,family,
    events:[],emaRateSamplesHz:[]});
  }
 }
 const indices=[...byIndex.keys()].sort((x,y)=>x-y);
 ensure(mappings.length===28&&indices.length===48,'Expected 28 mapped wing groups and 48 distinct MNs');
 const packets=captured.motorEvents,steps=captured.evaluation.steps;
 ensure(Array.isArray(packets)&&packets.length===steps+1,'Missing initial or per-block event packets');
 ensure(captured.evaluation.simSeconds*1000>=WINDOW.timeMs-1e-7,'Capture does not cover the complete (100, 280] ms analysis window');
 let previous=null,lastTimes=new Map(),totalEvents=0;
 for(const [p,packet]of packets.entries()){
  same(packet.indices,indices,'packet motor indices');
  ensure(Array.isArray(packet.ratesHz)&&packet.ratesHz.length===48&&packet.ratesHz.every(x=>Number.isFinite(x)&&x>=0),'Invalid packet rates');
  ensure(Array.isArray(packet.counts)&&packet.counts.length===48&&packet.counts.every(x=>Number.isInteger(x)&&x>=0&&x<2**24),'Invalid packet counts');
  ensure(Array.isArray(packet.events)&&packet.initialized===(p===0),'Invalid packet initialization or events');
  ensure(packet.timeMs===p*2&&Number.isFinite(packet.wingPhaseRadians)&&packet.wingPhaseRadians>=0&&packet.wingPhaseRadians<TAU&&Number.isFinite(packet.wingFrequencyHz)&&packet.wingFrequencyHz>0,'Invalid packet time or table phase');
  ensure(near(packet.bodyTimeSeconds*1000,p===0?0:(p-1)*2),'Motor event packet is not at the pre-body block boundary');
  if(!previous){
   ensure(packet.fromTimeMs===null&&packet.events.length===0&&packet.counts.every(x=>x===0),'Capture must start from the fresh zero-count brain');
  }else{
   ensure(packet.fromTimeMs===previous.timeMs,'Skipped, repeated, or reordered event interval');
   if(p>1){
    const predicted=phase(previous.wingPhaseRadians+TAU*previous.wingFrequencyHz*.002);
    ensure(Math.abs(Math.atan2(Math.sin(packet.wingPhaseRadians-predicted),Math.cos(packet.wingPhaseRadians-predicted)))<1e-7,'Table phase did not advance consistently within the prior body block');
   }else ensure(near(packet.wingPhaseRadians,previous.wingPhaseRadians),'Table phase changed before first body step');
   const seen=new Set();let priorEvent=null;
   for(const event of packet.events){
    ensure(byIndex.has(event.index)&&!seen.has(event.index),'Unknown or repeated MN in a 2 ms event packet');
    ensure(Number.isFinite(event.timeMs)&&Number.isInteger(event.timeMs/.5)&&event.timeMs>packet.fromTimeMs&&event.timeMs<=packet.timeMs,'Event timestamp outside its exact neural interval');
    ensure(!priorEvent||event.timeMs>priorEvent.timeMs||(event.timeMs===priorEvent.timeMs&&event.index>priorEvent.index),'Events must be sorted by time then index');
    ensure(!lastTimes.has(event.index)||event.timeMs-lastTimes.get(event.index)>=2.5,'Event train violates the supported refractory interval');
    seen.add(event.index);lastTimes.set(event.index,event.timeMs);priorEvent=event;totalEvents++;
    if(event.timeMs>WINDOW.fromTimeMs&&event.timeMs<=WINDOW.timeMs){
     byIndex.get(event.index).events.push({timeMs:event.timeMs,tablePhaseRadians:phase(packet.wingPhaseRadians+TAU*packet.wingFrequencyHz*(event.timeMs-packet.bodyTimeSeconds*1000)/1000)});
    }
   }
   for(let k=0;k<indices.length;k++)ensure(packet.counts[k]-previous.counts[k]===(seen.has(indices[k])?1:0),'Event list disagrees with cumulative counts');
   if(packet.timeMs>WINDOW.fromTimeMs&&packet.timeMs<=WINDOW.timeMs){
    for(let k=0;k<indices.length;k++)byIndex.get(indices[k]).emaRateSamplesHz.push(packet.ratesHz[k]);
   }
  }
  previous=packet;
 }
 const seconds=(WINDOW.timeMs-WINDOW.fromTimeMs)/1000;
 const perNeuron=indices.map(index=>{
  const {events,emaRateSamplesHz,...identity}=byIndex.get(index),isi=events.slice(1).map((event,k)=>event.timeMs-events[k].timeMs);
  const slot=indices.indexOf(index);
  ensure(events.length===packets[WINDOW.timeMs/2].counts[slot]-packets[WINDOW.fromTimeMs/2].counts[slot],'Window event total disagrees with endpoint cumulative counts');
  return {...identity,count:events.length,rawRateHz:events.length/seconds,emaRateHz:stats(emaRateSamplesHz),withinWindowInterSpikeIntervalMs:stats(isi),
   phase:phaseStats(events.map(event=>event.tablePhaseRadians)),events};
 });
 const grouping=new Map();
 for(const neuron of perNeuron){
  const key=neuron.muscle+':'+neuron.side;
  if(!grouping.has(key))grouping.set(key,[]);
  grouping.get(key).push(neuron);
 }
 const perMuscle=[...grouping.values()].map(rows=>({muscle:rows[0].muscle,side:rows[0].side,family:rows[0].family,
  motorIndices:rows.map(row=>row.index),totalEvents:rows.reduce((sum,row)=>sum+row.count,0),
  perNeuronRawRateHz:stats(rows.map(row=>row.rawRateHz)),phase:phaseStats(rows.flatMap(row=>row.events.map(event=>event.tablePhaseRadians)))}));
 const dlmCoincidences=['left','right','both'].map(side=>{
  const rows=perNeuron.filter(row=>row.family==='DLM'&&(side==='both'||row.side===side)),ticks=new Map();
  for(const row of rows)for(const event of row.events){if(!ticks.has(event.timeMs))ticks.set(event.timeMs,[]);ticks.get(event.timeMs).push(row.index);}
  const histogram=Array(rows.length+1).fill(0);histogram[0]=Math.round((WINDOW.timeMs-WINDOW.fromTimeMs)/.5)-ticks.size;
  for(const ids of ticks.values())histogram[ids.length]++;
  const simultaneous=[...ticks].filter(([,ids])=>ids.length>1).map(([timeMs,motorIndices])=>({timeMs,motorIndices}));
  return {side,motorIndices:rows.map(row=>row.index),tickWidthMs:.5,occupiedTicks:ticks.size,totalSpikes:rows.reduce((sum,row)=>sum+row.count,0),
   ticksByNumberOfSpikingNeurons:histogram,simultaneousTicks:simultaneous.length,spikesOnSimultaneousTicks:simultaneous.reduce((sum,row)=>sum+row.motorIndices.length,0),simultaneous};
 });
 return {schemaVersion:1,kind:'offline-flight-motor-event-analysis',window:{...WINDOW,durationSeconds:seconds},
  interpretation:'One short startup transient; descriptive model output, not a biological firing-rate or phase fit.',
  phaseDefinition:{reference:'Wingbeat-table oscillator, extrapolated within each 2 ms block from its pre-body phase and frequency. Not measured hinge or muscle phase.',formula:'mod(wingPhaseRadians + 2*pi*wingFrequencyHz*(event.timeMs - 1000*bodyTimeSeconds)/1000, 2*pi)',bins:BINS,binEdgesRadians:Array.from({length:BINS+1},(_,k)=>k*TAU/BINS),inference:'Circular statistics are descriptive; no null model, confidence interval, or physiological phase-locking claim.'},
  pairing:{verified:true,capturedAssignment:captured.assignment,controlAssignment:control.assignment,configHash:captured.configHash,modelFingerprint:captured.modelFingerprint,
   backendLabel:captured.backendLabel,nativeWebGPU:captured.nativeWebGPU,sourceHashes:captured.sourceHashes,ioSha256,physicsDigest:captured.physicsDigest,
   digestScope:'Byte-identical SHA-256 digest of the listed float64 state fields at initialization and every completed 2 ms body block. Does not independently compare native integration steps between the 2 ms samples.',
   result:{reason:captured.evaluation.reason,success:captured.evaluation.success,simSeconds:captured.evaluation.simSeconds,steps:captured.evaluation.steps,return:captured.evaluation.return}},
  coverage:{packets:packets.length,totalEvents,wingMotorNeurons:indices.length,wingMappings:mappings.length},perNeuron,perMuscle,dlmCoincidences};
}

function markdown(report){
 const n=value=>value===null?'—':value.toFixed(2),lines=[
  '# Wing motor events: paired observation audit','',report.interpretation,'',
  `Capture-on/off settings, evaluation results, source pins, and recorded physics digests match. Both ended with **${report.pairing.result.reason}** at ${report.pairing.result.simSeconds.toFixed(3)} s.`,
  '',report.pairing.digestScope,'',`Window: **${report.window.boundary}**, ${report.window.durationSeconds.toFixed(3)} s. Rate resolution is ${(1/report.window.durationSeconds).toFixed(2)} Hz per spike.`,
  '',report.phaseDefinition.reference,'',
  '| Muscle | Side | MN index | Cell type | Spikes | Raw Hz | EMA mean Hz | Phase R |','| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |'];
 for(const row of [...report.perNeuron].sort((a,b)=>a.muscle.localeCompare(b.muscle)||a.side.localeCompare(b.side)||a.index-b.index))lines.push(`| ${row.muscle} | ${row.side} | ${row.index} | ${row.cellType} | ${row.count} | ${n(row.rawRateHz)} | ${n(row.emaRateHz?.mean??null)} | ${n(row.phase.resultantLength)} |`);
 lines.push('','Phase R is descriptive circular concentration, not evidence of physiological entrainment; one event necessarily has R = 1. EMA rates are block-end samples of the existing 50 ms estimate, distinct from counted events. Inter-spike statistics include only intervals whose two endpoints lie inside the window.','','## DLM same-tick coincidences','');
 for(const row of report.dlmCoincidences)lines.push(`- ${row.side}: ${row.simultaneousTicks} occupied 0.5 ms ticks contained multiple DLM neurons; ${row.spikesOnSimultaneousTicks}/${row.totalSpikes} spikes occurred on those ticks.`);
 lines.push('','Coincidences are descriptive: the report supplies no independent-rate null model, latency compensation, or evidence of native biological synchronization. Per-neuron event times, extrapolated phases, histograms, identities, and provenance are retained in `analysis.json`.','');
 return lines.join('\n');
}

async function main(){
 const args=process.argv.slice(2);
 ensure(args.length===3,'Usage: node scripts/analyze-flight-motor-events.mjs capture.json control.json reports/new-output');
 const inputPaths=args.slice(0,2).map(value=>path.resolve(value)),output=path.resolve(args[2]),reports=path.join(ROOT,'reports');
 ensure(output.startsWith(reports+path.sep),'Output must be a new directory inside reports');
 const ioPath=path.join(ROOT,'data/prepared/banc888/io.json');
 const [first,second,ioBytes]=await Promise.all([fs.readFile(inputPaths[0]),fs.readFile(inputPaths[1]),fs.readFile(ioPath)]);
 const report=analyzeMotorEventPair(JSON.parse(first),JSON.parse(second),JSON.parse(ioBytes),sha(ioBytes));
 report.provenance={createdAt:new Date().toISOString(),inputs:inputPaths.map((file,i)=>({file,sha256:sha(i===0?first:second)})),ioFile:ioPath,scriptSha256:sha(await fs.readFile(fileURLToPath(import.meta.url))),nodeVersion:process.version};
 const realReports=await fs.realpath(reports);
 await fs.mkdir(path.dirname(output),{recursive:true});
 const realParent=await fs.realpath(path.dirname(output));
 ensure(realParent===realReports||realParent.startsWith(realReports+path.sep),'Output escaped reports through a symlink');
 await fs.mkdir(output); // Refuse to replace or mix any earlier report.
 await fs.writeFile(path.join(output,'analysis.json'),JSON.stringify(report,null,2)+'\n',{flag:'wx'});
 await fs.writeFile(path.join(output,'README.md'),markdown(report),{flag:'wx'});
 console.log(JSON.stringify({output,pairedPhysicsIdentical:true,window:report.window,wingMotorNeurons:report.perNeuron.length,reason:report.pairing.result.reason}));
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url))main().catch(error=>{console.error(error.stack||error.message);process.exitCode=1;});
