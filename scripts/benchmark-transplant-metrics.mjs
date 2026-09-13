// Pure validation/statistics for the browser harness; no browser or simulator.
export function median(values){
  if(!values.length||!values.every(Number.isFinite))return null;
  const sorted=[...values].sort((a,b)=>a-b),middle=Math.floor(sorted.length/2);
  return sorted.length%2?sorted[middle]:(sorted[middle-1]+sorted[middle])/2;
}

export function sampleValidity(sample,{seconds}){
  const issues=[],{before,after,population,mode,dataset}=sample;
  if(!before||!after)return ['missing measurement endpoints'];
  if(sample.errors?.length)issues.push('application errors');
  if(!(Number.isFinite(sample.wallMs)&&sample.wallMs>=seconds*1000))issues.push('measurement shorter than requested');
  if(!(Number.isFinite(sample.neuralMs)&&sample.neuralMs>0))issues.push('neural clock did not advance');
  for(const [name,point] of [['before',before],['after',after]]){
    const c=point.configuration;
    if(point.paused!==false)issues.push(`${name}: simulation paused`);
    if(point.population!==population)issues.push(`${name}: population mismatch`);
    if(!c||c.movementMode!=='direct'||c.movementControl!=='direct'||c.bodyClock!=='neural'||c.fast!==(mode==='fast')||c.referenceController)
      issues.push(`${name}: requested direct/neural/mode configuration not active`);
    if(!c||!c.flightEnabled||!c.motorCoupling||!c.sensorySwitches||Object.keys(c.sensorySwitches).length!==4||!Object.values(c.sensorySwitches).every(v=>v===true))issues.push(`${name}: required coupling or sensory switch disabled`);
    if(c?.bodyBackend!==(dataset==='banc'?'mujoco-wasm':'kinematic'))issues.push(`${name}: unexpected body backend`);
    if(dataset==='flywire'&&point.backend!=='wasm')issues.push(`${name}: unexpected FlyWire brain backend`);
    if(dataset==='banc'&&!['wasm','webgpu'].includes(point.backend))issues.push(`${name}: unknown BANC brain backend`);
    if(point.eyesReady!==population||point.gradedReady!==population||point.colorReady!==population)issues.push(`${name}: sensory inputs not ready`);
    if(!point.bodies||point.bodies.length!==population)issues.push(`${name}: missing body observations`);
    for(const body of point.bodies||[]){
      if(!Array.isArray(body.position)||body.position.length!==3||![body.time,body.neuralMs,...body.position].every(Number.isFinite))issues.push(`${name}: body ${body.id} is nonfinite`);
      const tolerance=dataset==='banc'?.0025:1/60+.0025;
      if(Math.abs(body.time-body.neuralMs/1000)>tolerance)issues.push(`${name}: body ${body.id} and neural clocks disagree`);
    }
    if(point.errors?.length)issues.push(`${name}: displayed error`);
  }
  if(JSON.stringify(before.configuration)!==JSON.stringify(after.configuration)||before.backend!==after.backend)issues.push('configuration changed during measurement');
  for(const body of after.bodies||[]){
    const old=before.bodies?.find(b=>b.id===body.id);
    if(!old||!(body.time>old.time))issues.push(`body ${body.id} did not advance`);
    if(old&&after.neural-before.neural>=50&&!(body.eyeSequence>old.eyeSequence))issues.push(`body ${body.id} retinal frames did not advance`);
  }
  if(!(after.anatomy?.frames>before.anatomy?.frames)||!(after.anatomy?.scanPixels>0))issues.push('anatomy inspection did not render during measurement');
  if(!(after.rendererFrames>before.rendererFrames))issues.push('scene renderer did not advance');
  return [...new Set(issues)];
}

export function compareSamples(samples,{population,mode,trials}){
  const select=dataset=>samples.filter(s=>s.population===population&&s.mode===mode&&s.dataset===dataset);
  const old=select('flywire'),banc=select('banc'),validOld=old.filter(s=>s.valid),validBanc=banc.filter(s=>s.valid);
  const validTrialIds=rows=>rows.length===trials&&new Set(rows.map(s=>s.trial)).size===trials&&rows.every(s=>s.trial>=1&&s.trial<=trials);
  const signature=s=>JSON.stringify({backend:s.after?.backend,configuration:s.after?.configuration});
  const complete=validTrialIds(validOld)&&validTrialIds(validBanc)&&old.length===trials&&banc.length===trials;
  const configurationsConsistent=[validOld,validBanc].every(rows=>new Set(rows.map(signature)).size<=1);
  const eligible=complete&&configurationsConsistent;
  const oldMedian=median(validOld.map(s=>s.cohortSpeed)),bancMedian=median(validBanc.map(s=>s.cohortSpeed));
  const paired=validOld.flatMap(a=>{const b=validBanc.find(b=>b.trial===a.trial);return b?[{trial:a.trial,slowdown:a.cohortSpeed/b.cohortSpeed}]:[];});
  const ratio=oldMedian!==null&&bancMedian>0?oldMedian/bancMedian:null;
  return {population,mode,expectedTrialsPerDataset:trials,validTrials:{flywire:validOld.length,banc:validBanc.length},complete,configurationsConsistent,eligible,
    flywireMedianSpeed:oldMedian,bancMedianSpeed:bancMedian,slowdown:eligible?ratio:null,pairedTrialSlowdowns:paired,
    pairedSlowdownRange:paired.length?[Math.min(...paired.map(p=>p.slowdown)),Math.max(...paired.map(p=>p.slowdown))]:null,
    threshold:10,withinOneOrderOfMagnitude:eligible?ratio<=10:null,
    everyPairedTrialWithinOneOrderOfMagnitude:eligible?paired.every(p=>p.slowdown<=10):null,
    limitation:'Short-window whole-product throughput: shared 3D console, different intended neural and body models. A <=10x median slowdown does not establish equal speed, long-run performance, or behavioral parity.'};
}
