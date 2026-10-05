// Read-only reduction of recorded flight-feedback assays. Never fits weights.
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {root,sha} from './flight-feedback-runtime.mjs';
const files=process.argv.slice(2);if(!files.length)throw new Error('Supply completed .assay-N.json files');
const norm=x=>Math.hypot(...x),max=x=>x.length?Math.max(...x):0;
const unpack=s=>{const b=Buffer.from(s,'base64');return new Float32Array(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));};
const eventDelta=(a,b,ids)=>{
  const keep=new Set(ids),ea=a.trace.flatMap(t=>t.events).filter(e=>keep.has(e.index)),eb=b.trace.flatMap(t=>t.events).filter(e=>keep.has(e.index));
  const key=e=>e.index+':'+e.timeMs,sa=new Set(ea.map(key)),sb=new Set(eb.map(key));
  const changed=[...ea.filter(e=>!sb.has(key(e))),...eb.filter(e=>!sa.has(key(e)))];
  return {changedEvents:changed.length,changedCells:[...new Set(changed.map(e=>e.index))],firstMs:changed.length?Math.min(...changed.map(e=>e.timeMs))-a.startTimeMs:null};
};
const flatOutput=o=>[...o.power,...o.steering.flat()];
for(const file of files){
  const bytes=await fs.readFile(file),report=JSON.parse(bytes);assert.equal(report.completed,true);
  const n=report.afferentIndices.length,rows=[];
  for(const group of report.groups){
    assert(group.duplicateShamExact&&group.unsignedPairExact,'Failed branch controls');
    const [sham,,positive,negative]=group.arms;
    let currentDiffPeak=0,voltageDiffPeak=0,releaseDiffPeak=0,requestedCurrentPeak=0;
    let currentSum=0,currentCount=0,afferentRatePeak=0;
    const rateMean=[0,0,0];
    for(let t=0;t<positive.trace.length;t++){
      const p=positive.trace[t],m=negative.trace[t];
      assert.equal(p.actualCurrentsPa.length,n,'This reduction requires identical driven/readout afferent sets');
      assert.equal(m.actualCurrentsPa.length,n,'This reduction requires identical driven/readout afferent sets');
      for(let k=0;k<n;k++){
        currentDiffPeak=Math.max(currentDiffPeak,Math.abs(p.actualCurrentsPa[k]-m.actualCurrentsPa[k]));
        requestedCurrentPeak=Math.max(requestedCurrentPeak,p.actualCurrentsPa[k],m.actualCurrentsPa[k]);
        currentSum+=p.actualCurrentsPa[k]+m.actualCurrentsPa[k];currentCount+=2;
      }
      const a=unpack(p.state9),b=unpack(m.state9),s=unpack(sham.trace[t].state9);
      for(let k=0;k<n;k++){
        voltageDiffPeak=Math.max(voltageDiffPeak,Math.abs(a[k*9]-b[k*9]));
        releaseDiffPeak=Math.max(releaseDiffPeak,Math.abs(a[k*9+5]-b[k*9+5]));
        afferentRatePeak=Math.max(afferentRatePeak,a[k*9+4],b[k*9+4]);
        for(const [j,state]of [s,a,b].entries())rateMean[j]+=state[k*9+4]/(n*positive.trace.length);
      }
    }
    const outputDiffPeak=Array(8).fill(0),outputRms=Array(8).fill(0);let firstOutputDifferenceMs=null;
    for(let t=0;t<positive.decoder.length;t++){
      const p=flatOutput(positive.decoder[t].output),m=flatOutput(negative.decoder[t].output);
      for(let k=0;k<8;k++){
        const d=p[k]-m[k];outputDiffPeak[k]=Math.max(outputDiffPeak[k],Math.abs(d));outputRms[k]+=d*d/positive.decoder.length;
        if(d!==0)firstOutputDifferenceMs??=positive.decoder[t].timeMs-report.startTimeMs;
      }
    }
    rows.push({capPa:group.maxCurrentPa,phaseRadians:group.phaseOffsetRadians,
      branchControlsExact:true,afferents:report.afferentIndices.length,gradedAfferents:report.gradedAfferents.length,
      requestedCurrentPeakPa:requestedCurrentPeak,requestedCurrentMeanPa:currentSum/currentCount,currentDifferencePeakPa:currentDiffPeak,
      afferentVoltageDifferencePeakMv:voltageDiffPeak,afferentReleaseDifferencePeak:releaseDiffPeak,
      afferentRatePeakHz:afferentRatePeak,afferentMeanRateHz:{sham:rateMean[0],positive:rateMean[1],negative:rateMean[2]},
      afferentEventContrast:eventDelta(positive,negative,report.afferentIndices),wingEventContrast:eventDelta(positive,negative,report.wingIndices),
      wingNetCountChangedCells:group.response.wingCounts.filter(x=>x.odd!==0).length,
      firstCurrentDifferenceMs:group.response.firstDifferentCurrentMs,firstFeatureDifferenceMs:group.response.firstDifferentDecoderFeatureMs,
      featureOddPeakNorm:max(group.response.featureResponse.map(x=>x.oddNorm)),featureOddMeanNorm:norm(group.response.meanOddFeature),
      firstOutputDifferenceMs,outputDifferencePeak:{power:outputDiffPeak.slice(0,2),steering:outputDiffPeak.slice(2)},
      outputDifferenceRms:{power:outputRms.slice(0,2).map(Math.sqrt),steering:outputRms.slice(2).map(Math.sqrt)}});
  }
  const summary={source:path.relative(root,path.resolve(file)),sourceSha256:sha(bytes),startTimeMs:report.startTimeMs,
    nativeSample:report.nativeSample,protocol:report.protocol,rows,phaseConsistency:report.phaseConsistency,
    aggregationWindow:'Entire 10ms pulse plus30ms recovery; exact event onsets are reported separately.',
    interpretation:'Signed contrasts from two prescribed absolute pitch rates in one frozen state. Different spikes are evidence of a causal response, not correct steering or sufficient information for flight. No body integration, parameter update, significance, or absence-of-information claim.'};
  const out=file.replace(/\.json$/,'.analysis.json');assert.notEqual(out,file);
  await fs.writeFile(out,JSON.stringify(summary,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({output:out,startTimeMs:summary.startTimeMs,nativeSample:summary.nativeSample,rows:rows.map(r=>({
    capPa:r.capPa,phaseRadians:r.phaseRadians,currentDifferencePeakPa:r.currentDifferencePeakPa,
    afferentChangedCells:r.afferentEventContrast.changedCells.length,wingChangedCells:r.wingEventContrast.changedCells.length,
    firstWingDifferenceMs:r.wingEventContrast.firstMs,firstFeatureDifferenceMs:r.firstFeatureDifferenceMs,
    outputDifferencePeak:r.outputDifferencePeak}))}));
}
