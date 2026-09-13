// Sequential one-brain calibration diagnostic. Does not modify production data.
import fs from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'playwright');
const browser=await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,args:['--enable-unsafe-webgpu']});
const page=await browser.newPage();
page.on('console',message=>console.log(message.text()));
try{
  await page.goto('http://127.0.0.1:7842/banc/verify.html');
  const result=await page.evaluate(async()=>{
    const {loadBancModel,WebGPUBrain}=await import('/banc-engine/src/index.js');
    const model=await loadBancModel(),groups=await (await fetch('/banc-data/console/groups.json')).json();
    const weights=new Float32Array(model.edges.buffer),original=weights.slice(),input=new Float32Array(model.manifest.neuron_count),results=[];
    const motorIds=Uint32Array.from(model.io.motor_neurons.map(c=>c.index)),lookup=new Map(Array.from(motorIds,(v,k)=>[v,k]));
    const read=async brain=>{
      const s=await brain.readState(motorIds),kind=name=>{
        const ids=model.io.muscles.filter(m=>m.kind===name).flatMap(m=>m.indices),rates=ids.map(id=>s[lookup.get(id)*8+4]);
        return {mean:rates.reduce((a,b)=>a+b,0)/Math.max(1,rates.length),max:Math.max(0,...rates)};
      };
      return {ms:brain.timeMs,totalSpikes:s.totalSpikes,wing:kind('asynchronous_wing'),leg:kind('leg'),proboscis:kind('proboscis_assumption'),pump:kind('pump')};
    };
    for(const scale of [1,.5,.2,.1,.05]){
      for(let e=0;e<model.manifest.chemical_edges;e++)weights[e*4+1]=original[e*4+1]*(.5*scale/(model.manifest.synaptic_weight_ns_ms_per_contact??.5));
      const brain=await WebGPUBrain.create(model),sample={scale,weightPerContact:.5*scale,rest:[],taste:[]};
      try{
        input.fill(0);
        for(let k=0;k<5;k++){await brain.step(80,input,{hunger:.65,akh:.65,insulin:0});sample.rest.push(await read(brain));}
        for(const i of groups.sweet){const p=i*16,hz=150;input[i]=model.params[p+1]*(model.params[p+3]-model.params[p+2])*(1-Math.exp(-hz/10))+hz*model.params[p]*(model.params[p+3]-model.params[p+4])/1000;}
        for(let k=0;k<5;k++){await brain.step(80,input,{hunger:.65,akh:.65,insulin:0});sample.taste.push(await read(brain));}
      }finally{brain.dispose();}
      results.push(sample);console.log(JSON.stringify({scale,rest:sample.rest.at(-1),taste:sample.taste.at(-1)}));
    }
    return {date:new Date().toISOString(),scope:'One brain at a time. External inputs zero for 200 ms, followed by isolated 150 Hz sugar-afferent drive for 200 ms. Chemical conductance scale is the only varied parameter; graph topology/contact counts stay fixed.',results};
  });
  await fs.writeFile('reports/banc-rest-calibration.json',JSON.stringify(result,null,2)+'\n');
}finally{await browser.close();}
