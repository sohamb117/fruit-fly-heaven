// Sensory-only physiology assay. One isolated full BANC brain at a time.
import fs from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'playwright');
const args=Object.fromEntries(process.argv.slice(2).map(s=>s.replace(/^--/,'').split('=')));
const browser=await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,args:['--enable-unsafe-webgpu']});
const page=await browser.newPage();page.on('console',m=>console.log(m.text()));
try{
 await page.goto('http://127.0.0.1:7842/banc/verify.html');
 const result=await page.evaluate(async options=>{
  const {loadBancModel,WebGPUBrain}=await import('/banc-engine/src/index.js');
  const model=await loadBancModel(),sensors=await (await fetch('/banc-data/console/sensory-inputs.json')).json();
  const ids=Uint32Array.from(model.io.motor_neurons.map(c=>c.index)),lookup=new Map(Array.from(ids,(v,k)=>[v,k]));
  const input=new Float32Array(model.manifest.neuron_count),results=[];
  const hzInput=(i,hz)=>{const p=i*16;input[i]=hz>0?model.params[p+1]*(model.params[p+3]-model.params[p+2])*(1-Math.exp(-hz/10))+hz*model.params[p]*(model.params[p+3]-model.params[p+4])/1000:0;};
  const selected=new Set();
  if(options.targets){for(const m of model.io.muscles.filter(m=>options.targets.split(',').includes(m.target)))for(const id of m.indices)for(let e=model.offsets[id];e<model.offsets[id+1];e++)if(model.edges[e*4+2]===0)selected.add(model.edges[e*4]);}
  const afferents=sensors.body_transducers.filter(s=>s.kind==='rotation'&&(!options.targets||selected.has(s.index)));
  for(const hz of (options.hz||'0,50,100,235').split(',').map(Number)){
   input.fill(0);for(const s of afferents)hzInput(s.index,hz);
   const brain=await WebGPUBrain.create(model),samples=[];
   try{
    for(let k=0;k<8;k++){
     await brain.step(100,input,{hunger:.65,akh:.65,insulin:0});const state=await brain.readState(ids);
     const muscles=model.io.muscles.filter(m=>m.joint.includes('wing')).map(m=>({target:m.target,side:m.joint.endsWith('left')?'left':'right',hz:m.indices.reduce((sum,i)=>sum+state[lookup.get(i)*8+4],0)/m.indices.length}));
     samples.push({ms:brain.timeMs,muscles});
    }
   }finally{brain.dispose();}
   const row={sensoryHz:hz,samples};results.push(row);console.log(JSON.stringify({hz,afferents:afferents.length,last:samples.at(-1)}));
  }
  return {date:new Date().toISOString(),options,afferents:afferents.map(s=>s.index),scope:'Constant drive to annotated wing/haltere sensory neurons only. No body or motor injection. Each trial starts a fresh full BANC brain.',results};
 },args);
 await fs.writeFile(args.output||'reports/banc-flight-sensory-probe.json',JSON.stringify(result,null,2)+'\n');
}finally{await browser.close();}
