// Usage: PLAYWRIGHT_MODULE=/absolute/path/to/playwright/index.mjs node scripts/verify-banc-browser.mjs
// Serve the project on 7842 first. Chrome is used to exercise actual WebGPU.
import fs from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'playwright');
const executablePath=process.env.CHROME_PATH||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const browser=await chromium.launch({executablePath,headless:true,args:['--enable-unsafe-webgpu']});
const page=await browser.newPage({viewport:{width:1400,height:1050}}),errors=[];
page.on('pageerror',error=>errors.push(error.message));
try{
  await page.goto('http://127.0.0.1:7842/banc/verify.html');
  await page.waitForFunction(()=>typeof window.verify==='function');
  const parity=await page.evaluate(async()=>{try{return await window.verify();}catch(error){throw new Error(error.stack||error.message||String(error));}});
  const fallback=await page.evaluate(()=>window.verifyFallback());
  if(process.argv.includes('--benchmark')){
    const benchmark=await page.evaluate(()=>window.benchmark());
    await fs.writeFile('reports/banc-compute-benchmark.json',JSON.stringify(benchmark,null,2)+'\n');
  }
  await page.goto('http://127.0.0.1:7842/banc-lab.html');
  await page.waitForFunction(()=>window.bancSnapshot||document.getElementById('status').classList.contains('error'),{},{timeout:120000});
  const initial=await page.evaluate(()=>({ready:window.bancReady,status:document.getElementById('status').textContent,body:window.bancSnapshot}));
  if(!initial.ready||!initial.body)throw new Error(initial.status);
  if(initial.ready.backend!=='webgpu')throw new Error('Full BANC test fell back from WebGPU');
  await page.waitForFunction(seconds=>window.bancSnapshot.time>=seconds,process.argv.includes('--long')?1:.05,{timeout:120000});
  await page.getByRole('button',{name:'Pause',exact:true}).click();
  await page.waitForTimeout(600);
  const paused=await page.evaluate(()=>window.bancSnapshot.time);await page.waitForTimeout(200);
  if(await page.evaluate(()=>window.bancSnapshot.time)!==paused)throw new Error('Pause advanced body time');
  await page.getByRole('button',{name:'Step 5 ms',exact:true}).click();
  await page.waitForFunction(t=>window.bancSnapshot.time>t,paused);
  const stepped=await page.evaluate(()=>window.bancSnapshot.time);
  if(Math.abs(stepped-paused-.005)>1e-9)throw new Error('Single step did not advance 5 ms');
  await page.screenshot({path:'/tmp/fruit-fly-banc-verified.png',fullPage:true});
  const final=await page.evaluate(()=>({time:window.bancSnapshot.time,x:window.bancSnapshot.x,y:window.bancSnapshot.y,events:window.bancSnapshot.events,backend:window.bancReady.backend,status:document.getElementById('status').textContent}));
  if(process.argv.includes('--long')){
    const body=await page.evaluate(()=>({...window.bancSnapshot,muscleState:Array.from(window.bancSnapshot.muscleState),jointState:Array.from(window.bancSnapshot.jointState)}));
    await fs.writeFile('reports/banc-closed-loop.json',JSON.stringify({backend:initial.ready.backend,adapter:initial.ready.adapter,body,errors},null,2)+'\n');
  }
  if(errors.length)throw new Error(errors.join('\n'));
  const report={parity,fallback,full_graph:{neurons:initial.ready.manifest.neuron_count,edges:initial.ready.manifest.chemical_edges,adapter:initial.ready.adapter,...final},pause_and_single_step:true,browser:browser.version(),errors};
  await fs.writeFile('reports/banc-browser-validation.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
}finally{await browser.close();}
