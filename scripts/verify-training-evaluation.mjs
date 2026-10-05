import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'playwright');
const directory='reports/training-independent-checkpoint';await fs.mkdir(directory,{recursive:true});
const checkpoint=await (await fetch('http://127.0.0.1:7850/api/training/checkpoint')).json();
await fs.writeFile(directory+'/shared-checkpoint.json',JSON.stringify(checkpoint,null,2));
const report={date:new Date().toISOString(),scope:'Independent evaluation of the generation-one shared candidate on three configured test seeds, with actual BANC/FlyBody; no update or model selection uses these returns.',errors:[]};
const browser=await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,args:['--enable-unsafe-webgpu']});
try{
 const page=await browser.newPage({viewport:{width:1440,height:1000}});page.on('pageerror',e=>report.errors.push(e.message));
 await page.goto('http://127.0.0.1:7842/train.html');await page.waitForFunction(()=>window.heavenTraining?.state.phase==='ready');
 await page.setInputFiles('#load-checkpoint',directory+'/shared-checkpoint.json');
 await page.waitForFunction(generation=>heavenTraining.state.generation===generation,checkpoint.generation);await page.selectOption('#computer-budget','high');await page.selectOption('#preview-quality','low');
 await page.click('#validate-checkpoint');
 await page.waitForFunction(()=>heavenTraining.state.phase==='error'||heavenTraining.client.testValidation,null,{timeout:240000});
 report.state=await page.evaluate(()=>heavenTraining.state);if(report.state.phase==='error')throw new Error(report.state.error);
 report.checkpoint=await page.evaluate(()=>heavenTraining.client.exportCheckpoint());
 assert.deepEqual(report.checkpoint.parameters,checkpoint.parameters);assert.equal(report.checkpoint.testValidation.results.length,3);assert.equal(report.state.completedEpisodes,3);assert.equal(report.state.contributedEpisodes,0);
 assert(report.state.history.every(r=>r.role==='test'));assert.equal(report.errors.length,0);
 await page.screenshot({path:directory+'/independent-evaluation.png',fullPage:true});await page.evaluate(()=>heavenTraining.client.stop());
 report.passed=true;
}catch(error){report.passed=false;report.failure=error.stack;process.exitCode=1;}
finally{await browser.close();await fs.writeFile(directory+'/result.json',JSON.stringify(report,null,2)+'\n');}
console.log(JSON.stringify({passed:report.passed,validation:report.checkpoint?.testValidation,report:directory+'/result.json',failure:report.failure}));
