import fs from 'node:fs';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import {createBrainModule} from '../packages/fly-brain-wasm/dist/index.js';
import {createColorModule,loadFlyColorModel} from '../packages/fly-color-wasm/dist/index.js';
import {createVisionModule} from '../packages/fly-vision-wasm/dist/index.js';
import {ColorVision} from '../web/color-vision.js';
import {compileVisualModel,compileVisualProjection,GradedVision} from '../web/graded-vision.js';
import {SensoryEncoder} from '../web/sensory-encoder.js';
import {BodyWorld} from '../web/body-world.js';
import {SIMULATION_MODES} from '../web/simulation-modes.js';
const root=new URL('../',import.meta.url),json=p=>JSON.parse(fs.readFileSync(new URL(p,root)));
const meta=json('data/prepared/metadata.json'),mapping=json('web/color-inputs.json'),sense=json('web/sensory-inputs.json'),groups=json('data/prepared/groups.json'),habitat=json('web/habitat.json'),visual=json('web/visual-projections.json'),motors=json('web/motor-outputs.json');
const data={neuronCount:meta.neurons_per_brain};
for(const [name,key,T]of [['indptr.bin','rowOffsets',Uint32Array],['targets.bin','targets',Uint32Array],['weights.bin','weights',Float32Array]]){
  const b=fs.readFileSync(new URL('data/prepared/'+name,root));assert.equal(crypto.createHash('sha256').update(b).digest('hex'),meta.prepared_sha256[name]);data[key]=new T(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));
}
assert.equal(mapping.ids_sha256,meta.prepared_sha256['ids.bin']);
const colorModel=await loadFlyColorModel(),mapper=(await createColorModule()).createMapper(colorModel),compiled=compileVisualModel(json('web/visual-model.json')),vm=(await createVisionModule()).createModel(compiled.csr),projection=compileVisualProjection(compiled,visual);
const colors={blue:[0,0,205],green:[0,255,0]},targets=new Set(mapping.cells.map(c=>c.index)),outputIds=new Set(motors.channels.flatMap(c=>c.indices));
assert.ok(mapping.cells.every(c=>!outputIds.has(c.index)));
const receptorIds=Uint32Array.from(mapping.cells.map(c=>c.index));
const report={protocol:'600 ms full FlyWire graph per condition, matched seed and frozen pose. Identical constant luminance plane, FlyVis graded input, food and body feedback. Only the R7/R8 spectral input differs: blue, green, or disconnected. The first two stimuli have approximately matched total Rh5+Rh6 capture. This checks signal delivery and propagation, not validated color perception.',model:colorModel.metadata,graphSha256:meta.prepared_sha256,results:[]};
for(const [name,mode]of Object.entries(SIMULATION_MODES)){
  const graph=(await createBrainModule({precision:mode.precision})).createConnectome(data),snapshots=[];
  try{
    for(const condition of ['off','blue','green']){
      const brain=graph.createBrain({...mode.parameters,seed:20360917}),world=new BodyWorld(habitat.fruit,[structuredClone(habitat.flies[0])]),encoder=new SensoryEncoder(sense,groups,world.habitat,{visualMapping:visual,colorMapping:mapping}),color=new ColorVision(mapper,mapping),graded=new GradedVision(vm,compiled,visual,projection);
      brain.setRefractoryPeriod(Uint32Array.from([...groups.odor_left,...groups.odor_right,...groups.sweet]),0);
      try{
        const rgb=colors[condition]||colors.blue;
        for(let ms=0;ms<600;ms+=20){
          const frame={sequence:ms/20,bodyTime:ms/1000,pixels:new Uint8Array(1024).fill(128),rgb:Uint8Array.from({length:3072},(_,i)=>rgb[i%3])};
          color.update(frame,condition!=='off');graded.update(frame,ms);brain.setPoissonInputs(encoder.update(world.poses()[0],frame,{graded,color}));brain.step(20);
        }
        const all=brain.readActivations({field:'spikeCount'});snapshots.push(all);
        report.results.push({mode:name,condition,rgb:condition==='off'?null:rgb,captures:color.summary.left,totalSpikes:brain.totalSpikes,receptorSpikes:Object.fromEntries(colorModel.metadata.channels.map(rh=>[rh,mapping.cells.filter(c=>c.opsin===rh).reduce((s,c)=>s+all[c.index],0)]))});
      }finally{brain.dispose();graded.dispose();}
    }
    const a=snapshots[1],b=snapshots[2];let changedReceptors=0,changedOtherNeurons=0;
    for(let i=0;i<a.length;i++)if(a[i]!==b[i]){if(targets.has(i))changedReceptors++;else changedOtherNeurons++;}
    assert.ok(changedReceptors>100);assert.ok(changedOtherNeurons>100);
    const sums=report.results.slice(-3);assert.ok(sums[1].receptorSpikes.Rh5>sums[2].receptorSpikes.Rh5);assert.ok(sums[2].receptorSpikes.Rh6>sums[1].receptorSpikes.Rh6);
    report.results.push({mode:name,changedReceptors,changedOtherNeurons});
    console.log(JSON.stringify(report.results.slice(-4)));
  }finally{graph.dispose();}
}
mapper.dispose();vm.dispose();fs.writeFileSync(new URL('../reports/color-validation.json',import.meta.url),JSON.stringify(report,null,2)+'\n');
