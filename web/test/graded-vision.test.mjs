import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createVisionModule} from '../../packages/fly-vision-wasm/dist/index.js';
import {compileVisualModel,compileVisualProjection,eyeInput,GradedVision} from '../graded-vision.js';

const json=async name=>JSON.parse(await readFile(new URL('../'+name,import.meta.url)));
const spec=await json('visual-model.json'),mapping=await json('visual-projections.json'),motors=await json('motor-outputs.json');
const compiled=compileVisualModel(spec),runtime=await createVisionModule(),model=runtime.createModel(compiled.csr);
const types=['T4a','T4b','T4c','T4d','T5a','T5b','T5c','T5d'];
const centers=Object.fromEntries(types.map(t=>[t,compiled.layers[t].filter(i=>Math.abs(compiled.nodes[i].u)<8&&Math.abs(compiled.nodes[i].v)<8)]));
const mean=(a,ids)=>ids.reduce((s,i)=>s+Math.max(0,a[i]),0)/ids.length;
const grating=(frame,direction)=>Uint8Array.from({length:1024},(_,i)=>255*(.5+.45*Math.sin(2*Math.PI*((i%32)/31*3-direction*frame*.02*2))));

test('published graph retains its complete size and only maps into identified visual cells',()=>{
  assert.equal(compiled.nodes.length,45669);assert.equal(compiled.csr.weights.length,1513231);assert.equal(spec.nodes.length,65);
  assert.equal(mapping.cells.length,14746);assert.equal(new Set(mapping.cells.map(c=>c.index)).size,14746);
  const motorIds=new Set(motors.channels.flatMap(c=>c.indices));
  assert.ok(mapping.cells.every(c=>!motorIds.has(c.index)&&c.index<138639&&/^\d+$/.test(c.root_id)));
  assert.ok(compileVisualProjection(compiled,mapping).every((p,i)=>compiled.nodes[p.node].type===mapping.cells[i].type));
});

test('reversing rendered gratings reverses the graded T4 direction response',()=>{
  const result=[];
  for(const direction of [1,-1]){
    const network=model.createNetwork(),sums=Object.fromEntries(types.map(t=>[t,0]));
    network.step(new Float32Array(model.inputCount).fill(.5),50);
    for(let f=0;f<100;f++){
      network.step(eyeInput(compiled,grating(f,direction)));
      if(f>=50){const a=network.readActivations();for(const t of types)sums[t]+=mean(a,centers[t])/50;}
    }
    result.push(sums);network.dispose();
  }
  assert.ok(result[0].T4b>result[0].T4a*1.3);assert.ok(result[1].T4a>result[1].T4b*1.2);
  assert.ok(result[1].T5c>result[0].T5c*5);assert.ok(result[0].T5d>result[1].T5d*2);
});

test('bright and dark moving edges preferentially recruit ON and OFF populations',()=>{
  const responses=[];
  for(const polarity of [1,-1]){
    const network=model.createNetwork(),sums={T4:0,T5:0};
    network.step(new Float32Array(model.inputCount).fill(polarity>0?.1:.9),75);
    for(let f=0;f<75;f++){
      const pixels=Uint8Array.from({length:1024},(_,i)=>255*((i%32)/31<(f-8)/60?(polarity>0?.9:.1):(polarity>0?.1:.9)));
      network.step(eyeInput(compiled,pixels));const a=network.readActivations();
      for(const t of types)sums[t.slice(0,2)]+=mean(a,centers[t])/75;
    }
    responses.push(sums);network.dispose();
  }
  assert.ok(responses[0].T4>responses[1].T4*3);
  assert.ok(responses[1].T5>responses[0].T5*2);
});

test('eyes have independent history and disconnecting vision clears bridge drive',()=>{
  const projection=compileVisualProjection(compiled,mapping),a=new GradedVision(model,compiled,mapping,projection),b=new GradedVision(model,compiled,mapping,projection);
  for(let f=0;f<35;f++){
    const pixels=grating(f,1),other=grating(f,-1);pixels.fill(128,512);other.fill(128,512);
    a.update({sequence:f,bodyTime:f*.02,pixels},f*20);b.update({sequence:f,bodyTime:f*.02,pixels:other},f*20);
  }
  assert.deepEqual(a.eyes[1].readActivations(),b.eyes[1].readActivations());
  assert.notDeepEqual(a.eyes[0].readActivations(),b.eyes[0].readActivations());
  assert.ok(a.ratesHz.some(n=>n>0));assert.ok(a.ratesHz.every(n=>n>=0&&n<=mapping.max_rate_hz));
  a.update(null,700,false);assert.ok(a.ratesHz.every(n=>n===0));assert.equal(a.summary.ready,false);
  a.dispose();b.dispose();
});

test.after(()=>model.dispose());
