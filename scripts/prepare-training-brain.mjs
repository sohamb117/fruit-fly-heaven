#!/usr/bin/env node
// A small display-only sample of measured BANC anchors, in prepared graph order.
import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');

export function sampleBrain({config,graph,metadata,idsBytes,positionsBytes,ownersBytes,neuronsBytes,count=4096}){
  if(!Number.isInteger(count)||count<1||count>4096)throw new Error('Brain sample must contain 1–4096 anchors');
  if(graph.dataset!=='BANC'||graph.materialization!==888||metadata.dataset!=='BANC v888'||metadata.neuronCount!==graph.neuron_count)
    throw new Error('Brain geometry and neural graph identity differ');
  if(!/^[a-f0-9]{64}$/.test(config.modelFingerprint))throw new Error('Missing model identity');
  const check=(bytes,record)=>{if(!record||bytes.length!==record.bytes||sha(bytes)!==record.sha256)throw new Error('Brain source checksum mismatch');};
  check(idsBytes,graph.files['ids.bin']);check(positionsBytes,metadata.files['positions.bin']);
  check(ownersBytes,metadata.files['neuron-indices.bin']);check(neuronsBytes,metadata.files['neurons.json']);
  const neurons=JSON.parse(neuronsBytes),n=graph.neuron_count,m=ownersBytes.length/4;
  if(idsBytes.length!==n*8||!Number.isInteger(m)||positionsBytes.length!==m*12||!Array.isArray(neurons.ids)||neurons.ids.length!==n||neurons.labels?.length!==n)
    throw new Error('Invalid brain source dimensions');
  for(let i=0;i<n;i++)if(neurons.ids[i]!==idsBytes.readBigUInt64LE(i*8).toString())throw new Error('Brain labels do not match prepared neuron identities');
  const owners=new Set();
  for(let i=0;i<m;i++){
    const owner=ownersBytes.readUInt32LE(i*4);
    if(owner>=n||owners.has(owner))throw new Error('Invalid or duplicate brain anchor owner');
    owners.add(owner);
    for(let k=0;k<3;k++)if(!Number.isFinite(positionsBytes.readFloatLE(i*12+k*4)))throw new Error('Nonfinite brain coordinate');
  }
  if(!m)throw new Error('No measured brain anchors');
  const size=Math.min(count,m),indices=[],positions=[],ids=[],labels=[];
  for(let k=0;k<size;k++){
    const point=size===1?0:Math.floor(k*(m-1)/(size-1)),index=ownersBytes.readUInt32LE(point*4);
    indices.push(index);ids.push(neurons.ids[index]);labels.push(neurons.labels[index]);
    for(let axis=0;axis<3;axis++)positions.push(positionsBytes.readFloatLE(point*12+axis*4));
  }
  return {schemaVersion:1,dataset:'BANC v888',modelFingerprint:config.modelFingerprint,
    preparedIdsSha256:graph.files['ids.bin'].sha256,neuronCount:n,sampleCount:size,mappedNeurons:m,
    units:'µm',positionMeaning:metadata.positionMeaning,selection:'Evenly spaced entries from the measured anchor list; display sample only.',
    sources:{positionsSha256:metadata.files['positions.bin'].sha256,ownersSha256:metadata.files['neuron-indices.bin'].sha256,neuronsSha256:metadata.files['neurons.json'].sha256},
    indices,positions,ids,labels};
}

export async function prepareTrainingBrain(config,{prepared=path.join(root,'data/prepared/banc888'),count=4096}={}){
  const anatomy=path.join(prepared,'anatomy');
  const [graphBytes,metadataBytes,idsBytes,positionsBytes,ownersBytes,neuronsBytes]=await Promise.all([
    fs.readFile(path.join(prepared,'manifest.json')),fs.readFile(path.join(anatomy,'metadata.json')),
    fs.readFile(path.join(prepared,'ids.bin')),fs.readFile(path.join(anatomy,'positions.bin')),
    fs.readFile(path.join(anatomy,'neuron-indices.bin')),fs.readFile(path.join(anatomy,'neurons.json'))]);
  if(sha(graphBytes)!==config.assets?.['/banc-data/manifest.json'])throw new Error('Prepared graph differs from the training configuration');
  return sampleBrain({config,graph:JSON.parse(graphBytes),metadata:JSON.parse(metadataBytes),idsBytes,positionsBytes,ownersBytes,neuronsBytes,count});
}

if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  const args=Object.fromEntries(process.argv.slice(2).map(arg=>{const match=/^--(config|output|prepared)=(.+)$/.exec(arg);if(!match)throw new Error('Use --config=PATH --output=PATH [--prepared=PATH]');return [match[1],match[2]];}));
  if(!args.config||!args.output)throw new Error('Both --config and --output are required');
  const config=JSON.parse(await fs.readFile(args.config));
  const sample=await prepareTrainingBrain(config,{...(args.prepared?{prepared:args.prepared}:{})});
  await fs.writeFile(args.output,JSON.stringify(sample)+'\n');
  console.log(JSON.stringify({output:args.output,sampleCount:sample.sampleCount,neuronCount:sample.neuronCount,modelFingerprint:sample.modelFingerprint}));
}
