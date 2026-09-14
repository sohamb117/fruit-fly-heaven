#!/usr/bin/env node
// Static recorded-pose report only. Never imports an evaluator or starts a worker.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const EXPERIMENT=path.join(ROOT,'reports/motor-decoder-v1/experiment');
const OUTPUT=path.join(ROOT,'reports/motor-decoder-v1/preview');
const number=x=>typeof x==='number'&&Number.isFinite(x);
function vector(value,n,label){
  if(!Array.isArray(value)||value.length!==n||!value.every(number))throw new Error('Invalid recorded '+label);
  return value.slice();
}
function numericFields(value,names){
  return Object.fromEntries(names.filter(name=>number(value?.[name])).map(name=>[name,value[name]]));
}
function ellipsoid(value){
  return {position:vector(value.position,3,'ellipsoid position'),size:vector(value.size,3,'ellipsoid size'),rotation:vector(value.rotation,9,'ellipsoid rotation')};
}
export function sanitizeFrame(frame){
  const result={position:vector(frame.position,3,'position'),quaternion:vector(frame.quaternion,4,'quaternion'),
    ...numericFields(frame,['time','simSeconds','nativeTimeSeconds','episodeTimeSeconds']),warmup:frame.warmup===true};
  if(!number(result.nativeTimeSeconds))result.nativeTimeSeconds=result.time;
  if(!number(result.nativeTimeSeconds))throw new Error('Missing recorded frame time');
  result.feet=(frame.feet||[]).map(p=>vector(p,3,'foot'));
  result.legs=(frame.legs||[]).map(points=>points.map(p=>vector(p,3,'leg point')));
  result.wings=(frame.wings||[]).map(ellipsoid);
  result.mouth={ellipsoids:(frame.mouth?.ellipsoids||[]).map(ellipsoid)};
  result.food=(frame.food||[]).map(food=>({kind:food.kind==='banana'?'banana':'apple',
    position:vector(food.position,3,'food position'),...numericFields(food,['radiusCm','lengthCm','rotation'])}));
  result.bowl={...numericFields(frame.bowl,['radiusCm','ceilingCm']),
    floor:numericFields(frame.bowl?.floor,['baseCm','radialCoefficientPerCm','capRadiusCm'])};
  result.contacts={...numericFields(frame.contacts,['environment','food'])};
  for(const [key,n]of [['legs',6],['mouth',2],['wings',2]])
    if(frame.contacts?.[key])result.contacts[key]=vector(frame.contacts[key],n,'contact counts');
  result.metrics=numericFields(frame.metrics,['return','elapsed','flightSeconds','bestFlightSeconds','totalQualifiedFlightSeconds']);
  result.metrics.observation=numericFields(frame.metrics?.observation,['height','up','angularSpeed','verticalSpeed','speedCmPerSecond']);
  return result;
}

export function sanitizeRecord(record,{baseline=false,fallbackName='Trial'}={}){
  const evaluation=record?.evaluation;
  if(!evaluation||record.complete===false||record.error||record.sourceVerificationError||evaluation.cancelled===true||
    !number(evaluation.return)||!number(evaluation.simSeconds)||!Array.isArray(record.frames)||!record.frames.length)return null;
  if(!(evaluation.terminated===true||evaluation.truncated===true||['time_limit','stage_success'].includes(evaluation.reason)))return null;
  const name=baseline?'Baseline':typeof record.assignment?.jobId==='string'&&/^g\d+-(?:p|a)\d+-(?:pos|neg|candidate|incumbent)$/.test(record.assignment.jobId)?record.assignment.jobId:fallbackName;
  const reason=typeof evaluation.reason==='string'&&/^[a-z_]{1,48}$/.test(evaluation.reason)?evaluation.reason:'completed';
  const frames=record.frames.map(sanitizeFrame);
  for(let i=1;i<frames.length;i++)if(frames[i].nativeTimeSeconds<frames[i-1].nativeTimeSeconds)throw new Error('Recorded frame clock regressed');
  return {name,metrics:{...numericFields(evaluation,['return','simSeconds']),
    ...numericFields(evaluation.metrics,['bestFlightSeconds','totalQualifiedFlightSeconds']),success:evaluation.success===true,reason},frames};
}

const HTML=`<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; worker-src 'none'; object-src 'none'; base-uri 'none'">
<title>Flight replay</title><link rel="stylesheet" href="./training/train.css"><link rel="stylesheet" href="./report.css"></head>
<body><main class="training-shell"><header class="training-header"><h1>Flight replay</h1><div class="status-light"><i class="paused"></i><span>Recorded · slowed playback</span></div></header>
<div class="run-bar"><div class="run-actions"><button id="play" class="primary" disabled>Play</button><button id="recenter">Recenter</button></div>
<div><label for="trial">Trial</label><select id="trial" disabled></select></div><button id="refresh">Refresh recordings</button></div>
<p id="error" class="error-banner" role="alert" hidden></p>
<div class="workspace"><section class="instrument"><div class="titlebar"><h2>Recorded flight</h2><span class="instrument-id">3D</span></div>
<div class="preview-wrap"><div id="training-preview"></div><div class="preview-overlay"><span id="phase">Loading</span><span>320 × 180</span></div>
<div class="preview-caption"><span>Drag to orbit · scroll to zoom</span><span id="render-status"></span></div></div>
<div class="panel-body"><div class="range-label"><label for="frame">Frame</label><output id="frame-label" for="frame">—</output></div><input id="frame" type="range" min="0" max="0" step="1" value="0" disabled>
<p>Discrete recorded frames · no interpolation</p></div>
<dl class="preview-readouts"><div><dt>Native time</dt><dd id="native-time">—</dd></div><div><dt>Scored time</dt><dd id="scored-time">—</dd></div><div><dt>Playback</dt><dd id="playback-state">Paused</dd></div></dl></section>
<section class="instrument"><div class="titlebar"><h2>Trial result</h2></div><div class="panel-body"><h3 id="trial-name">—</h3><p id="result">—</p>
<dl class="result-metrics"><div><dt>Longest qualified flight</dt><dd id="best-flight">—</dd></div><div><dt>Total qualified flight</dt><dd id="total-flight">—</dd></div><div><dt>Scored duration</dt><dd id="duration">—</dd></div><div><dt>Return</dt><dd id="score">—</dd></div></dl>
<p id="recording-summary"></p></div></section></div></main><script type="module" src="./player.js"></script></body></html>`;

const CSS=`.run-bar{grid-template-columns:auto minmax(180px,1fr) auto}.result-metrics{display:grid;gap:22px;margin:25px 0}.result-metrics dd{margin-top:5px}.preview-wrap{min-height:360px}.panel-body p{margin-bottom:0}#render-status{font-size:9px}#trial-name{overflow-wrap:anywhere}@media(max-width:760px){.run-bar{grid-template-columns:1fr}.run-actions{grid-column:1}.preview-wrap{min-height:240px}.result-metrics{grid-template-columns:1fr 1fr}.training-header h1{font-size:45px;letter-spacing:-2px}}\n`;

const PLAYER=`import {NativeFlyPreview} from './training/preview.js';
const $=id=>document.getElementById(id),seconds=x=>Number.isFinite(x)?x.toFixed(3)+' s':'—';
let trials=[],trial=null,index=0,timer=null,playing=false;
const preview=new NativeFlyPreview($('training-preview'),{onStatus:status=>{
  if(status.error){$('error').textContent=status.text;$('error').hidden=false;pause();}
  else if(status.hasFrame)$('render-status').textContent='Recorded';
}});preview.setQuality('low');
function pause(){playing=false;clearTimeout(timer);timer=null;$('play').textContent='Play';$('playback-state').textContent='Paused';}
function show(){
  const frame=trial.frames[index];preview.setFrame(frame);$('frame').value=String(index);
  $('frame-label').textContent=(index+1)+' / '+trial.frames.length;
  $('frame').setAttribute('aria-valuetext','Frame '+(index+1)+' of '+trial.frames.length);
  $('native-time').textContent=seconds(frame.nativeTimeSeconds);
  $('scored-time').textContent=frame.warmup?'Unscored':seconds(frame.episodeTimeSeconds??frame.simSeconds);
  $('phase').textContent=frame.warmup?'Warmup · unscored':'Recorded · scored episode';
}
function schedule(){
  if(!playing)return;if(index>=trial.frames.length-1){pause();return;}
  const delta=Math.max(0,trial.frames[index+1].nativeTimeSeconds-trial.frames[index].nativeTimeSeconds);
  // At least four times slower than native time, with a minimum readable dwell.
  timer=setTimeout(()=>{timer=null;if(!playing)return;index++;show();schedule();},Math.max(500,delta*4000));
}
function select(){
  pause();trial=trials[Number($('trial').value)];index=0;
  $('frame').max=String(trial.frames.length-1);$('frame').disabled=false;$('play').disabled=trial.frames.length<2;
  $('trial-name').textContent=trial.name;
  $('result').textContent=trial.metrics.success?'Stage success':trial.metrics.reason.replaceAll('_',' ');
  $('best-flight').textContent=seconds(trial.metrics.bestFlightSeconds);
  $('total-flight').textContent=seconds(trial.metrics.totalQualifiedFlightSeconds);
  $('duration').textContent=seconds(trial.metrics.simSeconds);$('score').textContent=trial.metrics.return.toFixed(3);
  $('recording-summary').textContent=trial.frames.length+' recorded frames. Playback ends at the last saved pose.';show();
}
$('play').addEventListener('click',()=>{if(!trial)return;if(playing){pause();return;}if(index===trial.frames.length-1)index=0;playing=true;$('play').textContent='Pause';$('playback-state').textContent='Slowed';show();schedule();});
$('frame').addEventListener('input',()=>{pause();index=Number($('frame').value);show();});
$('trial').addEventListener('change',select);$('recenter').addEventListener('click',()=>preview.recenter());
$('refresh').addEventListener('click',()=>location.reload());
document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
window.addEventListener('pagehide',()=>{pause();preview.dispose();},{once:true});
try{const response=await fetch('./recordings.json',{cache:'no-store'});if(!response.ok)throw new Error('Recordings could not be loaded');
 const data=await response.json();trials=data.trials;if(!trials.length)throw new Error('No completed recorded trials yet. Regenerate this snapshot after a trial finishes.');
 for(const [i,t]of trials.entries()){const option=document.createElement('option');option.value=String(i);option.textContent=t.name;$('trial').append(option);}
 $('trial').disabled=false;select();
}catch(error){$('error').textContent=error.message;$('error').hidden=false;}
`;

async function linkAsset(relative,target){
  const destination=path.join(OUTPUT,relative);await fs.mkdir(path.dirname(destination),{recursive:true});
  const link=path.relative(path.dirname(destination),path.join(ROOT,target));
  try{const entry=await fs.lstat(destination);if(!entry.isSymbolicLink()||await fs.readlink(destination)!==link)
    throw new Error('Unexpected existing preview asset: '+relative);}
  catch(error){if(error.code!=='ENOENT')throw error;await fs.symlink(link,destination);}
}
async function writeSnapshot(name,value){
  const temporary=path.join(OUTPUT,'.'+name+'.tmp');await fs.writeFile(temporary,value);await fs.rename(temporary,path.join(OUTPUT,name));
}

export async function preparePreview(){
  const trials=[];let skipped=0;
  for(const directory of ['baseline','contributions']){
    const folder=path.join(EXPERIMENT,directory);
    const files=(await fs.readdir(folder)).filter(name=>name.endsWith('.json')&&!name.startsWith('run-')&&name!=='checkpoint.json').sort((a,b)=>a.localeCompare(b,undefined,{numeric:true}));
    for(const [i,file]of files.entries()){
      let record;try{record=JSON.parse(await fs.readFile(path.join(folder,file),'utf8'));}
      catch(error){if(error instanceof SyntaxError){skipped++;continue;}throw error;}
      const sanitized=sanitizeRecord(record,{baseline:directory==='baseline',fallbackName:'Trial '+(i+1)});
      if(sanitized)trials.push(sanitized);else skipped++;
    }
  }
  await fs.mkdir(OUTPUT,{recursive:true});
  // Selective links, not a link to the entire training directory: no worker,
  // configuration, checkpoint or original contribution is served here.
  for(const [relative,target]of [['training/preview.js','web/training/preview.js'],['training/train.css','web/training/train.css'],
    ['vendor/three.module.js','web/vendor/three.module.js'],['vendor/three.core.js','web/vendor/three.core.js'],['vendor/LICENSE.three','web/vendor/LICENSE.three']])await linkAsset(relative,target);
  const recordings={trials};
  for(const [name,value]of [['index.html',HTML],['player.js',PLAYER],['report.css',CSS],['recordings.json',JSON.stringify(recordings)+'\n']])await writeSnapshot(name,value);
  return {output:OUTPUT,trials:trials.length,frames:trials.reduce((n,t)=>n+t.frames.length,0),skipped,
    serve:'.venv/bin/python -m http.server 7873 --bind 127.0.0.1 --directory reports/motor-decoder-v1/preview'};
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url))
  preparePreview().then(result=>console.log(JSON.stringify(result))).catch(error=>{console.error(error.message);process.exitCode=1;});
