import {NativeFlyPreview,PREVIEW_QUALITIES} from './preview.js';
import {TrainingClient,trainingCoordinatorURL,browserTrainingAvailable} from './client.js';

const $=id=>document.getElementById(id);
const controls={start:$('start-training'),pause:$('pause-training'),stop:$('stop-training'),budget:$('computer-budget'),quality:$('preview-quality'),save:$('save-checkpoint')};
const coordinatorURL=trainingCoordinatorURL(window.location,document.querySelector('meta[name="heaven-training-development"]')?.content==='same-origin');
const pending=new Set();
const trainerNote=document.createElement('p');trainerNote.id='trainer-note';trainerNote.className='download-status';trainerNote.hidden=true;
trainerNote.textContent='This run is using the connected trainer.';$('training-controls').after(trainerNote);
const tasks={takeoff:{label:'Take off',description:'Lift off and gain height.'},flight:{label:'Stay airborne',description:'Take off and maintain controlled flight.'},maintained_flight:{label:'Maintain flight',description:'Stay airborne and in control.'},landing:{label:'Take off, fly & land',description:'Lift off, stay airborne, then land on your feet.'}};
let client=null,config=null,state={phase:'idle',history:[],coordinator:{connected:false}},frame=null;
let pollTimer=null,polling=false,disposed=false,previewState={},recordedFrameKey=null,errorSource=null;
const nativeRun=()=>!!config&&!browserTrainingAvailable(config);
const text=(id,value)=>{const node=$(id),next=String(value);if(node.textContent!==next)node.textContent=next;};
const numeric=(value,digits=2)=>Number.isFinite(value)?value.toLocaleString(undefined,{minimumFractionDigits:digits,maximumFractionDigits:digits}):'—';
const integer=value=>Number.isFinite(value)?Math.max(0,Math.floor(value)).toLocaleString():'—';
const seconds=value=>Number.isFinite(value)?`${numeric(value,value<10?1:0)} s`:'—';
const errorMessage=value=>typeof value==='string'?value:value?.message||'';
const connectionError=error=>/network|fetch|offline|connection|unavailable|timeout|abort|coordinator.*(?:reach|connect)|returned 5\d\d/i.test(errorMessage(error));
const active=()=>['training','loading','evaluating','validating'].includes(state.phase);
const quality=()=>PREVIEW_QUALITIES[controls.quality.value]||PREVIEW_QUALITIES.balanced;
const stageInfo=id=>tasks[id]||{label:'Task',description:''};
function publicError(error,operation){
  const message=errorMessage(error);
  if(operation==='initialize')return 'Could not load training. Check your connection and reload.';
  if(error?.code==='native_trainer_required')return 'This run is using the connected trainer.';
  if(error?.code==='build_mismatch')return 'This page is out of date. Reload and try again.';
  if(/different|mismatch|checksum|configuration|out of date|reload/i.test(message))return 'This page is out of date. Reload and try again.';
  if(operation==='save')return 'Could not download. Check your connection and try again.';
  if(connectionError(error))return nativeRun()?'Connection lost. Reconnecting automatically.':'Connection lost. Check your connection and press Start to retry.';
  if(/memory|device|gpu|adapter|allocation/i.test(message))return 'Training could not start. Close other tabs and try again.';
  return 'Training stopped. Press Start to try again.';
}
const showError=(error,operation)=>{const node=$('training-error');node.textContent=publicError(error,operation);node.hidden=false;errorSource=operation;};
const clearError=()=>{$('training-error').hidden=true;errorSource=null;};
function refreshPreview(){
  const status=previewState;
  $('preview-empty').hidden=!!status.hasFrame&&!status.off&&!status.error;
  $('preview-empty').querySelector('strong').textContent=status.off?'Preview off':status.error?'Preview unavailable':nativeRun()?'This run is using the connected trainer.':'Press Start';
  const received=state.coordinator?.latestFrame;
  let caption=status.error?'Try turning the preview off and on.':'';
  if(nativeRun()&&received?.frame&&!status.off&&!status.error){
    const stale=received.stale||!state.coordinator.connected||!Number.isFinite(received.serverReceivedAt)||Date.now()/1000-received.serverReceivedAt>15;
    caption=stale?'Last received frame':`Updated ${new Date(received.serverReceivedAt*1000).toLocaleTimeString(undefined,{hour:'2-digit',minute:'2-digit',second:'2-digit'})}`;
  }
  text('preview-status',caption);
  controls.quality.disabled=false;$('recenter-preview').disabled=!status.hasFrame||!!status.off||!!status.error;
}
const preview=new NativeFlyPreview($('training-preview'),{onStatus:status=>{previewState=status;refreshPreview();}});
$('training-preview').setAttribute('aria-label','3D fly preview. Drag or use arrow keys to rotate, plus and minus to zoom.');

function budgetOptions(){return {dutyCycle:Number(controls.budget.value)/100,previewHz:quality().hz};}
function refreshStatus(){
  let label='Ready';
  if(pending.has('connect')||!client?.config)label='Connecting…';
  else if(nativeRun())label=!state.coordinator?.connected?'Connection lost':state.coordinator.jobs?.leased>0?'Training':'Watching';
  else if(pending.has('start')||state.phase==='loading')label='Starting…';
  else if(state.phase==='paused')label='Paused';
  else if(state.phase==='training')label='Running';
  else if(state.phase==='error')label=connectionError(state.error)?'Connection lost':'Stopped';
  else if(!state.coordinator?.connected)label='Connection lost';
  else if(state.phase==='stopped')label='Stopped';
  text('run-status',label);
  $('run-light').className=['Running','Training'].includes(label)?'running':label==='Paused'?'paused':'';
}
function refreshButtons(){
  const ready=!!client?.config&&!!config&&!['idle','loading'].includes(state.phase),busy=active(),paused=state.phase==='paused';
  controls.start.disabled=nativeRun()||!ready||busy||paused||pending.has('start')||pending.has('connect');
  controls.pause.disabled=nativeRun()||!client||!['training','paused'].includes(state.phase)||pending.has('pause');
  controls.pause.textContent=paused?'Resume':'Pause';
  controls.pause.setAttribute('aria-label',paused?'Resume training':'Pause training');
  controls.stop.disabled=nativeRun()||!client||(!['training','paused','evaluating','validating','error'].includes(state.phase)&&!client.running&&!client.starting)||pending.has('stop');
  controls.budget.disabled=nativeRun();
  for(const button of [controls.start,controls.pause,controls.stop]){
    if(nativeRun())button.title='This run is using the connected trainer.';else button.removeAttribute('title');
  }
  controls.save.disabled=!client?.config||pending.has('save');
  text('budget-value',`${controls.budget.value}%`);refreshStatus();
}
async function invoke(key,action){
  if(pending.has(key))return;
  pending.add(key);clearError();refreshButtons();
  try{await action();}catch(error){if(error.name!=='AbortError')showError(error,key==='connect'?'sync':key);}finally{pending.delete(key);refreshButtons();}
}

function renderStages(){
  if(!config)return;
  const selected=state.stage||config.stage,info=stageInfo(selected);
  text('stage-title',info.label);text('stage-description',info.description);
  const curriculum=Array.isArray(state.curriculum)&&state.curriculum.length?state.curriculum:config.stages;
  const completed=curriculum.filter(s=>['validated','passed','complete'].includes(s.status)).length;
  text('stage-progress-value',`${completed} / ${curriculum.length}`);
  $('stage-progress').max=Math.max(1,curriculum.length);$('stage-progress').value=completed;
  const list=$('stage-list');list.replaceChildren();
  for(const stage of curriculum){
    const li=document.createElement('li'),label=document.createElement('span'),status=document.createElement('span');
    const done=['validated','passed','complete'].includes(stage.status),current=stage.id===selected;
    li.className=done?'complete':current?'current':'';label.textContent=stageInfo(stage.id).label;
    status.className='stage-marker';status.textContent=done?'Done':current?'Current':'';
    li.append(label,status);list.append(li);
  }
}

const svgNS='http://www.w3.org/2000/svg';
function svgElement(tag,attributes,textValue){const element=document.createElementNS(svgNS,tag);for(const [key,value]of Object.entries(attributes))element.setAttribute(key,String(value));if(textValue!==undefined)element.textContent=String(textValue);return element;}
function renderHistory(){
  const remote=nativeRun(),source=remote?state.coordinator?.recentTrials:state.history;
  const history=(Array.isArray(source)?source:[]).filter(item=>Number.isFinite(item.return)),visible=history.slice(-120);
  const completed=remote?state.coordinator?.acceptedResults:state.completedEpisodes,update=state.coordinator?.lastParameterUpdate;
  text('timeline-heading',remote?'Training progress':'Your trials');
  $('uploaded-count').previousElementSibling.textContent=remote?'Changed values':'Uploaded';
  $('work-time').previousElementSibling.textContent=remote?'Recent time':'Time running';
  text('history-count',integer(completed??0));text('evaluations-count',integer(completed??0));
  text('uploaded-count',remote?Number.isFinite(update?.changedCount)?`${integer(update.changedCount)} / ${integer(update.parameterCount)}`:'—':integer(state.contributedEpisodes??0));
  text('latest-reward',numeric(remote?history.at(-1)?.return:state.lastReturn??history.at(-1)?.return,3));
  text('work-time',seconds(remote?state.coordinator?.recentWallSeconds:state.wallSeconds??0));
  $('history-empty').hidden=visible.length>0;$('reward-chart').toggleAttribute('hidden',!visible.length);
  if(visible.length){
    let min=Math.min(...visible.map(p=>p.return)),max=Math.max(...visible.map(p=>p.return));const padding=Math.max(.01,(max-min)*.15);
    if(max===min){min-=.5;max+=.5;}else{min-=padding;max+=padding;}
    const x=i=>42+(visible.length===1?.5:i/(visible.length-1))*585,y=v=>151-(v-min)/(max-min)*133;
    $('reward-path').setAttribute('d',visible.map((p,i)=>`${i?'L':'M'} ${x(i).toFixed(2)} ${y(p.return).toFixed(2)}`).join(' '));
    $('reward-dots').replaceChildren(...visible.map((p,i)=>svgElement('circle',{cx:x(i),cy:y(p.return),r:2.3})));
    const grid=[];for(let i=0;i<4;i++){const value=min+(max-min)*i/3,py=y(value);grid.push(svgElement('line',{x1:40,x2:628,y1:py,y2:py}),svgElement('text',{x:34,y:py+3,'text-anchor':'end'},numeric(value,2)));}
    $('reward-grid').replaceChildren(...grid);
    $('reward-chart').setAttribute('aria-label',`Scores from ${visible.length} trials. Latest score ${numeric(visible.at(-1).return,3)}.`);
  }
  text('reward-range',visible.length?`Last ${visible.length} trials`:'Score');
  const rows=$('candidate-rows');rows.replaceChildren();
  for(const entry of history.slice(-5).reverse()){
    const tr=document.createElement('tr');
    for(const value of [entry.episode??'—',stageInfo(entry.stage).label,numeric(entry.return,3),entry.success===true?'Success':entry.success===false?'Not yet':'—']){
      const td=document.createElement('td');td.textContent=String(value);tr.append(td);
    }rows.append(tr);
  }
}

function render(next){
  state=next||state;
  config=client?.config||config;
  trainerNote.hidden=!nativeRun();
  if(state.syncError)showError(state.syncError,'sync');
  else if(state.error)showError(state.error,'run');
  else if(errorSource==='sync'||errorSource==='run')clearError();
  text('preview-heading',nativeRun()?'Recorded preview':'Live preview');
  const recorded=state.coordinator?.latestFrame;
  if(nativeRun()&&recorded?.frame){
    const key=JSON.stringify([recorded.jobId,recorded.serverReceivedAt]);
    if(key!==recordedFrameKey){recordedFrameKey=key;renderFrame(recorded.frame);text('preview-candidate','—');}
  }
  renderStages();renderHistory();refreshPreview();refreshButtons();
}

function renderFrame(next){
  frame=next;preview.setFrame(frame);
  text('preview-label',stageInfo(frame.stage).label);
  text('preview-clock',seconds(frame.time??frame.simSeconds));text('preview-time',seconds(frame.time??frame.simSeconds));
  text('preview-candidate',integer(state.episode));text('preview-reward',numeric(frame.metrics?.return??frame.metrics?.reward,3));
}

controls.start.addEventListener('click',()=>invoke('start',()=>client.start({mode:'shared',coordinatorUrl:coordinatorURL,...budgetOptions()})));
controls.pause.addEventListener('click',()=>invoke('pause',()=>state.phase==='paused'?client.resume():client.pause()));
controls.stop.addEventListener('click',()=>invoke('stop',()=>client.stop()));
controls.budget.addEventListener('input',()=>{text('budget-value',`${controls.budget.value}%`);if(client?.setBudget)client.setBudget(budgetOptions());});
controls.quality.addEventListener('change',()=>{preview.setQuality(controls.quality.value);invoke('quality',async()=>{if(client?.setBudget)await client.setBudget(budgetOptions());});});
$('recenter-preview').addEventListener('click',()=>preview.recenter());
controls.save.addEventListener('click',()=>invoke('save',async()=>{
  const checkpoint=await client.downloadSharedCheckpoint();if(!checkpoint)throw new Error('Download failed');
  const blob=new Blob([JSON.stringify(checkpoint,null,2)+'\n'],{type:'application/json'}),url=URL.createObjectURL(blob),link=document.createElement('a');
  link.href=url;link.download=`heaven-checkpoint-generation-${checkpoint.generation}.json`;document.body.append(link);link.click();link.remove();
  setTimeout(()=>URL.revokeObjectURL(url),1000);text('checkpoint-status','Checkpoint downloaded.');$('checkpoint-status').hidden=false;
}));
function schedulePoll(){
  clearTimeout(pollTimer);
  if(!disposed&&!document.hidden)pollTimer=setTimeout(pollStatus,15000);
}
async function pollStatus(){
  clearTimeout(pollTimer);
  if(disposed||document.hidden||polling)return;
  polling=true;
  try{if(client?.config&&!pending.has('connect')&&!client.starting)await client.refreshCoordinatorStatus();}
  catch{/* The client publishes an actionable connection status. */}
  finally{polling=false;if(!disposed){refreshPreview();schedulePoll();}}
}
document.addEventListener('visibilitychange',()=>{if(document.hidden)clearTimeout(pollTimer);else pollStatus();});
window.addEventListener('pagehide',()=>{disposed=true;clearTimeout(pollTimer);client.statusRequest=(client.statusRequest||0)+1;preview.dispose();},{once:true});

async function initialize(){
  try{
    client=new TrainingClient({sharedOnly:true,coordinatorUrl:coordinatorURL});
    client.addEventListener('state',event=>render(event.detail));client.addEventListener('frame',event=>renderFrame(event.detail));
    globalThis.heavenTraining={client,preview,get state(){return state;},get frame(){return frame;}};
    await client.initialize();
    await invoke('connect',()=>client.connectCoordinator(coordinatorURL));
    schedulePoll();
  }catch(error){showError(error,'initialize');text('run-status',connectionError(error)?'Connection lost':'Stopped');}
}
initialize();
