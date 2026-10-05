// Rendering-only fixtures: no Worker, coordinator, neural model or physics.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';

const source=(await fs.readFile(new URL('../training/view.js',import.meta.url),'utf8'))
  .replace(/^import .*;\n/gm,'').replaceAll('import.meta.url',JSON.stringify('http://localhost/training/view.js'))
  .replace(/^initialize\(\);\s*$/m,'');

class Element {
  constructor(){this.textContent='';this.children=[];this.attributes={};this.handlers={};this.value='';this.hidden=false;this.disabled=false;}
  append(...children){this.children.push(...children);}
  after(){}
  addEventListener(name,handler){this.handlers[name]=handler;}
  focus(){this.focused=true;}
  setAttribute(name,value){this.attributes[name]=String(value);}
  removeAttribute(name){delete this.attributes[name];}
  toggleAttribute(name,enabled){if(enabled)this.setAttribute(name,'');else this.removeAttribute(name);}
  replaceChildren(...children){this.children=children;}
  querySelector(){return this.child??=new Element();}
  get previousElementSibling(){return this.previous??=new Element();}
}

function harness(config,state){
  const elements=new Map(),get=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
  get('preview-quality').value='balanced';get('computer-budget').value='60';
  const context=vm.createContext({console,Intl,URL,Blob,setTimeout,clearTimeout,
    document:{getElementById:get,createElement:()=>new Element(),createElementNS:()=>new Element(),querySelector:()=>null,addEventListener(){}},
    window:{location:new URL('http://localhost/train.html'),addEventListener(){}},
    NativeFlyPreview:class{setQuality(){}},TrainingBrainPreview:class{setActive(){}},TrainingEyePreview:class{setActive(){}},TrainingClient:class{},
    PREVIEW_QUALITIES:{balanced:{hz:6},off:{hz:0}},trainingCoordinatorURL:()=>'/api/training',browserTrainingAvailable:()=>true});
  vm.runInContext(source,context);
  const run=code=>vm.runInContext(code,context);
  run(`config=${JSON.stringify(config)};client={config,running:false,setBudget(){},setBrainObservation(){},setEyeObservation(value){this.eyeRequest=value;}};state=${JSON.stringify(state)};`);
  return {get,run,read:code=>JSON.parse(run(`JSON.stringify(${code})`))};
}

const phases=[
  {id:'legs',stage:'maintained_flight',label:'Leg feedback',description:'Calibrate leg feedback.'},
  {id:'antenna',stage:'maintained_flight',label:'Antenna feedback',description:'Calibrate antenna feedback.'},
  {id:'recover',stage:'recovery',label:'Recover',description:'Recover from a disturbance.'},
];
function fixture(){
  const motor=Array.from({length:672},(_,index)=>({name:`decoder_${index}`,side:index%2?'right':'left',target:'wing_muscle',kind:'power',unitIndex:1000+index}));
  return {stage:'maintained_flight',stages:[{id:'maintained_flight'},{id:'recovery'}],trainingSequence:{phases},
    motorDecoderContract:{parameters:motor},parameters:[...Array.from({length:24},(_,index)=>({name:`sensory_gain_${index}`})),...motor.map(({name})=>({name}))]};
}

test('curriculum uses phase identities and configured descriptions when multiple phases share physics',()=>{
  const config=fixture(),curriculum=phases.map((phase,i)=>({...phase,status:i===0?'complete':i===1?'running':'pending'}));
  const {get,run}=harness(config,{phase:'ready',stage:'maintained_flight',curriculum,
    coordinator:{connected:true,sequence:{status:'running',phaseId:'antenna'},curriculum}});
  run('renderStages()');
  assert.equal(get('stage-title').textContent,'Antenna feedback');
  assert.equal(get('stage-description').textContent,'Calibrate antenna feedback.');
  assert.equal(get('stage-progress-value').textContent,'1 / 3');
  assert.deepEqual(get('stage-list').children.map(row=>[row.className,...row.children.map(child=>child.textContent)]),
    [['complete','Leg feedback','Done'],['current','Antenna feedback','Current'],['','Recover','']]);
});

test('sequence initializes from configured phases before remote status instead of listing physical tasks',()=>{
  const {get,run}=harness(fixture(),{phase:'ready',stage:'maintained_flight',curriculum:[{id:'maintained_flight'}],coordinator:{connected:false}});
  run('renderStages()');
  assert.equal(get('stage-title').textContent,'Leg feedback');
  assert.equal(get('stage-progress-value').textContent,'0 / 3');
  assert.equal(get('stage-list').children.length,3);
});

test('terminal sequence statuses prevent starting or resuming and use plain public labels',()=>{
  for(const [status,label] of [['complete','Completed'],['needs-review','Needs review']]){
    const {get,run}=harness(fixture(),{phase:'paused',coordinator:{connected:true,sequence:{status}}});
    run('refreshButtons()');
    assert.equal(get('start-training').disabled,true);
    assert.equal(get('pause-training').disabled,true);
    assert.equal(get('run-status').textContent,label);
    assert.doesNotMatch(get('run-status').textContent,/shared|coordinator/i);
  }
  const {get,run}=harness(fixture(),{phase:'ready',coordinator:{connected:true,sequence:{status:'running'}}});
  run('refreshButtons()');assert.equal(get('start-training').disabled,false);assert.equal(get('run-status').textContent,'Ready');
});

test('motor metadata is joined by name after all 24 sensory entries and reads only current job values',()=>{
  const config=fixture(),values=config.parameters.map((_,index)=>index/1000);
  const {get,run,read}=harness(config,{phase:'ready',episode:4,activity:'evaluating',activeJob:{parameters:values,generation:2},coordinator:{connected:true}});
  run('renderParameters()');
  assert.equal(get('parameter-count').textContent,'696');
  assert.equal(read('parameterDefinitions[0].label'),'sensory gain 0');
  assert.equal(read('parameterDefinitions[24].label'),'Left wing · weight');
  assert.equal(read('parameterDefinitions[24].neuron'),1000);
  assert.equal(read('parameterDefinitions[695].neuron'),1671);
  assert.equal(get('parameter-rows').children[0].children[0].children.length,0);
  assert.equal(get('parameter-rows').children[24].children[0].children[0].textContent,'Neuron 1000');
  assert.equal(get('parameter-rows').children[24].children[1].textContent,'0.02400');
  run('state.activeJob=null;renderParameters()');
  assert.equal(get('parameter-rows').children[24].children[1].textContent,'—');
});

test('legacy physical curriculum remains usable and recovery has a descriptive label',()=>{
  const config={stage:'recovery',stages:[{id:'recovery'}],parameters:[]};
  const {get,run}=harness(config,{phase:'ready',stage:'recovery',coordinator:{connected:true}});
  run('renderStages();refreshButtons()');
  assert.equal(get('stage-title').textContent,'Recover');
  assert.equal(get('stage-description').textContent,'Regain stable flight after a disturbance.');
  assert.equal(get('start-training').disabled,false);
});

test('fit records remain visible with no behavioral score and never enter the score trajectory',()=>{
  const history=[{episode:1,stage:'recovery',mode:'evaluation',return:2.5,success:true},
    {episode:2,stage:'recovery',mode:'decoder-fit',reason:'fit_candidate',return:0,success:false},
    {episode:3,stage:'recovery',mode:'decoder-fit',reason:'fit_failed',return:0,success:false}];
  const {get,run}=harness(fixture(),{phase:'ready',history,lastReturn:0,completedEpisodes:3,coordinator:{connected:true}});
  run('renderHistory()');
  assert.equal(get('latest-reward').textContent,'2.500');
  assert.equal(get('reward-dots').children.length,1);
  assert.equal(get('reward-range').textContent,'Last 1 trials');
  assert.deepEqual(get('candidate-rows').children.map(row=>row.children.slice(2).map(child=>child.textContent)),
    [['—','Fit incomplete'],['—','Fit candidate'],['2.500','Success']]);
  run('state.history=state.history.slice(1);renderHistory()');
  assert.equal(get('latest-reward').textContent,'—');
  assert.equal(get('reward-chart').attributes.hidden,'');
  assert.equal(get('history-empty').textContent,'No flight scores yet');
});

test('Eyes tab requests images only while visible and enabled, with three-tab keyboard navigation',async()=>{
  const {get,run,read}=harness({...fixture(),vision:true},{phase:'ready',coordinator:{connected:true}});
  await run('selectPreview("eyes")');
  assert.equal(get('show-eyes').attributes['aria-selected'],'true');
  assert.equal(get('training-eyes').hidden,false);assert.equal(get('training-preview').hidden,true);
  assert.equal(get('recenter-preview').hidden,true);
  assert.equal(read('client.eyeRequest.enabled'),true);
  run('document.hidden=true;syncInspection()');assert.equal(read('client.eyeRequest.enabled'),false);
  run('document.hidden=false');get('preview-quality').value='off';run('syncInspection()');
  assert.equal(read('client.eyeRequest.enabled'),false);
  get('preview-quality').value='balanced';
  let prevented=false;
  get('show-eyes').handlers.keydown({key:'Home',preventDefault(){prevented=true;}});
  assert.equal(prevented,true);assert.equal(get('show-fly').focused,true);
  assert.equal(read('client.eyeRequest.enabled'),false);
  get('show-brain').handlers.keydown({key:'ArrowRight',preventDefault(){}});
  assert.equal(get('show-eyes').focused,true);assert.equal(read('client.eyeRequest.enabled'),true);
  run('config.vision=false;syncInspection();refreshPreview()');
  assert.equal(read('client.eyeRequest.enabled'),false);
  assert.equal(get('preview-empty').querySelector().textContent,'Eye images are off for this run.');
});
