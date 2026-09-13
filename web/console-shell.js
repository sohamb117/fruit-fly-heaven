import {createWindowManager} from './window-manager.js';

// Reparent the existing instruments so their IDs, listeners, and live state stay
// intact. The console is a workspace around the simulation, not another model.
export function mountConsole(){
  const take=selector=>document.querySelector(selector);
  const content={controls:take('.simulation-control'),motion:take('.habitat-controls'),scene:take('.scene-wrap'),subject:take('aside'),optics:take('.sensory-observatory'),cortex:take('#anatomy-view'),circuits:take('#circuit-inspector'),motor:take('.motor-map'),model:take('#model-notes'),telemetry:take('.telemetry'),summary:take('.model-summary'),population:take('#population-summary'),pause:take('#pause'),connection:take('.connection')};
  const old=[take('body > header'),take('body > main'),take('body > footer')];
  const shell=document.createElement('div');shell.className='console-shell';
  shell.innerHTML=`
    <header class="console-header">
      <div class="identity"><div class="identity-kicker">FRUIT FLY / NEURAL OBSERVATION SYSTEM</div><h1>HEAVEN<span>PROJECT</span></h1></div>
      <div class="system-diagram" aria-label="Perception, connectome, actuation">
        <svg viewBox="0 0 260 80" aria-hidden="true"><path d="M16 40 38 8h60l22 32-22 32H38ZM140 40l22-32h60l22 32-22 32h-60Z"/><path d="m88 40 20-32h44l20 32-20 32h-44Z"/><path d="M0 40h16m228 0h16"/><text x="58" y="38">INPUT</text><text x="58" y="51">01</text><text x="130" y="35">NEURAL</text><text x="130" y="51">02</text><text x="201" y="38">MOTOR</text><text x="201" y="51">03</text></svg>
        <span>PERCEPTION / CONNECTOME / ACTUATION</span>
      </div>
      <div class="system-meta"><span class="system-label">DROSOPHILA MELANOGASTER</span><div class="system-build">FLYWIRE <b>783</b></div><div id="connection-slot"></div></div>
    </header>
    <main id="console-workspace" aria-label="Live observation workspace"></main>
    <div id="window-layer"></div>
    <footer class="console-footer"><div id="telemetry-slot"></div><div class="console-status"><span class="status-bracket">[ SYSTEM ]</span><span id="workspace-status" role="status">Drag a title bar to undock · ↗ opens a separate window</span><a class="model-link instrument-link" id="training-console-link" href="/train.html">Train ↗</a><button class="model-link instrument-link" id="model-details" type="button">Model &amp; sources ↗</button></div></footer>`;
  document.body.prepend(shell);
  take('#connection-slot').append(content.connection);
  take('#telemetry-slot').append(content.telemetry,content.pause);
  content.cortex.hidden=false;
  content.scene.querySelector('.scene-label').textContent='FALSE-COLOR OBSERVER';
  content.circuits.open=true;content.motor.open=true;content.model.open=true;
  content.cortex.querySelector('#back-to-bowl').textContent='Close cortex';
  content.subject.querySelector('.inspect-header .eyebrow').textContent='SUBJECT TRACKING / INDIVIDUAL STATE';
  // Put the live neural trace above the detailed body readout.
  content.subject.querySelector('.state-row').after(content.subject.querySelector('.neural'));
  const environment=content.subject.querySelector('.sensory');content.motion.append(environment);
  const circuitLink=document.createElement('button');circuitLink.type='button';circuitLink.className='eyebrow instrument-link';circuitLink.textContent='NEURAL OUTPUT RATES ↗';circuitLink.title='Inspect the neural circuits';
  content.subject.querySelector('.signals > .eyebrow').replaceWith(circuitLink);
  const manager=createWindowManager();
  const definitions=[
    {id:'habitat',title:'Habitat',nodes:[content.scene],primary:true},
    {id:'subject',title:'Subject',nodes:[content.subject],primary:true},
    {id:'controls',title:'Control',nodes:[content.controls,content.motion,content.population],primary:true},
    {id:'optics',title:'Optics',nodes:[content.optics],primary:true},
    {id:'cortex',title:'Cortex',nodes:[content.cortex],opener:take('#brain-view')},
    {id:'circuits',title:'Circuits',nodes:[content.circuits,content.motor],opener:circuitLink},
    {id:'model',title:'Model',nodes:[content.summary,content.model],opener:take('#model-details')},
  ];
  const panels=Object.fromEntries(definitions.map(d=>[d.id,manager.register(d)]));
  circuitLink.addEventListener('click',()=>manager.open('circuits'));
  take('#model-details').addEventListener('click',()=>manager.open('model'));
  old.forEach(node=>node.remove());
  return {...manager,panels};
}
