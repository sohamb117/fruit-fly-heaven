// Update dataset-specific text in the existing instruments. Layout, IDs and
// window behavior remain owned by the original console.
export function labelBancConsole(meta){
  const set=(selector,text)=>{const el=document.querySelector(selector);if(el)el.textContent=text;};
  set('.system-build','BANC 888');
  set('.matrix-legend','Membrane voltage: blue below −52 mV, yellow near −45 mV. BANC rest and thresholds vary by cell. Each pixel is one neuron; the matrix is not anatomy.');
  const forward=document.querySelector('#walking')?.parentElement.querySelector('span');if(forward)forward.textContent='Forward · coxa motor neurons';
  set('.signals > .caption','Population means · 50 ms release smoothing · bar scale 0–100 Hz. Direct BANC movement uses individual motor-neuron muscle mappings; channel means are inspection summaries.');
  const subject=document.querySelector('#actuator-status');
  if(subject){const internal=document.createElement('p');internal.id='internal-state';internal.className='caption';subject.after(internal);}
  set('.graded-vision-panel > p:last-child','The trained FlyVis network processes the rendered eyes and drives matching BANC visual cells. Graded activity is shown in model units; firing rates belong to the BANC physiology model.');
  set('.body-sense-panel > p','Native contact forces supply per-leg load. Joint motion and rotation are sensed separately. Sensory tuning remains a modeling assumption.');
  const sensory=document.querySelector('#sensory-mapping-summary')?.parentElement;
  if(sensory){
    const paragraphs=sensory.querySelectorAll(':scope > p');
    if(paragraphs[1])paragraphs[1].textContent='The same binocular RGB cameras, graded visual model and spectral color conversion feed BANC-specific input mappings. L1–L3 carry modeled luminance drive because this snapshot lacks annotated R1–R6. Retinotopy, opsin assignments and current gains are assumptions. Body feedback enters annotated peripheral sensory cells and propagates through the VNC.';
    const link=sensory.querySelector('a');if(link)link.href='/banc-data/console/sensory-inputs.json';
  }
  const circuit=document.querySelector('#circuit-inspector');
  if(circuit){
    const caption=circuit.querySelector('.caption');if(caption)caption.textContent='Simulated BANC voltages and spikes, with graded input neurons reported separately. Brain, VNC and motor readouts retain their measured neuron identities. The separate FlyVis front end is shown above.';
    const headers=circuit.querySelectorAll('th');for(const h of headers)if(h.textContent==='FlyWire root ID')h.textContent='BANC root ID';
  }
  const motor=document.querySelector('.motor-map');
  if(motor){
    const paragraphs=motor.querySelectorAll(':scope > p');
    if(paragraphs[0])paragraphs[0].textContent='Direct BANC movement follows brain → VNC → motor neurons → modeled muscle activation and force → FlyBody joints and MuJoCo contact mechanics. Head motor pathways remain in their measured locations. Behavior mode preserves the original assisted walking and flight controller for comparison.';
    if(paragraphs[1])paragraphs[1].textContent='The twelve original channel readouts now select BANC motor populations. The direct body uses 135 muscle groups mapped from 454 motor neurons; 351 other motor neurons remain simulated without a muscle target. Antennal motion retains the original kinematic boundary. Gains and force scales are uncalibrated assumptions.';
    const link=motor.querySelector('a');if(link)link.href='/banc-data/console/motor-outputs.json';
  }
  const telemetry=document.querySelectorAll('.telemetry > div');
  for(const cell of telemetry)if(cell.textContent.includes('SYNAPSES PER BRAIN'))cell.querySelector('strong').textContent=meta.synapses_represented.toLocaleString();
  const notes=document.querySelector('#model-notes .notes-grid');
  if(notes){
    const sections=[
      ['Measured brain and nerve cord',`${meta.neurons_per_brain.toLocaleString()} neuronal identities and ${meta.connection_rows.toLocaleString()} directed chemical connections from BANC v888/v3. Glial, tracheal and non-neuronal annotations are excluded. Unknown transmitter edges retain topology with zero conductance pending assignment. Display geometry is sampled independently of the complete simulation graph.`],
      ['Modeled physiology','Cell-type priors supply conductance-based spiking or graded dynamics, receptor rise/decay, monoamine sensitivity, adaptation and selected electrical coupling. Reference uses 0.5 ms Float32 steps; Fast uses 1 ms. Parameters are uncalibrated priors, not measured physiology of individual BANC cells.'],
      ['Muscles, body and internal state','Direct mode passes annotated motor-neuron activity into activation, fatigue and force, then the FlyBody joints, contacts and wing aerodynamics in MuJoCo WASM. Hunger, crop filling, energy, insulin-like and AKH-like states feed back into configured neural sensitivities. Finite food transfer requires mouth contact, proboscis activity and pumping. Behavior mode preserves the original assisted controller for comparison. Its flight program is not evidence of autonomous neural control.'],
      ['Target and validation','The target remains food localization → approach → landing → probing/feeding → takeoff/flight. The event monitor observes behavior; it does not issue commands. This sequence has not been validated as an autonomous neural behavior. Cameras, body mechanics, sensory gains, retinotopy and the neuromuscular boundary remain modeled approximations.'],
    ];
    notes.replaceChildren(...sections.map(([title,text])=>{const box=document.createElement('div'),h=document.createElement('h2'),p=document.createElement('p');h.textContent=title;p.textContent=text;box.append(h,p);return box;}));
  }
  const sources=document.querySelector('#model-notes .sources');
  if(sources){sources.replaceChildren(...[['BANC paper','https://doi.org/10.1038/s41586-026-10735-w'],['Graph provenance','/banc-data/manifest.json'],['Motor mappings','/banc-data/io.json'],['FlyBody physics','/body-model/flybody-mujoco.json']].map(([title,url])=>{const a=document.createElement('a');a.textContent=title+' ↗';a.href=url;a.target='_blank';a.rel='noreferrer';return a;}));}
}
