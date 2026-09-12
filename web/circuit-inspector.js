// The inspector displays measured WASM state; it never controls the simulation.
export function createCircuitInspector(host){
  let current=null,flyId=null,lastKey='';
  const $=id=>host.querySelector('[data-probe="'+id+'"]');
  const number=(n,d=1)=>n.toLocaleString(undefined,{maximumFractionDigits:d,minimumFractionDigits:d});
  const row=(...values)=>{const element=document.createElement('tr');for(const value of values){const cell=document.createElement('td');cell.textContent=value;element.append(cell);}return element;};
  function render(){
    if(!current){$('time').textContent='Collecting 100 ms of selected-brain activity. Resume if paused.';$('results').hidden=true;return;}
    $('results').hidden=false;
    $('time').textContent=`Fly ${String(flyId).padStart(3,'0')} · neural ${(current.timeMs/1000).toFixed(2)} s · ${number(current.windowMs,0)} ms window`;
    $('stages').replaceChildren(...current.groups.map(g=>row(g.label,`${g.active.toLocaleString()} / ${g.n.toLocaleString()}`,number(g.meanHz,3)+' Hz',number(g.meanMv)+' mV')));
    const retina=current.groups.find(g=>g.key==='retina'),motion=current.groups.find(g=>g.key==='motion');
    $('visual-status').textContent=retina.spikes>0&&motion.spikes===0?'Photoreceptors fired, but T4/T5 produced no spikes in this window. Check the graded visual activity and Vision switch above.':motion.spikes>0?'T4/T5 are firing in the FlyWire graph. They receive inputs from the separate graded visual network; its live activity is shown above.':'No photoreceptor spikes in this window. Check Vision and the eye input rates.';
    const channel=current.channels.find(c=>c.key===$('channel').value)||current.channels[0];
    $('motor-status').textContent=channel.spikes>0?`${channel.active} / ${channel.n} cells fired · mean ${number(channel.meanHz,2)} Hz. Direct actuation uses a separate 100 ms smoothed rate with a 2 Hz dead zone. Behavior mode also has a modeled flight program.`:
      `No spikes from any of the ${channel.n} cells. ${channel.maxMv<current.restMv?'All are below the resting voltage at this sample.':channel.maxMv<current.thresholdMv?'Their sampled voltages are below firing threshold.':'A cell is at threshold now; the next neural step will process it.'} This direct channel received zero spikes in this window. Behavior mode can still walk or run its flight program from other active populations.`;
    $('threshold').textContent=`Rest ${current.restMv} mV · firing threshold ${current.thresholdMv} mV · mean net synaptic drive ${number(channel.meanDriveMv)} mV`;
    $('cells').replaceChildren(...channel.cells.map(c=>row(`${c.type} · ${c.side}`,number(c.rateHz,2)+' Hz',number(c.voltageMv)+' mV',number(current.thresholdMv-c.voltageMv)+' mV',c.root_id)));
    for(const [name,entries]of [['positive',channel.incomingPositive],['negative',channel.incomingNegative]]){
      const list=$(name);list.replaceChildren();
      for(const e of entries){const item=document.createElement('li'),label=document.createElement('span'),value=document.createElement('strong');label.textContent=e.type;value.textContent=(e.weightedHz>0?'+':'')+number(e.weightedHz,0);item.append(label,value);list.append(item);}
      if(!entries.length){const item=document.createElement('li');item.textContent='No active inputs of this sign in this window';list.append(item);}
    }
  }
  $('channel').addEventListener('change',render);
  return {update(sample,id){const key=id+':'+(sample?.timeMs??'waiting');if(key===lastKey)return;lastKey=key;current=sample;flyId=id;render();}};
}
