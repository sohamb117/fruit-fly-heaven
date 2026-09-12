// Live UI nodes may be adopted into a same-origin popout. They keep their event
// handlers and canvases; no second simulation or mirrored controls are created.
const documents=new Set([document]);
export function uiElement(id){for(const doc of documents){const found=doc.getElementById(id);if(found)return found;}return null;}
export function uiQueryAll(selector){return [...documents].flatMap(doc=>[...doc.querySelectorAll(selector)]);}
export function registerUIDocument(doc){documents.add(doc);doc.addEventListener('visibilitychange',schedule);schedule();return()=>{doc.removeEventListener('visibilitychange',schedule);documents.delete(doc);schedule();};}
export function isUIVisible(){return [...documents].some(doc=>!doc.hidden&&doc.defaultView&&!doc.defaultView.closed);}
const callbacks=new Set();let scheduledWindow=null,frame=0;
function visibleWindow(){return [...documents].find(doc=>!doc.hidden&&doc.defaultView&&!doc.defaultView.closed)?.defaultView;}
function tick(){scheduledWindow=null;frame=0;try{for(const callback of callbacks)callback(performance.now());}finally{schedule();}}
function schedule(){
  const next=visibleWindow();
  if(next===scheduledWindow&&frame)return;
  if(scheduledWindow&&frame){try{scheduledWindow.cancelAnimationFrame(frame);}catch{}}
  scheduledWindow=null;frame=0;
  if(next&&callbacks.size){scheduledWindow=next;frame=next.requestAnimationFrame(tick);}
}
export function onUIFrame(callback){callbacks.add(callback);schedule();return()=>{callbacks.delete(callback);schedule();};}
document.addEventListener('visibilitychange',schedule);
// Recover a pending RAF if its popout was closed or backgrounded before firing.
setInterval(schedule,500);
