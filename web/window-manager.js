import {registerUIDocument} from './ui-elements.js';
const STORAGE='heaven-console-layout-v1';
const make=(tag,className,text)=>{const node=document.createElement(tag);node.className=className;if(text)node.textContent=text;return node;};
const clamp=(n,lo,hi)=>Math.max(lo,Math.min(hi,n));
export function createWindowManager(){
  const workspace=document.getElementById('console-workspace'),layer=document.getElementById('window-layer'),status=document.getElementById('workspace-status');
  const panels=new Map();let z=20,unloading=false;
  let saved={};try{saved=JSON.parse(localStorage.getItem(STORAGE))||{};}catch{}
  const announce=message=>{status.textContent=message;};
  function save(){try{localStorage.setItem(STORAGE,JSON.stringify(Object.fromEntries([...panels].map(([id,p])=>[id,{open:!p.node.hidden,mode:p.mode==='popout'?'float':p.mode,rect:p.rect}]))));}catch{}}
  function geometry(p,rect){
    const maxWidth=Math.max(260,innerWidth-16),maxHeight=Math.max(180,innerHeight-24);
    const w=clamp(Number(rect.w)||680,Math.min(300,maxWidth),maxWidth),h=clamp(Number(rect.h)||480,180,maxHeight);
    const x=clamp(Number(rect.x)||0,8,Math.max(8,innerWidth-w-8)),y=clamp(Number(rect.y)||0,8,Math.max(8,innerHeight-h-8));
    p.rect={x,y,w,h};Object.assign(p.node.style,{left:x+'px',top:y+'px',width:w+'px',height:h+'px'});
  }
  function focus(p){p.node.style.zIndex=String(++z);for(const q of panels.values())q.node.classList.toggle('is-focused',p===q);}
  function notify(p){
    p.slot.classList.toggle('is-vacant',p.mode!=='dock'||p.node.hidden);
    p.slot.hidden=!p.primary&&(p.mode!=='dock'||p.node.hidden);
    p.placeholderLabel.textContent=p.node.hidden?'WINDOW ON STANDBY':p.mode==='popout'?'OPEN IN SEPARATE WINDOW':'WINDOW UNDOCKED';
    p.node.dispatchEvent(new CustomEvent('windowvisibility',{bubbles:true,detail:{visible:!p.node.hidden}}));
    p.node.querySelector('[data-action="maximize"]').setAttribute('aria-label',`${p.maximized?'Restore':'Maximize'} ${p.title}`);
    p.node.querySelector('[data-action="maximize"]').title=p.maximized?'Restore window':'Maximize window';
  }
  function restorePopout(p,close=true){
    if(!p.popup)return;
    const popup=p.popup;p.popup=null;p.unregister?.();p.unregister=null;
    layer.append(document.adoptNode(p.node));p.node.classList.remove('is-popout');p.mode='float';
    if(close&&!popup.closed)popup.close();
  }
  function dock(p){restorePopout(p);p.mode='dock';p.maximized=false;p.node.hidden=false;p.node.classList.remove('is-floating','is-maximized');p.node.style.cssText='';p.slot.append(p.node);notify(p);focus(p);save();announce(`${p.title} docked`);}
  function float(p,rect=p.rect){
    restorePopout(p);if(!rect){const r=p.node.getBoundingClientRect();rect=r.width?{x:r.x,y:r.y,w:r.width,h:r.height}:{x:80,y:120,w:800,h:560};}
    p.mode='float';p.maximized=false;p.node.hidden=false;p.node.classList.remove('is-maximized');p.node.classList.add('is-floating');layer.append(p.node);geometry(p,rect);notify(p);focus(p);save();
  }
  function open(id){const p=panels.get(id);if(!p)return;if(p.popup&&!p.popup.closed){p.popup.focus();return;}p.node.hidden=false;if(p.mode==='dock')dock(p);else float(p);p.titlebar.focus({preventScroll:true});notify(p);save();}
  function hide(id){const p=panels.get(id);if(!p)return;restorePopout(p);p.node.hidden=true;notify(p);(p.opener||p.reopen).focus({preventScroll:true});save();announce(`${p.title} hidden.`);}
  function maximize(p){
    if(p.mode==='popout'){p.popup.focus();return;}
    if(p.maximized){p.maximized=false;geometry(p,p.beforeMax);p.node.classList.remove('is-maximized');}
    else{if(p.mode==='dock')float(p);p.beforeMax={...p.rect};p.maximized=true;geometry(p,{x:8,y:8,w:innerWidth-16,h:innerHeight-16});p.node.classList.add('is-maximized');}
    notify(p);focus(p);save();
  }
  function popout(p){
    if(p.popup&&!p.popup.closed){p.popup.focus();return;}
    const popup=window.open('',`heaven-${p.id}`,`popup,width=${Math.round(p.rect?.w||860)},height=${Math.round(p.rect?.h||650)}`);
    if(!popup){float(p);announce('The browser blocked a separate window. The panel is floating here; allow popups to detach it.');return;}
    const doc=popup.document;doc.title=`${p.title} / HEAVEN`;
    const base=doc.createElement('base');base.href=document.baseURI;
    const viewport=doc.createElement('meta');viewport.name='viewport';viewport.content='width=device-width,initial-scale=1';
    const style=doc.createElement('link');style.rel='stylesheet';style.href=new URL('console.css',document.baseURI).href;
    doc.head.replaceChildren(base,viewport,style);doc.documentElement.lang='en';doc.body.className='console-popout';
    p.unregister=registerUIDocument(doc);p.mode='popout';p.popup=popup;p.node.hidden=false;p.node.classList.remove('is-floating','is-maximized');p.node.classList.add('is-popout');p.node.style.cssText='';doc.body.append(doc.adoptNode(p.node));
    popup.addEventListener('pagehide',()=>{if(p.popup===popup&&!unloading){restorePopout(p,false);float(p);announce(`${p.title} returned to the console`);}});
    // pagehide also covers refreshing the detached document.
    notify(p);save();announce(`${p.title} detached. Close its window to return it here.`);
  }
  function register({id,title,nodes,primary=false,opener=null}){
    const slot=make('div',`window-slot slot-${id}`),placeholder=make('div','window-placeholder'),placeholderLabel=make('span','','WINDOW ON STANDBY'),reopen=make('button','','Recall window');
    placeholder.append(placeholderLabel,reopen);slot.append(placeholder);if(!primary)slot.classList.add('secondary-slot');workspace.append(slot);
    const node=make('section',`console-window panel-${id}`);node.dataset.window=id;node.setAttribute('aria-label',title+' window');
    const titlebar=make('div','window-titlebar');titlebar.tabIndex=0;titlebar.setAttribute('role','button');titlebar.setAttribute('aria-label',`Move ${title} window`);titlebar.title='Drag to move · arrow keys to move · Shift + arrows for larger steps · double-click to maximize';
    const label=make('div','window-title');label.append(make('h2','',title));titlebar.append(label);
    const actions=make('div','window-actions');
    for(const [action,symbol,description]of [['dock','⊞','Dock'],['float','⧉','Float'],['popout','↗','Pop out'],['maximize','□','Maximize'],['hide','×','Hide']]){const b=make('button','',symbol);b.type='button';b.dataset.action=action;b.setAttribute('aria-label',`${description} ${title}`);b.title=`${description} window`;actions.append(b);}
    titlebar.append(actions);const body=make('div','window-body');nodes.forEach(n=>body.append(n));
    const resize=make('button','window-resize');resize.setAttribute('aria-label',`Resize ${title}`);resize.title='Drag to resize · arrow keys to resize';resize.innerHTML='<span aria-hidden="true">◢</span>';
    node.append(titlebar,body,resize);slot.append(node);
    node.id=`window-${id}`;
    const p={id,title,node,slot,body,titlebar,reopen,opener,placeholderLabel,primary,mode:primary?'dock':'float',rect:null,maximized:false,popup:null};panels.set(id,p);
    const initial=saved[id];if(initial&&['dock','float'].includes(initial.mode)){p.mode=initial.mode;p.rect=initial.rect;}node.hidden=initial?initial.open===false:!primary;
    if(p.mode==='float'){layer.append(node);node.classList.add('is-floating');geometry(p,p.rect||{x:60+panels.size*18,y:90+panels.size*12,w:Math.min(980,innerWidth-90),h:Math.min(690,innerHeight-110)});}
    node.addEventListener('pointerdown',()=>focus(p));
    reopen.addEventListener('click',()=>dock(p));
    actions.addEventListener('click',e=>{const action=e.target.closest('button')?.dataset.action;if(action==='dock')dock(p);if(action==='float')float(p);if(action==='popout')popout(p);if(action==='maximize')maximize(p);if(action==='hide')hide(id);});
    titlebar.addEventListener('dblclick',e=>{if(!e.target.closest('button'))maximize(p);});
    let drag=null;
    function begin(e,resizing){
      if(e.button!==0||p.mode==='popout'||(!resizing&&e.target.closest('button')))return;
      e.preventDefault();if(p.maximized)maximize(p);if(p.mode==='dock')float(p);
      drag={x:e.clientX,y:e.clientY,rect:{...p.rect},resizing,target:e.currentTarget};e.currentTarget.setPointerCapture(e.pointerId);node.classList.add('is-dragging');
    }
    function move(e){if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y,r=drag.rect;geometry(p,drag.resizing?{...r,w:r.w+dx,h:r.h+dy}:{...r,x:r.x+dx,y:r.y+dy});}
    function end(){if(!drag)return;drag=null;node.classList.remove('is-dragging');save();}
    for(const [handle,resizing]of [[titlebar,false],[resize,true]]){
      handle.addEventListener('pointerdown',e=>begin(e,resizing));handle.addEventListener('pointermove',move);handle.addEventListener('pointerup',end);handle.addEventListener('pointercancel',end);handle.addEventListener('lostpointercapture',end);
      handle.addEventListener('keydown',e=>{
        if(e.target!==handle||p.mode==='popout')return;
        if(!resizing&&(e.key==='Enter'||e.key===' ')){e.preventDefault();p.mode==='dock'?float(p):dock(p);return;}
        if(e.key==='Escape'){dock(p);return;}
        if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key))return;e.preventDefault();if(p.mode==='dock')float(p);if(p.maximized)maximize(p);
        const step=e.shiftKey?40:10,dx=e.key==='ArrowLeft'?-step:e.key==='ArrowRight'?step:0,dy=e.key==='ArrowUp'?-step:e.key==='ArrowDown'?step:0;
        geometry(p,resizing?{...p.rect,w:p.rect.w+dx,h:p.rect.h+dy}:{...p.rect,x:p.rect.x+dx,y:p.rect.y+dy});save();
      });
    }
    notify(p);return p;
  }
  window.addEventListener('resize',()=>{for(const p of panels.values())if(p.mode==='float')geometry(p,p.maximized?{x:8,y:8,w:innerWidth-16,h:innerHeight-16}:p.rect);});
  window.addEventListener('beforeunload',()=>{unloading=true;for(const p of panels.values())if(p.popup&&!p.popup.closed)p.popup.close();});
  return {register,open,hide,isOpen:id=>{const p=panels.get(id);return !!p&&!p.node.hidden;}};
}
