"""Local UI server. Simulation and snapshots never share mutable brain state."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import json
import signal
import threading
import time
from urllib.parse import urlparse, parse_qs
from world import World, FRUIT

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--port',type=int,default=7841)
    p.add_argument('--flies',type=int,default=100)
    p.add_argument('--workers',type=int,default=6)
    args=p.parse_args()
    if not 1<=args.flies<=100: p.error('--flies must be between 1 and 100')
    world=World(args.flies,args.workers)
    lock=threading.Lock();stop=threading.Event()
    state={'snapshot':world.snapshot(),'error':None,'paused':False,'odor':True,'taste':True}
    metadata=json.loads((ROOT/'data/prepared/metadata.json').read_text())
    runtime=ROOT/'runtime';runtime.mkdir(exist_ok=True)
    def loop():
        try:
            while not stop.is_set():
                with lock:
                    pause=state['paused'];world.odor_enabled=state['odor'];world.taste_enabled=state['taste']
                if pause:
                    stop.wait(.05);continue
                world.advance()
                snap=world.snapshot()
                # Save numeric evidence and the most recent actual neural events.
                with lock:
                    state['snapshot']=snap
                    state['recent']=[f['recent'] for f in world.flies]
                (runtime/'latest.json.tmp').write_text(json.dumps(snap,allow_nan=False))
                (runtime/'latest.json.tmp').replace(runtime/'latest.json')
        except Exception as exc:
            with lock:state['error']=repr(exc);state['paused']=True
            print('Simulation error:',repr(exc),flush=True)
    worker=threading.Thread(target=loop,daemon=True);worker.start()
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*a,**kw):super().__init__(*a,directory=str(ROOT/'web'),**kw)
        def log_message(self,*args):pass
        def reply(self,obj,status=200):
            data=json.dumps(obj,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(data)))
            self.end_headers();self.wfile.write(data)
        def do_GET(self):
            url=urlparse(self.path)
            if url.path=='/api/meta':return self.reply({**metadata,'flies':args.flies,'fruit':FRUIT})
            if url.path=='/api/state':
                try:i=int(parse_qs(url.query).get('fly',['1'])[0])-1
                except ValueError:return self.reply({'error':'Invalid fly ID'},400)
                if not 0<=i<args.flies:return self.reply({'error':'Invalid fly ID'},400)
                with lock:
                    snap={**state['snapshot'],'paused':state['paused'],'error':state['error']}
                    snap['selected_spikes']=state.get('recent',[[]]*args.flies)[i]
                return self.reply(snap)
            return super().do_GET()
        def do_POST(self):
            if self.headers.get('Origin') not in (None,f'http://127.0.0.1:{args.port}',f'http://localhost:{args.port}'):
                return self.reply({'error':'Origin rejected'},403)
            if self.path!='/api/control':return self.reply({'error':'Unknown action'},404)
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<1024:raise ValueError('Invalid request size')
                data=json.loads(self.rfile.read(size))
                if not isinstance(data,dict):raise ValueError('Expected object')
                if any(k not in ('paused','odor','taste') or type(v) is not bool for k,v in data.items()):
                    raise ValueError('Controls must be booleans')
                with lock:state.update(data)
                return self.reply({'ok':True})
            except (ValueError,TypeError) as e:return self.reply({'error':str(e)},400)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'Fruit Fly Heaven: http://127.0.0.1:{args.port} — {args.flies} independent connectomes',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        stop.set();worker.join();server.server_close();world.close()

if __name__=='__main__':main()
