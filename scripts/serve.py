"""Static example server. Brains run in browser WASM Workers, not in Python."""
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse,unquote
import argparse

ROOT=Path(__file__).resolve().parents[1]
class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*a,**kw):super().__init__(*a,directory=str(ROOT/'web'),**kw)
    def translate_path(self,path):
        path=unquote(urlparse(path).path)
        for prefix,base in [('/engine/',ROOT/'packages/fly-brain-wasm/dist'),('/view-engine/',ROOT/'packages/brain-view-wasm/dist'),('/anatomy/',ROOT/'data/anatomy'),('/connectome/',ROOT/'data/prepared')]:
            if path.startswith(prefix):
                target=(base/path[len(prefix):]).resolve()
                if not target.is_relative_to(base.resolve()):return str(ROOT/'web/__missing__')
                return str(target)
        return super().translate_path(path)
    def end_headers(self):
        self.send_header('Cache-Control','no-store')
        super().end_headers()
    def log_message(self,*a):pass

p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=7842);args=p.parse_args()
print(f'Fruit Fly Heaven (WASM): http://127.0.0.1:{args.port}',flush=True)
ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
