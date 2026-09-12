"""Package locally; never publishes to an external registry or creates a tag."""
from pathlib import Path
import gzip
import hashlib
import io
import json
import tarfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('package', nargs='?', default='fly-brain-wasm', choices=['fly-brain-wasm','brain-view-wasm'])
args=parser.parse_args()
PACKAGE=ROOT/'packages'/args.package
meta=json.loads((PACKAGE/'package.json').read_text())
name=f"{meta['name']}-{meta['version']}"
OUT=ROOT/'releases';OUT.mkdir(exist_ok=True)
files=[PACKAGE/p for p in ['package.json','README.md','LICENSE','CHANGELOG.md']]
files+=sorted((PACKAGE/'dist').glob('*'))
files.append(PACKAGE/'build.sh')
for directory in ['native','src','third-party-licenses'] + (['test'] if 'test' in meta.get('files', []) else []):
    files+=sorted(p for p in (PACKAGE/directory).rglob('*') if p.is_file())
required={'core.js','core.wasm','index.js','index.d.ts','worker.js'}
if args.package=='fly-brain-wasm':required.update({'core-f32.js','core-f32.wasm'})
assert required.issubset({p.name for p in files}), 'Build all release files first'
assert (PACKAGE/'dist/core.wasm').read_bytes()[:4]==b'\0asm'
(OUT/f'{name}.wasm').write_bytes((PACKAGE/'dist/core.wasm').read_bytes())
manifest={'name':meta['name'],'version':meta['version'],'compiler':'Emscripten 4.0.23',
          'published':False,'dataset_included':False,'files':{}}
if args.package=='fly-brain-wasm':
    payload=(PACKAGE/'dist/core-f32.wasm').read_bytes()
    assert payload[:4]==b'\0asm'
    (OUT/f'{name}-f32.wasm').write_bytes(payload)
    manifest['precision_variants']={'float64':'dist/core.wasm','float32':'dist/core-f32.wasm'}
for path in files:
    payload=path.read_bytes()
    manifest['files'][str(path.relative_to(PACKAGE))]={'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
archive=io.BytesIO()
with tarfile.open(fileobj=archive,mode='w',format=tarfile.USTAR_FORMAT) as tar:
    for path in files:
        payload=path.read_bytes();info=tarfile.TarInfo('package/'+str(path.relative_to(PACKAGE)))
        info.size=len(payload);info.mode=0o644;info.mtime=0
        tar.addfile(info,io.BytesIO(payload))
with (OUT/f'{name}.tgz').open('wb') as output:
    with gzip.GzipFile(filename='',mode='wb',fileobj=output,mtime=0) as compressed:
        compressed.write(archive.getvalue())
with zipfile.ZipFile(OUT/f'{name}.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        info=zipfile.ZipInfo(name+'/'+str(path.relative_to(PACKAGE)),date_time=(2026,1,1,0,0,0))
        info.compress_type=zipfile.ZIP_DEFLATED;archive.writestr(info,path.read_bytes())
(OUT/f'{name}.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
checks=[]
for path in sorted(OUT.iterdir()):
    if path.is_file() and path.name != 'SHA256SUMS' and any(path.name.endswith(suffix) for suffix in ['.tgz','.zip','.wasm','.manifest.json']):
        checks.append(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}')
(OUT/'SHA256SUMS').write_text('\n'.join(checks)+'\n')
print('\n'.join(str(OUT/f'{name}.{suffix}') for suffix in ['tgz','zip','manifest.json']))
