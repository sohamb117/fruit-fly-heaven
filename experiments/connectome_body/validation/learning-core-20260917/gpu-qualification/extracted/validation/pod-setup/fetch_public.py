import concurrent.futures, hashlib, json, os, pathlib, urllib.request, time
root = pathlib.Path.cwd()
sources = json.loads((root / 'configs/sources.json').read_text())
def fetch(job):
    label, name, spec = job
    dest = root / 'data/raw' / label / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + '.download')
    start = time.monotonic()
    with urllib.request.urlopen(spec['url'], timeout=120) as response, tmp.open('wb') as out:
        while chunk := response.read(4*1024*1024):
            out.write(chunk)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if digest != spec['sha256']:
        raise ValueError(f'Checksum mismatch: {label}/{name}')
    os.replace(tmp, dest)
    print(json.dumps({'file': str(dest.relative_to(root)), 'bytes': dest.stat().st_size, 'seconds': time.monotonic()-start, 'sha256': digest}), flush=True)
jobs = [(label, name, spec) for label in ('banc','malecns') for name, spec in sources['datasets'][label]['files'].items()]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(fetch, jobs))
print('PUBLIC_DOWNLOADS_COMPLETE', flush=True)
