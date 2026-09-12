"""Fetch measured FlyWire morphology and a coarse EM volume for the viewer.

Only display geometry is sampled. This never alters the simulation connectome.
Public source files stay separate from the reusable WASM release.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import shutil
import time
import urllib.request
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw'
OUT = ROOT / 'data/anatomy'
OUT.mkdir(parents=True, exist_ok=True)
CACHE = RAW / 'skeletons-783'
CACHE.mkdir(exist_ok=True)
COMMIT = '8587524c1748ce5ef2080822a2fc890fc03bf597'
ANNOTATIONS = f'https://raw.githubusercontent.com/flyconnectome/flywire_annotations/{COMMIT}/supplemental_files/Supplemental_file1_neuron_annotations.tsv'
SKELETONS = 'https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/flywire_skeletons_783'
SURFACE = 'https://storage.googleapis.com/flywire_neuropil_meshes/whole_neuropil/brain_mesh_v141.surf/mesh/1:0:0'
EM = 'https://storage.googleapis.com/flywire_em/aligned/v1'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def download(url, path):
    if path.exists():
        return path.read_bytes()
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                data = response.read()
            path.write_bytes(data)
            return data
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)

download(ANNOTATIONS, RAW / 'anatomy_annotations.tsv')
annotation_bytes = (RAW / 'anatomy_annotations.tsv').read_bytes()
annotation_blob = hashlib.sha1(b'blob ' + str(len(annotation_bytes)).encode() + b'\0' + annotation_bytes).hexdigest()
if annotation_blob != 'afea3e15a5671f5da0b9f7dd2e932d328c3b57a0':
    raise ValueError('Cached annotation file does not match the pinned Git commit')
ids = np.fromfile(ROOT / 'data/prepared/ids.bin', dtype='<i8')
table = pd.read_csv(RAW / 'anatomy_annotations.tsv', sep='\t', low_memory=False).set_index('root_id').reindex(ids)
raw = table[['pos_x', 'pos_y', 'pos_z']].to_numpy(float)
valid = np.isfinite(raw).all(axis=1)
positions = (raw[valid] * [0.004, 0.004, 0.040]).astype('<f4')
positions.tofile(OUT / 'positions.bin')
np.flatnonzero(valid).astype('<u4').tofile(OUT / 'neuron-indices.bin')
labels = table['cell_type'].fillna(table['hemibrain_type']).fillna(table['cell_class']).fillna(table['super_class']).fillna('Unclassified')
(OUT / 'neurons.json').write_text(json.dumps({'ids': [str(i) for i in ids], 'labels': labels.tolist()}, separators=(',', ':')))

# Preserve every edge of each displayed skeleton. Pick a reproducible background
# sample across cell classes, plus every neuron used by the habitat's I/O adapter.
groups = json.loads((ROOT / 'data/prepared/groups.json').read_text())
chosen = set(sum(groups.values(), []))
rng = np.random.default_rng(783)
for _, group in table.reset_index().groupby('cell_class', dropna=False):
    candidates = group.index.to_numpy()
    chosen.update(rng.choice(candidates, min(8, len(candidates)), replace=False).tolist())
chosen.update(rng.choice(np.flatnonzero(valid), 180, replace=False).tolist())
chosen = sorted(chosen)
print(f'Coordinates: {valid.sum():,}/{len(ids):,}; fetching {len(chosen)} exact neuron skeletons', flush=True)

def fetch_skeleton(index):
    path = CACHE / f'{ids[index]}.bin'
    data = download(f'{SKELETONS}/{ids[index]}', path)
    n, e = np.frombuffer(data, '<u4', count=2)
    expected = 8 + int(n) * 16 + int(e) * 8
    if len(data) != expected:
        raise ValueError(f'Unexpected skeleton size for {ids[index]}')
    vertices = np.frombuffer(data, '<f4', count=int(n)*3, offset=8).reshape(-1, 3).copy() / 1000
    edges = np.frombuffer(data, '<u4', count=int(e)*2, offset=8+int(n)*12).reshape(-1, 2).copy()
    if not np.isfinite(vertices).all() or (edges.size and edges.max() >= n):
        raise ValueError('Invalid skeleton geometry')
    return index, vertices, edges, sha(path)

items, failures = [], []
with ThreadPoolExecutor(max_workers=6) as pool:
    jobs = {pool.submit(fetch_skeleton, i): i for i in chosen}
    for done, future in enumerate(as_completed(jobs), 1):
        try:
            items.append(future.result())
        except Exception as error:
            failures.append({'index': jobs[future], 'id': str(ids[jobs[future]]), 'error': str(error)})
        if done % 100 == 0:
            print(f'Skeletons {done}/{len(chosen)}', flush=True)
items.sort(key=lambda item: item[0])
vertices, owners, edges, ranges = [], [], [], []
offset = edge_offset = 0
for index, v, e, digest in items:
    vertices.append(v); owners.append(np.full(len(v), index, '<u4')); edges.append(e + offset)
    ranges.append({'index': index, 'id': str(ids[index]), 'vertexOffset': offset, 'vertexCount': len(v), 'edgeOffset': edge_offset, 'edgeCount': len(e), 'sha256': digest})
    offset += len(v); edge_offset += len(e)
np.concatenate(vertices).astype('<f4').tofile(OUT / 'skeleton-positions.bin')
np.concatenate(owners).astype('<u4').tofile(OUT / 'skeleton-neurons.bin')
np.concatenate(edges).astype('<u4').tofile(OUT / 'skeleton-edges.bin')

surface = download(SURFACE, RAW / 'brain-surface.bin')
n = int(np.frombuffer(surface, '<u4', count=1)[0])
v = np.frombuffer(surface, '<f4', count=n*3, offset=4).reshape(-1, 3) / 1000
f = np.frombuffer(surface, '<u4', offset=4+n*12).reshape(-1, 3)
assert np.isfinite(v).all() and f.max() < n
v.astype('<f4').tofile(OUT / 'surface-positions.bin')
f.astype('<u4').tofile(OUT / 'surface-triangles.bin')

print('Fetching original EM at 2.048 × 2.048 × 1.280 µm per voxel', flush=True)
from cloudvolume import CloudVolume
volume = CloudVolume('precomputed://' + EM, mip=9, progress=False, cache=False, parallel=1, fill_missing=False, use_https=True)
info = volume.scales[9]
volume_path = OUT / 'em-volume.bin'
if not volume_path.exists():
    image = np.asarray(volume[:])[:, :, :, 0]
    assert list(image.shape) == info['size']
    image.ravel(order='F').astype('u1').tofile(volume_path)
assert volume_path.stat().st_size == int(np.prod(info['size']))

manifest = {
    'schema': 'fruit-fly-anatomy-v1', 'dataset': 'FlyWire FAFB v783', 'units': 'µm',
    'neuronCount': len(ids), 'mappedNeurons': int(valid.sum()), 'missingIndices': np.flatnonzero(~valid).tolist(),
    'positionMeaning': 'One annotated backbone anchor per neuron; not necessarily its soma.',
    'bounds': [v.min(axis=0).tolist(), v.max(axis=0).tolist()],
    'skeletonCount': len(items), 'skeletonVertices': offset, 'skeletonEdges': edge_offset,
    'skeletonRanges': ranges, 'skeletonFailures': failures,
    'surfaceVertices': n, 'surfaceTriangles': len(f),
    'volume': {'dimensions': info['size'], 'spacing': (np.array(info['resolution'])/1000).tolist(),
               'origin': (np.array(info.get('voxel_offset',[0,0,0]))*info['resolution']/1000).tolist(), 'order': 'x-fastest'},
    'sources': {'annotations': ANNOTATIONS, 'annotationCommit': COMMIT, 'annotationSha256': sha(RAW/'anatomy_annotations.tsv'),
                'skeletons': SKELETONS, 'surface': SURFACE, 'surfaceSha256': sha(RAW/'brain-surface.bin'), 'em': EM},
    'notes': ['Display skeletons are a sample; the full simulation connectivity is unchanged.',
              'Microscopy is static measured anatomy; activity colors and traces are simulated.',
              'The coarse EM overview cannot resolve individual synapses.',
              'Coordinates retain the original FlyWire imagery orientation.'],
    'files': {},
}
for path in sorted(OUT.glob('*.bin')):
    manifest['files'][path.name] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
manifest['files']['neurons.json'] = {'bytes': (OUT/'neurons.json').stat().st_size, 'sha256': sha(OUT/'neurons.json')}
(OUT/'metadata.json').write_text(json.dumps(manifest, indent=2)+'\n')
shutil.copy2(OUT/'metadata.json', ROOT/'reports/anatomy-provenance.json')
print(f'Prepared {len(items)} skeletons, {offset:,} vertices, {edge_offset:,} edges and {volume_path.stat().st_size:,} EM voxels.', flush=True)
