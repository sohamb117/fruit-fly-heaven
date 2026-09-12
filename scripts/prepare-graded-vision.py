"""Export the authors' trained FlyVis model, without executing checkpoint code.

Preparation mirrors FlyVis's ordered type/offset parameter sharing and hull fill.
The browser's full FlyWire graph remains separate and unchanged.
"""
from pathlib import Path
from collections import OrderedDict
import hashlib
import io
import json
import pickle
import zipfile
import os
os.environ.setdefault('MPLCONFIGDIR', str(Path('/tmp/fruit-fly-mpl')))
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from matplotlib.path import Path as Polygon

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'references/motion-research/flyvis'
COMMIT = '92b3845cc426dd309a1a0e1b3890156c42e14021'
ARCHIVE_SHA = '71c78d4070556a536b13b23ee3139cd2788aa2a9d07d430a223b4edead281db1'

def checkpoint(path):
    # Only tensor reconstruction and OrderedDict are permitted; arbitrary pickle
    # globals, code, and external storage are rejected.
    with zipfile.ZipFile(path) as archive:
        prefix = next(n[:-8] for n in archive.namelist() if n.endswith('data.pkl'))
        def rebuild(storage, offset, shape, strides, *unused):
            return np.ndarray(shape, dtype=storage.dtype, buffer=storage,
                              offset=offset*storage.itemsize,
                              strides=tuple(s*storage.itemsize for s in strides)).copy()
        class Tensors(pickle.Unpickler):
            def find_class(self, module, name):
                allowed = {('collections', 'OrderedDict'): OrderedDict,
                           ('torch._utils', '_rebuild_tensor_v2'): rebuild,
                           ('torch', 'FloatStorage'): np.dtype('<f4'),
                           ('torch', 'LongStorage'): np.dtype('<i8')}
                if (module, name) not in allowed:
                    raise ValueError('Unsupported checkpoint global')
                return allowed[module, name]
            def persistent_load(self, pid):
                kind, dtype, key, location, count = pid
                assert kind == 'storage' and str(key).isdigit()
                value = np.frombuffer(archive.read(prefix+'data/'+key), dtype=dtype)
                assert value.size == count
                return value
        return Tensors(io.BytesIO(archive.read(prefix+'data.pkl'))).load()['network']

def filled_offsets(edge):
    offsets = edge['offsets']
    if len(offsets) < 3:
        return offsets
    pts = np.array([v[0] for v in offsets])
    hull = ConvexHull((1+1e-6)*pts, False, 'QJ')
    grid = np.concatenate(np.dstack(np.mgrid[pts[:,0].min():pts[:,0].max()+1,
                                            pts[:,1].min():pts[:,1].max()+1]))
    inside = grid[Polygon(pts[hull.vertices]).contains_points(grid)]
    known = set(map(tuple, pts))
    return offsets+[[[int(u),int(v)],1] for u,v in inside if (u,v) not in known]

def main():
    archive=SOURCE/'results_pretrained_models.zip'
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == ARCHIVE_SHA
    spec_path=SOURCE/'flyvis/connectome/fib25-fib19_v2.2.json'
    spec=json.loads(spec_path.read_text())
    ckpt=SOURCE/'pretrained/results/flow/0000/000/chkpts/chkpt_00000'
    params=checkpoint(ckpt)
    positions={n['name']:{(u,v) for u in range(-15,16) for v in range(max(-15,-15-u),min(15,15-u)+1)
                         if u % n['pattern'][1][0] == 0 and v % n['pattern'][1][1] == 0} for n in spec['nodes']}
    offsets=OrderedDict()
    for e in spec['edges']:
        for (u,v),count in filled_offsets(e):
            if not any((x+u,y+v) in positions[e['tar']] for x,y in positions[e['src']]):
                continue
            offsets.setdefault((e['src'],e['tar'],u,v),[]).append(count)
    pairs=list(dict.fromkeys((k[0],k[1]) for k in offsets))
    assert len(params['nodes_bias']) == len(spec['nodes'])
    assert len(params['edges_sign']) == len(pairs) == len(params['edges_syn_strength'])
    assert len(params['edges_syn_count']) == len(offsets)
    # Prove the checkpoint parameter order matches the published graph, including
    # filled offsets; a wrong ordering can look active but destroys visual tuning.
    np.testing.assert_allclose(np.exp(params['edges_syn_count']),[np.mean(v) for v in offsets.values()],rtol=2e-6)
    pair_index={p:i for i,p in enumerate(pairs)}
    sign=np.array([e['alpha'] for e in spec['edges'] if (e['src'],e['tar']) in pairs])
    np.testing.assert_array_equal(params['edges_sign'],sign)
    kernels=[]
    for k,(src,tar,u,v) in enumerate(offsets):
        j=pair_index[src,tar]
        weight=params['edges_sign'][j]*np.exp(params['edges_syn_count'][k])*params['edges_syn_strength'][j]
        kernels.append([src,tar,u,v,float(weight)])
    nodes=[dict(type=n['name'],stride=n['pattern'][1],bias=float(params['nodes_bias'][i]),
                tauSeconds=float(params['nodes_time_const'][i])) for i,n in enumerate(spec['nodes'])]
    result=dict(schema_version=1,model='FlyVis flow/0000/000',extent=15,stepSeconds=.02,
                nodes=nodes,kernels=kernels,inputs=spec['input_units'],outputs=spec['output_units'],
                source='https://github.com/TuragaLab/flyvis/tree/'+COMMIT,
                paper='https://www.nature.com/articles/s41586-024-07939-3',
                archive_sha256=ARCHIVE_SHA,checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
                graph_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(),
                notes='Published trained graded visual model, distinct from FlyWire v783. Dimensionless activity; not Hz or mV. Retinotopic transfer to FlyWire and camera optics are host assumptions.')
    (ROOT/'web/visual-model.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    (ROOT/'web/visual-model.LICENSE').write_text((SOURCE/'license').read_text())
    ids=np.fromfile(ROOT/'data/prepared/ids.bin','<i8')
    types=pd.read_csv(ROOT/'data/raw/consolidated_cell_types.csv.gz').set_index('root_id').reindex(ids).primary_type.fillna('')
    columns=pd.read_csv(ROOT/'data/raw/column_assignment.csv.gz').set_index('root_id')
    target_types=['T2','T3','T4a','T4b','T4c','T4d','T5a','T5b','T5c','T5d']
    mapped=[]
    for side in ['left','right']:
        eye=columns[columns.hemisphere.eq(side)]
        x0,x1,y0,y1=eye.x.min(),eye.x.max(),eye.y.min(),eye.y.max()
        for i in np.flatnonzero(types.isin(target_types)):
            if ids[i] not in eye.index: continue
            c=eye.loc[ids[i]];u=float((c.x-x0)/(x1-x0));v=float(1-(c.y-y0)/(y1-y0))
            mapped.append(dict(index=int(i),root_id=str(ids[i]),type=str(types.iloc[i]),side=side,u=1-u if side=='left' else u,v=v))
    mapping=dict(schema_version=1,neuron_count=len(ids),cells=mapped,rate_scale_hz=20,max_rate_hz=60,
                 ids_sha256=hashlib.sha256((ROOT/'data/prepared/ids.bin').read_bytes()).hexdigest(),
                 model_sha256=hashlib.sha256((ROOT/'web/visual-model.json').read_bytes()).hexdigest(),
                 method='Match published cell type and nearest normalized retinotopic position. Transfer nonnegative graded activity to Poisson drive in existing FlyWire visual neurons. No motor neurons are directly stimulated.',
                 limitation='Different specimens and visual field geometries; activity-to-Hz gain and optical correspondence are host assumptions, not a physiological calibration.')
    (ROOT/'web/visual-projections.json').write_text(json.dumps(mapping,separators=(',',':'))+'\n')
    print(json.dumps(dict(types=len(nodes),kernels=len(kernels),bias_range=[float(params['nodes_bias'].min()),float(params['nodes_bias'].max())],tau_range=[float(params['nodes_time_const'].min()),float(params['nodes_time_const'].max())])))

if __name__ == '__main__': main()
