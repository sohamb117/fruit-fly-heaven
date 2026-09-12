"""Assign RGB-derived catches only to existing, annotated R7/R8 neurons.

Positions use the R1–6 adapter's weighted postsynaptic-column method. Opsin
subtypes are absent from this dataset: a documented deterministic 30:70 mosaic
pairs Rh3/Rh5 or Rh4/Rh6 at each column. It is not a measured subtype assignment.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
FILES=['data/prepared/ids.bin','data/prepared/indptr.bin','data/prepared/targets.bin','data/prepared/weights.bin','data/raw/classification.csv.gz','data/raw/consolidated_cell_types.csv.gz','data/raw/column_assignment.csv.gz']
ids=np.fromfile(ROOT/FILES[0],'<i8'); ptr=np.fromfile(ROOT/FILES[1],'<u4')
targets=np.memmap(ROOT/FILES[2],'<u4','r'); weights=np.memmap(ROOT/FILES[3],'<f4','r')
ann=pd.read_csv(ROOT/FILES[4]).set_index('root_id').reindex(ids).fillna('')
types=pd.read_csv(ROOT/FILES[5]).set_index('root_id').reindex(ids).fillna('').primary_type
columns=pd.read_csv(ROOT/FILES[6]).set_index('root_id'); positions=columns.reindex(ids)[['p','q']].to_numpy(); sides=columns.reindex(ids).hemisphere.fillna('').to_numpy()
bounds={side:[float(c.x.min()),float(c.x.max()),float(c.y.min()),float(c.y.max())] for side in ['left','right'] for c in [columns[columns.hemisphere.eq(side)]]}
cells=[];unmapped=[]
for i in np.flatnonzero(types.isin(['R7','R8'])):
    side=ann.side.iloc[i]; t=targets[ptr[i]:ptr[i+1]]; w=np.abs(weights[ptr[i]:ptr[i+1]])
    valid=np.isfinite(positions[t]).all(axis=1)&(sides[t]==side)&(w>0)
    if side not in bounds or not valid.any():unmapped.append(str(ids[i]));continue
    scores={}
    for pq,strength in zip(positions[t[valid]],w[valid]):
        key=tuple(map(int,pq));scores[key]=scores.get(key,0)+float(strength)
    p,q=max(sorted(scores),key=scores.get);x,y=(q-p)/2,p+q;x0,x1,y0,y1=bounds[side]
    u=(x-x0)/(x1-x0);u=1-u if side=='left' else u
    # Stable independent of iteration order; R7 and R8 in a column share subtype.
    value=int.from_bytes(hashlib.sha256(f'color-mosaic-783:{side}:{p}:{q}'.encode()).digest()[:4],'little')/2**32
    pale=value<.3; typ=str(types.iloc[i]);opsin=('Rh3' if pale else 'Rh4') if typ=='R7' else ('Rh5' if pale else 'Rh6')
    cells.append(dict(index=int(i),root_id=str(ids[i]),type=typ,side=side,p=p,q=q,u=round(float(u),6),v=round(float(1-(y-y0)/(y1-y0)),6),opsin=opsin,subtype_assigned=True))
model=json.loads((ROOT/'packages/fly-color-wasm/model/fly-color-model.json').read_text())
result=dict(schema_version=1,neuron_count=len(ids),ids_sha256=hashlib.sha256((ROOT/FILES[0]).read_bytes()).hexdigest(),lut_sha256=model['lut_sha256'],
    cells=cells,unmapped_root_ids=unmapped,base_rate_hz=2,rate_scale_hz=60,max_rate_hz=80,
    source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in FILES},
    column_method='https://github.com/hsseung/OpticLobe.jl/blob/3352e97c37b0f96ab08f27b70b4420f3d4ac2726/src/columnassignment.jl',
    subtype_source='https://elifesciences.org/articles/71858',
    assumptions=['Retinotopy estimated from strongest same-side postsynaptic column, fitted to each pinhole camera.',
      'Dataset labels R7/R8 only. Opsin identities use an assumed 30:70 pale/yellow mosaic, paired by column. Dorsal-rim specializations and coexpression are not modeled.',
      'Relative captures become bounded Poisson drive at existing photoreceptors. Gains are assumed, with no calibrated adaptation or phototransduction. No motor cells are directly driven.'])
assert cells and len({c['index'] for c in cells})==len(cells)
(ROOT/'web/color-inputs.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
print(json.dumps(dict(mapped=len(cells),unmapped=len(unmapped),opsins={rh:sum(c['opsin']==rh for c in cells) for rh in model['channels']})))
