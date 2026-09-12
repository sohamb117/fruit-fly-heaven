"""Build read-only circuit probes from the exact graph used by the habitat."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = [ROOT/'data/prepared'/name for name in ['ids.bin', 'indptr.bin', 'targets.bin', 'weights.bin']]
    paths += [ROOT/'data/raw'/name for name in ['classification.csv.gz', 'consolidated_cell_types.csv.gz']]
    ids = np.fromfile(paths[0], dtype='<i8')
    rows = np.fromfile(paths[1], dtype='<u4')
    targets = np.fromfile(paths[2], dtype='<u4')
    weights = np.fromfile(paths[3], dtype='<f4')
    ann = pd.read_csv(paths[4]).set_index('root_id').reindex(ids).fillna('')
    types = pd.read_csv(paths[5]).set_index('root_id').reindex(ids).fillna('').primary_type
    motor = json.loads((ROOT/'web/motor-outputs.json').read_text())
    sensory = json.loads((ROOT/'web/sensory-inputs.json').read_text())
    groups = [dict(key='retina', label='Photoreceptors · R1–6', indices=[r['index'] for r in sensory['vision']['receptors']])]
    for key, label, names in [
        ('lamina', 'First relay · L1 / L2', ['L1', 'L2']),
        ('on_relay', 'ON pathway relay · Mi1 / Tm3', ['Mi1', 'Tm3']),
        ('off_relay', 'OFF pathway relay · Tm1 / Tm2', ['Tm1', 'Tm2']),
        ('motion', 'Motion pathways · T4 / T5', [f'T{n}{s}' for n in [4, 5] for s in 'abcd']),
        ('looming', 'Looming pathways · LC4 / LPLC2', ['LC4', 'LPLC2']),
    ]:
        groups.append(dict(key=key, label=label, indices=np.flatnonzero(types.isin(names)).tolist()))
    channels = [c for c in motor['channels'] if c['key'] in ['forward', 'wing_left', 'wing_right']]
    monitored = [i for c in channels for i in c['indices']]
    mask = np.isin(targets, monitored)
    edges = np.flatnonzero(mask)
    sources = np.searchsorted(rows, edges, side='right') - 1
    incoming = {str(i): [] for i in monitored}
    for e, src in zip(edges, sources):
        incoming[str(int(targets[e]))].append([int(src), float(weights[e])])
    cells = {str(i): dict(index=int(i), root_id=str(ids[i]), type=str(types.iloc[i]), side=str(ann.side.iloc[i]))
             for i in sorted(set(monitored) | set(map(int, sources)))}
    retinal_rows = np.repeat(np.isin(np.arange(len(ids)), groups[0]['indices']), np.diff(rows))
    retinal_lamina = retinal_rows & np.isin(targets, groups[1]['indices'])
    positive, negative = weights[retinal_lamina & (weights > 0)], weights[retinal_lamina & (weights < 0)]
    sign_audit = dict(pathway='Mapped R1–6 → L1/L2', positive_edges=len(positive), negative_edges=len(negative),
                      positive_weight=float(positive.sum()), negative_weight=float(negative.sum()),
                      note='Signs preserved from the upstream computational model, not inferred again from physiology. Photoreceptor histaminergic inhibition and graded signaling need separate validation.')
    result = dict(schema_version=1, neuron_count=len(ids), groups=groups,
                  channels=[dict(key=c['key'], label=c['label'], indices=c['indices']) for c in channels],
                  incoming=incoming, cells=cells, visual_sign_audit=sign_audit,
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  note='Read-only diagnostic annotations. Incoming-weight × presynaptic-rate is a ranking proxy, not delivered current or a causal attribution.')
    (ROOT/'web/circuit-probe.json').write_text(json.dumps(result, separators=(',', ':'))+'\n')
    print(json.dumps(dict(groups={g['key']:len(g['indices']) for g in groups}, motor_cells=len(monitored), incoming_edges=len(edges))))


if __name__ == '__main__':
    main()
