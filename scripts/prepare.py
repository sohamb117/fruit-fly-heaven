"""Preserve every connection in the source v783 computational model.

The sparse matrix is shared by all brains. Only the dynamic state is replicated.
No connection-strength threshold, graph sampling, or generated edges are used.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'references/fly-brain'
OUT = ROOT / 'data/prepared'
OUT.mkdir(parents=True, exist_ok=True)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    comp = SOURCE / 'data/2025_Completeness_783.csv'
    conn = SOURCE / 'data/2025_Connectivity_783.parquet'
    ids = pd.read_csv(comp, index_col=0).index.to_numpy(dtype=np.int64)
    df = pd.read_parquet(conn)
    n = len(ids)
    pre = df['Presynaptic_Index'].to_numpy()
    post = df['Postsynaptic_Index'].to_numpy()
    assert np.array_equal(ids[pre], df['Presynaptic_ID'])
    assert np.array_equal(ids[post], df['Postsynaptic_ID'])
    order = np.argsort(pre, kind='stable')
    ptr = np.zeros(n + 1, dtype='<u4')
    ptr[1:] = np.cumsum(np.bincount(pre, minlength=n))
    ptr.tofile(OUT / 'indptr.bin')
    post[order].astype('<u4').tofile(OUT / 'targets.bin')
    df['Excitatory x Connectivity'].to_numpy()[order].astype('<f4').tofile(OUT / 'weights.bin')
    ids.astype('<i8').tofile(OUT / 'ids.bin')
    ann = pd.read_csv(ROOT / 'data/raw/classification.csv.gz').set_index('root_id').reindex(ids).fillna('')
    types = pd.read_csv(ROOT / 'data/raw/consolidated_cell_types.csv.gz').set_index('root_id').reindex(ids).fillna('')
    typ = types.primary_type.to_numpy()
    side = ann.side.to_numpy()
    cls = ann['class'].to_numpy()
    sub = ann.sub_class.to_numpy()
    # DM1/VA2 food-odor channels: Semmelhack & Wang (2009), PMC2702439.
    # This selects sensory stimulation sites, never neurons to omit from the graph.
    food_odor = np.isin(typ, ['ORN_DM1', 'ORN_VA2'])
    masks = {
        'odor_left': food_odor & (side == 'left'),
        'odor_right': food_odor & (side == 'right'),
        'sweet': sub == 'sugar/water',
        'walk': typ == 'DNp09',
        'steer_left': np.isin(typ, ['DNa01', 'DNa02']) & (side == 'left'),
        'steer_right': np.isin(typ, ['DNa01', 'DNa02']) & (side == 'right'),
        'feed': np.isin(sub, ['proboscis_motor_neuron', 'haustellum_motor_neuron', 'ingestion_motor_neuron']),
        'antenna': sub == 'antennal_motor_neuron',
    }
    groups = {name: np.flatnonzero(mask).tolist() for name, mask in masks.items()}
    assert all(groups.values()), 'An I/O population is missing from the measured annotations'
    (OUT / 'groups.json').write_text(json.dumps(groups))
    with (OUT / 'groups.bin').open('wb') as f:
        for group in groups.values():
            np.array([len(group), *group], dtype='<u4').tofile(f)
    selected = sorted(set(i for name in ['walk', 'steer_left', 'steer_right', 'feed', 'antenna'] for i in groups[name]))
    annotations = [{"index": i, "flywire_id": str(ids[i]), "type": str(typ[i]),
                    "class": str(cls[i]), "sub_class": str(sub[i]), "side": str(side[i])} for i in selected]
    meta = {
        'dataset': 'FlyWire FAFB v783 / Eon whole-brain model',
        'neurons_per_brain': n,
        'connection_rows': len(df),
        'synapses_represented': int(df.Connectivity.sum()),
        'connection_rows_removed': 0,
        'classification_neurons': len(pd.read_csv(ROOT / 'data/raw/classification.csv.gz')),
        'source_model_neuron_difference': 139255 - n,
        'groups': {k: len(v) for k, v in groups.items()},
        'motor_annotations': annotations,
        'sources': [
            'https://github.com/eonsystemspbc/fly-brain',
            'https://www.nature.com/articles/s41586-024-07558-y',
            'https://www.nature.com/articles/s41586-024-07763-9',
            'https://codex.flywire.ai',
            'https://pmc.ncbi.nlm.nih.gov/articles/PMC2702439/',
        ],
        'upstream_commit': subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [comp, conn, ROOT/'data/raw/classification.csv.gz', ROOT/'data/raw/consolidated_cell_types.csv.gz']},
        'prepared_sha256': {p.name: sha(p) for p in sorted(OUT.glob('*.bin'))},
        'parameters': {'dt_ms': .1, 'membrane_tau_ms': 20, 'synapse_tau_ms': 5,
                       'rest_mv': -52, 'threshold_mv': -45, 'reset_mv': -52,
                       'refractory_ms': 2.2, 'synaptic_delay_ms': 1.8,
                       'mv_per_synapse': .275},
        'limits': [
            'The upstream computational model contains 138,639 of the 139,255 classified brain neurons; no additional neurons or edges are removed here.',
            'Connectivity and neuron identities are measured; neurotransmitter signs are predictions supplied by the upstream model.',
            'LIF electrical dynamics and sensory rates are assumptions. This is not a validated emulation of a living fly.',
            'Odor stimulates annotated left/right ORN_DM1 and ORN_VA2 food-odor channels. Rates and spatial odor fields are assumed; banana/apple chemistry is not measured.',
            'The ventral nerve cord is absent. Walking and steering use a supplied kinematic decoder of descending-neuron spikes; leg cycles are visual animation.',
            'Food is inexhaustible. Metabolism, development, reproduction, learning, and subjective experience are not modeled.',
        ],
    }
    (OUT / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    (ROOT / 'reports/data-provenance.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps({k: meta[k] for k in ['neurons_per_brain', 'connection_rows', 'synapses_represented', 'groups']}, indent=2))

if __name__ == '__main__':
    main()
