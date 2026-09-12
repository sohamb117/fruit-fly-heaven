"""Map rendered eyes and body feedback into the existing v783 brain.

This creates an environment adapter, never new neurons or synaptic connections.
R1-6 columns follow the authors' weighted postsynaptic-column assignment method.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLUMN_URL = 'https://storage.googleapis.com/flywire-data/codex/data/fafb/783/column_assignment.csv.gz'
COLUMN_METHOD = 'https://github.com/hsseung/OpticLobe.jl/blob/3352e97c37b0f96ab08f27b70b4420f3d4ac2726/src/columnassignment.jl'

def main():
    files = ['data/prepared/ids.bin', 'data/prepared/indptr.bin', 'data/prepared/targets.bin',
             'data/prepared/weights.bin', 'data/raw/classification.csv.gz',
             'data/raw/consolidated_cell_types.csv.gz', 'data/raw/column_assignment.csv.gz']
    ids = np.fromfile(ROOT/files[0], '<i8')
    ptr = np.fromfile(ROOT/files[1], '<u4')
    targets = np.memmap(ROOT/files[2], '<u4', 'r')
    weights = np.memmap(ROOT/files[3], '<f4', 'r')
    ann = pd.read_csv(ROOT/files[4]).set_index('root_id').reindex(ids).fillna('')
    typ = pd.read_csv(ROOT/files[5]).set_index('root_id').reindex(ids).fillna('').primary_type
    columns = pd.read_csv(ROOT/files[6]).set_index('root_id')
    coordinates = columns.reindex(ids)[['p', 'q']].to_numpy()
    hemispheres = columns.reindex(ids).hemisphere.fillna('').to_numpy()
    retina, unmapped = [], []
    # x = (q-p)/2 points posteriorly; y = p+q points dorsally. Spatial columns
    # are measured; fitting this lattice to a pinhole camera is an approximation.
    bounds = {}
    for side in ['left', 'right']:
        c = columns[columns.hemisphere.eq(side)]
        bounds[side] = [float(c.x.min()), float(c.x.max()), float(c.y.min()), float(c.y.max())]
    for i in np.flatnonzero(typ.eq('R1-6')):
        side = ann.side.iloc[i]
        t = targets[ptr[i]:ptr[i+1]]
        w = np.abs(weights[ptr[i]:ptr[i+1]])
        valid = np.isfinite(coordinates[t]).all(axis=1) & (hemispheres[t] == side) & (w > 0)
        if side not in bounds or not valid.any():
            unmapped.append(str(ids[i])); continue
        scores = {}
        for pq, strength in zip(coordinates[t[valid]], w[valid]):
            key = tuple(map(int, pq)); scores[key] = scores.get(key, 0) + float(strength)
        p, q = max(sorted(scores), key=scores.get)
        x, y = (q-p)/2, p+q
        x0, x1, y0, y1 = bounds[side]
        u = (x-x0)/(x1-x0)
        # Camera image right points posteriorly in the right eye, anteriorly in
        # the left eye. Mirror the latter; never mirror the actual connectome.
        if side == 'left': u = 1-u
        retina.append(dict(index=int(i), root_id=str(ids[i]), side=side, p=p, q=q,
                           u=round(float(u), 6), v=round(float(1-(y-y0)/(y1-y0)), 6)))
    assert retina and all(0 <= c['u'] <= 1 and 0 <= c['v'] <= 1 for c in retina)
    channels = []
    for side in ['left', 'right']:
        for key, label, mask, evidence, assumption in [
            ('self_motion', 'Body posture / motion', ann.sub_class.eq('AN_AVLP'),
             'https://www.nature.com/articles/s41593-023-01281-z',
             'Population-level boundary proxy: support, joint angles and joint/body velocity. These individual ANs do not have validated tuning to this feature mixture.'),
            ('antenna', 'Antennal deflection / airflow', ann.sub_class.eq('wind_gravity') & ann['class'].eq('mechanosensory'),
             'https://www.nature.com/articles/s41586-024-07686-5',
             'JO wind/gravity population receives a bounded proxy for antenna deflection, tilt and air-relative movement; tuning/gains are assumed.'),
            ('touch', 'Head contact', typ.eq('BM_Ant'),
             'https://www.nature.com/articles/s41586-024-07686-5',
             'Annotated antennal bristles receive nearby head/antenna surface-contact drive; exact receptive fields are not modeled.'),
            ('vibration', 'Footfall / impact', typ.eq('SA_DLV'),
             'https://www.nature.com/articles/s41586-025-08925-z',
             'Sensory-ascending boundary proxy for footfall and impact vibration. No claim of individual leg-joint receptor tuning.'),
        ]:
            inds = np.flatnonzero(mask & ann.side.eq(side)).tolist()
            assert inds, key
            channels.append(dict(key=f'{key}_{side}', label=f'{label} · {side}', indices=inds,
                cells=[dict(index=i, root_id=str(ids[i]), type=str(typ.iloc[i]), side=side) for i in inds],
                source=evidence, encoding=assumption))
    result = dict(schema_version=1, neuron_count=len(ids),
        source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},
        sources=dict(columns=COLUMN_URL, column_method=COLUMN_METHOD,
                     annotations='https://www.nature.com/articles/s41586-024-07686-5'),
        vision=dict(width=32, height=16, eye_yaw_degrees=60, vertical_fov_degrees=120,
                    frame_interval_body_seconds=.05, receptors=retina, unmapped_root_ids=unmapped,
                    mapping='R1-6 assigned to the same-side postsynaptic (p,q) column with greatest absolute connection weight; p/q lattice fitted to each camera image.',
                    limitations=['Two low-resolution pinhole cameras; this manifest maps their luminance plane. The separate color-inputs.json manifest maps RGB-derived R7/R8 input.',
                                 'Light and adaptation are encoded as Poisson drive in the existing LIF engine; real photoreceptors use graded signals.',
                                 'Unmapped R1-6 photoreceptors stay in the graph but receive no added luminance drive. Color receptor mapping and assumptions are recorded separately.']),
        channels=channels,
        scope='Brain only; body-sense rates approximate the missing peripheral/VNC computation at existing input neurons. Wiring is unchanged.')
    (ROOT/'web/sensory-inputs.json').write_text(json.dumps(result, separators=(',', ':'))+'\n')
    print(json.dumps(dict(visual_receptors=len(retina), unmapped=len(unmapped), body_channels={c['key']:len(c['indices']) for c in channels}), indent=2))

if __name__ == '__main__': main()
