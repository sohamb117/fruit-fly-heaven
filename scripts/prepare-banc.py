"""Prepare the pinned BANC v888/v3 graph, without a connection-count cutoff.

uv run --with numpy --with pyarrow scripts/prepare-banc.py --download
Root IDs stay strings or uint64: never pass them through floating point.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def profile(row, config):
    name = 'default'
    for rule in config['rules']:
        if row.get(rule['field']) == rule['equals']:
            name = rule['profile']
    values = list(config['profiles'][name])
    for key, value in config['cell_overrides'].get(row['banc_888_id'], {}).items():
        values[config['parameter_order'].index(key)] = value
    if len(values) != 16 or not np.isfinite(values).all() or min(values[0], values[1], values[6], values[10]) <= 0:
        raise ValueError(f'Invalid physiology profile for {row["banc_888_id"]}')
    return name, values


def muscle_target(row):
    """Exact annotation-to-joint rules, never DN names or connectivity guesses."""
    if row.get('super_class') != 'motor' or row.get('side') not in ('left', 'right'):
        return None
    part, fn, target, side = (row.get(k) or '' for k in ('body_part_effector', 'cell_function_detailed', 'peripheral_target_type', 'side'))
    segment = {'front_leg': 'T1', 'middle_leg': 'T2', 'hind_leg': 'T3'}.get(part)
    rules = {'move_coxa_anterior': ('coxa', 1), 'move_coxa_posterior': ('coxa', -1),
             'flex_coxa_trochanter_joint': ('femur', 1), 'extend_coxa_trochanter_joint': ('femur', -1),
             'flex_femur_tibia_joint': ('tibia', 1), 'extend_femur_tibia_joint': ('tibia', -1),
             'flex_tibia_tarsus_joint': ('tarsus', 1), 'extend_tibia_tarsus_joint': ('tarsus', -1)}
    if segment and fn in rules:
        joint, sign = rules[fn]
        return f'{joint}_{segment}_{side}', sign, 'leg'
    if segment and fn == 'pull_long_tendon':
        # Pretarsal depressor / long tendon -> claw engagement. FlyBody's
        # adhesive contact is a reduced substitute for microscopic claw grip.
        return f'adhere_claw_{segment}_{side}', 1, 'claw_grip_assumption'
    if part == 'wing' and target in ('dorsal_longitudinal_muscle', 'dorsoventral_muscle'):
        return f'wing_power_{side}', 1, 'asynchronous_wing'
    if part == 'wing' and fn in ('tonic_wing_steering', 'phasic_wing_steering'):
        # Sign is deliberately unassigned until a muscle-specific calibration exists.
        return f'wing_steer_{side}', 1, 'wing_steering_assumption'
    if part == 'haltere' and target == 'haltere_dorsoventral_muscle':
        return f'haltere_power_{side}', 1, 'asynchronous_haltere'
    if part == 'haltere' and target in ('hi1_muscle', 'hi2_muscle', 'hiii2_muscle', 'haltere_basalare_muscle'):
        return f'haltere_steer_{side}', 1, 'haltere_steering_assumption'
    if part in ('proboscis', 'pharynx') and fn == 'pharyngeal_pumping':
        return 'pump', 1, 'pump'
    if part == 'proboscis' and fn in ('proboscis_positioning', 'labellar_spreading'):
        return 'proboscis', 1, 'proboscis_assumption'
    return None


def prepare(meta_path, edges_path, output, config, lock, report_path=None):
    metadata = feather.read_table(meta_path)
    rows = metadata.to_pylist()
    excluded = {r['banc_888_id'] for r in rows if r.get('super_class') in ('glia', 'trachea', 'not_a_neuron')}
    edges = feather.read_table(edges_path, columns=['pre', 'post', 'count'])
    keep = pc.and_(pc.invert(pc.is_in(edges['pre'], value_set=pa.array(sorted(excluded)))),
                   pc.invert(pc.is_in(edges['post'], value_set=pa.array(sorted(excluded)))))
    excluded_edges = len(edges) - int(pc.sum(keep).as_py())
    edges = edges.filter(keep)
    # Include all remaining annotations and all edge endpoints, including unknown cells.
    ids = sorted(set(r['banc_888_id'] for r in rows if r['banc_888_id'] not in excluded)
                 | set(pc.unique(edges['pre']).to_pylist()) | set(pc.unique(edges['post']).to_pylist()), key=int)
    id_array = pa.array(ids)
    by_id = {r['banc_888_id']: r for r in rows}
    rows = [by_id.get(i, {'banc_888_id': i}) for i in ids]
    n = len(ids)
    pre = pc.index_in(edges['pre'], value_set=id_array).to_numpy().astype('<u4')
    post = pc.index_in(edges['post'], value_set=id_array).to_numpy().astype('<u4')
    count = edges['count'].to_numpy()
    if np.any(count <= 0):
        raise ValueError('Nonpositive synapse count')
    order = np.argsort(post, kind='stable')
    pre, post, count = pre[order], post[order], count[order]
    assignments = [profile(r, config) for r in rows]
    params = np.array([p for _, p in assignments], dtype='<f4')
    tx = [r.get('neurotransmitter_verified') or r.get('neurotransmitter_predicted') for r in rows]
    receptor = np.array([config['transmitter_receptor'].get(t, 0) for t in tx], dtype='<u4')[pre]
    known = np.array([t in config['transmitter_receptor'] for t in tx])[pre]
    weight = (count * config['synaptic_weight_ns_ms_per_contact'] * known).astype('<f4')
    index = {rid: i for i, rid in enumerate(ids)}
    for override in config['receptor_overrides']:
        selection = (pre == index[override['pre']]) & (post == index[override['post']])
        if not selection.any() or override['receptor'] not in range(9) or not override.get('source'):
            raise ValueError('Invalid or unreferenced receptor override')
        receptor[selection] = override['receptor']
        weight[selection] = count[selection] * config['synaptic_weight_ns_ms_per_contact']
        known[selection] = True
    delay = round(config['delay_ms'] / config['dt_ms'])
    if not 1 <= delay < 32:
        raise ValueError('Transmission delay must be 1..31 time steps')
    packed = np.empty((len(pre), 4), dtype='<u4')
    packed[:, 0], packed[:, 1], packed[:, 2], packed[:, 3] = pre, weight.view('<u4'), receptor, delay
    chemical_offsets = np.r_[0, np.cumsum(np.bincount(post, minlength=n))].astype('<u4')
    gaps = []
    for g in config['gap_junctions']:
        if not g.get('source') or g['a'] == g['b'] or not np.isfinite(g['conductance_ns']) or g['conductance_ns'] <= 0:
            raise ValueError('Gap junction needs two distinct neurons, positive conductance and a source')
        a, b = index[g['a']], index[g['b']]
        gaps.extend([(b, a, g['conductance_ns']), (a, b, g['conductance_ns'])])
    gaps.sort()
    gap_packed = np.zeros((len(gaps), 4), dtype='<u4')
    for k, (_, source, conductance) in enumerate(gaps):
        gap_packed[k] = (source, np.float32(conductance).view(np.uint32), 0, 0)
    gap_offsets = (len(pre) + np.r_[0, np.cumsum(np.bincount([g[0] for g in gaps], minlength=n))]).astype('<u4')
    muscles = defaultdict(list)
    sensory, unassigned = [], []
    for i, row in enumerate(rows):
        target = muscle_target(row)
        if target:
            joint, sign, kind = target
            muscles[(joint, sign, kind, row['peripheral_target_type'])].append(i)
        elif row.get('super_class') == 'motor':
            unassigned.append(i)
        cls = row.get('cell_class') or ''
        kind = {'olfactory_receptor_neuron': 'odor', 'photoreceptor_neuron': 'vision',
                'taste_bristle_gustatory_neuron': 'taste', 'chordotonal_organ_neuron': 'proprioception',
                'campaniform_sensillum_neuron': 'load'}.get(cls)
        # Unassigned laterality is excluded from binocular/odor drive, reported below.
        if kind and (row.get('side') in ('left', 'right') or kind in ('taste', 'proprioception', 'load')):
            sensory.append({'index': i, 'kind': kind, 'side': row.get('side'), 'body_part': row.get('body_part_sensory'), 'cell_type': row.get('cell_type')})
    muscle_list = [{'joint': joint, 'sign': sign, 'kind': kind, 'target': target, 'indices': members,
                    'root_ids': [ids[i] for i in members], 'mapping_status': 'annotated target; reduced mechanics assumed'}
                   for (joint, sign, kind, target), members in muscles.items()]
    output.mkdir(parents=True, exist_ok=True)
    arrays = {'ids.bin': np.array(ids, dtype='<u8'), 'offsets.bin': np.r_[chemical_offsets, gap_offsets],
              'edges.bin': np.concatenate([packed, gap_packed]), 'params.bin': params}
    for filename, array in arrays.items():
        array.tofile(output / filename)
    inspect = [{'index': i, 'root_id': ids[i], **{k: rows[i].get(k) for k in ('cell_type', 'super_class', 'region', 'side', 'peripheral_target_type')}}
               for i in sorted({i for m in muscle_list for i in m['indices']} | set(unassigned))]
    (output / 'io.json').write_text(json.dumps({'muscles': muscle_list, 'sensory': sensory, 'motor_neurons': inspect, 'unassigned_motor_indices': unassigned}, separators=(',', ':')))
    summary = {'schema': 1, 'dataset': 'BANC', 'materialization': 888, 'synapse_table': 'v3', 'neuron_count': n,
               'chemical_edges': len(pre), 'synapse_count': int(count.sum()), 'electrical_directed_edges': len(gaps),
               'excluded_non_neuronal_annotations': len(excluded), 'excluded_non_neuronal_edges': excluded_edges,
               'unknown_transmitter_edges_zeroed': int((~known).sum()), 'count_threshold': None,
               'regions': dict(Counter(r.get('region') or 'unknown' for r in rows)),
               'physiology_profiles': dict(Counter(name for name, _ in assignments)),
               'motor_neurons': len(inspect), 'mapped_motor_neurons': sum(len(m['indices']) for m in muscle_list),
               'unassigned_motor_neurons': len(unassigned), 'muscle_groups': len(muscle_list), 'sensory_neurons': len(sensory),
               'dt_ms': config['dt_ms'], 'delay_slots': 32, 'receptors': config['receptors'],
               'synaptic_weight_ns_ms_per_contact': config['synaptic_weight_ns_ms_per_contact'],
               'physiology_status': config['status'], 'source_lock': lock, 'physiology_sha256': digest(ROOT / 'configs/banc-physiology.json'),
               'files': {name: {'bytes': (output/name).stat().st_size, 'sha256': digest(output/name)} for name in [*arrays, 'io.json']}}
    (output / 'manifest.json').write_text(json.dumps(summary, indent=2) + '\n')
    if report_path:
        report_path.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('files', 'source_lock', 'receptors')}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    lock = json.loads((ROOT / 'configs/banc-v888.lock.json').read_text())
    raw = ROOT / 'data/raw/banc888'
    raw.mkdir(parents=True, exist_ok=True)
    for source in lock['sources'].values():
        path = raw / source['file']
        if not path.exists() and args.download:
            temporary = path.with_suffix('.download')
            urllib.request.urlretrieve(source['url'], temporary)
            if digest(temporary) != source['sha256']:
                raise ValueError(f'Download checksum mismatch: {path.name}; upstream snapshot changed')
            temporary.replace(path)
        if not path.exists() or digest(path) != source['sha256']:
            raise ValueError(f'Missing or checksum-mismatched {path}; use --download for missing files')
    config = json.loads((ROOT / 'configs/banc-physiology.json').read_text())
    prepare(raw/lock['sources']['meta']['file'], raw/lock['sources']['edges']['file'], ROOT/'data/prepared/banc888', config, lock, ROOT/'reports/banc-provenance.json')


if __name__ == '__main__':
    main()
