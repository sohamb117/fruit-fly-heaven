#!/usr/bin/env python3
"""Write only the new, immutable BANC v888 leg annotation catalog.

Run with .venv/bin/python scripts/prepare-banc-leg-proprioception.py.
This does not regenerate the graph, original IO, or existing console assets.
Family physiology is not an assertion of individual preferred angle/sign.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    'data/raw/banc888/meta.feather': '86ccf5df0c67419f8c5f43e93a7ed38d23a080e9f7fde26737290252f3780098',
    'data/prepared/banc888/ids.bin': 'dd942f6fd3bf27b155b4112c94b31a8348564264070933f7f916bc18e9dd1542',
    'data/prepared/banc888/io.json': '1b19cd8c91ff64d1fe2af8f3fdd09d98792ecfca229e1c102ecb08a96da1ce72',
}
PARTS = {'front_leg': 0, 'middle_leg': 1, 'hind_leg': 2}
FUNCTIONS = {'claw': 'position', 'hook': 'direction', 'club': 'vibro_tactile'}
RAW_FIELDS = ('super_class', 'cell_class', 'cell_sub_class', 'cell_type',
              'body_part_sensory', 'peripheral_target_type', 'cell_function',
              'cell_function_detailed', 'nerve')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def classify(row):
    part, cls = row['body_part_sensory'], row['cell_class']
    if cls == 'hair_plate_neuron':
        return None, 'hair_plate_joint_axis_unregistered'
    if cls == 'campaniform_sensillum_neuron':
        return None, 'strain_receptor_axis_unregistered'
    if part not in PARTS:
        return None, 'leg_joint_unregistered'
    if cls != 'chordotonal_organ_neuron':
        return None, 'receptor_class_unsupported'
    if row['peripheral_target_type'] != 'chordotonal_organ':
        return None, 'peripheral_target_unregistered'
    for family, function in FUNCTIONS.items():
        expected = f'{part}_{family}_chordotonal_organ_neuron'
        if row['cell_sub_class'] == expected:
            return ((family, None) if row['cell_function_detailed'] == function
                    else (None, 'family_function_conflict'))
    return None, 'receptor_subclass_unregistered'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'models/banc-leg-proprioception-v1.json')
    parser.add_argument('--check', action='store_true', help='Compare with existing artifact without writing.')
    args = parser.parse_args()
    for path, expected in EXPECTED.items():
        actual = sha_bytes((ROOT/path).read_bytes())
        if actual != expected:
            raise ValueError(f'Pinned source differs: {path}: {actual}')
    ids = np.fromfile(ROOT/'data/prepared/banc888/ids.bin', dtype='<u8')
    if len(ids) != 175401:
        raise ValueError('Unexpected prepared neuron count')
    rows = feather.read_table(ROOT/'data/raw/banc888/meta.feather',
                              columns=['banc_888_id', 'side', *RAW_FIELDS]).to_pylist()
    by_id = {str(r['banc_888_id']): r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError('Duplicate raw BANC root ID')
    sensory = json.loads((ROOT/'data/prepared/banc888/console/sensory-inputs.json').read_text())
    channels = []
    cells = []
    seen = set()
    for side, count in [('left', 539), ('right', 527)]:
        matches = [c for c in sensory['channels'] if c['key'] == f'self_motion_{side}']
        if len(matches) != 1 or len(matches[0]['indices']) != count:
            raise ValueError('Unexpected self-motion channel coverage')
        channel = matches[0]
        channels.append({'key': channel['key'], 'indices': channel['indices']})
        for index in channel['indices']:
            if index in seen:
                raise ValueError('Duplicate channel index')
            seen.add(index)
            root = str(ids[index])
            row = by_id[root]
            if (row['side'] != side or row['super_class'] not in ('sensory', 'sensory_ascending')
                    or 'leg' not in (row['body_part_sensory'] or '')):
                raise ValueError('Prepared channel and raw sensory annotation disagree')
            family, abstention = classify(row)
            part = row['body_part_sensory']
            leg = PARTS[part] + (3 if side == 'right' else 0) if part in PARTS else None
            cells.append({'index': int(index), 'root_id': root, 'side': side,
                          'channel': channel['key'], 'leg': leg,
                          **{k: row[k] for k in RAW_FIELDS},
                          'family': family, 'abstention': abstention,
                          'anatomical_polarity': None})
    counts = Counter(c['family'] or 'abstained' for c in cells)
    if dict(counts) != {'claw': 160, 'hook': 137, 'abstained': 434, 'club': 335}:
        raise ValueError(f'Annotation coverage changed: {dict(counts)}')
    catalog = {
        'schema': 1, 'kind': 'banc-leg-proprioception-annotations-v1',
        'dataset': 'BANC', 'materialization': 888, 'neuron_count': len(ids),
        'source_files': EXPECTED,
        'channel_identity_sha256': sha_bytes(canonical(channels).encode()),
        'evidence': {
            'raw_identity': 'Exact BANC v888 root, side, leg, class, subclass and detailed-function join; no sign inferred from SNpp name.',
            'family_physiology': ['https://doi.org/10.1016/j.neuron.2018.09.009', 'https://doi.org/10.7554/eLife.60299'],
            'joint_transfer': 'Femoral chordotonal claw/hook/club family physiology is mapped to the native femur-tibia hinge as a family-level cross-model assumption.',
            'polarity': 'No individual preferred angle, flexion/extension sign, or native-coordinate polarity is present in these annotations.',
            'club_scope': 'Bidirectional joint movement only; vibration is unmodeled. Collision times joint velocity is not used as vibration.',
        },
        'coverage': dict(counts),
        'abstention_counts': dict(sorted(Counter(c['abstention'] for c in cells if c['abstention']).items())),
        'channels': channels, 'cells': cells,
    }
    output = canonical(catalog) + '\n'
    if args.check:
        if args.output.read_text() != output:
            raise ValueError('Existing catalog differs from deterministic preparation')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(json.dumps({'output': str(args.output), 'checked': args.check, 'coverage': dict(counts),
                      'canonicalSha256': sha_bytes(canonical(catalog).encode()),
                      'fileSha256': sha_bytes(output.encode()), 'abstentions': catalog['abstention_counts']}, indent=2))


if __name__ == '__main__':
    main()
