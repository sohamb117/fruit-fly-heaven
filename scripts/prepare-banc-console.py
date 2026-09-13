"""Map BANC identities into the existing console's sensory and inspection API.

This never changes or samples the simulated connectivity. Spatial input tuning,
opsin mosaics, and the neuromuscular boundary are declared model assumptions.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pyarrow.feather as feather

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/prepared/banc888'
OUT = DATA / 'console'
SOURCE = 'https://doi.org/10.1038/s41586-026-10735-w'
CONTACT_CHEMOSENSORY_FUNCTIONS = frozenset(('sugar', 'low_salt', 'contact_pheromone'))
EXTERNAL_TASTE_ORGANS = frozenset(('labellum', 'wing_margin', 'front_leg', 'middle_leg', 'hind_leg'))
AUDITORY_FREQUENCY_FUNCTIONS = frozenset(('auditory_low_frequency', 'auditory_high_frequency'))
JO_MODALITY_SOURCE = 'https://doi.org/10.3389/fphys.2014.00179'
SUGAR_IDENTITY_EVIDENCE = json.loads((ROOT/'models/banc-sensory-annotation-evidence.json').read_text())
SUGAR_IDENTITY_ROWS = {row['banc_root_id']: row for row in SUGAR_IDENTITY_EVIDENCE['rows']}


def sugar_identity_evidence(row):
    evidence = SUGAR_IDENTITY_ROWS.get(row.get('banc_888_id'))
    if evidence and row.get('fafb_match') != evidence['fafb_match']:
        raise ValueError('BANC taste evidence does not match annotated FAFB identity')
    return evidence


def sugar_modality_exclusion_reason(row):
    evidence = sugar_identity_evidence(row)
    return evidence['decision'] if evidence and not evidence['supports_external_sugar'] else None


def function_tokens(row):
    return {token.strip().lower() for token in (row.get('cell_function_detailed') or '').split(',')}


def is_external_sugar_sensor(row):
    # Broad classes omit taste-peg, BM_Taste, LgLG4 and SNch11 cells. Exact
    # function and external organ, rather than class or substring, define this
    # group. In particular, hemolymph_sugar is an internal-state sensor.
    return (row.get('super_class') == 'sensory' and 'sugar' in function_tokens(row)
            and row.get('body_part_sensory') in EXTERNAL_TASTE_ORGANS and not sugar_modality_exclusion_reason(row))


def auditory_transducer_exclusion_reason(row):
    if (row.get('body_part_sensory') == 'antenna'
            and row.get('cell_class') == 'chordotonal_organ_neuron'
            and AUDITORY_FREQUENCY_FUNCTIONS.intersection(function_tokens(row))):
        return ('Explicit auditory frequency function; whole-body tilt, speed and static antennal angle are not an acoustic or vibration transducer. '
                'Native antennal vibration and frequency tuning are unmodeled, so no added auditory drive is assigned. Neural dynamics and connectivity are retained.')
    return None


def has_contact_chemosensory_function(row):
    """Explicit v888 function labels take precedence over broad bristle class."""
    return bool(CONTACT_CHEMOSENSORY_FUNCTIONS.intersection(
        token.strip().lower() for token in (row.get('cell_function_detailed') or '').split(',')))


def bristle_transducer_exclusion_reason(row):
    """Abstain when explicit bristle function does not support collision drive."""
    if row.get('cell_class') != 'bristle_neuron':
        return None
    if has_contact_chemosensory_function(row):
        return 'Explicit chemosensory function; mechanical collision drive is unsupported. External sugar is routed separately by annotated organ and side; no pheromone or salt tuning is assigned.'
    if 'joint_angle' in {token.strip().lower() for token in (row.get('cell_function_detailed') or '').split(',')}:
        return 'Explicit joint_angle function; mechanical collision drive is unsupported. Joint identity and tuning are unassigned, so no substitute position drive is assigned.'
    return None


def body_transducer_exclusion_reason(row):
    return bristle_transducer_exclusion_reason(row) or auditory_transducer_exclusion_reason(row)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    ids = np.fromfile(DATA / 'ids.bin', '<u8')
    annotations = {r['banc_888_id']: r for r in feather.read_table(ROOT / 'data/raw/banc888/meta.feather').to_pylist()}
    assert sha(ROOT/'data/raw/banc888/meta.feather') == SUGAR_IDENTITY_EVIDENCE['sources']['data/raw/banc888/meta.feather']['sha256']
    rows = [annotations.get(str(i), {}) for i in ids]
    manifest = json.loads((DATA / 'manifest.json').read_text())
    params = np.fromfile(DATA / 'params.bin', '<f4').reshape(-1, 16)
    n = len(ids)
    base = {'schema_version': 1, 'neuron_count': n}
    provenance = {'data/prepared/' + name: item['sha256'] for name, item in manifest['files'].items()}
    base['source_sha256'] = provenance

    def select(predicate):
        return [i for i, r in enumerate(rows) if predicate(r)]

    def typ(r):
        return r.get('cell_type') or r.get('fafb_cell_type') or ''

    def annotation(i):
        r = rows[i]
        return {'index': i, 'root_id': str(ids[i]), 'type': typ(r) or r.get('cell_class') or 'Unclassified',
                'side': r.get('side') or 'unassigned', 'region': r.get('region'),
                'restMv': float(params[i, 2]), 'thresholdMv': float(params[i, 3]), 'graded': bool(params[i, 8])}

    groups = {
        'odor_left': select(lambda r: typ(r) in ('ORN_DM1', 'ORN_VA2') and r.get('side') == 'left'),
        'odor_right': select(lambda r: typ(r) in ('ORN_DM1', 'ORN_VA2') and r.get('side') == 'right'),
        'sweet': select(is_external_sugar_sensor),
        'walk': select(lambda r: typ(r) == 'DNp09'),
        'steer_left': select(lambda r: typ(r) in ('DNa01', 'DNa02') and r.get('side') == 'left'),
        'steer_right': select(lambda r: typ(r) in ('DNa01', 'DNa02') and r.get('side') == 'right'),
        'feed': select(lambda r: r.get('cell_class') in ('proboscis_motor_neuron', 'pharynx_motor_neuron')),
        'antenna': select(lambda r: r.get('cell_class') == 'antenna_motor_neuron'),
    }
    assert all(groups.values()), {k: len(v) for k, v in groups.items()}

    # Coordinates are representative points, not measured receptive fields.
    # Fit each cell-type/side in its own y-z plane. Do not recycle FlyWire indices.
    def spatial(types):
        cells, missing = [], []
        for cell_type in types:
            for side in ('left', 'right'):
                indices, positions = [], []
                for i, r in enumerate(rows):
                    if typ(r) != cell_type or r.get('side') != side:
                        continue
                    text = r.get('root_position_nm') or r.get('position')
                    if not text:
                        missing.append(str(ids[i]))
                        continue
                    p = np.fromstring(text.strip('[]()'), sep=',')
                    if len(p) != 3 or not np.isfinite(p).all():
                        missing.append(str(ids[i]))
                        continue
                    if not r.get('root_position_nm'):
                        p *= [4, 4, 45]
                    indices.append(i)
                    positions.append(p[1:])
                if not positions:
                    continue
                pos = np.array(positions)
                lo, hi = np.quantile(pos, [.01, .99], axis=0)
                uv = np.clip((pos - lo) / np.maximum(hi - lo, 1), 0, 1)
                for i, (u, v) in zip(indices, uv):
                    cells.append({'index': i, 'root_id': str(ids[i]), 'type': cell_type, 'side': side,
                                  'u': round(float(u if side == 'left' else 1-u), 6), 'v': round(float(v), 6)})
        missing += [str(ids[i]) for i, r in enumerate(rows) if typ(r) in types and r.get('side') not in ('left', 'right')]
        return cells, missing

    luminance, lum_missing = spatial(['L1', 'L2', 'L3'])
    projected, projection_missing = spatial(['T2', 'T3', 'T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d'])
    color, color_missing = spatial(['R7', 'R8'])
    for cell in color:
        column = round(cell['u']*31) + 32*round(cell['v']*15)
        pale = (column * 2654435761 % 100) < 30
        cell['opsin'] = ('Rh3' if pale else 'Rh4') if cell['type'] == 'R7' else ('Rh5' if pale else 'Rh6')
        cell['subtype_assigned'] = True
    channels = []
    for key in ('self_motion', 'antenna', 'touch', 'vibration'):
        for side in ('left', 'right'):
            def predicate(r):
                if r.get('super_class') not in ('sensory', 'sensory_ascending') or r.get('side') != side:
                    return False
                cls, part = r.get('cell_class') or '', r.get('body_part_sensory') or ''
                if key == 'self_motion':
                    return 'leg' in part and cls in ('chordotonal_organ_neuron', 'hair_plate_neuron', 'campaniform_sensillum_neuron')
                if key == 'antenna':
                    return part == 'antenna' and cls == 'chordotonal_organ_neuron' and not auditory_transducer_exclusion_reason(r)
                if key == 'touch':
                    return 'leg' in part and cls == 'bristle_neuron' and not bristle_transducer_exclusion_reason(r)
                return part in ('haltere', 'wing_base') and cls == 'campaniform_sensillum_neuron'
            selected = select(predicate)
            assert selected, (key, side)
            channels.append({'key': key+'_'+side, 'label': key.replace('_', ' ').title()+' '+side, 'indices': selected})
    visual_note = ('BANC has no annotated R1–R6 in this snapshot. Rendered luminance enters existing graded L1/L2/L3 cells. '
                   'Representative-point y/z coordinates estimate binocular retinotopy; this is not measured receptive-field calibration. '
                   'Cells without annotated laterality/coordinates are retained in the graph without added visual drive.')
    transducers = []
    legs = {'front_leg': 0, 'middle_leg': 1, 'hind_leg': 2}
    for channel in channels:
        for index in channel['indices']:
            row = rows[index]
            part, cls = row.get('body_part_sensory'), row.get('cell_class')
            if part not in legs and part not in ('haltere', 'wing_base'):
                continue
            function = row.get('cell_function_detailed') or ''
            kind = ('rotation' if part in ('haltere', 'wing_base') else
                    'load' if cls == 'campaniform_sensillum_neuron' else
                    'touch' if cls == 'bristle_neuron' else
                    'velocity' if function == 'direction' else
                    'vibration' if 'vibro' in function else 'position')
            transducers.append({'index': index, 'kind': kind,
                'leg': legs[part] + (3 if row['side'] == 'right' else 0) if part in legs else None,
                'side': row['side'], 'function': function,
                'organ': part, 'cell_type': typ(row),
                'annotation': row.get('cell_sub_class'),
                'tuning_status': 'Annotated organ, leg and modality; numerical tuning and direction preference remain priors.'})
    sensory = {**base, 'vision': {'width': 32, 'height': 16, 'frame_interval_body_seconds': .05, 'receptors': luminance, 'unmapped_root_ids': lum_missing,
                                'label': 'L1–L3 luminance inputs', 'mapping_note': visual_note}, 'channels': channels,
               'body_transducers': transducers,
               'sugar_annotation_exclusions': [
                   {'index': i, 'root_id': str(ids[i]), 'side': row.get('side'), 'organ': row.get('body_part_sensory'),
                    'cell_type': typ(row), 'function': row.get('cell_function_detailed'),
                    'reason': sugar_modality_exclusion_reason(row), 'identity_evidence': sugar_identity_evidence(row)}
                   for i, row in enumerate(rows) if sugar_modality_exclusion_reason(row)],
               'body_transducer_exclusions': [
                   {'index': i, 'root_id': str(ids[i]), 'side': row.get('side'), 'organ': row.get('body_part_sensory'),
                    'cell_type': typ(row), 'cell_class': row.get('cell_class'), 'function': row.get('cell_function_detailed'),
                    'reason': body_transducer_exclusion_reason(row),
                    **({'source': JO_MODALITY_SOURCE} if auditory_transducer_exclusion_reason(row) else {})}
                   for i, row in enumerate(rows)
                   if row.get('super_class') in ('sensory', 'sensory_ascending') and row.get('side') in ('left', 'right')
                   and (('leg' in (row.get('body_part_sensory') or '') and bristle_transducer_exclusion_reason(row))
                        or auditory_transducer_exclusion_reason(row))]}
    projection = {**base, 'cells': projected, 'unmapped_root_ids': projection_missing, 'rate_scale_hz': 20, 'max_rate_hz': 60,
                  'ids_sha256': sha(DATA/'ids.bin'), 'model_sha256': sha(ROOT/'web/visual-model.json'), 'method': visual_note}
    old_color = json.loads((ROOT/'web/color-inputs.json').read_text())
    color_mapping = {**base, 'cells': color, 'unmapped_root_ids': color_missing, 'ids_sha256': sha(DATA/'ids.bin'),
                     'lut_sha256': old_color['lut_sha256'], 'base_rate_hz': 2, 'rate_scale_hz': 60, 'max_rate_hz': 80,
                     'assumptions': [visual_note, 'Opsins use an assumed 30:70 pale/yellow mosaic, paired by projected camera bin.']}

    motors = []
    for key, label, predicate in [
        ('forward', 'Forward', lambda r: r.get('cell_function_detailed') == 'move_coxa_posterior'),
        ('reverse', 'Reverse', lambda r: r.get('cell_function_detailed') == 'move_coxa_anterior'),
        ('turn_left', 'Turn left', lambda r: r.get('side') == 'right' and r.get('cell_function_detailed') == 'move_coxa_posterior'),
        ('turn_right', 'Turn right', lambda r: r.get('side') == 'left' and r.get('cell_function_detailed') == 'move_coxa_posterior'),
        ('wing_left', 'Left wing', lambda r: r.get('side') == 'left' and r.get('cell_function') == 'wing_power'),
        ('wing_right', 'Right wing', lambda r: r.get('side') == 'right' and r.get('cell_function') == 'wing_power'),
        ('takeoff', 'Takeoff', lambda r: r.get('cell_function') == 'jump_escape' or typ(r) == 'tergotrochanter_extensor'),
        ('landing', 'Landing posture', lambda r: r.get('body_part_effector') == 'front_leg' and r.get('cell_function_detailed') == 'extend_coxa_trochanter_joint'),
        ('groom', 'Foreleg grooming', lambda r: r.get('body_part_effector') == 'front_leg' and r.get('cell_function_detailed') == 'flex_coxa_trochanter_joint'),
        ('proboscis', 'Proboscis', lambda r: r.get('cell_class') == 'proboscis_motor_neuron'),
        ('antenna_left', 'Left antenna', lambda r: r.get('cell_class') == 'antenna_motor_neuron' and r.get('side') == 'left'),
        ('antenna_right', 'Right antenna', lambda r: r.get('cell_class') == 'antenna_motor_neuron' and r.get('side') == 'right'),
    ]:
        selected = select(lambda r: r.get('super_class') == 'motor' and predicate(r))
        assert selected, key
        motors.append({'key': key, 'label': label, 'indices': selected, 'cells': [annotation(i) for i in selected],
                       'source': SOURCE, 'decoder': 'Annotated BANC motor neurons. Muscle activation and force provide the modeled body interface.'})
    motor = {**base, 'channels': motors}

    edges = np.memmap(DATA/'edges.bin', dtype='<u4', mode='r').reshape(-1, 4)
    weights = edges.view('<f4')[:, 1]
    offsets = np.fromfile(DATA/'offsets.bin', '<u4')
    incoming, cells = {}, {}
    for i in sorted({i for channel in motors for i in channel['indices']}):
        pairs = []
        for e in range(int(offsets[i]), int(offsets[i+1])):
            j, receptor = int(edges[e, 0]), int(edges[e, 2])
            sign = -1 if receptor in (1, 2, 3) else 1
            pairs.append([j, float(weights[e])*sign])
            cells[j] = annotation(j)
        incoming[i] = pairs
        cells[i] = annotation(i)
    probe = {**base, 'dataset': 'BANC v888', 'driveUnit': 'pA', 'incomingUnit': 'nS·ms × Hz',
             'groups': [{'key': 'retina', 'label': 'Graded luminance / color', 'indices': [c['index'] for c in luminance+color]},
                        {'key': 'motion', 'label': 'T4 / T5 motion', 'indices': [c['index'] for c in projected if c['type'].startswith(('T4','T5'))]},
                        {'key': 'descending', 'label': 'Descending', 'indices': select(lambda r: r.get('super_class') == 'descending')},
                        {'key': 'vnc', 'label': 'Ventral nerve cord', 'indices': select(lambda r: r.get('region') == 'ventral_nerve_cord')},
                        {'key': 'motor', 'label': 'Motor neurons', 'indices': select(lambda r: r.get('super_class') == 'motor')}],
             'channels': [{k: c[k] for k in ('key', 'label', 'indices')} for c in motors], 'cells': cells, 'incoming': incoming}
    metadata = {'dataset': 'BANC v888', 'neurons_per_brain': n, 'connection_rows': manifest['chemical_edges'],
                'synapses_represented': manifest['synapse_count'], 'parameters': {'rest_mv': -60, 'threshold_mv': -45, 'dt_ms': .5},
                'prepared_sha256': {k: v['sha256'] for k, v in manifest['files'].items()}, 'groups': {k: len(v) for k,v in groups.items()},
                'sources': [SOURCE], 'limits': [manifest['physiology_status'], visual_note]}
    for name, value in [('metadata', metadata), ('groups', groups), ('sensory-inputs', sensory), ('motor-outputs', motor),
                        ('visual-projections', projection), ('color-inputs', color_mapping), ('circuit-probe', probe)]:
        (OUT/(name+'.json')).write_text(json.dumps(value, separators=(',', ':'))+'\n')
    # Keep graph/IO preparation immutable. Supply the missing anatomical facts
    # under the existing URL, bound to exact prepared IDs, IO and raw metadata.
    # The filename is retained for compatibility; this now covers every
    # external sugar sensor omitted by the broad prepared sensory classifier.
    io = json.loads((DATA/'io.json').read_text())
    io_sensory_ids = {row['index'] for row in io['sensory']}
    supplement = {'schema': 1, 'kind': 'BANC_v888_verified_sensory_annotation_supplement',
        'source': {'path': 'data/raw/banc888/meta.feather', 'sha256': sha(ROOT/'data/raw/banc888/meta.feather'),
                   'url': 'https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888/banc_888_meta.feather'},
        'prepared_ids_sha256': sha(DATA/'ids.bin'), 'prepared_io_sha256': sha(DATA/'io.json'),
        'identity_evidence_file': 'models/banc-sensory-annotation-evidence.json',
        'identity_evidence_sha256': sha(ROOT/'models/banc-sensory-annotation-evidence.json'),
        'explanation': 'External sugar sensors omitted by the broad prepared IO class filter. Exact sugar token and supported external organ define inclusion. Index, organ, side, class, function and nerve are copied from BANC annotations; null side remains null. No connectivity or physiology parameters change. Contact-rate tuning remains a prior.',
        'annotations': [{'index': i, 'root_id': str(ids[i]), 'kind': 'taste', 'body_part': rows[i].get('body_part_sensory'),
            **{key: rows[i].get(key) for key in ('side', 'cell_type', 'cell_class', 'nerve', 'cell_function', 'cell_function_detailed', 'fafb_match')},
            **({'identity_evidence': sugar_identity_evidence(rows[i])} if sugar_identity_evidence(rows[i]) else {})}
            for i in groups['sweet'] if i not in io_sensory_ids]}
    (ROOT/'models/banc-taste-peg-annotations.json').write_text(json.dumps(supplement, indent=2)+'\n')
    print(json.dumps({'groups': metadata['groups'], 'luminance': len(luminance), 'graded_projection': len(projected),
                      'color': len(color), 'body_sense': {c['key']: len(c['indices']) for c in channels}, 'motor_channels': len(motors)}, indent=2))


if __name__ == '__main__':
    prepare()
