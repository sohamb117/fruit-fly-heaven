"""Resolve a documented actuator readout against the existing v783 brain IDs.

This adds output annotations only. It never edits connectivity or neural state.
"""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'atlas': 'https://www.nature.com/articles/s41586-026-10735-w',
    'descending': 'https://www.nature.com/articles/s41586-025-08925-z',
    'wing_power': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC9206711/',
    'landing': 'https://www.nature.com/articles/s41593-019-0413-4',
    'annotations': 'https://www.nature.com/articles/s41586-024-07686-5',
}

def main():
    files = [ROOT/'data/prepared/ids.bin', ROOT/'data/raw/classification.csv.gz', ROOT/'data/raw/consolidated_cell_types.csv.gz']
    import numpy as np
    ids = np.fromfile(files[0], dtype='<i8')
    ann = pd.read_csv(files[1]).set_index('root_id').reindex(ids).fillna('')
    types = pd.read_csv(files[2]).set_index('root_id').reindex(ids).fillna('')
    channels = []

    def add(key, label, mask, source, evidence, decoder):
        indices = np.flatnonzero(mask).tolist()
        if not indices:
            raise ValueError(f'No annotated v783 neurons for {key}; do not substitute another population')
        cells = [{'index': i, 'root_id': str(ids[i]), 'type': str(types.primary_type.iloc[i]),
                  'side': str(ann.side.iloc[i]), 'subclass': str(ann.sub_class.iloc[i])} for i in indices]
        channels.append(dict(key=key, label=label, indices=indices, cells=cells, source=SOURCES[source], evidence=evidence, decoder=decoder))

    typ = types.primary_type
    add('forward', 'Forward walking', typ.isin(['DNg97', 'DNg100']), 'atlas',
        'These cell types have documented forward-walking roles.', 'Positive forward drive; no baseline speed.')
    add('reverse', 'Backward walking', typ.eq('MDN'), 'descending',
        'Moonwalker descending neurons are associated with backward walking.', 'Opposes forward drive.')
    for side in ['left', 'right']:
        add(f'turn_{side}', f'Turn {side}', typ.isin(['DNa01', 'DNa02']) & ann.side.eq(side), 'descending',
            'Identified walking-steering descending populations.', 'Left-minus-right rate sets yaw torque; side-to-torque gain is a decoder assumption.')
        # v783 subdivides the experimentally named DNg02 population into a–h.
        add(f'wing_{side}', f'Wing power {side}', typ.str.fullmatch(r'DNg02(?:_[a-h])?') & ann.side.eq(side), 'wing_power',
            'DNg02 population activity modulates wingbeat amplitude during flight.', 'Mean bilateral activity sets lift/thrust; asymmetry sets flight yaw. Force gains are assumptions.')
    add('takeoff', 'Escape takeoff', typ.isin(['DNp01', 'DNp02', 'DNp04', 'DNp06', 'DNp11']), 'atlas',
        'These descending types have documented escape-takeoff roles.', 'A rising command produces a bounded jump impulse; no periodic launch timer.')
    add('landing', 'Landing extension', typ.isin(['DNp07', 'DNp10']), 'landing',
        'Activation elicits landing-associated leg extension.', 'Extends legs and brakes the flight actuator. Throttle braking is an approximation, not a measured transfer function.')
    add('groom', 'Antennal grooming', typ.isin(['DNg62', 'DNge011', 'DNge012', 'DNge078']), 'atlas',
        'These descending types have documented antennal-grooming roles.', 'Front-leg strokes advance only while this output is active.')
    add('proboscis', 'Proboscis', ann.sub_class.isin(['proboscis_motor_neuron', 'haustellum_motor_neuron']), 'annotations',
        'Motor-cell identities are supplied by the measured brain annotations.', 'Controls proboscis extension; fruit contact determines whether it counts as feeding.')
    for side in ['left', 'right']:
        add(f'antenna_{side}', f'Antenna {side}', ann.sub_class.eq('antennal_motor_neuron') & ann.side.eq(side), 'annotations',
            'Annotated antennal motor neurons on this side.', 'Controls that antenna’s deflection and motion amplitude.')

    result = {
        'schema_version': 1, 'dataset': 'FAFB-FlyWire v783 / existing 138,639-neuron computational model',
        'neuron_count': len(ids), 'rate_window_ms': 100,
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'scope': 'Brain only. No BANC, FANC, MANC, or ventral nerve cord connectivity is loaded. The atlas is used only as a functional reference for named brain cells.',
        'limitations': ['Published behavioral associations do not establish a validated firing-rate-to-force transfer function.',
                       'Group means and rate thresholds are an explicit actuator decoder, not muscle-resolved circuitry.',
                       'The habitat supplies odor, sugar, rendered visual input and an approximate body-sense boundary adapter; see sensory-inputs.json for its mapping and limitations.'],
        'channels': channels,
    }
    output = ROOT/'web/motor-outputs.json'
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({c['key']:len(c['indices']) for c in channels}))

if __name__ == '__main__':
    main()
