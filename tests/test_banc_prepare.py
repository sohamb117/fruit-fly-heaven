"""Run with .venv/bin/python -m unittest discover -s tests -p test_banc_prepare.py."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_banc', ROOT/'scripts/prepare-banc.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BancPreparationTest(unittest.TestCase):
    def test_exact_ids_artifact_filtering_and_unknown_transmitter(self):
        ids = ['720575941000000001', '720575941000000002', '720575941000000003', '720575941000000004']
        rows = [dict(banc_888_id=rid, super_class=cls, side='left', cell_function_detailed='flex_femur_tibia_joint',
                     body_part_effector='front_leg', peripheral_target_type='tibia_flexor_muscle', neurotransmitter_predicted=tx)
                for rid, cls, tx in zip(ids[:3], ['descending', 'motor', 'glia'], ['acetylcholine', 'glutamate', None])]
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            feather.write_feather(pa.Table.from_pylist(rows), p/'meta.feather')
            feather.write_feather(pa.table({'pre': [ids[0], ids[3], ids[2]], 'post': [ids[1], ids[1], ids[0]], 'count': [1, 3, 50]}), p/'edges.feather')
            config = json.loads((ROOT/'configs/banc-physiology.json').read_text())
            config['gap_junctions'] = []
            with contextlib.redirect_stdout(io.StringIO()):
                module.prepare(p/'meta.feather', p/'edges.feather', p/'out', config, {})
            actual = np.fromfile(p/'out/ids.bin', dtype='<u8').astype(str).tolist()
            self.assertEqual(actual, [ids[0], ids[1], ids[3]])
            manifest = json.loads((p/'out/manifest.json').read_text())
            self.assertEqual(manifest['chemical_edges'], 2)
            self.assertEqual(manifest['excluded_non_neuronal_edges'], 1)
            self.assertEqual(manifest['unknown_transmitter_edges_zeroed'], 1)
            edges = np.fromfile(p/'out/edges.bin', dtype='<u4').reshape(-1, 4)
            self.assertEqual(edges[:, 0].tolist(), [0, 2])
            np.testing.assert_allclose(edges[:, 1].copy().view('<f4'),
                                       [config['synaptic_weight_ns_ms_per_contact'], 0])
            io_data = json.loads((p/'out/io.json').read_text())
            self.assertEqual(io_data['muscles'][0]['root_ids'], [ids[1]])
            self.assertEqual(io_data['muscles'][0]['joint'], 'tibia_T1_left')

    def test_descending_and_unknown_muscles_are_never_actuators(self):
        row = dict(super_class='descending', side='left', body_part_effector='wing', peripheral_target_type='dorsal_longitudinal_muscle')
        self.assertIsNone(module.muscle_target(row))
        row['super_class'] = 'motor'
        row['peripheral_target_type'] = 'unknown'
        self.assertIsNone(module.muscle_target(row))

    def test_cell_specific_override_does_not_modify_other_cells(self):
        config = json.loads((ROOT/'configs/banc-physiology.json').read_text())
        config['cell_overrides'] = {'1': {'capacitance_pf': 99}}
        self.assertEqual(module.profile({'banc_888_id': '1'}, config)[1][0], 99)
        self.assertEqual(module.profile({'banc_888_id': '2'}, config)[1][0], 20)

    def test_tarsus_and_claw_targets_follow_annotated_limb_and_side(self):
        row = dict(super_class='motor', side='right', body_part_effector='hind_leg',
                   cell_function_detailed='pull_long_tendon')
        self.assertEqual(module.muscle_target(row), ('adhere_claw_T3_right', 1, 'claw_grip_assumption'))
        row['cell_function_detailed'] = 'extend_tibia_tarsus_joint'
        self.assertEqual(module.muscle_target(row), ('tarsus_T3_right', -1, 'leg'))
        row['side'] = None
        self.assertIsNone(module.muscle_target(row))

    def test_haltere_mapping_requires_an_annotated_muscle_target(self):
        row = dict(super_class='motor', side='left', body_part_effector='haltere',
                   peripheral_target_type='haltere_dorsoventral_muscle')
        self.assertEqual(module.muscle_target(row), ('haltere_power_left', 1, 'asynchronous_haltere'))
        row.update(side='right', peripheral_target_type='hi2_muscle')
        self.assertEqual(module.muscle_target(row), ('haltere_steer_right', 1, 'haltere_steering_assumption'))
        row.update(peripheral_target_type=None, cell_function_detailed='haltere_power')
        self.assertIsNone(module.muscle_target(row))


if __name__ == '__main__':
    unittest.main()
