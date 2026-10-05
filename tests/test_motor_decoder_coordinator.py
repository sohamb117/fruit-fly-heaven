"""The explicit decoder contract fits through the existing shared coordinator."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('decoder_coordinator', ROOT / 'scripts/training_coordinator.py')
COORDINATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COORDINATOR)


def config():
    value = json.loads((ROOT / 'web/training/config.json').read_text())
    value['parameterContract'] = 'banc-masked-motor-decoder-v1'
    value['freezeNeuralParameters'] = True
    value['wingEventExcitation'] = {}
    value['parameters'] = [dict(name=f'decoder_{i}', min=-.25, max=.25, initial=0.) for i in range(672)]
    return value


def rules(value):
    result = COORDINATOR.CoordinatorRules()
    result._configure(value, config_hash=hashlib.sha256(COORDINATOR.canonical(value).encode()).hexdigest())
    return result


class DecoderCoordinatorTests(unittest.TestCase):
    def test_declared_decoder_preserves_all_coordinates_in_assignments(self):
        instance = rules(config())
        _, jobs = instance._generation_records(0, instance.initial)
        self.assertEqual(len(instance.names), 672)
        for job in jobs:
            self.assertEqual(len(job['parameters']), 672)
            self.assertTrue(all(-.25 <= value <= .25 for value in job['parameters']))

    def test_large_vectors_require_exact_explicit_contract(self):
        valid = config()
        variants = []
        for key in ('parameterContract', 'freezeNeuralParameters', 'wingEventExcitation'):
            value = copy.deepcopy(valid)
            del value[key]
            variants.append(value)
        for length in (671, 673):
            value = copy.deepcopy(valid)
            value['parameters'] = [dict(name=f'p{i}', min=-1, max=1, initial=0) for i in range(length)]
            variants.append(value)
        for value in variants:
            with self.subTest(length=len(value['parameters'])), self.assertRaises(ValueError):
                rules(value)


if __name__ == '__main__':
    unittest.main()
