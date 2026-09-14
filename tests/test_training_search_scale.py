"""Pure optimizer fixtures: no coordinator database, server, brain, or body."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads('{"schemaVersion":1,"environmentVersion":"banc-flybody-flight-objective-v3","modelFingerprint":"4d53af18ac499bf90ecc2d98342cd596bc65e12ddbbf9f92c66d87a3e6757cc6","assets":{"fixture.wasm":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},"algorithm":"antithetic-evolution-strategies","dtMs":0.5,"bodyBlockMs":2,"parameters":[{"name":"gain_log","min":-2,"max":2,"initial":0}],"optimizer":{"populationPairs":2,"sigma":0.25,"learningRate":0.035,"maximumUpdate":0.15,"seed":888},"objective":{"min":-10,"max":10,"direction":"maximize"},"stage":"landing","durationSeconds":8,"stages":[{"id":"takeoff","durationSeconds":3},{"id":"flight","durationSeconds":5},{"id":"landing","durationSeconds":8}],"contribution":{"leaseSeconds":10,"maxRequestBytes":1024}}')


spec = importlib.util.spec_from_file_location('training_search_scale_coordinator', ROOT / 'scripts/training_coordinator.py')
STAGED = importlib.util.module_from_spec(spec)
spec.loader.exec_module(STAGED)


def rules(config, implementation=STAGED, config_hash=None):
    value = implementation.CoordinatorRules()
    value._configure(copy.deepcopy(config), config_hash=config_hash, clock=lambda: 1000.)
    return value


def fixture(scales=(.5, .25), bounds=((-10, 10), (-10, 10)), initial=(0., 0.)):
    config = copy.deepcopy(CONFIG)
    config['parameters'] = [dict(name=f'p{k}', min=bounds[k][0], max=bounds[k][1], initial=initial[k],
                                 **({} if scale is None else {'searchScale': scale})) for k, scale in enumerate(scales)]
    config['optimizer'].update(populationPairs=2, sigma=.25, learningRate=.035, maximumUpdate=.15)
    return config


def score(jobs):
    for job in jobs:
        job['score'] = math.tanh(sum((k+1)*x for k, x in enumerate(job['parameters'])))
    return jobs


class SearchScaleTests(unittest.TestCase):
    def test_omission_preserves_legacy_generation_update_signature(self):
        config = fixture((None, None))
        current = rules(config)
        self.assertEqual(current.config, config)
        self.assertEqual(current.search_scales, [1., 1.])
        records = []
        for generation in range(32):
            record, jobs = current._generation_records(generation, current.initial)
            score(jobs)
            records.append([record, jobs, current._next_center(record, jobs)])
        # Pinned pre-searchScale implementation, same fixed fixture and clock.
        digest = hashlib.sha256(STAGED.canonical(records).encode()).hexdigest()
        self.assertEqual(digest, '10e9e7e5fd1ab87f642d145a5cca19bcb6e9ba917d29ff8234aa727e5e00b1a3')

    def test_explicit_one_and_scaled_perturbations_at_same_random_stream(self):
        missing = rules(fixture((None, None)))
        ones = rules(fixture((1., 1.)), config_hash=missing.config_hash)
        scaled = rules(fixture(), config_hash=missing.config_hash)
        for generation in range(16):
            _, base = missing._generation_records(generation, missing.initial)
            self.assertEqual(ones._generation_records(generation, ones.initial)[1], base)
            _, actual = scaled._generation_records(generation, scaled.initial)
            for a, b in zip(actual, base):
                self.assertEqual(a['noise'], b['noise'])
                self.assertEqual(a['seed'], b['seed'])
                self.assertEqual(a['parameters'], [x*s for x, s in zip(b['parameters'], scaled.search_scales)])
        # Explicit metadata changes the real namespace and therefore its RNG.
        self.assertNotEqual(rules(fixture((1., 1.))).config_hash, missing.config_hash)

    def test_scaled_clipping_and_web_coordinator_aggregate_parity(self):
        config = fixture((.5, .125), ((-1, 1), (-.001, .001)), (.999, 0.))
        current = rules(config)
        record, jobs = current._generation_records(0, current.initial)
        clipped = 0
        for job in jobs:
            for k, x in enumerate(job['parameters']):
                p = config['parameters'][k]
                raw = current.initial[k]+job['sign']*current.sigma*p['searchScale']*job['noise'][k]
                self.assertEqual(x, max(p['min'], min(p['max'], raw)))
                clipped += x != raw
        self.assertGreater(clipped, 0)
        score(jobs)
        pairs = []
        for i in range(0, len(jobs), 2):
            positive, negative = jobs[i:i+2]
            pairs.append({'noise': positive['noise'], 'jobs': [{'sign': j['sign'], 'parameters': j['parameters']} for j in (positive, negative)],
                          'results': {str(j['sign']): {'return': j['score'], 'success': False, 'simSeconds': 1, 'steps': 500} for j in (positive, negative)}})
        payload = {'config': config, 'round': {'baseline': current.initial, 'pairs': pairs}}
        js = "import fs from 'node:fs'; import {validateConfig,updateGeneration} from " + json.dumps((ROOT/'web/training/optimizer.js').as_uri()) + "; const x=JSON.parse(fs.readFileSync(0,'utf8')); console.log(JSON.stringify(updateGeneration(x.round,validateConfig(x.config))));"
        completed = subprocess.run(['node', '--input-type=module', '-e', js], input=json.dumps(payload), text=True, capture_output=True, check=True)
        expected = json.loads(completed.stdout)
        self.assertEqual(current._next_center(record, jobs), expected)
        stored = copy.deepcopy(jobs)
        for job in stored:
            job['parameters'] = json.dumps(job['parameters'])
        self.assertEqual(current._next_center(record, stored), expected)

    def test_normalized_coordinate_equivalence_with_and_without_candidate_clipping(self):
        for bounds in (((-10, 10), (-10, 10)), ((-.001, .001), (-.002, .002))):
            config = fixture(bounds=bounds)
            current = rules(config)
            record, jobs = current._generation_records(3, current.initial)
            score(jobs)
            normalized = copy.deepcopy(config)
            for p in normalized['parameters']:
                scale = p['searchScale']
                for key in ('min', 'max', 'initial'):
                    p[key] /= scale
                p['searchScale'] = 1
            y = rules(normalized)
            y_record, y_jobs = copy.deepcopy(record), copy.deepcopy(jobs)
            y_record['center'] = [x/s for x, s in zip(record['center'], current.search_scales)]
            for job in y_jobs:
                job['parameters'] = [x/s for x, s in zip(job['parameters'], current.search_scales)]
            actual = current._next_center(record, jobs)
            expected = [v*s for v, s in zip(y._next_center(y_record, y_jobs), current.search_scales)]
            for a, b in zip(actual, expected):
                self.assertAlmostEqual(a, b, places=14)

    def test_s_squared_linear_response_and_retained_physical_cap(self):
        config = fixture()
        config['optimizer']['learningRate'] = .1
        current = rules(config)
        gradient, noises, jobs = (2., -3.), ((math.sqrt(2), 0.), (0., math.sqrt(2))), []
        for i, noise in enumerate(noises):
            for sign in (1, -1):
                p = [sign*current.sigma*s*n for s, n in zip(current.search_scales, noise)]
                jobs.append({'pair_id': str(i), 'sign': sign, 'parameters': p, 'score': sum(x*g for x, g in zip(p, gradient))})
        record = {'center': [0., 0.]}
        actual = current._next_center(record, jobs)
        for a, s, g in zip(actual, current.search_scales, gradient):
            self.assertAlmostEqual(a, current.learning_rate*s*s*g, places=14)
        current.maximum_update = .005
        self.assertEqual(current._next_center(record, jobs), [.005, -.005])

    def test_equal_returns_fixed_and_invalid_scales_rejected(self):
        current = rules(fixture())
        record, jobs = current._generation_records(0, current.initial)
        for job in jobs:
            job['score'] = 1.
        self.assertEqual(current._next_center(record, jobs), current.initial)
        for value in (0., -1., 1.0001, math.nan, math.inf, -math.inf, None, True, False, '0.5', [], {}):
            config = fixture()
            config['parameters'][0]['searchScale'] = value
            with self.subTest(value=value), self.assertRaises((ValueError, STAGED.APIError)):
                rules(config)
        for value in (5e-324, .01, .5, 1.):
            config = fixture()
            config['parameters'][0]['searchScale'] = value
            self.assertEqual(rules(config).search_scales[0], value)


if __name__ == '__main__':
    unittest.main(verbosity=2)
