"""Staged coordinator tests: isolated SQLite and algebraic returns, no fly."""
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


module = load('acceptance_staged', ROOT/'scripts/training_coordinator.py')


def fixture(guarded=True, pairs=4):
    value = {
        'schemaVersion': 2 if guarded else 1,
        'environmentVersion': 'guard-algebra-v1',
        'modelFingerprint': hashlib.sha256(('fixture.wasm:'+'b'*64+'\n').encode()).hexdigest(),
        'assets': {'fixture.wasm': 'b'*64}, 'algorithm': 'antithetic-evolution-strategies',
        'dtMs': .5, 'bodyBlockMs': 2,
        'parameters': [{'name': 'gain_log', 'min': -2, 'max': 2, 'initial': 0}],
        'optimizer': {'populationPairs': pairs, 'sigma': .25, 'learningRate': .035, 'maximumUpdate': .15, 'seed': 888},
        'objective': {'min': -10, 'max': 10, 'direction': 'maximize'},
        'stage': 'landing', 'durationSeconds': 8,
        'stages': [{'id': 'landing', 'durationSeconds': 8}],
        'validation': {'seeds': [190888, 290888], 'testSeeds': [1190888, 1290888, 1390888]},
        'contribution': {'leaseSeconds': 10, 'maxRequestBytes': 65536},
    }
    if guarded:
        value['optimizer']['acceptance'] = {'profile': 1, 'proposal': 'best-search-job', 'seedCount': 3,
            'nativeExecution': {'backend': 'dawn-metal', 'moduleSha256': 'c'*64, 'packageLockSha256': 'd'*64}}
    return value


def rules(config, implementation=module, config_hash=None):
    value = implementation.CoordinatorRules()
    value._configure(config, config_hash=config_hash, clock=lambda: 1000.)
    return value


def identity(coordinator, owner):
    return {'contributorId': owner, 'configHash': coordinator.config_hash, 'modelFingerprint': coordinator.model_fingerprint}


def result(coordinator, job, owner, score):
    provenance = {key: job[key] for key in ('modelFingerprint', 'configHash', 'parametersHash', 'seed', 'stage', 'durationSeconds', 'sign', 'pairId', 'generation')}
    provenance.update(environmentVersion=coordinator.environment_version, backend='webgpu', bodyBackend='mujoco-wasm', dtMs=.5, bodyBlockMs=2,
        neuralEngine='dawn-metal', nativeWebGPU={'backend': 'dawn-metal', 'moduleSha256': 'c'*64, 'packageLockSha256': 'd'*64,
                                              'adapter': {'isFallbackAdapter': False, 'vendor': 'fixture'}})
    return {**identity(coordinator, owner), 'jobId': job['jobId'], 'leaseToken': job['leaseToken'],
            'parameters': job['parameters'], 'objective': score,
            'metrics': {'success': False, 'terminated': False, 'cancelled': False, 'simSeconds': 8., 'steps': 4000, 'reason': 'time_limit'}, 'provenance': provenance}


def wasm_fixture(pairs=2):
    value = fixture(pairs=pairs)
    value['assets']['/banc-engine/dist/core.wasm'] = 'e'*64
    value['modelFingerprint'] = hashlib.sha256(''.join(f'{key}:{digest}\n' for key, digest in value['assets'].items()).encode()).hexdigest()
    value['optimizer']['acceptance']['nativeExecution'] = {'backend': 'wasm', 'moduleSha256': 'e'*64}
    return value


def wasm_result(coordinator, job, owner, score):
    value = result(coordinator, job, owner, score)
    del value['provenance']['nativeWebGPU']
    value['provenance'].update(backend='wasm', neuralEngine='wasm',
                              parameters=copy.deepcopy(job['parameters']),
                              wasmExecution=copy.deepcopy(coordinator.acceptance_config['nativeExecution']))
    return value


class WasmGuardTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.config = wasm_fixture()
        self.coordinator = module.TrainingCoordinator(Path(self.directory.name)/'wasm.sqlite3', self.config, clock=lambda: 1000.)

    def tearDown(self):
        self.coordinator.close()
        self.directory.cleanup()

    def lease(self):
        return self.coordinator.lease(identity(self.coordinator, 'browser'))['job']

    def test_wasm_pin_requires_exact_configured_module_and_its_own_config_identity(self):
        self.assertTrue(self.coordinator.guarded)
        self.assertNotEqual(self.coordinator.config_hash, rules(fixture(pairs=2)).config_hash)
        for execution in ({'backend': 'wasm', 'moduleSha256': 'f'*64},
                          {'backend': 'wasm', 'moduleSha256': 'E'*64},
                          {'backend': 'wasm'},
                          {'backend': 'wasm', 'moduleSha256': 'e'*64, 'packageLockSha256': 'd'*64},
                          {'backend': 'webgpu', 'moduleSha256': 'e'*64}):
            config = copy.deepcopy(self.config); config['optimizer']['acceptance']['nativeExecution'] = execution
            with self.subTest(execution=execution), self.assertRaises(ValueError):
                rules(config)

    def test_wasm_search_and_three_seed_comparison_keep_guarded_accept_reject_semantics(self):
        for generation, comparison_gain in ((0, .1), (1, -.1)):
            incumbent = self.coordinator.checkpoint()['parameters']
            for _ in range(4):
                job = self.lease()
                self.coordinator.result(wasm_result(self.coordinator, job, 'browser', job['parameters'][0]))
            self.assertEqual(self.coordinator.status()['generationPhase'], 'comparison')
            self.assertEqual(self.coordinator.checkpoint()['parameters'], incumbent)
            for _ in range(6):
                job = self.lease()
                self.coordinator.result(wasm_result(self.coordinator, job, 'browser', comparison_gain if job['sign'] == 1 else 0.))
            state = self.coordinator.status()
            self.assertEqual(state['generation'], generation+1)
            if comparison_gain > 0:
                self.assertNotEqual(state['checkpoint']['parameters'], incumbent)
            else:
                self.assertEqual(state['checkpoint']['parameters'], incumbent)
            self.assertFalse(state['checkpoint']['heldOutValidated'])
        self.assertEqual(self.coordinator.status()['acceptedResults'], 20)

    def test_wrong_backend_pin_or_fabricated_native_provenance_cannot_enter_wasm_pool(self):
        job = self.lease(); valid = wasm_result(self.coordinator, job, 'browser', 0.)
        for key, value in (('backend', 'webgpu'), ('neuralEngine', 'dawn-metal'),
                           ('wasmExecution', None), ('wasmExecution', {'backend': 'wasm', 'moduleSha256': 'f'*64}),
                           ('wasmExecution', {'backend': 'wasm', 'moduleSha256': 'e'*64, 'extra': 1}),
                           ('nativeWebGPU', None), ('nativeWebGPU', {'backend': 'dawn-metal'})):
            body = copy.deepcopy(valid); body['provenance'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(module.APIError) as caught:
                self.coordinator.result(body)
            self.assertEqual(caught.exception.code, 'provenance_mismatch')
        self.assertEqual(self.coordinator.status()['acceptedResults'], 0)
        self.assertTrue(self.coordinator.result(valid)['accepted'])

    def test_actual_browser_payload_shape_requires_no_native_top_level_vector_or_duration(self):
        job = self.lease()
        body = wasm_result(self.coordinator, job, 'browser', 0.)
        # web/training/client.js uploads the actual vector only inside
        # provenance; environment.js supplies these four assignment fields.
        del body['parameters']
        del body['provenance']['durationSeconds']
        self.assertTrue(self.coordinator.result(body)['accepted'])
        self.assertTrue(self.coordinator.result(body)['duplicate'])
        self.assertEqual(self.coordinator.status()['acceptedResults'], 1)

    def test_guarded_wasm_rejects_missing_or_wrong_applied_vector_despite_matching_claimed_hash(self):
        job = self.lease()
        valid = wasm_result(self.coordinator, job, 'browser', 0.)
        for vector in (None, [], [True], [job['parameters'][0] + .01], 'not a vector'):
            body = copy.deepcopy(valid)
            body['provenance']['parameters'] = vector
            with self.subTest(vector=vector), self.assertRaises(module.APIError) as caught:
                self.coordinator.result(body)
            self.assertEqual(caught.exception.code, 'provenance_mismatch')
        body = copy.deepcopy(valid)
        del body['provenance']['parameters']
        # Even a correct top-level vector must not replace the applied-vector
        # field that the browser records in its execution provenance.
        with self.assertRaises(module.APIError) as caught:
            self.coordinator.result(body)
        self.assertEqual(caught.exception.code, 'provenance_mismatch')
        self.assertEqual(self.coordinator.status()['acceptedResults'], 0)
        self.assertTrue(self.coordinator.result(valid)['accepted'])

    def test_guarded_wasm_requires_exact_assignment_identity_in_provenance(self):
        job = self.lease()
        valid = wasm_result(self.coordinator, job, 'browser', 0.)
        for key, wrong in (('parametersHash', '0'*64), ('sign', -job['sign']),
                           ('pairId', 'g0-p999'), ('generation', job['generation'] + 1)):
            for missing in (True, False):
                body = copy.deepcopy(valid)
                if missing:
                    del body['provenance'][key]
                else:
                    body['provenance'][key] = wrong
                with self.subTest(key=key, missing=missing), self.assertRaises(module.APIError) as caught:
                    self.coordinator.result(body)
                self.assertEqual(caught.exception.code, 'provenance_mismatch')
        self.assertEqual(self.coordinator.status()['acceptedResults'], 0)
        self.assertTrue(self.coordinator.result(valid)['accepted'])

    def test_guarded_wasm_native_producer_payload_with_both_parameter_fields_remains_valid(self):
        job = self.lease()
        body = wasm_result(self.coordinator, job, 'browser', 0.)
        for key in ('parametersHash', 'durationSeconds', 'generation', 'sign', 'pairId'):
            body[key] = job[key]
        self.assertEqual(body['parameters'], body['provenance']['parameters'])
        self.assertTrue(self.coordinator.result(body)['accepted'])

    def test_wasm_clock_completion_and_cancellation_rules_remain_strict(self):
        job = self.lease(); valid = wasm_result(self.coordinator, job, 'browser', 0.)
        for edits in ({'cancelled': True}, {'terminated': None}, {'steps': 3999},
                      {'simSeconds': 1., 'steps': 500}, {'reason': 'simulation_error'},
                      {'reason': 'invalid_observation', 'terminated': True}):
            body = copy.deepcopy(valid); body['metrics'].update(edits)
            with self.subTest(edits=edits), self.assertRaises(module.APIError) as caught:
                self.coordinator.result(body)
            self.assertEqual(caught.exception.code, 'incomplete_evaluation')
        valid['metrics'].update(simSeconds=.2, steps=100, terminated=True, reason='excessive_rotation')
        self.assertTrue(self.coordinator.result(valid)['accepted'])
        self.assertTrue(self.coordinator.result(valid)['duplicate'])

    def test_wasm_reports_do_not_enter_existing_dawn_cohort(self):
        native = module.TrainingCoordinator(Path(self.directory.name)/'native.sqlite3', fixture(), clock=lambda: 1000.)
        try:
            job = native.lease(identity(native, 'browser'))['job']
            body = result(native, job, 'browser', 0.)
            del body['provenance']['nativeWebGPU']
            body['provenance'].update(backend='wasm', neuralEngine='wasm',
                                      wasmExecution=self.config['optimizer']['acceptance']['nativeExecution'])
            with self.assertRaises(module.APIError) as caught:
                native.result(body)
            self.assertEqual(caught.exception.code, 'provenance_mismatch')
            self.assertEqual(native.status()['acceptedResults'], 0)
        finally:
            native.close()

    def test_warmstart_vector_initializes_a_fresh_cohort_without_inherited_fitness(self):
        config = copy.deepcopy(self.config); config['parameters'][0]['initial'] = .25
        warm = module.TrainingCoordinator(Path(self.directory.name)/'warm.sqlite3', config, clock=lambda: 1000.)
        try:
            state = warm.status()
            self.assertEqual(state['checkpoint']['parameters'], [.25])
            self.assertEqual(state['generation'], 0)
            self.assertEqual(state['acceptedResults'], 0)
            self.assertEqual(state['recentTrials'], [])
            self.assertFalse(state['checkpoint']['biologicalSuccessValidated'])
            self.assertNotEqual(warm.config_hash, self.coordinator.config_hash)
        finally:
            warm.close()


class RulesTests(unittest.TestCase):
    def test_strict_opt_in_schema_and_native_pins(self):
        self.assertTrue(rules(fixture()).guarded)
        self.assertFalse(rules(fixture(False)).guarded)
        bad = []
        for schema in (True, 0, 3, 2.0):
            value = fixture(); value['schemaVersion'] = schema; bad.append(value)
        value = fixture(False); value['optimizer']['acceptance'] = None; bad.append(value)
        value = fixture(); del value['optimizer']['acceptance']; bad.append(value)
        for key, value in [('profile', True), ('profile', 2), ('proposal', 'es'), ('seedCount', 2), ('seedCount', 3.0), ('extra', 1)]:
            config = fixture(); config['optimizer']['acceptance'][key] = value; bad.append(config)
        for key, value in [('backend', 'wasm'), ('moduleSha256', 'C'*64), ('packageLockSha256', 'bad'), ('extra', 1)]:
            config = fixture(); config['optimizer']['acceptance']['nativeExecution'][key] = value; bad.append(config)
        for config in bad:
            with self.subTest(config=config), self.assertRaises((ValueError, module.APIError)):
                rules(config)

    def test_vanilla_complete_records_and_updates_are_unchanged(self):
        config = fixture(False)
        actual, records = rules(config), []
        for generation in range(32):
            a, aj = actual._generation_records(generation, actual.initial)
            for job in aj:
                job['score'] = job['parameters'][0]
            records.append([a, aj, actual._next_center(a, aj)])
        # Fixed fixture signature generated from the pre-guard implementation.
        self.assertEqual(hashlib.sha256(module.canonical(records).encode()).hexdigest(),
                         '1cf2f4fa894b28ee12dbe26621ba0377c12dd13d655cdde05f090925104c25b1')

    def test_comparison_seeds_are_fresh_disjoint_deterministic_and_reserved_safe(self):
        current = rules(fixture())
        previous_seeds = set()
        for generation in range(100):
            seeds = current._guard_comparison_seeds(generation)
            self.assertEqual(seeds, current._guard_comparison_seeds(generation))
            self.assertEqual(len(set(seeds)), 3)
            self.assertFalse(set(seeds) & previous_seeds)
            self.assertFalse(set(seeds) & current.guard_reserved_seeds)
            self.assertTrue(all(seed >= 2**31 for seed in seeds))
            self.assertTrue(all(job['seed'] < 2**31 for job in current._generation_records(generation, current.initial)[1]))
            previous_seeds.update(seeds)
        first = current._guard_comparison_seeds(0)
        config = fixture(); config['validation']['seeds'] = first
        # Fix the hash solely to isolate exclusion from the otherwise new RNG namespace.
        excluded = rules(config, config_hash=current.config_hash)
        self.assertFalse(set(excluded._guard_comparison_seeds(0)) & set(first))
        search = current._generation_records(0, current.initial)[1]
        config = fixture(); config['validation']['seeds'] = [job['seed'] for job in search]
        excluded = rules(config, config_hash=current.config_hash)
        new_jobs = excluded._generation_records(0, excluded.initial)[1]
        self.assertFalse({job['seed'] for job in new_jobs} & excluded.guard_reserved_seeds)
        self.assertEqual(len({job['seed'] for job in new_jobs}), excluded.pairs)
        with self.assertRaises(module.APIError):
            current._guard_comparison_seeds(2**22)

    def test_signed_zero_nomination_is_a_noop(self):
        config = fixture(); config['parameters'][0].update(initial=-0., searchScale=5e-324)
        current = rules(config)
        generation, jobs = current._generation_records(0, current.initial)
        for job in jobs:
            job.update(state='completed', score=1. if job['parameters'][0] == 0. else 0.)
        transition = current._guard_transition(generation, jobs, 1001.)
        self.assertEqual(transition['generation']['acceptance']['decision'], 'identical_parameters')
        self.assertEqual(module.canonical(transition['nextGeneration'][0]['center']), '[-0.0]')


class CrossPlatformNoiseTests(unittest.TestCase):
    def setUp(self):
        self.rules = rules(fixture())
        self.generation, self.jobs = self.rules._generation_records(0, self.rules.initial)

    def shifted(self, ulps):
        jobs = copy.deepcopy(self.jobs)
        for row in jobs:
            noise = []
            for value in row['noise']:
                for _ in range(ulps):
                    value = math.nextafter(value, math.inf)
                noise.append(value)
            row['noise'] = noise
            row['parameters'] = self.rules._candidate_parameters(self.generation['center'], noise, row['sign'])
            row['parameters_hash'] = module.fingerprint(row['parameters'])
        return jobs

    def rejected(self, jobs, generation=None):
        with self.assertRaises(module.APIError) as caught:
            self.rules._validate_guard_batch(generation or self.generation, jobs)
        self.assertEqual(caught.exception.code, 'invalid_history')

    def test_bounded_libm_noise_drift_preserves_exact_stored_vectors_and_hashes(self):
        for distance in (1, module.GAUSSIAN_NOISE_MAX_ULPS):
            rows = self.shifted(distance)
            before = module.canonical(rows)
            validated = self.rules._validate_guard_batch(self.generation, rows)['search']
            self.assertEqual(module.canonical(validated), before)
            self.assertNotEqual(rows[0]['parameters_hash'], self.jobs[0]['parameters_hash'])
            # The same stored assignments survive SQLite's JSON representation.
            serialized = copy.deepcopy(rows)
            for row in serialized:
                for field in ('noise', 'parameters'):
                    row[field] = module.canonical(row[field])
            self.rules._validate_guard_batch(self.generation, serialized)

    def test_noise_beyond_hard_ulp_budget_is_rejected_even_with_consistent_hash(self):
        self.rejected(self.shifted(module.GAUSSIAN_NOISE_MAX_ULPS+1))
        rows = self.shifted(0)
        for row in rows[:2]:
            row['noise'][0] += 1e-12
            row['parameters'] = self.rules._candidate_parameters(self.generation['center'], row['noise'], row['sign'])
            row['parameters_hash'] = module.fingerprint(row['parameters'])
        self.rejected(rows)

    def test_pair_noise_must_agree_exactly_within_the_allowed_host_drift(self):
        rows = self.shifted(1)
        rows[1] = copy.deepcopy(self.jobs[1])
        self.rejected(rows)

    def test_candidate_coordinate_and_hash_receive_no_tolerance(self):
        rows = self.shifted(1)
        rows[0]['parameters'][0] = math.nextafter(rows[0]['parameters'][0], math.inf)
        rows[0]['parameters_hash'] = module.fingerprint(rows[0]['parameters'])
        self.rejected(rows)
        rows = self.shifted(1); rows[0]['parameters_hash'] = self.jobs[0]['parameters_hash']
        self.rejected(rows)

    def test_assignment_identity_and_dimensions_remain_exact(self):
        for field, value in [('seed', self.jobs[0]['seed']+1), ('sign', -1), ('pair_id', 'g0-p99'),
                             ('generation', 1), ('noise', []), ('noise', [True])]:
            rows = self.shifted(1); rows[0][field] = value
            self.rejected(rows)

    def test_comparison_vectors_and_recorded_scores_remain_exact_after_noise_drift(self):
        rows = self.shifted(1)
        for row in rows:
            row.update(state='completed', score=row['parameters'][0])
        transition = self.rules._guard_transition(self.generation, rows, 1001.)
        current = transition['generation']; comparisons = transition['jobs']
        self.rules._validate_guard_batch(current, rows+comparisons)
        changed = copy.deepcopy(comparisons)
        changed[0]['parameters'][0] = math.nextafter(changed[0]['parameters'][0], math.inf)
        changed[0]['parameters_hash'] = module.fingerprint(changed[0]['parameters'])
        self.rejected(rows+changed, current)
        for row in comparisons:
            row.update(state='completed', score=.1 if row['sign'] == 1 else 0.)
        finished = self.rules._guard_transition(current, rows+comparisons, 1002.)['generation']
        self.rules._validate_guard_batch(finished, rows+comparisons)
        changed = copy.deepcopy(finished)
        changed['acceptance']['meanPairedDifference'] = math.nextafter(changed['acceptance']['meanPairedDifference'], math.inf)
        self.rejected(rows+comparisons, changed)


class SQLiteGuardTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name)/'guard.sqlite3'
        self.config = fixture()
        self.now = 1000.
        self.coordinator = module.TrainingCoordinator(self.path, self.config, clock=lambda: self.now)

    def tearDown(self):
        self.coordinator.close()
        self.directory.cleanup()

    def lease(self, owner):
        return self.coordinator.lease(identity(self.coordinator, owner))['job']

    def generation(self, number=0):
        value = dict(self.coordinator.db.execute('SELECT * FROM generations WHERE generation=?', (number,)).fetchone())
        return self.coordinator._normalize_guard_generation(value)

    def rows(self, number=0):
        return [dict(row) for row in self.coordinator.db.execute('SELECT * FROM jobs WHERE generation=? ORDER BY pair_id,sign DESC', (number,))]

    def search(self, score=None):
        values = []
        for index in range(8):
            owner = f's{index}'
            job = self.lease(owner)
            payload = result(self.coordinator, job, owner, score(job) if score else job['parameters'][0])
            response = self.coordinator.result(payload)
            self.assertFalse(response['nextGenerationCreated'])
            values.append((job, payload))
            self.assertEqual(self.coordinator.checkpoint()['parameters'], [0.])
        self.assertEqual(self.generation()['status'], 'checking')
        return values

    def comparison(self, differences):
        values = []
        for index in range(6):
            owner = f'a{index}'
            job = self.lease(owner)
            pair = int(job['pairId'].rsplit('a', 1)[1])
            reward = differences[pair] if job['sign'] == 1 else 0.
            payload = result(self.coordinator, job, owner, reward)
            response = self.coordinator.result(payload)
            self.assertEqual(response['nextGenerationCreated'], index == 5)
            values.append((job, payload))
        return values

    def test_search_barrier_best_nomination_and_persisted_schema(self):
        values = self.search()
        generation = self.generation()
        nominated = min(values, key=lambda item: (-item[1]['objective'], item[0]['jobId']))[0]
        self.assertEqual(generation['acceptance']['proposalJobId'], nominated['jobId'])
        self.assertEqual(generation['acceptance']['proposedParameters'], nominated['parameters'])
        self.assertEqual(len(generation['acceptance']['comparisonJobIds']), 6)
        self.assertEqual(generation['acceptance']['decision'], 'pending')
        self.assertEqual(self.coordinator.status()['generationPhase'], 'comparison')
        self.assertEqual(self.coordinator.status()['totalJobs'], 14)
        self.assertEqual(self.coordinator.db.execute("SELECT value FROM meta WHERE key='schemaVersion'").fetchone()[0], '2')
        self.assertEqual(self.coordinator.db.execute('SELECT COUNT(*) FROM generations').fetchone()[0], 1)
        comparisons = self.coordinator._validate_guard_batch(generation, self.rows())['comparison']
        for row in comparisons:
            parameters = json.loads(row['parameters'])
            self.assertEqual(parameters, nominated['parameters'] if row['sign'] == 1 else [0.])

    def test_positive_mean_accepts_despite_two_losing_seeds(self):
        self.search()
        proposed = self.generation()['acceptance']['proposedParameters']
        self.coordinator._next_center = lambda *_: (_ for _ in ()).throw(AssertionError('Guard cannot aggregate comparison rows as ES'))
        self.comparison([1., -.2, -.2])
        decision = self.generation()['acceptance']
        self.assertEqual(decision['decision'], 'accepted')
        self.assertAlmostEqual(decision['meanPairedDifference'], .2)
        self.assertEqual(decision['pairedDifferences'], [1., -.2, -.2])
        self.assertEqual(self.coordinator.checkpoint()['parameters'], proposed)
        self.assertEqual(self.coordinator.checkpoint()['generation'], 1)
        self.assertEqual(self.coordinator.status()['acceptedResults'], 14)
        self.assertEqual(self.coordinator.checkpoint()['status'], 'unverified')
        self.assertFalse(self.coordinator.checkpoint()['heldOutValidated'])
        self.assertEqual(self.coordinator._guard_selected_center(self.generation(), self.rows()), proposed)

    def test_ties_and_worse_means_retain_incumbent_and_advance(self):
        for differences in ([0., 0., 0.], [-.2, .1, -.2]):
            with self.subTest(differences=differences):
                self.search()
                self.comparison(differences)
                self.assertEqual(self.coordinator.checkpoint()['parameters'], [0.])
                generation = self.coordinator.checkpoint()['generation']-1
                self.assertEqual(self.generation(generation)['acceptance']['decision'], 'rejected')
                # Start fresh for this fixture's generation-zero helper methods.
                if differences == [0., 0., 0.]:
                    self.coordinator.close()
                    self.path = Path(self.directory.name)/'second.sqlite3'
                    self.coordinator = module.TrainingCoordinator(self.path, self.config, clock=lambda: self.now)

    def test_deterministic_tie_nomination(self):
        values = self.search(lambda _: 0.)
        self.assertEqual(self.generation()['acceptance']['proposalJobId'], min(job['jobId'] for job, _ in values))

    def test_restart_preserves_comparison_lease_and_incumbent(self):
        values = self.search()
        lease = self.lease('resume')
        self.coordinator.close()
        self.coordinator = module.TrainingCoordinator(self.path, self.config, clock=lambda: self.now)
        self.assertEqual(self.lease('resume'), lease)
        self.assertEqual(self.coordinator.checkpoint()['parameters'], [0.])
        self.assertTrue(self.coordinator.result(values[-1][1])['duplicate'])
        self.assertEqual(self.coordinator.status()['acceptedResults'], 8)

    def test_duplicate_results_across_both_barriers_are_noops(self):
        search = self.search()
        self.assertTrue(self.coordinator.result(search[-1][1])['duplicate'])
        comparisons = self.comparison([.1, .1, .1])
        for _, payload in (search[-1], comparisons[-1]):
            self.assertTrue(self.coordinator.result(payload)['duplicate'])
            changed = copy.deepcopy(payload); changed['objective'] += 1
            with self.assertRaises(module.APIError) as caught:
                self.coordinator.result(changed)
            self.assertEqual(caught.exception.code, 'result_conflict')
        self.assertEqual(self.coordinator.status()['acceptedResults'], 14)

    def test_concurrent_final_results_create_each_phase_once(self):
        jobs = [(f's{i}', self.lease(f's{i}')) for i in range(8)]
        payloads = [result(self.coordinator, job, owner, job['parameters'][0]) for owner, job in jobs]
        with ThreadPoolExecutor(max_workers=8) as pool:
            replies = list(pool.map(self.coordinator.result, payloads+payloads[-1:]*3))
        self.assertEqual(sum(not value['duplicate'] for value in replies), 8)
        self.assertEqual(self.coordinator.status()['totalJobs'], 14)
        jobs = [(f'a{i}', self.lease(f'a{i}')) for i in range(6)]
        payloads = [result(self.coordinator, job, owner, .1 if job['sign'] == 1 else 0.) for owner, job in jobs]
        with ThreadPoolExecutor(max_workers=6) as pool:
            replies = list(pool.map(self.coordinator.result, payloads+payloads[-1:]*3))
        self.assertEqual(sum(value.get('nextGenerationCreated', False) for value in replies), 1)
        self.assertEqual(self.coordinator.status()['acceptedResults'], 14)
        self.assertEqual(self.coordinator.db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0], 22)

    def test_comparison_expiry_release_and_stale_token(self):
        self.search()
        job = self.lease('expired'); payload = result(self.coordinator, job, 'expired', 1.)
        self.now += 11
        new = self.lease('replacement')
        self.assertEqual(new['jobId'], job['jobId'])
        with self.assertRaises(module.APIError) as caught:
            self.coordinator.result(payload)
        self.assertEqual(caught.exception.code, 'stale_lease')
        credentials = {**identity(self.coordinator, 'replacement'), 'jobId': new['jobId'], 'leaseToken': new['leaseToken']}
        self.assertTrue(self.coordinator.heartbeat(credentials)['renewed'])
        self.assertTrue(self.coordinator.release(credentials)['released'])
        self.assertTrue(self.coordinator.release(credentials)['duplicate'])

    def test_all_guard_jobs_reject_wrong_execution_or_incomplete_episodes(self):
        job = self.lease('owner'); payload = result(self.coordinator, job, 'owner', 1.)
        mutations = [lambda p: p['provenance'].update(backend='wasm'),
                     lambda p: p['provenance'].update(neuralEngine='browser'),
                     lambda p: p['provenance']['nativeWebGPU'].update(moduleSha256='f'*64),
                     lambda p: p['provenance']['nativeWebGPU']['adapter'].update(isFallbackAdapter=True),
                     lambda p: p['metrics'].update(reason='simulation_error'),
                     lambda p: p['metrics'].update(reason='invalid_observation'),
                     lambda p: p['metrics'].pop('reason'),
                     lambda p: p['metrics'].update(reason='unknown'),
                     lambda p: p['metrics'].update(reason='stage_success', success=False),
                     lambda p: p['metrics'].update(reason='time_limit', success=True),
                     lambda p: p['metrics'].update(reason='time_limit', terminated=True),
                     lambda p: p['metrics'].update(reason='stage_success', success=True, terminated=False),
                     lambda p: p['metrics'].update(cancelled=True),
                     lambda p: p['metrics'].update(simSeconds=.1, steps=50, reason='overturned', terminated=False),
                     lambda p: p['metrics'].update(simSeconds=.1, steps=50, reason='incomplete'),
                     lambda p: p['metrics'].update(steps=1)]
        for mutate in mutations:
            altered = copy.deepcopy(payload); mutate(altered)
            with self.assertRaises(module.APIError):
                self.coordinator.result(altered)
            self.assertEqual(self.coordinator.status()['acceptedResults'], 0)
        payload['metrics'].update(simSeconds=.1, steps=50, reason='excessive_rotation', terminated=True)
        self.assertTrue(self.coordinator.result(payload)['accepted'])

    def test_missing_job_or_tampered_pending_decision_fails_closed(self):
        self.search()
        generation = self.generation()
        tampered = copy.deepcopy(generation); tampered['acceptance']['proposedParameters'][0] += .01
        with self.assertRaises(module.APIError):
            self.coordinator._validate_guard_batch(tampered, self.rows())
        with self.assertRaises(module.APIError):
            self.coordinator._validate_guard_batch(generation, self.rows()[:-1])
        extra = copy.deepcopy(self.rows()[0]); extra['job_id'] = 'g0-unexpected'
        with self.assertRaises(module.APIError):
            self.coordinator._validate_guard_batch(generation, self.rows()+[extra])

    def test_finished_decision_tampering_fails_recomputation(self):
        self.search(); self.comparison([.2, .1, .3])
        generation = self.generation(); rows = self.rows()
        self.coordinator._validate_guard_batch(generation, rows)
        for key, value in [('decision', 'rejected'), ('candidateMean', 9.), ('meanPairedDifference', -.2), ('pairedDifferences', [0., 0., 0.])]:
            altered = copy.deepcopy(generation); altered['acceptance'][key] = value
            with self.assertRaises(module.APIError):
                self.coordinator._validate_guard_batch(altered, rows)

    def test_identical_proposal_skips_comparison_and_records_noop(self):
        self.coordinator.close()
        config = fixture(); config['parameters'][0].update(initial=1., searchScale=5e-324)
        self.path = Path(self.directory.name)/'identical.sqlite3'
        self.coordinator = module.TrainingCoordinator(self.path, config, clock=lambda: self.now)
        for i in range(8):
            job = self.lease(str(i))
            response = self.coordinator.result(result(self.coordinator, job, str(i), 1.))
            self.assertEqual(response['nextGenerationCreated'], i == 7)
        acceptance = self.generation()['acceptance']
        self.assertEqual(acceptance['decision'], 'identical_parameters')
        self.assertEqual(acceptance['comparisonSeeds'], [])
        self.assertEqual(acceptance['comparisonJobIds'], [])
        self.assertIsNone(acceptance['candidateMean'])
        self.assertEqual(self.coordinator.checkpoint()['parameters'], [1.])
        self.coordinator._validate_guard_batch(self.generation(), self.rows())

    def test_guard_config_cannot_mutate_or_resume_a_vanilla_database(self):
        path = Path(self.directory.name)/'vanilla.sqlite3'
        vanilla = module.TrainingCoordinator(path, fixture(False), clock=lambda: self.now)
        before = [tuple(row) for row in vanilla.db.execute('PRAGMA table_info(generations)')]
        with self.assertRaises(ValueError):
            module.TrainingCoordinator(path, self.config, clock=lambda: self.now)
        self.assertEqual([tuple(row) for row in vanilla.db.execute('PRAGMA table_info(generations)')], before)
        self.assertEqual(vanilla.checkpoint()['parameters'], [0.])
        vanilla.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
