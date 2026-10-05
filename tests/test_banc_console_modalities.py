"""Annotation semantics only; no connectivity generation or neural tuning."""
import importlib.util
from collections import Counter
import copy
import json
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('banc_console', ROOT/'scripts/prepare-banc-console.py')
console = importlib.util.module_from_spec(spec)
spec.loader.exec_module(console)


class ConsoleModalityTest(unittest.TestCase):
    def test_sugar_function_and_external_organ_override_broad_class(self):
        for cell_class in ('bristle_neuron', 'taste_bristle_neuron', 'taste_bristle_gustatory_neuron', 'taste_peg_gustatory_neuron'):
            for organ in console.EXTERNAL_TASTE_ORGANS:
                self.assertTrue(console.is_external_sugar_sensor(dict(super_class='sensory', cell_class=cell_class,
                    body_part_sensory=organ, cell_function_detailed='sugar, low_salt, Gr64f, Ir56b')))

    def test_internal_sugar_and_non_sugar_chemistry_do_not_become_food_contact(self):
        row = dict(super_class='sensory', body_part_sensory='front_leg')
        for function in ('hemolymph_sugar, Gr43a', 'low_salt', 'contact_pheromone', 'sugar_response_unknown', None):
            self.assertFalse(console.is_external_sugar_sensor({**row, 'cell_function_detailed': function}))
        self.assertFalse(console.is_external_sugar_sensor({**row, 'cell_function_detailed': 'sugar', 'body_part_sensory': 'hemolymph'}))
        self.assertFalse(console.is_external_sugar_sensor({**row, 'cell_function_detailed': 'sugar', 'super_class': 'motor'}))

    def test_unknown_side_is_preserved_for_mapper_abstention(self):
        self.assertTrue(console.is_external_sugar_sensor(dict(super_class='sensory', cell_class='bristle_neuron',
            side=None, body_part_sensory='labellum', cell_function_detailed='sugar, Gr64f')))

    def test_exact_crossmatched_sugar_identity_takes_precedence_over_generic_type_name(self):
        for evidence in console.SUGAR_IDENTITY_EVIDENCE['rows']:
            row = dict(banc_888_id=evidence['banc_root_id'], fafb_match=evidence['fafb_match'],
                super_class='sensory', cell_class='bristle_neuron', cell_type='BM_Taste', body_part_sensory='labellum',
                side=evidence['banc_side'], cell_function_detailed='sugar, Gr64f')
            self.assertEqual(console.is_external_sugar_sensor(row), evidence['supports_external_sugar'])
            with self.assertRaisesRegex(ValueError, 'identity'):
                console.is_external_sugar_sensor({**row, 'fafb_match': 'different_cell'})

    def test_only_explicit_antennal_auditory_frequency_functions_are_excluded(self):
        row = dict(body_part_sensory='antenna', cell_class='chordotonal_organ_neuron', cell_function='auditory')
        for function in console.AUDITORY_FREQUENCY_FUNCTIONS:
            self.assertIn('vibration', console.auditory_transducer_exclusion_reason({**row, 'cell_function_detailed': function}))
        # D/mixed, C/E/static deflection and generic JO classes must survive;
        # the broad "auditory" field alone is not sufficient for exclusion.
        for function in ('position', 'direction', 'vibro_position', None, 'auditory_high_frequency_unknown'):
            self.assertIsNone(console.auditory_transducer_exclusion_reason({**row, 'cell_function_detailed': function}))
        self.assertIsNone(console.auditory_transducer_exclusion_reason({**row,
            'body_part_sensory': 'front_leg', 'cell_function_detailed': 'auditory_high_frequency'}))

    def test_chemical_sugar_mapping_does_not_restore_collision_drive(self):
        row = dict(super_class='sensory', cell_class='bristle_neuron', body_part_sensory='front_leg',
            cell_function_detailed='sugar, low_salt, Gr64f, Ir56b')
        self.assertTrue(console.is_external_sugar_sensor(row))
        self.assertIn('mechanical collision drive is unsupported', console.body_transducer_exclusion_reason(row))


class ProximalHairPlateExclusionTest(unittest.TestCase):
    ROW = dict(super_class='sensory', side='left', cell_class='hair_plate_neuron',
               cell_sub_class='front_leg_hair_plate_neuron', cell_type='SNpp45',
               body_part_sensory='coxa,front_leg', cell_function_detailed='joint_angle',
               peripheral_target_type='CoHP8,hair_plate', nerve='left_prothoracic_leg_nerve')

    def test_exact_compound_tokens_and_function_are_required(self):
        for part in ('coxa,front_leg', 'front_leg,coxa', ' front_leg , trochanter ', 'coxa, front_leg,coxa'):
            row = {**self.ROW, 'body_part_sensory': part, 'cell_function_detailed': 'other, JOINT_ANGLE '}
            self.assertIn('no verified receptive-joint/axis transfer', console.body_transducer_exclusion_reason(row))
            self.assertIsNotNone(console.body_transducer_exclusion(row, 12, 'root'))
        for part in ('front_leg', 'coxa', 'front_leg_extra,coxa', 'front_leg,coxa_like',
                     'hind_leg,coxa', 'front_leg,middle_leg,coxa', 'front_leg,coxa,trochanter', None):
            self.assertIsNone(console.hair_plate_transducer_exclusion_reason({**self.ROW, 'body_part_sensory': part}))
        for function in ('joint_angle_like', 'position', None):
            self.assertIsNone(console.hair_plate_transducer_exclusion_reason({**self.ROW, 'cell_function_detailed': function}))
        for cls in ('chordotonal_organ_neuron', 'campaniform_sensillum_neuron', 'bristle_neuron'):
            self.assertIsNone(console.hair_plate_transducer_exclusion_reason({**self.ROW, 'cell_class': cls}))

    def test_emitter_preserves_raw_anatomy_without_assigning_a_joint_or_direction(self):
        row = {**self.ROW, 'body_part_sensory': ' front_leg , coxa ', 'peripheral_target_type': 'hair_plate, CoHP8'}
        original = copy.deepcopy(row)
        for side, leg in (('left', 0), ('right', 3)):
            excluded = console.body_transducer_exclusion({**row, 'side': side}, 42, 123)
            self.assertEqual(excluded['index'], 42)
            self.assertEqual(excluded['root_id'], '123')
            self.assertEqual(excluded['side'], side)
            self.assertEqual(excluded['leg'], leg)
            self.assertEqual(excluded['organ'], row['body_part_sensory'])
            self.assertEqual(excluded['body_part_sensory_tokens'], ['coxa', 'front_leg'])
            self.assertEqual(excluded['peripheral_target_type'], row['peripheral_target_type'])
            self.assertEqual(excluded['peripheral_target_type_tokens'], ['CoHP8', 'hair_plate'])
            self.assertEqual(excluded['cell_sub_class'], row['cell_sub_class'])
            self.assertEqual(excluded['nerve'], row['nerve'])
            self.assertTrue({'kind', 'joint', 'axis', 'sign', 'gain'}.isdisjoint(excluded))
        self.assertEqual(row, original)
        for side in (None, 'both', 'unknown'):
            self.assertIsNone(console.body_transducer_exclusion({**row, 'side': side}, 42, 123))
        self.assertIsNotNone(console.body_transducer_exclusion({**row, 'super_class': 'sensory_ascending'}, 42, 123))
        self.assertIsNone(console.body_transducer_exclusion({**row, 'super_class': 'motor'}, 42, 123))

    def test_prior_bristle_and_auditory_emission_rules_remain(self):
        chemical = {**self.ROW, 'cell_class': 'bristle_neuron', 'body_part_sensory': 'front_leg',
                    'cell_function_detailed': 'sugar, low_salt'}
        self.assertIn('Explicit chemosensory', console.body_transducer_exclusion(chemical, 1, 'a')['reason'])
        self.assertIsNone(console.body_transducer_exclusion({**chemical, 'body_part_sensory': 'labellum'}, 1, 'a'))
        auditory = {**self.ROW, 'cell_class': 'chordotonal_organ_neuron', 'body_part_sensory': 'antenna',
                    'cell_function_detailed': 'auditory_high_frequency'}
        self.assertEqual(console.body_transducer_exclusion(auditory, 2, 'b')['source'], console.JO_MODALITY_SOURCE)
        self.assertIsNone(console.body_transducer_exclusion({**auditory, 'cell_function_detailed': 'position'}, 2, 'b'))


class RealProximalHairPlateAuditTest(unittest.TestCase):
    # Independently audited prepared indices, bound through ids.bin to the raw
    # BANC annotation. Two additional raw cells have null side and no channel.
    EXPECTED = frozenset((141265, 15918, 45876, 124123, 125703, 11741, 19614, 51050, 140611, 99560,
        42976, 115216, 78273, 93998, 105312, 110882, 149419, 63642, 11590, 63253, 106745, 90320,
        40559, 31162, 137583, 79320, 147824, 51886, 172733, 91542, 113289, 142958, 134694,
        78293, 125151, 6240, 65924, 171630, 34124, 106611, 167954, 83689, 146297, 7620,
        48687, 105612, 15181, 112747, 132069, 2326))

    @classmethod
    def setUpClass(cls):
        raw = ROOT/'data/raw/banc888/meta.feather'
        if not raw.exists():
            raise unittest.SkipTest('Real BANC metadata is not installed')
        fields = ['banc_888_id', 'side', 'super_class', 'cell_class', 'cell_sub_class', 'cell_type',
                  'fafb_cell_type', 'body_part_sensory', 'cell_function_detailed', 'peripheral_target_type', 'nerve']
        source_rows = console.feather.read_table(raw, columns=fields).to_pylist()
        cls.ids = console.np.fromfile(console.DATA/'ids.bin', '<u8')
        by_id = {str(root_id): i for i, root_id in enumerate(cls.ids)}
        cls.rows = [(by_id[row['banc_888_id']], row) for row in source_rows
                    if row['banc_888_id'] in by_id and row['cell_class'] == 'hair_plate_neuron'
                    and row['body_part_sensory'] in ('front_leg,trochanter', 'coxa,front_leg')]
        cls.manifest = json.loads((console.OUT/'sensory-inputs.json').read_text())
        cls.groups = json.loads((console.OUT/'groups.json').read_text())
        cls.exclusions = [item for index, row in cls.rows
                          if (item := console.body_transducer_exclusion(row, index, cls.ids[index])) is not None]

    def test_all_and_only_50_lateralized_real_cells_are_emitted_and_remain_in_channels(self):
        self.assertEqual(len(self.rows), 52)
        self.assertEqual({row['index'] for row in self.exclusions}, self.EXPECTED)
        self.assertEqual(Counter(row['organ'] for row in self.exclusions),
                         {'front_leg,trochanter': 25, 'coxa,front_leg': 25})
        self.assertEqual(Counter(row['side'] for row in self.exclusions), {'left': 27, 'right': 23})
        channels = {index: channel['key'] for channel in self.manifest['channels'] for index in channel['indices']}
        transducers = {row['index'] for row in self.manifest['body_transducers']}
        for item in self.exclusions:
            self.assertEqual(item['root_id'], str(self.ids[item['index']]))
            self.assertEqual(channels[item['index']], 'self_motion_'+item['side'])
            self.assertNotIn(item['index'], transducers)
            self.assertEqual(item['cell_sub_class'], 'front_leg_hair_plate_neuron')
            self.assertEqual(item['function'], 'joint_angle')
            self.assertIn('hair_plate', item['peripheral_target_type_tokens'])
        unassigned = [(index, row) for index, row in self.rows if row['side'] is None]
        self.assertEqual(len(unassigned), 2)
        self.assertTrue(all(index not in channels for index, _ in unassigned))

    def test_real_50_exclusions_block_native_fallback_without_changing_neighbor_inputs(self):
        # Build candidate metadata only in memory. The native encoder is real;
        # there is no MuJoCo/neural step and no prepared-data regeneration.
        program = r'''
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {SensoryEncoder} from './web/sensory-encoder.js';
const {manifest,groups,exclusions}=JSON.parse(fs.readFileSync(0,'utf8'));
const excluded=new Set(exclusions.map(row=>row.index));assert.equal(excluded.size,50);
const baseline={...manifest,body_transducer_exclusions:(manifest.body_transducer_exclusions||[]).filter(row=>!excluded.has(row.index))};
const candidate={...baseline,body_transducer_exclusions:[...baseline.body_transducer_exclusions,...exclusions]};
assert.deepEqual(candidate.channels,manifest.channels);assert.deepEqual(candidate.body_transducers,manifest.body_transducers);
const environment={odor:()=>0,surface:()=>({y:0,contact:false})};
const oldEncoder=new SensoryEncoder(baseline,groups,environment),newEncoder=new SensoryEncoder(candidate,groups,environment);
const neighbors=manifest.body_transducers.filter(row=>[0,3].includes(row.leg)&&['position','velocity','load'].includes(row.kind));
assert(neighbors.some(row=>row.kind==='position'));assert(neighbors.some(row=>row.kind==='velocity'));
let resting;
for(const moving of [false,true]){
 const legs=Array.from({length:6},()=>({loadBodyWeights:moving?.2:0,support:moving?1:0,collision:0,
  tibiaAngle:moving?.6:0,coxaAngle:moving?.2:0,tibiaVelocity:moving?4:0,vibration:0,angle:moving?.8:0,speed:moving?4:0}));
 const pose={x:0,y:0,z:0,heading:0,bodyTime:moving?.002:0,contact:false,feedback:{legs,speed:moving?20:0,yaw:moving?3:0}};
 const options={odor:false,taste:false,vision:false},old=oldEncoder.update(pose,null,options),next=newEncoder.update(pose,null,options);
 assert.deepEqual(next.indices,old.indices);
 const oldRates=new Map(Array.from(old.indices,(index,k)=>[index,old.ratesHz[k]]));
 const rates=new Map(Array.from(next.indices,(index,k)=>[index,next.ratesHz[k]]));
 for(const index of excluded){assert.equal(rates.get(index),0);if(moving)assert(oldRates.get(index)>0,'Baseline must exercise the bad aggregate fallback');}
 for(const [index,rate]of rates)if(!excluded.has(index))assert.equal(rate,oldRates.get(index),'A neighboring input changed');
 if(!moving)resting=rates;
 else for(const neighbor of neighbors)assert(rates.get(neighbor.index)>resting.get(neighbor.index),'Supported native leg receptor stopped responding');
}
console.log(JSON.stringify({excluded:excluded.size,responsiveNeighbors:neighbors.length}));
'''
        result = subprocess.run(['node', '--input-type=module', '-e', program], cwd=ROOT, text=True,
                                input=json.dumps({'manifest': self.manifest, 'groups': self.groups,
                                                  'exclusions': self.exclusions}),
                                capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertEqual(json.loads(result.stdout)['excluded'], 50)


if __name__ == '__main__':
    unittest.main()
