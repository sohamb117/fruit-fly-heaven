"""Annotation semantics only; no connectivity generation or neural tuning."""
import importlib.util
from pathlib import Path
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


if __name__ == '__main__':
    unittest.main()
