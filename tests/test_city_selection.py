import unittest
from collections import Counter

from resworld.city_selection import apply_selection, balanced_scene_sample
from resworld.bev_design import RoadEnvelope


class Entries:
    entries = ['train-a', 'train-b', 'train-c']


class CitySelectionTests(unittest.TestCase):
    def test_rejects_tokens_outside_split_and_duplicates(self):
        with self.assertRaises(ValueError):
            apply_selection(Entries(), ['validation-token'])
        with self.assertRaises(ValueError):
            apply_selection(Entries(), ['train-a', 'train-a'])
        subset = apply_selection(Entries(), ['train-c', 'train-a'])
        self.assertEqual(subset.indices, [2, 0])

    def test_balanced_sampling_is_reproducible_and_scene_diverse(self):
        candidates = [dict(token=f'{command}-{scene}-{i}', command=command, scene=scene)
                      for command in range(3) for scene in ('a', 'b', 'c') for i in range(4)]
        selected = balanced_scene_sample(candidates, 3, seed=12)
        self.assertEqual(selected, balanced_scene_sample(candidates, 3, seed=12))
        self.assertEqual(Counter(row['command'] for row in selected), {0:3, 1:3, 2:3})
        for command in range(3):
            self.assertEqual({row['scene'] for row in selected if row['command']==command}, {'a', 'b', 'c'})

    def test_missing_route_fails_instead_of_silently_training_unbalanced_set(self):
        with self.assertRaises(ValueError):
            balanced_scene_sample([dict(token='a', command=1, scene='s')], 1, 0)

    def test_urban_reference_is_distinct_from_highway_reference(self):
        urban = RoadEnvelope(road_type='urban_reference', grid_spacing_m=.5).derive()
        highway = RoadEnvelope().derive()
        self.assertEqual(urban['reference']['stopping_sight_distance_m'], 70)
        self.assertEqual(urban['model_grid']['x_max'], 75)
        self.assertEqual(urban['model_grid']['bev_height'], 221)
        self.assertEqual(urban['model_grid']['bev_width'], 221)
        self.assertNotEqual(urban['reference']['standard'], highway['reference']['standard'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
