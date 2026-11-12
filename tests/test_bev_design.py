import unittest

from resworld.bev_design import RoadEnvelope
from resworld.temporal_alignment import BEVGrid


class RoadEnvelopeTests(unittest.TestCase):
    def test_reference_and_assumptions_are_separate(self):
        result = RoadEnvelope().derive()
        self.assertEqual(result['reference']['lane_width_m'], 3.5)
        self.assertEqual(result['reference']['stopping_sight_distance_m'], 75)
        self.assertEqual(result['calculations']['planning_travel_m'], 50)
        self.assertEqual(result['model_grid']['x_max'], 80)
        self.assertNotIn('dataset', result['assumptions'])

    def test_intersection_observation_exceeds_road_width(self):
        road = RoadEnvelope(include_cross_traffic=False).derive()['model_grid']
        intersection = RoadEnvelope(include_cross_traffic=True).derive()['model_grid']
        self.assertEqual(road['y_max'], 35)
        self.assertEqual(intersection['y_max'], 55)

    def test_higher_design_speed_expands_forward_range(self):
        low = RoadEnvelope(design_speed_kmh=60).derive()['model_grid']
        high = RoadEnvelope(design_speed_kmh=80).derive()['model_grid']
        self.assertEqual(high['x_max'], 115)
        self.assertGreater(high['x_max'], low['x_max'])

    def test_derived_raster_preserves_requested_spacing(self):
        result = RoadEnvelope(grid_spacing_m=.5).derive()['model_grid']
        grid = BEVGrid(**{k:v for k,v in result.items() if k not in ('bev_height', 'bev_width')},
                       height=result['bev_height'], width=result['bev_width'])
        self.assertEqual((grid.x_max-grid.x_min)/(grid.height-1), .5)
        self.assertEqual((grid.y_max-grid.y_min)/(grid.width-1), .5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
