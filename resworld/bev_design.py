"""Derive candidate BEV extents from an explicit road/motion design envelope.

Road design sight distances are engineering references, not neural-network BEV
requirements. See BEV_DESIGN.md for sources, applicability, and assumptions.
No dataset samples or labels are read by this module.
"""
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path


@dataclass
class RoadEnvelope:
    road_type: str = 'highway_reference'
    design_speed_kmh: int = 60
    planning_seconds: float = 3.
    total_motor_lanes: int = 6
    median_width_m: float = 3.
    roadside_width_m: float = 5.
    include_cross_traffic: bool = True
    cross_speed_kmh: float = 60.
    cross_interaction_seconds: float = 3.
    rear_closing_speed_mps: float = 10.
    lane_change_seconds: float = 3.
    front_margin_m: float = 5.
    side_margin_m: float = 5.
    rear_margin_m: float = 5.
    grid_spacing_m: float = 1.

    def derive(self):
        # JTG B01-2014 tables 4.0.2 and 4.0.15-1/-2. For this road class only.
        stopping = {20:20., 30:30., 40:40., 60:75., 80:110., 100:160., 120:210.}
        lanes = {20:3., 30:3.25, 40:3.5, 60:3.5, 80:3.75, 100:3.75, 120:3.75}
        standard, lane_table, stopping_table = 'JTG B01-2014', '4.0.2', '4.0.15-1/-2'
        source = 'https://xxgk.mot.gov.cn/2020/jigou/glj/202006/P020230628577830212881.pdf'
        if self.road_type == 'urban_reference':
            # Urban design reference, not a statement of mandatory ADS requirements.
            stopping = {20:20., 30:30., 40:40., 50:60., 60:70., 80:110., 100:160.}
            lanes = {speed:3.5 if speed <= 60 else 3.75 for speed in stopping}
            standard, lane_table = 'CJJ 37-2012 (2016 edition), mixed-traffic lane reference', '5.3.2'
            stopping_table = 'urban stopping sight-distance reference'
            source = 'https://zrzyt.zj.gov.cn/module/download/downfile.jsp?classid=0&filename=a41ed183b99444258d2c9f4d1282de5e.pdf'
        elif self.road_type != 'highway_reference':
            raise ValueError('unknown road reference type')
        if self.design_speed_kmh not in stopping:
            raise ValueError('choose a design speed covered by the cited JTG reference table')
        if self.total_motor_lanes < 1 or self.grid_spacing_m <= 0:
            raise ValueError('lane count and grid spacing must be positive')
        assumptions = asdict(self)
        if any(value < 0 for key,value in assumptions.items()
               if isinstance(value, (float, int)) and not isinstance(value, bool)):
            raise ValueError('design distances, speeds, and times must be nonnegative')
        lane = lanes[self.design_speed_kmh]
        travel = self.design_speed_kmh/3.6*self.planning_seconds
        front = max(travel, stopping[self.design_speed_kmh]) + self.front_margin_m
        # Ego can be in an outer lane, not necessarily on the road centreline.
        road_side = ((self.total_motor_lanes-.5)*lane + self.median_width_m + self.roadside_width_m)
        cross_side = (self.cross_speed_kmh/3.6*self.cross_interaction_seconds
                      if self.include_cross_traffic else 0.)
        side = max(road_side, cross_side) + self.side_margin_m
        rear = self.rear_closing_speed_mps*self.lane_change_seconds + self.rear_margin_m
        # Round coverage outward in 5m increments and raster endpoints outward.
        front, side, rear = (math.ceil(distance/5)*5. for distance in (front, side, rear))
        height = math.ceil((front+rear)/self.grid_spacing_m)+1
        width = math.ceil(2*side/self.grid_spacing_m)+1
        return dict(
            status='engineering_candidate_not_a_standard_mandated_bev',
            reference=dict(standard=standard, lane_table=lane_table,
                           stopping_table=stopping_table, lane_width_m=lane,
                           stopping_sight_distance_m=stopping[self.design_speed_kmh],
                           source=source),
            assumptions=assumptions,
            calculations=dict(planning_travel_m=travel, road_side_envelope_m=road_side,
                              cross_traffic_reach_m=cross_side),
            model_grid=dict(x_min=-rear, x_max=front, y_min=-side, y_max=side,
                            bev_height=height, bev_width=width))


def main():
    envelope = RoadEnvelope()
    derivation = envelope.derive()
    directory = Path(__file__).resolve().parents[1]
    (directory/'configs/bev_road60_design.json').write_text(json.dumps(derivation, indent=2))
    # Keep perception capacity unchanged from the larger prototype; only derive its grid here.
    config = json.loads((directory/'configs/bev_attention_full.json').read_text())
    config.update(derivation['model_grid'])
    config.update(depth_max=100., depth_bins=70, temporal_offset_meters=4.)
    (directory/'configs/bev_attention_road60.json').write_text(json.dumps(config, indent=2))
    urban = RoadEnvelope(road_type='urban_reference', grid_spacing_m=.5).derive()
    (directory/'configs/bev_city60_design.json').write_text(json.dumps(urban, indent=2))
    config.update(urban['model_grid'])
    (directory/'configs/bev_attention_city60.json').write_text(json.dumps(config, indent=2))
    pilot_grid = RoadEnvelope(road_type='urban_reference', grid_spacing_m=1.).derive()['model_grid']
    config.update(pilot_grid, channels=32, tokens=8, heads=4, layers=1, spatial_layers=1)
    (directory/'configs/bev_attention_city60_pilot.json').write_text(json.dumps(config, indent=2))
    print(json.dumps(derivation, indent=2))


if __name__ == '__main__':
    main()
