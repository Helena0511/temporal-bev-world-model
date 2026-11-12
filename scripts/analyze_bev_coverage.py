"""Measure BEV coverage using training labels only; validation does not select bounds."""
import argparse
import json
from pathlib import Path

import torch

from resworld.driving_data import DrivingDataset
from resworld.driving_model import ModelConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='/mnt/ssd0/datasets/nuscenes')
    parser.add_argument('--version', default='v1.0-trainval')
    parser.add_argument('--model-config', default='configs/wide_bev_temporal.json')
    parser.add_argument('--output', default='runs/bev_coverage.json')
    args = parser.parse_args()
    torch.set_num_threads(4)
    dataset = DrivingDataset(args.root, args.version, 'train')
    config = ModelConfig(**json.loads(Path(args.model_config).read_text()))
    positions = []
    for token in dataset.entries:
        trajectory, valid, _ = dataset.labels(token)
        positions.append(trajectory[valid])
    positions = torch.cat(positions)
    levels = torch.tensor([0., .001, .01, .5, .99, .999, 1.])
    normalized = config.grid().source_points_to_grid(positions)
    payload = dict(version=args.version, split='train', sequences=len(dataset),
                   valid_waypoints=len(positions), quantile_levels=levels.tolist(),
                   x_quantiles=torch.quantile(positions[:, 0], levels).tolist(),
                   y_quantiles=torch.quantile(positions[:, 1], levels).tolist(),
                   outside_fraction=(normalized.abs() > 1).any(-1).float().mean().item(),
                   grid=dict(x=[config.x_min, config.x_max], y=[config.y_min, config.y_max],
                             x_spacing=(config.x_max-config.x_min)/(config.bev_height-1),
                             y_spacing=(config.y_max-config.y_min)/(config.bev_width-1)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
