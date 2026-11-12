"""Audit available files, labels, and a train/validation item before a long run."""
import argparse
import json
from pathlib import Path

import torch

from resworld.driving_data import DrivingDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='/mnt/ssd0/datasets/nuscenes')
    parser.add_argument('--version', default='v1.0-mini')
    parser.add_argument('--output', default='runs/data_audit.json')
    args = parser.parse_args()
    torch.set_num_threads(4)
    summary = {}
    scene_sets = []
    for split in ('train', 'val'):
        print(f'Loading and checking {args.version}/{split}...', flush=True)
        ds = DrivingDataset(args.root, args.version, split)
        commands = [0, 0, 0]
        full = 0
        for token in ds.entries:
            positions, mask, command = ds.labels(token)
            assert mask.any() and torch.isfinite(positions).all()
            assert not ((~mask[:-1]) & mask[1:]).any()
            commands[command] += 1
            full += int(mask.all())
        item = ds[0]
        assert torch.isfinite(item['images']).all()
        names = sorted({ds.scene_names[ds.samples[t]['scene_token']] for t in ds.entries})
        scene_sets.append(set(names))
        summary[split] = dict(sequences=len(ds), full_horizon=full, commands=commands,
                             scenes=names, missing_file_sequences=ds.missing_files,
                             image_shape=list(item['images'].shape),
                             supervised_depth_pixels=int((item['depth_targets'] >= 0).sum()))
        print(json.dumps({split: {k:v for k,v in summary[split].items() if k != 'scenes'}}), flush=True)
    assert scene_sets[0].isdisjoint(scene_sets[1])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(version=args.version, **summary), indent=2))


if __name__ == '__main__':
    main()
