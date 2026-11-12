"""Scene-disjoint route-balanced pilot selection within a past-speed envelope.

Speed filtering is a driving-condition proxy, not a validated road-type label.
Full route labels are required. This balanced pilot is not the official benchmark.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

import torch
from torch.utils.data import Subset

from .driving_data import DrivingDataset
from .nuscenes_io import ego_pose_to_planar_matrix


def apply_selection(dataset, tokens):
    if not tokens or len(tokens) != len(set(tokens)):
        raise ValueError('selected tokens must be nonempty and unique')
    indices = {token:i for i,token in enumerate(dataset.entries)}
    missing = set(tokens) - set(indices)
    if missing:
        raise ValueError(f'{len(missing)} selected tokens do not belong to the requested official split')
    return Subset(dataset, [indices[token] for token in tokens])


def past_speed(dataset, token):
    previous = dataset.samples[token]['prev']
    current_pose = ego_pose_to_planar_matrix(dataset.reference_pose(token))
    previous_pose = ego_pose_to_planar_matrix(dataset.reference_pose(previous))
    dt = (dataset.samples[token]['timestamp']-dataset.samples[previous]['timestamp']) / 1e6
    translation = torch.linalg.solve(current_pose, previous_pose)[:2, 2]
    return translation.norm().item()/dt


def balanced_scene_sample(candidates, per_route, seed):
    """Round-robin distinct scenes before taking more windows from one scene."""
    rng = random.Random(seed)
    selected = []
    for command in range(3):
        by_scene = defaultdict(list)
        for row in candidates:
            if row['command'] == command:
                by_scene[row['scene']].append(row)
        if sum(map(len, by_scene.values())) < per_route:
            raise ValueError(f'not enough candidates for command {command}')
        scene_names = sorted(by_scene)
        rng.shuffle(scene_names)
        for rows in by_scene.values():
            rng.shuffle(rows)
        route_selected = []
        while len(route_selected) < per_route:
            for scene in scene_names:
                if by_scene[scene]:
                    route_selected.append(by_scene[scene].pop())
                    if len(route_selected) == per_route:
                        break
        selected.extend(route_selected)
    rng.shuffle(selected)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='/mnt/ssd0/datasets/nuscenes')
    parser.add_argument('--train-per-route', type=int, default=16)
    parser.add_argument('--val-per-route', type=int, default=8)
    parser.add_argument('--speed-kmh', type=float, default=60.)
    parser.add_argument('--all-eligible', action='store_true', help='retain all eligible sequences for the full phase-one run')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', default='runs/city_phase1/selection.json')
    args = parser.parse_args()
    if min(args.train_per_route, args.val_per_route, args.speed_kmh) <= 0:
        raise ValueError('route counts and speed limit must be positive')
    torch.set_num_threads(4)
    manifest = dict(version='v1.0-trainval', seed=args.seed,
                    definition=('full future labels; past ego speed <= limit; all eligible sequences'
                                if args.all_eligible else
                                'full future labels; past ego speed <= limit; route-balanced scene-diverse pilot'),
                    limitations=['speed proxy does not verify urban road class',
                                 'no following or pedestrian-interaction ground-truth labels',
                                 'balanced pilot metrics differ from official validation distribution'],
                    max_past_speed_kmh=args.speed_kmh, train=[], val=[], summary={}, records={})
    scene_sets = []
    for split, count in (('train', args.train_per_route), ('val', args.val_per_route)):
        print(f'Preparing {split} candidates...', flush=True)
        dataset = DrivingDataset(args.root, 'v1.0-trainval', split)
        candidates = []
        for token in dataset.entries:
            _, valid, command = dataset.labels(token)
            if not valid.all():
                continue
            speed = past_speed(dataset, token)
            if speed > args.speed_kmh/3.6:
                continue
            candidates.append(dict(token=token, command=command, speed_mps=speed,
                                   scene=dataset.scene_names[dataset.samples[token]['scene_token']]))
        rows = candidates if args.all_eligible else balanced_scene_sample(candidates, count, args.seed)
        manifest[split] = [row['token'] for row in rows]
        manifest['records'][split] = rows
        scenes = {r['scene'] for r in rows}
        scene_sets.append(scenes)
        manifest['summary'][split] = dict(candidates=len(candidates), selected=len(rows),
            scene_count=len(scenes), route_counts=dict(Counter(r['command'] for r in rows)))
        print(json.dumps({split:manifest['summary'][split]}), flush=True)
    assert scene_sets[0].isdisjoint(scene_sets[1])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
