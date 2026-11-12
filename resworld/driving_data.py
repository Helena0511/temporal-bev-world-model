"""Raw nuScenes planning data with calibrated cameras and scene-disjoint splits."""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .nuscenes_io import CAMERAS, ego_pose_to_planar_matrix, get_current2history


MINI_TRAIN = {'scene-0061', 'scene-0553', 'scene-0655', 'scene-0757',
              'scene-0796', 'scene-1077', 'scene-1094', 'scene-1100'}
MINI_VAL = {'scene-0103', 'scene-0916'}


def iter_json_rows(path):
    """Stream a JSON array; trainval sample_data is >1GB before Python decoding."""
    decoder = json.JSONDecoder()
    with Path(path).open() as file:
        buffer, position, eof, started = '', 0, False, False
        while True:
            if position >= len(buffer) and not eof:
                buffer, position = file.read(1024*1024), 0
                eof = not buffer
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position >= len(buffer):
                if eof:
                    raise ValueError(f'incomplete JSON array: {path}')
                continue
            if not started:
                if buffer[position] != '[':
                    raise ValueError(f'expected JSON array: {path}')
                started = True
                position += 1
                continue
            if buffer[position] == ']':
                return
            if buffer[position] == ',':
                position += 1
                continue
            try:
                row, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if eof:
                    raise
                tail = file.read(1024*1024)
                buffer, position, eof = buffer[position:] + tail, 0, not tail
                continue
            yield row
            position = end


@lru_cache(maxsize=1)
def load_metadata(root, version):
    directory = Path(root) / version
    def table(name):
        return json.loads((directory / f'{name}.json').read_text())
    samples = {r['token']: r for r in table('sample')}
    calibrations = {r['token']: r for r in table('calibrated_sensor')}
    sensors = {r['token']: r['channel'] for r in table('sensor')}
    records, lidars, pose_tokens = {}, {}, set()
    for row in iter_json_rows(directory/'sample_data.json'):
        if not row['is_key_frame']:
            continue
        channel = sensors[calibrations[row['calibrated_sensor_token']]['sensor_token']]
        if channel in CAMERAS:
            records.setdefault(row['sample_token'], {})[channel] = row
            pose_tokens.add(row['ego_pose_token'])
        elif channel == 'LIDAR_TOP':
            lidars[row['sample_token']] = row
            pose_tokens.add(row['ego_pose_token'])
    poses = {r['token']: r for r in iter_json_rows(directory/'ego_pose.json')
             if r['token'] in pose_tokens}
    scenes = {s['token']: s['name'] for s in table('scene')}
    return samples, calibrations, records, lidars, poses, scenes


def pose_matrix(record):
    """nuScenes wxyz rotation and translation -> sensor/ego to parent, float64."""
    w, x, y, z = record['rotation']
    matrix = torch.eye(4, dtype=torch.float64)
    matrix[:3, :3] = torch.tensor([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=torch.float64)
    matrix[:3, 3] = torch.tensor(record['translation'], dtype=torch.float64)
    return matrix


class DrivingDataset(Dataset):
    def __init__(self, root, version='v1.0-mini', split='train',
                 image_size=(64, 160), future_steps=6, split_file=None,
                 check_files=True, depth_bins=16, depth_min=1., depth_max=35.,
                 input_mode='temporal'):
        if input_mode not in ('temporal', 'current', 'state'):
            raise ValueError('input_mode must be temporal, current, or state')
        self.input_mode = input_mode
        self.root = Path(root)
        self.image_size = tuple(image_size)  # H,W; divisible by encoder stride 8
        self.future_steps = future_steps
        self.depth_bins, self.depth_min, self.depth_max = depth_bins, depth_min, depth_max
        (self.samples, self.calibrations, self.records, self.lidars,
         self.poses, self.scene_names) = load_metadata(str(self.root), version)
        if split_file:
            splits = json.loads(Path(split_file).read_text())
            train_names, val_names = set(splits['train']), set(splits['val'])
            if train_names & val_names:
                raise ValueError('train/val scene names overlap')
        elif version == 'v1.0-mini':
            train_names, val_names = MINI_TRAIN, MINI_VAL
        elif version == 'v1.0-trainval':
            splits = json.loads((Path(__file__).resolve().parents[1]/'configs/nuscenes_splits.json').read_text())
            train_names, val_names = set(splits['train']), set(splits['val'])
        else:
            raise ValueError('trainval requires --split-file with official train/val scene names')
        if split not in ('train', 'val'):
            raise ValueError('split must be train or val')
        allowed = train_names if split == 'train' else val_names
        self.entries = []
        exists = {}
        def has_file(filename):
            if filename not in exists:
                exists[filename] = (self.root / filename).is_file()
            return exists[filename]
        self.missing_files = 0
        for row in self.samples.values():
            if self.scene_names[row['scene_token']] not in allowed:
                continue
            tokens = self.history_tokens(row['token'])
            if tokens is None or not row['next']:
                continue
            if not all(all(c in self.records.get(t, {}) for c in CAMERAS) for t in tokens):
                continue
            if check_files and not all(has_file(self.records[t][c]['filename'])
                                       for t in tokens for c in CAMERAS):
                self.missing_files += 1
                continue
            self.entries.append(row['token'])
        if not self.entries:
            raise RuntimeError(f'no usable {version}/{split} sequences; check data files and splits')

    def history_tokens(self, token):
        result = [token]
        for _ in range(2):
            token = self.samples[token]['prev']
            if not token:
                return None
            result.append(token)
        return result[::-1]

    def reference_pose(self, token):
        return self.poses[self.records[token]['CAM_FRONT']['ego_pose_token']]

    def labels(self, token):
        reference = ego_pose_to_planar_matrix(self.reference_pose(token))
        points, mask = [], []
        cursor = token
        for _ in range(self.future_steps):
            cursor = self.samples[cursor]['next'] if cursor else ''
            if cursor and 'CAM_FRONT' in self.records.get(cursor, {}):
                pose = self.reference_pose(cursor)
                point = torch.tensor([*pose['translation'][:2], 1.], dtype=torch.float64)
                points.append(torch.linalg.solve(reference, point)[:2].float())
                mask.append(True)
            else:
                points.append(torch.zeros(2))
                mask.append(False)
        positions = torch.stack(points)
        valid = torch.tensor(mask)
        # Oracle route instruction, as in common open-loop nuScenes protocols.
        # Only full-horizon labels define a turn; partial horizons use straight.
        y = positions[-1, 1].item() if valid.all() else 0.
        command = 0 if y > 2 else (2 if y < -2 else 1)
        return positions, valid, command

    def __len__(self):
        return len(self.entries)

    def lidar_global_points(self, token):
        row = self.lidars.get(token)
        if row is None or not (self.root / row['filename']).is_file():
            return None
        points = np.fromfile(self.root / row['filename'], dtype=np.float32).reshape(-1, 5)
        xyz = torch.from_numpy(points[:, :3].copy()).double()
        homogeneous = torch.cat((xyz, torch.ones(len(xyz), 1, dtype=torch.float64)), 1)
        transform = (pose_matrix(self.poses[row['ego_pose_token']]) @
                     pose_matrix(self.calibrations[row['calibrated_sensor_token']]))
        return homogeneous @ transform.T

    def depth_target(self, global_points, sensor_to_global, intrinsics):
        height, width = self.image_size
        target = torch.full((height//8, width//8), -1, dtype=torch.long)
        if global_points is None:
            return target
        camera = (global_points @ torch.linalg.inv(sensor_to_global).T)[:, :3].float()
        projected = camera @ intrinsics.T
        z = camera[:, 2]
        u = projected[:, 0] / z.clamp_min(1e-6)
        v = projected[:, 1] / z.clamp_min(1e-6)
        col, row = (u/8).round().long(), (v/8).round().long()
        keep = ((z >= self.depth_min) & (z <= self.depth_max) &
                (u >= 0) & (u < width) & (v >= 0) & (v < height) &
                (row >= 0) & (row < height//8) & (col >= 0) & (col < width//8))
        # Multiple returns per feature pixel: supervise the nearest surface.
        nearest = torch.full((height//8 * (width//8),), float('inf'))
        indices = row[keep] * (width//8) + col[keep]
        nearest.scatter_reduce_(0, indices, z[keep], reduce='amin', include_self=True)
        valid = torch.isfinite(nearest)
        bins = ((nearest[valid] - self.depth_min) / (self.depth_max-self.depth_min)
                * (self.depth_bins-1)).round().long()
        target.flatten()[valid] = bins
        return target

    def __getitem__(self, index):
        token = self.entries[index]
        tokens = self.history_tokens(token)
        frames, intrinsics, extrinsics, static_extrinsics, depth_targets = [], [], [], [], []
        height, width = self.image_size
        mean = torch.tensor([.485, .456, .406])[:, None, None]
        std = torch.tensor([.229, .224, .225])[:, None, None]
        image_tokens = tokens if self.input_mode == 'temporal' else (tokens[-1:] if self.input_mode == 'current' else [])
        for time_token in image_tokens:
            reference = self.reference_pose(time_token)
            global_to_ref = torch.linalg.inv(pose_matrix(reference))
            views, ks, es, static_es, targets = [], [], [], [], []
            global_points = self.lidar_global_points(time_token)
            for camera in CAMERAS:
                row = self.records[time_token][camera]
                calibration = self.calibrations[row['calibrated_sensor_token']]
                with Image.open(self.root / row['filename']) as image:
                    old_w, old_h = image.size
                    array = np.asarray(image.convert('RGB').resize(
                        (width, height), Image.Resampling.BILINEAR), dtype=np.float32).copy()
                views.append((torch.from_numpy(array).permute(2, 0, 1) / 255 - mean) / std)
                k = torch.tensor(calibration['camera_intrinsic'], dtype=torch.float32)
                k[0] *= width / old_w
                k[1] *= height / old_h
                ks.append(k)
                # Each camera has its own exposure timestamp and ego pose.
                sensor_to_global = pose_matrix(self.poses[row['ego_pose_token']]) @ pose_matrix(calibration)
                es.append((global_to_ref @ sensor_to_global).float())
                static_es.append(pose_matrix(calibration).float())
                targets.append(self.depth_target(global_points, sensor_to_global, k))
            frames.append(torch.stack(views))
            intrinsics.append(torch.stack(ks))
            extrinsics.append(torch.stack(es))
            static_extrinsics.append(torch.stack(static_es))
            depth_targets.append(torch.stack(targets))
        states = [ego_pose_to_planar_matrix(self.reference_pose(t)) for t in tokens]
        transforms = torch.stack([get_current2history(self.reference_pose(token),
                                  self.reference_pose(t))['current2history'] for t in tokens[:2]])
        # Backward finite differences only: no future motion enters model state.
        motion = torch.zeros(3, 3)
        for t in (1, 2):
            dt = (self.samples[tokens[t]]['timestamp'] - self.samples[tokens[t-1]]['timestamp']) / 1e6
            if dt <= 0:
                raise ValueError('non-increasing history timestamps')
            relative = torch.linalg.solve(states[t-1], states[t])
            motion[t, :2] = relative[:2, 2].float() / dt
            motion[t, 2] = torch.atan2(relative[1, 0], relative[0, 0]).float() / dt
        gt, valid, command = self.labels(token)
        history_dt = (self.samples[tokens[2]]['timestamp'] - self.samples[tokens[1]]['timestamp']) / 1e6
        def stack_or_empty(values, shape, dtype=torch.float32):
            return torch.stack(values) if values else torch.empty((0, *shape), dtype=dtype)
        return dict(images=stack_or_empty(frames, (6, 3, height, width)),
                    intrinsics=stack_or_empty(intrinsics, (6, 3, 3)),
                    camera_to_ego=stack_or_empty(extrinsics, (6, 4, 4)),
                    camera_to_ego_static=stack_or_empty(static_extrinsics, (6, 4, 4)), current2history=transforms,
                    motion=motion, depth_targets=stack_or_empty(depth_targets, (6, height//8, width//8), torch.long),
                    gt_positions=gt, future_valid=valid,
                    command=torch.tensor(command), history_dt=torch.tensor(history_dt), token=token,
                    scene=self.scene_names[self.samples[token]['scene_token']])
