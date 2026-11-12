"""Current-frame annotated box footprints for camera BEV supervision.

This is a three-class BEV segmentation proxy, not nuScenes detection mAP or
collision occupancy. No unannotated object or free-space claim is made.
"""
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .driving_data import iter_json_rows, pose_matrix


CLASS_NAMES = ('background', 'vehicle', 'human')


def load_annotations(root, version, tokens):
    directory = Path(root) / version
    categories = {r['token']: r['name'] for r in
                  json.loads((directory/'category.json').read_text())}
    instances = {r['token']: categories[r['category_token']] for r in
                 json.loads((directory/'instance.json').read_text())}
    selected = {token: [] for token in tokens}
    for row in iter_json_rows(directory/'sample_annotation.json'):
        if row['sample_token'] in selected:
            name = instances[row['instance_token']]
            label = 1 if name.startswith('vehicle.') else (2 if name.startswith('human.') else -1)
            selected[row['sample_token']].append({
                k: row[k] for k in ('translation', 'rotation', 'size')
            } | {'label': label})
    return selected


def footprint_target(annotations, reference_pose, grid):
    """Rasterize oriented boxes with nuScenes size=(width,length,height).

    Cell centres inside ignored-category boxes are ignored (255). Annotated
    vehicle/human boxes take precedence over ignored boxes and background.
    Tiny boxes cover their centre cell to avoid dropping pedestrians at 1m.
    """
    xy = grid.target_metric_points(1, 'cpu', torch.float64)[0, ..., :2]
    world_to_ego = torch.linalg.inv(pose_matrix(reference_pose))
    target = torch.zeros(xy.shape[:-1], dtype=torch.long)
    for annotation in sorted(annotations, key=lambda a: a['label']):
        transform = world_to_ego @ pose_matrix(annotation)
        centre = transform[:2, 3]
        yaw = torch.atan2(transform[1, 0], transform[0, 0])
        cs, sn = yaw.cos(), yaw.sin()
        delta = xy - centre
        local_x = cs*delta[..., 0] + sn*delta[..., 1]
        local_y = -sn*delta[..., 0] + cs*delta[..., 1]
        width, length, _ = annotation['size']
        inside = (local_x.abs() <= length/2) & (local_y.abs() <= width/2)
        if (not inside.any() and grid.x_min <= centre[0] <= grid.x_max
                and grid.y_min <= centre[1] <= grid.y_max):
            index = (xy-centre).square().sum(-1).argmin()
            inside.flatten()[index] = True
        label = annotation['label']
        target[inside] = label if label >= 0 else 255
    return target


class PerceptionTargets(Dataset):
    def __init__(self, dataset, base, annotations, grid):
        self.dataset, self.base = dataset, base
        self.annotations, self.grid = annotations, grid

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        batch = self.dataset[index]
        token = batch['token']
        batch['bev_target'] = footprint_target(self.annotations[token],
                                                self.base.reference_pose(token), self.grid)
        return batch
