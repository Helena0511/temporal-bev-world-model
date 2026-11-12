"""Export real LiDAR-to-camera projection overlays for a selected pilot sample."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1]/'runs/.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from resworld.driving_data import DrivingDataset, pose_matrix
from resworld.nuscenes_io import CAMERAS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='/mnt/ssd0/datasets/nuscenes')
    parser.add_argument('--selection', default='runs/city_phase1/selection.json')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--output', default='runs/city_phase1/projection')
    args = parser.parse_args()
    torch.set_num_threads(4)
    selection = json.loads(Path(args.selection).read_text())
    dataset = DrivingDataset(args.root, selection['version'], 'val', image_size=(256, 704))
    token = selection['val'][args.index]
    item = dataset[dataset.entries.index(token)]
    global_points = dataset.lidar_global_points(token)
    if global_points is None:
        raise RuntimeError('no LiDAR points available for projection inspection')
    reference_points = global_points @ torch.linalg.inv(pose_matrix(dataset.reference_pose(token))).T
    figure, axes = plt.subplots(2, 3, figsize=(18, 6))
    counts = {}
    for view, camera in enumerate(CAMERAS):
        points = (reference_points @ torch.linalg.inv(item['camera_to_ego'][2, view].double()).T)[:, :3].float()
        pixels = points @ item['intrinsics'][2, view].T
        depth = points[:, 2]
        uv = pixels[:, :2]/depth.clamp_min(1e-5)[:, None]
        valid = ((depth > 1) & (depth < 100) & (uv[:, 0] >= 0) & (uv[:, 0] < 704)
                 & (uv[:, 1] >= 0) & (uv[:, 1] < 256))
        image = item['images'][2, view] * torch.tensor([.229, .224, .225])[:, None, None]
        image += torch.tensor([.485, .456, .406])[:, None, None]
        ax = axes.flatten()[view]
        ax.imshow(image.permute(1, 2, 0).clamp(0, 1))
        ax.scatter(uv[valid, 0], uv[valid, 1], c=depth[valid], s=.4, cmap='turbo', vmin=0, vmax=70, alpha=.75)
        counts[camera] = int(valid.sum())
        ax.set_title(f'{camera}: {counts[camera]} points')
        ax.axis('off')
    figure.suptitle(f'{item["scene"]} / {token} / geometric projection inspection')
    figure.tight_layout()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output/'lidar_camera_overlay.png', dpi=150)
    plt.close(figure)
    (output/'projection.json').write_text(json.dumps(dict(token=token, scene=item['scene'],
        valid_points=counts, note='One real sample; visual inspection is not a full calibration or temporal-consistency audit.'), indent=2))
    print(output/'lidar_camera_overlay.png')


if __name__ == '__main__':
    main()
