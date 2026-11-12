"""Export held-out trajectory predictions and a reviewable PNG from a checkpoint."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1]/'runs/.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import default_collate

from resworld.driving_data import DrivingDataset
from resworld.driving_model import DrivingResWorld, ModelConfig


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--root', default='/mnt/ssd0/datasets/nuscenes')
    parser.add_argument('--output', default='runs/predictions')
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = ModelConfig(**checkpoint['model_config'])
    model = DrivingResWorld(config).eval()
    model.load_state_dict(checkpoint['model'])
    saved = checkpoint['args']
    dataset = DrivingDataset(args.root, saved['version'], 'val',
                             (saved['image_height'], saved['image_width']),
                             config.future_steps, saved['split_file'],
                             depth_bins=config.depth_bins, depth_min=config.depth_min,
                             depth_max=config.depth_max)
    indices = torch.linspace(0, len(dataset)-1, 4).round().long().tolist()
    figure, axes = plt.subplots(2, 4, figsize=(16, 8))
    records = []
    for col, index in enumerate(indices):
        item = dataset[index]
        batch = default_collate([item])
        outputs = model(batch)
        mode = item['command'].item()
        valid = item['future_valid']
        prediction = outputs['final_traj'][0, mode]
        prior = outputs['init_traj'][0, mode]
        gt = item['gt_positions']
        relative = torch.linalg.inv(item['current2history'][1])
        velocity = -relative[:2, 2] / item['history_dt']
        baseline = torch.arange(1, config.future_steps+1)[:, None] * .5 * velocity
        image = item['images'][2, 1] * torch.tensor([.229, .224, .225])[:, None, None]
        image += torch.tensor([.485, .456, .406])[:, None, None]
        axes[0, col].imshow(image.permute(1, 2, 0).clamp(0, 1))
        axes[0, col].set_title(f'{item["scene"]} / {["left", "straight", "right"][mode]}')
        axes[0, col].axis('off')
        ax = axes[1, col]
        for label, trajectory, style in [('GT', gt[valid], 'k-o'),
                                         ('Prior', prior[valid], 'b--'),
                                         ('Refined', prediction[valid], 'r-o'),
                                         ('Constant velocity', baseline[valid], 'g--')]:
            trajectory = torch.cat((torch.zeros(1, 2), trajectory))
            ax.plot(-trajectory[:, 1], trajectory[:, 0], style, label=label, markersize=3)
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, alpha=.3)
        ax.set_xlabel('Right (m)')
        ax.set_ylabel('Forward (m)')
        ax.legend(fontsize=7)
        records.append(dict(token=item['token'], scene=item['scene'], command=mode,
                            valid=valid.tolist(), gt=gt.tolist(),
                            prior=prior.tolist(), refined=prediction.tolist(),
                            constant_velocity=baseline.tolist()))
    figure.suptitle(f'Held-out {saved["version"]} validation / diagnostic checkpoint (not paper results)')
    figure.tight_layout()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output/'trajectories.png', dpi=150)
    plt.close(figure)
    (output/'predictions.json').write_text(json.dumps(records, indent=2))
    print(output/'trajectories.png')


if __name__ == '__main__':
    main()
