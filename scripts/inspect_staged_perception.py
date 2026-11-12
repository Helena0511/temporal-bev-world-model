"""Plot current-frame BEV annotations and predictions for fixed held-out samples."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch

from resworld.city_selection import apply_selection
from resworld.driving_data import DrivingDataset
from resworld.driving_model import ModelConfig
from resworld.perception_targets import PerceptionTargets, load_annotations
from resworld.staged_driving import StagedDrivingModel
from scripts.train_driving import make_loader, move


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--indices', type=int, nargs='+', default=[0, 4])
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    manifest = checkpoint['manifest']
    saved = manifest['args']
    config = ModelConfig(**manifest['model_config'])
    selection_bytes = Path(saved['selection_file']).read_bytes()
    if hashlib.sha256(selection_bytes).hexdigest() != manifest['selection_sha256']:
        raise ValueError('selection changed since training')
    selected = json.loads(selection_bytes)['val']
    base = DrivingDataset(saved['root'], saved['version'], 'val', tuple(manifest['image_size']),
        config.future_steps, saved['split_file'], depth_bins=config.depth_bins,
        depth_min=config.depth_min, depth_max=config.depth_max)
    # apply_selection validates official-split membership, including requested indices.
    dataset = apply_selection(base, [selected[i] for i in args.indices])
    annotations = load_annotations(saved['root'], saved['version'], [selected[i] for i in args.indices])
    dataset = PerceptionTargets(dataset, base, annotations, config.grid())
    model = StagedDrivingModel(config, 'perception', saved['ego_dropout']).eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    cmap = ListedColormap(['#eeeeee', '#1976d2', '#ef6c00', '#777777'])
    fig, axes = plt.subplots(len(dataset), 3, figsize=(13, 5*len(dataset)), squeeze=False)
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]
    for i, raw in enumerate(make_loader(dataset, 1)):
        batch = move(raw, 'cpu')
        outputs = model(batch)
        image = (batch['images'][0, -1, 1]*std+mean).clamp(0, 1).permute(1, 2, 0)
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(batch['scene'][0]+' / CAM_FRONT')
        axes[i, 0].axis('off')
        target = batch['bev_target'][0].clone()
        target[target == 255] = 3
        for axis, raster, title in ((axes[i, 1], target, 'Annotated box footprints'),
                                    (axes[i, 2], outputs['bev_logits'][0].argmax(0), 'Camera BEV prediction')):
            axis.imshow(raster, cmap=cmap, vmin=0, vmax=3,
                extent=(config.y_max, config.y_min, config.x_min, config.x_max), origin='upper')
            axis.scatter([0], [0], c='black', marker='x', s=25)
            axis.set_title(title)
            axis.set_xlabel('Left y (m)')
            axis.set_ylabel('Forward x (m)')
    fig.suptitle('Blue: vehicle; orange: human; gray: ignored annotation category', y=.995)
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
