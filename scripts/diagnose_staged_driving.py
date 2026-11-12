"""Image sensitivity and standalone branch errors for a staged checkpoint.

These are fixed-checkpoint sensitivity tests, not independently trained causal
ablations. Geometric alignment and the oracle route instruction are retained.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from resworld.city_selection import apply_selection
from resworld.driving_data import DrivingDataset
from resworld.driving_model import ModelConfig
from resworld.staged_driving import StagedDrivingModel
from scripts.train_driving import make_loader, trajectory_metrics


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    manifest = checkpoint['manifest']
    saved = manifest['args']
    if manifest['stage'] not in ('visual', 'fusion', 'joint'):
        raise ValueError('a visual/fusion/joint planning checkpoint is required')
    config = ModelConfig(**manifest['model_config'])
    selection_bytes = Path(saved['selection_file']).read_bytes()
    if hashlib.sha256(selection_bytes).hexdigest() != manifest['selection_sha256']:
        raise ValueError('selection changed since training')
    base = DrivingDataset(saved['root'], saved['version'], 'val', tuple(manifest['image_size']),
        config.future_steps, saved['split_file'], depth_bins=config.depth_bins,
        depth_min=config.depth_min, depth_max=config.depth_max)
    dataset = apply_selection(base, json.loads(selection_bytes)['val'])
    model = StagedDrivingModel(config, manifest['stage'], saved['ego_dropout']).eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    totals, shift_sum, valid_count = {}, 0., 0
    for batch in make_loader(dataset, saved['batch_size']):
        output = model(batch)
        index = torch.arange(len(batch['command']))
        selected = output['final_traj'][index, batch['command']]
        zero_output = model({**batch, 'images':torch.zeros_like(batch['images'])})
        zero_prediction = zero_output['final_traj'][index, batch['command']]
        predictions = {'base': selected, 'zero_images': zero_prediction}
        if 'visual_outputs' in output:
            predictions['visual_branch'] = output['visual_outputs']['final_traj'][index, batch['command']]
            predictions['ego_branch'] = model.ego(batch)['final_traj'][index, batch['command']]
        for name, prediction in predictions.items():
            acc = totals.setdefault(name, {})
            for key, value in trajectory_metrics(prediction, batch).items():
                acc[key] = acc.get(key, 0)+value
        valid = batch['future_valid']
        shift_sum += (zero_prediction-selected).norm(dim=-1)[valid].sum().item()
        valid_count += valid.sum().item()
    metrics = {name: {k[:-4]:v/sums[k[:-4]+'_count'] if sums[k[:-4]+'_count'] else None
                     for k,v in sums.items() if k.endswith('_sum')} for name,sums in totals.items()}
    result = dict(checkpoint=args.checkpoint, sequences=len(dataset), metrics=metrics,
                  zero_images_prediction_shift_m=shift_sum/max(1, valid_count),
                  interpretation=__doc__.strip())
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
