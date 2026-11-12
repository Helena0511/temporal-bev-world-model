"""Checkpoint diagnostics: errors by axis/route/speed and inference perturbations.

Perturbations measure sensitivity of one trained checkpoint, not the causal
benefit of a module. Architecture comparisons require independent retraining.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from resworld.driving_data import DrivingDataset
from resworld.driving_model import DrivingResWorld, ModelConfig
from resworld.city_selection import apply_selection
from scripts.train_driving import make_loader, trajectory_metrics


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='runs/mini_pipeline_check/best.pt')
    parser.add_argument('--root', default='/mnt/ssd0/datasets/nuscenes')
    parser.add_argument('--output', default='runs/mini_diagnosis.json')
    parser.add_argument('--split', choices=('train', 'val'), default='val')
    parser.add_argument('--full-split', action='store_true', help='evaluate the full official split instead of saved pilot selection')
    parser.add_argument('--skip-perturbations', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = ModelConfig(**checkpoint['model_config'])
    model = DrivingResWorld(config).eval()
    model.load_state_dict(checkpoint['model'])
    saved = checkpoint['args']
    ds = DrivingDataset(args.root, saved['version'], args.split,
                        (saved['image_height'], saved['image_width']),
                        config.future_steps, saved['split_file'],
                        depth_bins=config.depth_bins, depth_min=config.depth_min,
                        depth_max=config.depth_max)
    selection_file = saved.get('selection_file')
    if selection_file and not args.full_split:
        contents = Path(selection_file).read_bytes()
        if hashlib.sha256(contents).hexdigest() != checkpoint.get('selection_sha256'):
            raise ValueError('saved pilot selection has changed')
        ds = apply_selection(ds, json.loads(contents)[args.split])
    totals = {}
    sequence_errors = []
    outside, gt_outside, ref_count = 0, 0, 0
    gt_forward = []
    diagnostic = {'overlap_sum': 0., 'batches': 0}

    def accumulate(name, prediction, batch, groups):
        for i, group_names in enumerate(groups):
            subset = {k:batch[k][i:i+1] for k in ('gt_positions', 'future_valid')}
            values = trajectory_metrics(prediction[i:i+1], subset)
            delta = prediction[i] - batch['gt_positions'][i]
            valid = batch['future_valid'][i]
            values['forward_abs_sum'] = delta[valid, 0].abs().sum().item()
            values['forward_bias_sum'] = delta[valid, 0].sum().item()
            values['lateral_abs_sum'] = delta[valid, 1].abs().sum().item()
            for group in group_names:
                acc = totals.setdefault(name, {}).setdefault(group, {'sequences': 0})
                acc['sequences'] += 1
                for key, value in values.items():
                    acc[key] = acc.get(key, 0) + value

    for batch in make_loader(ds, 2):
        b = len(batch['command'])
        index = torch.arange(b)
        relative = torch.linalg.inv(batch['current2history'][:, 1])
        velocity = -relative[:, :2, 2] / batch['history_dt'][:, None]
        speed = velocity.norm(dim=-1)
        groups = []
        for i in range(b):
            bucket = '0-2mps' if speed[i] < 2 else ('2-5mps' if speed[i] < 5 else '5+mps')
            route = ('left', 'straight', 'right')[batch['command'][i].item()]
            groups.append(('all', f'route/{route}', f'speed/{bucket}'))
        base = model(batch)
        selected = base['final_traj'][index, batch['command']]
        for i in range(b):
            valid = batch['future_valid'][i]
            delta = selected[i]-batch['gt_positions'][i]
            last = int(valid.sum())-1
            sequence_errors.append(dict(token=batch['token'][i], scene=batch['scene'][i],
                command=int(batch['command'][i]), past_speed_mps=speed[i].item(),
                ade_m=delta[valid].norm(dim=-1).mean().item(),
                forward_bias_m=delta[valid,0].mean().item(),
                forward_abs_m=delta[valid,0].abs().mean().item(),
                lateral_abs_m=delta[valid,1].abs().mean().item(),
                predicted_last_xy=selected[i,last].tolist(), gt_last_xy=batch['gt_positions'][i,last].tolist()))
        accumulate('base', selected, batch, groups)
        accumulate('without_refinement', base['init_traj'][index, batch['command']], batch, groups)
        horizon = torch.arange(1, config.future_steps+1) * .5
        accumulate('constant_velocity', velocity[:, None] * horizon[None, :, None], batch, groups)
        refs = config.grid().source_points_to_grid(base['init_traj'][index, batch['command']])
        valid = batch['future_valid']
        outside += int(((refs.abs() > 1).any(-1) & valid).sum())
        gt_refs = config.grid().source_points_to_grid(batch['gt_positions'])
        gt_outside += int(((gt_refs.abs() > 1).any(-1) & valid).sum())
        gt_forward.append(batch['gt_positions'][..., 0][valid])
        ref_count += int(valid.sum())
        diagnostic['overlap_sum'] += base['overlap_mask'].float().mean().item()
        diagnostic['batches'] += 1
        for name in (() if args.skip_perturbations else ('zero_images', 'zero_motion', 'without_world_delta')):
            perturbed = dict(batch)
            hook = None
            if name == 'zero_images':
                # Zero normalized pixels correspond to the normalization mean.
                perturbed['images'] = torch.zeros_like(batch['images'])
            elif name == 'zero_motion':
                perturbed['motion'] = torch.zeros_like(batch['motion'])
            else:
                hook = model.token_fuser.register_forward_hook(
                    lambda module, inputs, output: torch.zeros_like(output))
            try:
                output = model(perturbed)['final_traj'][index, batch['command']]
            finally:
                if hook is not None:
                    hook.remove()
            accumulate(name, output, batch, groups)
            shift = (output-selected).norm(dim=-1)[valid]
            diagnostic[name+'_shift_sum'] = diagnostic.get(name+'_shift_sum', 0.) + shift.sum().item()

    report = {}
    for name, grouped in totals.items():
        report[name] = {}
        for group, sums in grouped.items():
            result = {'sequences': sums['sequences']}
            for metric in ('ade', 'fde', 'l2_1s', 'l2_2s', 'l2_3s'):
                count = sums.get(metric+'_count', 0)
                result[metric] = sums.get(metric+'_sum', 0)/count if count else None
            for metric in ('forward_abs', 'forward_bias', 'lateral_abs'):
                result[metric] = sums[metric+'_sum']/sums['ade_count']
            report[name][group] = result
    geometry = {'prior_waypoints_outside_bev_fraction': outside/max(1, ref_count),
                'gt_waypoints_outside_bev_fraction': gt_outside/max(1, ref_count),
                'gt_forward_position_quantiles_50_90_95_100': torch.quantile(
                    torch.cat(gt_forward), torch.tensor([.5, .9, .95, 1.])).tolist(),
                'history_overlap_batch_mean': diagnostic['overlap_sum']/diagnostic['batches']}
    sensitivity = {key.removesuffix('_shift_sum'):value/max(1, ref_count)
                   for key,value in diagnostic.items() if key.endswith('_shift_sum')}
    payload = dict(checkpoint=args.checkpoint, version=saved['version'], split=args.split,
                   selected_pilot=bool(selection_file and not args.full_split), metrics=report,
                   sequence_errors=sorted(sequence_errors,key=lambda row:row['ade_m'],reverse=True),
                   geometry=geometry, mean_prediction_shift_m=sensitivity,
                   interpretation='Inference perturbations are sensitivity tests; retrain for causal ablations.')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({'overall':{name:groups['all'] for name,groups in report.items()},
                      'geometry':geometry, 'mean_prediction_shift_m':sensitivity}, indent=2))


if __name__ == '__main__':
    main()
