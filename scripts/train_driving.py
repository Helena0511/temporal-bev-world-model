"""Train, overfit, or evaluate the native nuScenes ResWorld implementation."""
import argparse
from dataclasses import asdict
import json
import hashlib
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from resworld.driving_data import DrivingDataset
from resworld.driving_model import DrivingResWorld, ModelConfig, planning_loss
from resworld.city_selection import apply_selection


def move(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def make_loader(dataset, batch_size, workers=0, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=False)


def trajectory_metrics(prediction, batch):
    """Return sums/counts so partial trajectories never distort dataset averages."""
    error = (prediction - batch['gt_positions']).norm(dim=-1)
    valid = batch['future_valid']
    result = {'ade_sum': error[valid].sum().item(), 'ade_count': valid.sum().item()}
    axis_error = (prediction-batch['gt_positions']).abs()
    for axis, name in enumerate(('forward_abs', 'lateral_abs')):
        result[name+'_sum'] = axis_error[..., axis][valid].sum().item()
        result[name+'_count'] = valid.sum().item()
    for step in (2, 4, 6):
        if step > error.shape[1]:
            continue
        horizon_valid = valid[:, :step].all(1)
        result[f'l2_{step//2}s_sum'] = error[horizon_valid, step-1].sum().item()
        result[f'l2_{step//2}s_count'] = horizon_valid.sum().item()
    full = valid.all(1)
    result['fde_sum'] = error[full, -1].sum().item()
    result['fde_count'] = full.sum().item()
    return result


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    totals, grouped = {}, {}
    loss_sum, loss_count, sequences = 0., 0, 0
    for raw in loader:
        batch = move(raw, device)
        output = model(batch)
        terms = planning_loss(output, batch)
        b = batch['images'].shape[0]
        valid_steps = batch['future_valid'].sum().item()
        loss_sum += terms['planning'].item() * valid_steps
        loss_count += valid_steps
        sequences += b
        index = torch.arange(b, device=device)
        predictions = {
            'final': output['final_traj'][index, batch['command']],
            'init': output['init_traj'][index, batch['command']],
            'stationary': torch.zeros_like(batch['gt_positions']),
        }
        # Constant velocity from the past only, expressed in the current ego frame.
        relative = torch.linalg.inv(batch['current2history'][:, 1])
        dt = batch['history_dt'][:, None]
        velocity = -relative[:, :2, 2] / dt
        horizon = torch.arange(1, model.config.future_steps+1, device=device) * .5
        predictions['constant_velocity'] = velocity[:, None] * horizon[None, :, None]
        for name, prediction in predictions.items():
            acc = totals.setdefault(name, {})
            for key, value in trajectory_metrics(prediction, batch).items():
                acc[key] = acc.get(key, 0) + value
        for i, scene in enumerate(batch.get('scene', ['unknown']*b)):
            label = ('left', 'straight', 'right')[batch['command'][i].item()]
            speed = velocity[i].norm().item()
            speed_group = '0-2mps' if speed < 2 else ('2-5mps' if speed < 5 else '5+mps')
            for group in (f'command/{label}', f'scene/{scene}', f'speed/{speed_group}'):
                acc = grouped.setdefault(group, {})
                sliced = {key: batch[key][i:i+1] for key in ('gt_positions', 'future_valid')}
                for key, value in trajectory_metrics(predictions['final'][i:i+1], sliced).items():
                    acc[key] = acc.get(key, 0) + value
    result = {'sequences': sequences, 'planning_loss': loss_sum / max(1, loss_count)}
    for name, sums in totals.items():
        result[name] = {key[:-4]: value/sums[key[:-4]+'_count']
                        if sums[key[:-4]+'_count'] else None
                        for key, value in sums.items() if key.endswith('_sum')}
    result['final_by_group'] = {
        group: {key[:-4]: value/sums[key[:-4]+'_count'] if sums[key[:-4]+'_count'] else None
                for key, value in sums.items() if key.endswith('_sum')}
        for group, sums in grouped.items()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='/mnt/ssd0/datasets/nuscenes')
    parser.add_argument('--version', default='v1.0-mini')
    parser.add_argument('--split-file')
    parser.add_argument('--selection-file', help='JSON with train/val sample-token lists for a diagnostic pilot')
    parser.add_argument('--full-validation', action='store_true', help='evaluate the full official validation split despite pilot selection')
    parser.add_argument('--mode', choices=['smoke', 'overfit', 'train', 'eval'], default='smoke')
    parser.add_argument('--output', default='runs/nuscenes_smoke')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--steps', type=int, default=80)
    parser.add_argument('--limit-train', type=int, default=0)
    parser.add_argument('--limit-val', type=int, default=0)
    parser.add_argument('--image-height', type=int, default=64)
    parser.add_argument('--image-width', type=int, default=160)
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--model-config', help='JSON with ModelConfig fields')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume')
    args = parser.parse_args()
    if args.full_validation and args.mode != 'eval':
        raise ValueError('--full-validation is only available in eval mode')
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; choose --device cpu or repair GPU runtime')
    if args.image_height % 8 or args.image_width % 8:
        raise ValueError('image dimensions must be multiples of 8')
    if min(args.batch_size, args.epochs, args.steps) <= 0:
        raise ValueError('batch size, epochs, and steps must be positive')
    device = torch.device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = (ModelConfig(**json.loads(Path(args.model_config).read_text()))
              if args.model_config else ModelConfig(channels=args.channels))
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        requested_config = config
        config = ModelConfig(**checkpoint['model_config'])
        if args.model_config and json.dumps(asdict(requested_config),sort_keys=True) != json.dumps(asdict(config),sort_keys=True):
            raise ValueError('model config differs from checkpoint; changing BEV bounds requires a new experiment')
        saved = checkpoint['args']
        for key in ('version', 'split_file', 'image_height', 'image_width'):
            if getattr(args, key) != saved[key]:
                raise ValueError(f'resume/eval requires matching --{key.replace("_", "-")}')
        if bool(args.selection_file) != bool(saved.get('selection_file')):
            raise ValueError('resume/eval requires the same selection policy; content is checked by SHA256')
    if args.mode == 'eval' and checkpoint is None:
        raise ValueError('eval requires --resume checkpoint')
    if config.image_backbone == 'resnet50' and checkpoint is None:
        raise ValueError('fresh ResNet50 training uses train_staged_driving.py --backbone-weights for verified ImageNet initialization')
    datasets = {split: DrivingDataset(args.root, args.version, split,
                (args.image_height, args.image_width), config.future_steps, args.split_file,
                depth_bins=config.depth_bins, depth_min=config.depth_min, depth_max=config.depth_max)
                for split in ('train', 'val')}
    assert set(datasets['train'].entries).isdisjoint(datasets['val'].entries)
    data_summary = {split: {'sequences': len(ds), 'missing_file_sequences': ds.missing_files,
                            'scenes': sorted({ds.scene_names[ds.samples[t]['scene_token']]
                                              for t in ds.entries})}
                    for split, ds in datasets.items()}
    console_summary = {split: {**{k:v for k,v in summary.items() if k != 'scenes'},
                               'scene_count': len(summary['scenes'])}
                       for split, summary in data_summary.items()}
    print(json.dumps({'dataset': console_summary, 'device': str(device)}, ensure_ascii=False), flush=True)
    selection_hash = None
    if args.selection_file:
        contents = Path(args.selection_file).read_bytes()
        selection_hash = hashlib.sha256(contents).hexdigest()
        selection = json.loads(contents)
        if selection.get('version') != args.version:
            raise ValueError('selection dataset version mismatch')
        if set(selection['train']) & set(selection['val']):
            raise ValueError('selected train/val tokens overlap')
        for split in ('train', 'val'):
            if not (split == 'val' and args.full_validation):
                datasets[split] = apply_selection(datasets[split], selection[split])
        print(json.dumps({'pilot_selection':{s:len(ds) for s,ds in datasets.items()}}),flush=True)
    if checkpoint and checkpoint.get('selection_sha256') != selection_hash:
        raise ValueError('selection file content differs from checkpoint')
    for split, limit in (('train', args.limit_train), ('val', args.limit_val)):
        if limit:
            datasets[split] = Subset(datasets[split], range(min(limit, len(datasets[split]))))
    model = DrivingResWorld(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    start_epoch, global_step, best = 0, 0, float('inf')
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        if args.mode == 'train':
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            global_step = checkpoint['global_step']
            best = checkpoint['best_ade']
            torch.set_rng_state(checkpoint['torch_rng'])
            random.setstate(checkpoint['python_rng'])
            np.random.set_state(checkpoint['numpy_rng'])
            if device.type == 'cuda' and checkpoint.get('cuda_rng') is not None:
                torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    project_root = Path(__file__).resolve().parents[1]
    source_names = ('resworld/driving_data.py', 'resworld/driving_model.py',
                    'resworld/bev_attention.py', 'resworld/temporal_alignment.py',
                    'resworld/fgtr.py', 'resworld/mini_resworld.py',
                    'resworld/city_selection.py', 'resworld/image_backbone.py',
                    'scripts/train_driving.py', 'configs/nuscenes_splits.json')
    manifest = dict(args=vars(args), model_config=asdict(config), dataset=data_summary,
                    selected_dataset={s:len(ds) for s,ds in datasets.items()}, selection_sha256=selection_hash,
                    torch_version=torch.__version__, cuda=torch.cuda.is_available(),
                    source_sha256={name: hashlib.sha256((project_root/name).read_bytes()).hexdigest()
                                   for name in source_names})
    (output_dir/'config.json').write_text(json.dumps(manifest, indent=2))
    val_loader = make_loader(datasets['val'], args.batch_size, args.workers)

    def record(data):
        print(json.dumps(data, ensure_ascii=False), flush=True)
        with (output_dir/'metrics.jsonl').open('a') as file:
            file.write(json.dumps(data) + '\n')

    def save(name, epoch):
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                     model_config=asdict(config), args=vars(args), epoch=epoch,
                     global_step=global_step, best_ade=best,
                     selection_sha256=selection_hash,
                     torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                     numpy_rng=np.random.get_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
        temporary = output_dir/(name+'.tmp')
        torch.save(state, temporary)
        temporary.replace(output_dir/name)

    if args.mode == 'eval':
        result = evaluate(model, val_loader, device)
        record({'mode': 'eval', **result})
        return
    if args.mode in ('smoke', 'overfit'):
        batch = move(next(iter(make_loader(datasets['train'], args.batch_size))), device)
        fixed = [batch]
        before = evaluate(model, fixed, device)
        record({'mode': args.mode, 'phase': 'before', **before})
        branch_gradients = {}
        started = time.monotonic()
        for step in range(3 if args.mode == 'smoke' else args.steps):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            terms = planning_loss(prediction, batch)
            if not torch.isfinite(terms['total']):
                raise RuntimeError('non-finite loss')
            terms['total'].backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise RuntimeError(f'non-finite gradient in {name}')
            if step == 0:
                for prefix in ('camera_bev.backbone', 'camera_bev.depth_head', 'fusion',
                               'scene_tokenizer', 'residual_tokenizer', 'action_condition',
                               'token_fuser', 'sampler', 'refine_head'):
                    branch_gradients[prefix] = sum(p.grad.abs().sum().item() for n,p in model.named_parameters()
                                                  if n.startswith(prefix) and p.grad is not None)
                if config.temporal_fusion == 'attention':
                    branch_gradients.pop('fusion')
                    prefix = 'temporal_attention'
                    branch_gradients[prefix] = sum(p.grad.abs().sum().item() for n,p in model.named_parameters()
                                                  if n.startswith(prefix) and p.grad is not None)
                if config.perception == 'bev_query':
                    prefix = 'camera_bev.spatial'
                    branch_gradients[prefix] = sum(p.grad.abs().sum().item() for n,p in model.named_parameters()
                                                  if n.startswith(prefix) and p.grad is not None)
                if not config.use_world_delta:
                    for prefix in ('residual_tokenizer', 'action_condition', 'token_fuser'):
                        branch_gradients.pop(prefix, None)
                # Zero-initialized refinement initially blocks world/sampler gradients;
                # check all branch gradients after that head has taken its first update.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            global_step += 1
            if step == 1 or step == args.steps-1:
                branch_gradients = {prefix: sum(p.grad.abs().sum().item() for n,p in model.named_parameters()
                                               if n.startswith(prefix) and p.grad is not None)
                                    for prefix in branch_gradients}
            if (step+1) % 10 == 0:
                record({'step': step+1, 'loss': terms['total'].item()})
        after = evaluate(model, fixed, device)
        record({'mode': args.mode, 'phase': 'after', 'seconds': time.monotonic()-started,
                'branch_gradients': branch_gradients, **after})
        if args.mode == 'overfit':
            assert after['planning_loss'] < before['planning_loss'] * .7, 'overfit failed to reduce loss by 30%'
        assert all(value > 0 for value in branch_gradients.values()), 'disconnected gradient branch'
        save('last.pt', -1)
        record({'phase': 'held_out_validation', **evaluate(model, val_loader, device)})
        return

    loader = make_loader(datasets['train'], args.batch_size, args.workers, shuffle=True)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        started, total, sequences = time.monotonic(), 0., 0
        # Epoch-based cosine schedule; exact formula remains valid across resume.
        lr = args.lr * (.1 + .9*(1+np.cos(np.pi*epoch/args.epochs))/2)
        for group in optimizer.param_groups:
            group['lr'] = lr
        for raw in loader:
            batch = move(raw, device)
            optimizer.zero_grad(set_to_none=True)
            terms = planning_loss(model(batch), batch)
            if not torch.isfinite(terms['total']):
                raise RuntimeError('non-finite training loss')
            terms['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            b = batch['images'].shape[0]
            total += terms['total'].item()*b
            sequences += b
            global_step += 1
            if global_step % 20 == 0:
                record({'epoch': epoch+1, 'step': global_step, 'loss': terms['total'].item(),
                        'planning_loss': terms['planning'].item(), 'depth_loss': terms['depth'].item()})
        validation = evaluate(model, val_loader, device)
        improved = validation['final']['ade'] < best
        best = min(best, validation['final']['ade'])
        record({'epoch': epoch+1, 'step': global_step, 'lr': lr,
                'seconds': time.monotonic()-started, 'train_loss': total/sequences,
                'validation': validation})
        save('last.pt', epoch)
        if improved:
            save('best.pt', epoch)


if __name__ == '__main__':
    main()
