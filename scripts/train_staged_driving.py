"""Train independent modality baselines, perception pretraining and staged fusion.

All transfers validate data selection. Visual transfers also validate image size
and full model geometry; state-only transfers validate their channel/horizon sizes.
Stage transfers start a fresh optimizer; --init-checkpoint is not a resume flag.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from resworld.city_selection import apply_selection
from resworld.driving_data import DrivingDataset
from resworld.driving_model import ModelConfig
from resworld.perception_targets import CLASS_NAMES, PerceptionTargets, load_annotations
from resworld.staged_driving import STAGES, StagedDrivingModel, staged_loss
from scripts.train_driving import evaluate, make_loader, move


def transfer_checkpoint(model, checkpoint, manifest, component=None):
    """Validate before changing any parameters, then load an entire component."""
    saved = checkpoint['manifest']
    for key in ('model_config', 'version', 'selection_sha256', 'image_size', 'split_sha256'):
        if component == 'ego' and key == 'image_size':
            continue  # EgoTrajectoryBranch never reads images or camera features.
        previous = asdict(ModelConfig(**saved[key])) if key=='model_config' else saved[key]
        requested = asdict(ModelConfig(**manifest[key])) if key=='model_config' else manifest[key]
        if component == 'ego' and key == 'model_config':
            previous = {name: previous[name] for name in ('channels', 'future_steps')}
            requested = {name: requested[name] for name in ('channels', 'future_steps')}
        if json.dumps(previous, sort_keys=True) != json.dumps(requested, sort_keys=True):
            raise ValueError(f'checkpoint transfer mismatch: {key}')
    if component == 'visual' and saved['stage'] not in ('perception', 'visual'):
        raise ValueError('visual initialization requires a perception/visual checkpoint')
    if component == 'ego' and saved['stage'] != 'ego':
        raise ValueError('ego initialization requires an ego-only checkpoint')
    state = checkpoint['model']
    if component == 'visual':
        for name in ('visual', 'perception_head'):
            prefix = name+'.'
            getattr(model, name).load_state_dict({k[len(prefix):]: v for k, v in state.items()
                                                 if k.startswith(prefix)}, strict=True)
    elif component == 'ego':
        model.ego.load_state_dict({k[4:]: v for k, v in state.items() if k.startswith('ego.')}, strict=True)
    else:
        model.load_state_dict(state, strict=True)


@torch.no_grad()
def evaluate_stage(model, loader, device):
    result = {} if model.stage == 'perception' else evaluate(model, loader, device)
    if model.stage == 'ego':
        return result
    confusion = torch.zeros(3, 3, dtype=torch.long)
    depth_error, correct, depth_count, sequences = 0., 0, 0, 0
    original_stage = model.stage
    model.set_stage('perception')
    model.eval()
    try:
        for raw in loader:
            batch = move(raw, device)
            outputs = model(batch)
            target = batch['bev_target']
            valid = target != 255
            prediction = outputs['bev_logits'].argmax(1)
            confusion += torch.bincount((target[valid]*3+prediction[valid]).cpu(), minlength=9).reshape(3, 3)
            logits = outputs['depth_logits'][:, 0]
            depth_prediction = logits.argmax(2)
            depth_target = batch['depth_targets'][:, -1]
            known = depth_target >= 0
            depth_error += (depth_prediction[known]-depth_target[known]).abs().sum().item()
            correct += (depth_prediction[known] == depth_target[known]).sum().item()
            depth_count += known.sum().item()
            sequences += target.shape[0]
    finally:
        model.set_stage(original_stage)
    intersection = confusion.diag()
    union = confusion.sum(0)+confusion.sum(1)-intersection
    iou = {name: intersection[i].item()/union[i].item() if union[i] > 0 else None
           for i, name in enumerate(CLASS_NAMES)}
    actual,predicted=confusion.sum(1),confusion.sum(0)
    precision={name:intersection[i].item()/predicted[i].item() if predicted[i]>0 else None
               for i,name in enumerate(CLASS_NAMES)}
    recall={name:intersection[i].item()/actual[i].item() if actual[i]>0 else None
            for i,name in enumerate(CLASS_NAMES)}
    foreground = [iou[c] for c in CLASS_NAMES[1:] if iou[c] is not None]
    cfg = model.config
    result['perception'] = dict(sequences=sequences, footprint_iou=iou,
        footprint_precision=precision,footprint_recall=recall,
        actual_cells={name:actual[i].item() for i,name in enumerate(CLASS_NAMES)},
        predicted_cells={name:predicted[i].item() for i,name in enumerate(CLASS_NAMES)},
        foreground_miou=sum(foreground)/len(foreground) if foreground else None,
        confusion=confusion.tolist(), depth_pixels=depth_count,
        depth_bin_accuracy=correct/depth_count if depth_count else None,
        depth_quantized_mae_m=depth_error/max(1, depth_count)*(cfg.depth_max-cfg.depth_min)/max(1, cfg.depth_bins-1))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='/mnt/ssd0/datasets/nuscenes')
    parser.add_argument('--version', default='v1.0-trainval')
    parser.add_argument('--split-file')
    parser.add_argument('--selection-file', required=True)
    parser.add_argument('--model-config', default='configs/bev_attention_city60_pilot.json')
    parser.add_argument('--stage', choices=STAGES, required=True)
    parser.add_argument('--strict-single-frame', action='store_true')
    parser.add_argument('--perception-checkpoint')
    parser.add_argument('--ego-checkpoint')
    parser.add_argument('--init-checkpoint', help='transfer all weights; new optimizer and epoch counter')
    parser.add_argument('--from-scratch', action='store_true', help='joint-training comparison without pretraining')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--image-height', type=int, default=64)
    parser.add_argument('--image-width', type=int, default=160)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--encoder-lr-scale', type=float, default=.1, help='camera/ego LR multiplier in joint stage')
    parser.add_argument('--backbone-lr-scale',type=float,default=.1,help='pretrained ResNet body LR multiplier')
    parser.add_argument('--ego-dropout', type=float, default=.3)
    parser.add_argument('--visual-weight', type=float, default=.5)
    parser.add_argument('--bev-loss',choices=('balanced','ce_dice'),default='balanced')
    parser.add_argument('--backbone-weights',help='official ImageNet ResNet50 weights for fresh visual initialization')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads, args.image_height, args.image_width) < 1:
        raise ValueError('positive epochs, batch size, threads and image dimensions required')
    if args.image_height % 8 or args.image_width % 8:
        raise ValueError('image dimensions must be multiples of 8')
    if args.visual_weight < 0 or args.lr <= 0 or not 0 < args.encoder_lr_scale <= 1 or not 0 < args.backbone_lr_scale <= 1:
        raise ValueError('nonnegative visual weight and positive learning rate required')
    if args.from_scratch and (args.stage != 'joint' or args.init_checkpoint or
                             args.perception_checkpoint or args.ego_checkpoint or args.eval_only):
        raise ValueError('--from-scratch is for a joint-training comparison without checkpoint inputs')
    if args.init_checkpoint and (args.perception_checkpoint or args.ego_checkpoint):
        raise ValueError('choose whole-model initialization or individual components')
    if args.backbone_weights and (args.init_checkpoint or args.perception_checkpoint):
        raise ValueError('ImageNet initialization cannot overwrite a supplied visual checkpoint')
    if args.stage == 'fusion' and not (args.init_checkpoint or
                                        (args.perception_checkpoint and args.ego_checkpoint)):
        raise ValueError('fusion requires both pretrained components')
    if args.stage == 'joint' and not (args.init_checkpoint or args.from_scratch):
        raise ValueError('joint training requires a fusion initialization')
    if args.eval_only and not args.init_checkpoint:
        raise ValueError('evaluation requires --init-checkpoint')
    output = Path(args.output)
    if any((output/name).exists() for name in ('config.json', 'last.pt')):
        raise ValueError('output already contains an experiment; choose a new output directory')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if args.device.startswith('cuda') and torch.cuda.device_count() != 1:
        raise RuntimeError('single-GPU policy: set CUDA_VISIBLE_DEVICES to exactly one GPU')
    config = replace(ModelConfig(**json.loads(Path(args.model_config).read_text())),
                     use_ego_state=False, kinematic_prior=False)
    if args.strict_single_frame:
        config = replace(config, temporal_context='single_frame', use_world_delta=False)
    if args.backbone_weights and config.image_backbone != 'resnet50':
        raise ValueError('--backbone-weights requires the resnet50 image backbone')
    if config.image_backbone == 'resnet50' and args.stage != 'ego' and not (
            args.backbone_weights or args.init_checkpoint or args.perception_checkpoint):
        raise ValueError('fresh ResNet50 visual training requires --backbone-weights')
    selection_bytes = Path(args.selection_file).read_bytes()
    selection = json.loads(selection_bytes)
    if selection['version'] != args.version or set(selection['train']) & set(selection['val']):
        raise ValueError('invalid selection version or overlapping tokens')
    input_mode = 'state' if args.stage == 'ego' else ('current' if args.stage == 'perception' else 'temporal')
    bases = {s: DrivingDataset(args.root, args.version, s, (args.image_height, args.image_width),
                config.future_steps, args.split_file, depth_bins=config.depth_bins,
                depth_min=config.depth_min, depth_max=config.depth_max, input_mode=input_mode) for s in ('train', 'val')}
    datasets = {s: apply_selection(bases[s], selection[s]) for s in bases}
    project_root = Path(__file__).resolve().parents[1]
    split_path = Path(args.split_file) if args.split_file else project_root/'configs/nuscenes_splits.json'
    source_names = ('scripts/train_staged_driving.py', 'scripts/train_driving.py',
                    'resworld/staged_driving.py', 'resworld/perception_targets.py',
                    'resworld/driving_model.py', 'resworld/driving_data.py',
                    'resworld/bev_attention.py', 'resworld/city_selection.py',
                    'resworld/fgtr.py', 'resworld/temporal_alignment.py',
                    'resworld/mini_resworld.py', 'resworld/image_backbone.py',
                    'resworld/perception_objectives.py')
    manifest = dict(stage=args.stage, args=vars(args), model_config=asdict(config), version=args.version,
        input_mode=input_mode, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        gpu_name=torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None,
        cuda_device_count=torch.cuda.device_count() if args.device.startswith('cuda') else 0,
        selection_sha256=hashlib.sha256(selection_bytes).hexdigest(),
        split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
        image_size=[args.image_height, args.image_width], sizes={s:len(ds) for s,ds in datasets.items()},
        source_sha256={name: hashlib.sha256((project_root/name).read_bytes()).hexdigest()
                       for name in source_names})
    model = StagedDrivingModel(config, args.stage, args.ego_dropout).to(args.device)
    if args.backbone_weights:
        digest = model.visual.camera_bev.backbone.load_imagenet(args.backbone_weights)
        manifest['backbone_initialization'] = dict(path=args.backbone_weights,sha256=digest)
    initializations = []
    for path, component in ((args.init_checkpoint, None), (args.perception_checkpoint, 'visual'),
                            (args.ego_checkpoint, 'ego')):
        if path:
            state = torch.load(path, map_location='cpu', weights_only=False)
            if args.eval_only and state['manifest']['stage'] != args.stage:
                raise ValueError('evaluation stage must match the saved checkpoint')
            if args.eval_only and state['manifest']['args']['ego_dropout'] != args.ego_dropout:
                raise ValueError('evaluation requires the saved ego dropout setting')
            if args.stage == 'joint' and not args.eval_only and state['manifest']['stage'] not in ('fusion', 'joint'):
                raise ValueError('joint stage must follow a fusion/joint checkpoint')
            transfer_checkpoint(model, state, manifest, component)
            initializations.append(dict(path=path, component=component,
                sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(), stage=state['manifest']['stage']))
    manifest['initializations'] = initializations
    if args.stage != 'ego':
        print(json.dumps({'phase':'loading_annotations', 'sizes':manifest['sizes']}), flush=True)
        annotations = load_annotations(args.root, args.version, selection['train']+selection['val'])
        datasets = {s: PerceptionTargets(ds, bases[s], annotations, config.grid()) for s,ds in datasets.items()}
    loaders = {s: make_loader(ds, args.batch_size, args.workers, shuffle=(s == 'train'))
               for s,ds in datasets.items()}
    output.mkdir(parents=True, exist_ok=True)
    (output/'config.json').write_text(json.dumps(manifest, indent=2))

    def record(data):
        print(json.dumps(data), flush=True)
        with (output/'metrics.jsonl').open('a') as file:
            file.write(json.dumps(data)+'\n')

    record({'stage':args.stage, 'sizes':manifest['sizes'], 'trainable_parameters':
            sum(p.numel() for p in model.parameters() if p.requires_grad)})
    if args.eval_only:
        record({'mode':'eval', 'validation':evaluate_stage(model, loaders['val'], args.device)})
        return
    by_scale = {}
    for name,parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        encoder = name.startswith(('visual.camera_bev.','ego.'))
        scale = args.encoder_lr_scale if encoder and args.stage == 'joint' else 1.
        if name.startswith('visual.camera_bev.backbone.body.'):
            scale *= args.backbone_lr_scale
        by_scale.setdefault(scale,[]).append(parameter)
    groups = [dict(params=parameters,lr=args.lr*scale,lr_scale=scale)
              for scale,parameters in by_scale.items()]
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=1e-4)
    best, step = float('inf'), 0
    for epoch in range(args.epochs):
        model.train()
        if args.device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        started, total, sequences = time.monotonic(), 0., 0
        for group in optimizer.param_groups:
            group['lr'] = args.lr*group['lr_scale']*(.1+.9*(1+np.cos(np.pi*epoch/args.epochs))/2)
        for raw in loaders['train']:
            batch = move(raw, args.device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            terms = staged_loss(prediction, batch, args.stage, args.visual_weight,args.bev_loss)
            if not torch.isfinite(terms['total']):
                raise RuntimeError('non-finite loss')
            terms['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            b = batch['images'].shape[0]
            total += terms['total'].item()*b
            sequences += b
            step += 1
            if step % 12 == 0:
                data=dict(epoch=epoch+1,step=step,losses={k:v.item() for k,v in terms.items()})
                if 'bev_logits' in prediction:
                    labels=prediction['bev_logits'].detach().argmax(1)
                    target=batch['bev_target']
                    valid=target!=255
                    data['train_bev_iou']={}
                    data['train_bev_cells']={}
                    for i,name in enumerate(CLASS_NAMES):
                        actual=(target==i)&valid
                        predicted=(labels==i)&valid
                        intersection=(actual&predicted).sum().item()
                        union=(actual|predicted).sum().item()
                        data['train_bev_iou'][name]=intersection/union if union else None
                        data['train_bev_cells'][name]={'actual':actual.sum().item(),'predicted':predicted.sum().item()}
                record(data)
        validation = evaluate_stage(model, loaders['val'], args.device)
        score = (-validation['perception']['foreground_miou'] if args.stage == 'perception'
                 else validation['final']['ade'])
        improved = score < best
        best = min(best, score)
        record(dict(epoch=epoch+1, step=step, seconds=time.monotonic()-started,
                    train_loss=total/sequences, validation=validation,
                    cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if args.device.startswith('cuda') else None))
        checkpoint = dict(model=model.state_dict(), manifest=manifest, optimizer=optimizer.state_dict(),
                          epoch=epoch, step=step, best_score=best, torch_rng=torch.get_rng_state(),
                          python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                          cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else None)
        for name in (('last.pt', 'best.pt') if improved else ('last.pt',)):
            temporary = output/(name+'.tmp')
            torch.save(checkpoint, temporary)
            temporary.replace(output/name)


if __name__ == '__main__':
    main()
