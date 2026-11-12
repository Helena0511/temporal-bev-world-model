"""Fixed-checkpoint image, position-prior and range diagnostics for BEV perception.

This audits a predetermined validation subset; it does not retrain on validation
or turn image interventions into causal driving-performance claims.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Subset

from resworld.city_selection import apply_selection
from resworld.driving_data import DrivingDataset
from resworld.driving_model import ModelConfig
from resworld.perception_targets import PerceptionTargets, load_annotations, CLASS_NAMES
from resworld.staged_driving import StagedDrivingModel
from scripts.train_driving import make_loader, move


def summarize_confusion(confusion):
    c = confusion.double()
    result = {}
    for i, name in enumerate(CLASS_NAMES):
        tp, actual, predicted = c[i, i].item(), c[i].sum().item(), c[:, i].sum().item()
        union = actual+predicted-tp
        result[name] = dict(iou=tp/union if union else None,
            precision=tp/predicted if predicted else None, recall=tp/actual if actual else None,
            actual_cells=int(actual), predicted_cells=int(predicted), true_positive_cells=int(tp))
    foreground = [result[name]['iou'] for name in CLASS_NAMES[1:] if result[name]['iou'] is not None]
    return dict(classes=result, foreground_miou=sum(foreground)/len(foreground) if foreground else None,
                confusion=confusion.tolist())


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='runs/city_phase1/perception_full_gpu1/best.pt')
    parser.add_argument('--output', default='runs/city_phase1/perception_audit')
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.samples < 2:
        raise ValueError('at least two samples are required')
    if args.device.startswith('cuda') and (not torch.cuda.is_available() or torch.cuda.device_count()!=1):
        raise RuntimeError('CUDA diagnostics require exactly one visible GPU')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    manifest, saved = checkpoint['manifest'], checkpoint['manifest']['args']
    cfg = ModelConfig(**manifest['model_config'])
    contents = Path(saved['selection_file']).read_bytes()
    if hashlib.sha256(contents).hexdigest() != manifest['selection_sha256']:
        raise ValueError('selection file changed')
    selection = json.loads(contents)
    indices = torch.linspace(0, len(selection['val'])-1, min(args.samples,len(selection['val']))).round().long().tolist()
    tokens = [selection['val'][i] for i in indices]
    base = DrivingDataset(saved['root'], saved['version'], 'val', tuple(manifest['image_size']),
        cfg.future_steps, saved['split_file'], depth_bins=cfg.depth_bins,
        depth_min=cfg.depth_min, depth_max=cfg.depth_max, input_mode='current')
    dataset = apply_selection(base, tokens)
    annotations = load_annotations(saved['root'], saved['version'], tokens)
    dataset = PerceptionTargets(dataset, base, annotations, cfg.grid())
    cache = list(make_loader(dataset, 2, workers=2))
    model = StagedDrivingModel(cfg, 'perception', saved['ego_dropout']).to(args.device).eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    metric = cfg.grid().target_metric_points(1, args.device, torch.float32)[0, ..., :2]
    distance = metric.norm(dim=-1)
    ranges = {'all':torch.ones_like(distance,dtype=torch.bool), '0-20m':distance<20,
              '20-40m':(distance>=20)&(distance<40), '40+m':distance>=40}
    report, predictions, probability_changes = {}, {}, {}
    baseline_probabilities, baseline_classes = [], []
    visibility_counts = {name:dict(total=0, geometric_visible=0) for name in CLASS_NAMES}
    for variant in ('base', 'zero_images', 'swap_images', 'position_only'):
        totals = {name:torch.zeros(3,3,dtype=torch.long) for name in ranges}
        changes, valid_count, probability_shift = 0, 0, 0.
        predicted_maps = []
        with ExitStack() as stack:
            if variant == 'position_only':
                # Remove every image-derived attention update, retaining learned
                # BEV queries, positional embeddings, FFNs and the output head.
                for layer in model.visual.camera_bev.spatial:
                    hook = layer.output.register_forward_hook(lambda module,inputs,out:torch.zeros_like(out))
                    stack.callback(hook.remove)
            for j, raw in enumerate(cache):
                batch = move(raw,args.device)
                if variant == 'zero_images':
                    batch['images'] = torch.zeros_like(batch['images'])
                elif variant == 'swap_images':
                    batch['images'] = batch['images'].flip(0)
                outputs = model(batch)
                probability = outputs['bev_logits'].softmax(1)
                prediction = probability.argmax(1)
                target = batch['bev_target']
                valid = target!=255
                for name, region in ranges.items():
                    keep = valid & region[None]
                    totals[name] += torch.bincount((target[keep]*3+prediction[keep]).cpu(),minlength=9).reshape(3,3)
                predicted_maps.extend(prediction.cpu().unbind())
                if variant == 'base':
                    baseline_probabilities.append(probability.cpu())
                    baseline_classes.append(prediction.cpu())
                    visible = model.visual.camera_bev.last_visibility[:,0]
                    for i,name in enumerate(CLASS_NAMES):
                        cells=target==i
                        visibility_counts[name]['total'] += cells.sum().item()
                        visibility_counts[name]['geometric_visible'] += (cells&visible).sum().item()
                else:
                    changes += ((prediction.cpu()!=baseline_classes[j])&valid.cpu()).sum().item()
                    probability_shift += (probability.cpu()-baseline_probabilities[j]).abs().sum(1)[valid.cpu()].sum().item()
                    valid_count += valid.sum().item()
        report[variant] = {name:summarize_confusion(c) for name,c in totals.items()}
        predictions[variant] = torch.stack(predicted_maps)
        if variant != 'base':
            probability_changes[variant] = dict(class_flip_fraction=changes/max(1,valid_count),
                                               probability_l1_change=probability_shift/max(1,valid_count))
        print(json.dumps({'variant':variant,'metrics':report[variant]['all'],
                          'change':probability_changes.get(variant)}),flush=True)
    params=dict(total=sum(p.numel() for p in model.parameters() if p.requires_grad),
                image_backbone=sum(p.numel() for p in model.visual.camera_bev.backbone.parameters()),
                bev_queries=model.visual.camera_bev.queries.numel())
    result=dict(checkpoint=args.checkpoint,indices=indices,tokens=tokens,selection_sha256=manifest['selection_sha256'],
                samples=len(tokens),scenes=[base.scene_names[base.samples[t]['scene_token']] for t in tokens],
                parameters=params,metrics=report,image_interventions=probability_changes,
                geometric_visibility=visibility_counts,device=args.device)
    (output/'audit.json').write_text(json.dumps(result,indent=2))
    # Small diagnostic maps can be plotted without re-running the model.
    torch.save(dict(predictions=predictions,targets=torch.cat([b['bev_target'] for b in cache]),
                    front_images=torch.cat([b['images'][:,-1,1] for b in cache]),
                    config=manifest['model_config'],indices=indices,tokens=tokens),output/'maps.pt')
    print(json.dumps({'parameters':params,'geometric_visibility':visibility_counts,'output':str(output)}),flush=True)


if __name__ == '__main__':
    main()
