"""CPU-only audit of metric strictness, calibration and held-out BEV quality.

Never changes a checkpoint or training configuration. Threshold maxima and
spatial tolerances are diagnostics, not replacement benchmark results.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from resworld.city_selection import apply_selection
from resworld.driving_data import DrivingDataset, iter_json_rows, pose_matrix
from resworld.driving_model import ModelConfig
from resworld.nuscenes_io import CAMERAS
from resworld.perception_targets import CLASS_NAMES, PerceptionTargets, footprint_target
from resworld.staged_driving import StagedDrivingModel
from scripts.audit_bev_perception import summarize_confusion
from scripts.train_driving import make_loader


def dilate_disk(mask, radius):
    """Boolean Euclidean dilation in raster-cell units, without wraparound."""
    h, w = mask.shape[-2:]
    padded = F.pad(mask, (radius, radius, radius, radius))
    result = torch.zeros_like(mask)
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            if dx*dx+dy*dy <= radius*radius:
                result |= padded[..., radius+dy:radius+dy+h, radius+dx:radius+dx+w]
    return result


def confusion(target, prediction, valid):
    return torch.bincount(target[valid]*3+prediction[valid], minlength=9).reshape(3, 3)


def threshold_summary(probability, truth, valid, class_id):
    p = probability[:, class_id][valid].numpy()
    positive = (truth[valid] == class_id).numpy()
    order = np.argsort(-p, kind='stable')
    sorted_p = p[order]
    cumulative = positive[order].cumsum()
    ends = np.r_[np.flatnonzero(sorted_p[1:] != sorted_p[:-1]), len(p)-1]
    tp = cumulative[ends]
    fp = ends+1-tp
    actual = int(positive.sum())
    precision = tp/np.maximum(1, tp+fp)
    recall = tp/max(1, actual)
    iou = tp/np.maximum(1, actual+fp)
    def row(threshold):
        count = int(np.searchsorted(-sorted_p, -threshold, side='right'))
        true_positive = int(cumulative[count-1]) if count else 0
        false_positive = count-true_positive
        return dict(threshold=float(threshold), precision=true_positive/max(1,count),
                    recall=true_positive/max(1,actual),
                    iou=true_positive/max(1,actual+false_positive),predicted_cells=count)
    average_precision = float((np.diff(np.r_[0., recall])*precision).sum())
    best_index = int(iou.argmax())
    return dict(positive_cells=actual, scoring='Exact probability ranking; tied scores grouped.',
        grid_binary_ap=average_precision, oracle_best_iou=row(sorted_p[ends[best_index]]),
        curve=[row(value) for value in (1e-7,1e-6,1e-5,1e-4,.001,.005,.01,.025,.05,.075,.1,.15,.2,.3,.4,.5,.6,.7,.8,.9)],
        positive_probability_quantiles=np.quantile(p[positive], [0,.1,.25,.5,.75,.9,1]).tolist()
            if actual else None,
        interpretation='One-vs-rest grid scoring; best threshold uses these labels and is optimistic. Not object detection AP.')


def analyze(cache, cfg, base, annotations):
    target = cache['target']
    probability = cache['probability']
    prediction = probability.argmax(1)
    valid = target != 255
    xy = cfg.grid().target_metric_points(1, 'cpu', torch.float32)[0, ..., :2]
    distance = xy.norm(dim=-1)
    regions = {'all':torch.ones_like(distance, dtype=torch.bool), '0-20m':distance<20,
               '20-40m':(distance>=20)&(distance<40), '40+m':distance>=40}
    result = dict(sequences=len(target), distinct_scenes=len(set(cache['scenes'])),
                  ranges={}, tolerant_grid_coverage={}, calibration={}, object_coverage={})
    for name, region in regions.items():
        result['ranges'][name] = summarize_confusion(confusion(target, prediction, valid&region))
    c_np = np.bincount((target[valid]*3+prediction[valid]).numpy(), minlength=9).reshape(3,3)
    assert np.array_equal(c_np, np.array(result['ranges']['all']['confusion']))
    result['confusion_independent_numpy_match'] = True
    for class_id in (1,2):
        name = CLASS_NAMES[class_id]
        actual = (target == class_id)&valid
        predicted = (prediction == class_id)&valid
        result['tolerant_grid_coverage'][name] = {}
        for radius in (0,1,2):
            hit_actual = actual & dilate_disk(predicted, radius)
            hit_prediction = predicted & dilate_disk(actual, radius)
            result['tolerant_grid_coverage'][name][str(radius)] = dict(
                recall=float(hit_actual.sum()/actual.sum()) if actual.any() else None,
                precision=float(hit_prediction.sum()/predicted.sum()) if predicted.any() else None)
        result['calibration'][name] = threshold_summary(probability, target, valid, class_id)
    objects, timing = [], []
    restricted = target.clone()
    for i, token in enumerate(cache['tokens']):
        pose = base.reference_pose(token)
        retained = {c:torch.zeros_like(target[i],dtype=torch.bool) for c in (1,2)}
        predicted_xy = {c:xy[prediction[i] == c] for c in (1,2)}
        reference_time = base.lidars[token]['timestamp']
        for camera in CAMERAS:
            timing.append((base.records[token][camera]['timestamp']-reference_time)/1000)
        for annotation in annotations[token]:
            c = annotation['label']
            if c not in (1,2):
                continue
            mask = footprint_target([annotation],pose,cfg.grid()) == c
            mask &= target[i] == c
            if not mask.any():
                continue
            transform = torch.linalg.inv(pose_matrix(pose))@pose_matrix(annotation)
            centre = transform[:2,3].float()
            dist = centre.norm().item()
            visible = int(annotation['visibility_token'] or 0)
            points = annotation['num_lidar_pts']+annotation['num_radar_pts']
            if visible >= 3 and points > 0:
                retained[c] |= mask
            pred_xy = predicted_xy[c]
            nearest = (pred_xy-centre).norm(dim=-1).min().item() if len(pred_xy) else None
            objects.append(dict(token=token, annotation_token=annotation['token'],
                category=annotation['category'], class_name=CLASS_NAMES[c], distance_m=dist,
                visibility=visible, sensor_points=points, cells=int(mask.sum()),
                footprint_recalled_cells=int(((prediction[i] == c)&mask).sum()),
                nearest_predicted_cell_to_center_m=nearest,
                width_m=annotation['size'][0], length_m=annotation['size'][1]))
        for c in (1,2):
            restricted[i][(target[i] == c)&~retained[c]] = 255
    result['high_visibility_sensor_supported_within40m'] = summarize_confusion(
        confusion(restricted,prediction,(restricted != 255)&(distance < 40)))
    for name in CLASS_NAMES[1:]:
        selected = [row for row in objects if row['class_name']==name]
        group_result = {}
        predicates = {'all':lambda r:True, '0-20m':lambda r:r['distance_m']<20,
            '20-40m':lambda r:20<=r['distance_m']<40, '40+m':lambda r:r['distance_m']>=40,
            'visible60+_within40m':lambda r:r['visibility']>=3 and r['sensor_points']>0 and r['distance_m']<40,
            'visibility0-40':lambda r:r['visibility']==1,
            'visibility80-100':lambda r:r['visibility']==4,
            'no_sensor_points':lambda r:r['sensor_points']==0,
            'one_cell':lambda r:r['cells']==1, '4+_cells':lambda r:r['cells']>=4}
        for group, predicate in predicates.items():
            subset = [row for row in selected if predicate(row)]
            if not subset:
                continue
            group_result[group] = dict(objects=len(subset),
                any_same_class_footprint_hit=sum(r['footprint_recalled_cells']>0 for r in subset)/len(subset),
                center_within1m=sum(r['nearest_predicted_cell_to_center_m'] is not None and
                    r['nearest_predicted_cell_to_center_m']<=1 for r in subset)/len(subset),
                center_within2m=sum(r['nearest_predicted_cell_to_center_m'] is not None and
                    r['nearest_predicted_cell_to_center_m']<=2 for r in subset)/len(subset),
                grid_recall=sum(r['footprint_recalled_cells'] for r in subset)/sum(r['cells'] for r in subset))
        result['object_coverage'][name] = dict(groups=group_result,
            one_cell_fraction=sum(r['cells']==1 for r in selected)/max(1,len(selected)),
            category_counts=dict(Counter(r['category'] for r in selected)),
            interpretation='Coverage by any same-class raster cell, without instance matching or false-positive penalty; not detection recall.')
    result['camera_minus_lidar_timestamp_ms_quantiles'] = np.quantile(timing,[0,.05,.5,.95,1]).tolist()
    return result, objects


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',default='runs/city_phase1/perception_resnet50_ce_dice_v1/best.pt')
    parser.add_argument('--output',default='runs/city_phase1/resnet50_metric_audit_v1')
    parser.add_argument('--val-samples',type=int,default=128)
    parser.add_argument('--train-samples',type=int,default=64)
    parser.add_argument('--intervention-samples',type=int,default=16)
    args = parser.parse_args()
    if min(args.val_samples,args.train_samples,args.intervention_samples)<2:
        raise ValueError('at least two samples per analysis required')
    torch.set_num_threads(4)
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    manifest=checkpoint['manifest']; saved=manifest['args']; cfg=ModelConfig(**manifest['model_config'])
    contents=Path(saved['selection_file']).read_bytes()
    assert hashlib.sha256(contents).hexdigest()==manifest['selection_sha256']
    split_path=Path(saved['split_file']) if saved['split_file'] else Path(__file__).resolve().parents[1]/'configs/nuscenes_splits.json'
    assert hashlib.sha256(split_path.read_bytes()).hexdigest()==manifest['split_sha256']
    selection=json.loads(contents)
    chosen={split:torch.linspace(0,len(selection[split])-1,min(count,len(selection[split]))).round().long().tolist()
            for split,count in [('val',args.val_samples),('train',args.train_samples)]}
    tokens={s:[selection[s][i] for i in indices] for s,indices in chosen.items()}
    directory=Path(saved['root'])/saved['version']
    category={r['token']:r['name'] for r in json.loads((directory/'category.json').read_text())}
    instances={r['token']:category[r['category_token']] for r in json.loads((directory/'instance.json').read_text())}
    annotations={t:[] for ts in tokens.values() for t in ts}
    print(json.dumps(dict(phase='loading_annotations',samples={s:len(ts) for s,ts in tokens.items()})),flush=True)
    for row in iter_json_rows(directory/'sample_annotation.json'):
        if row['sample_token'] in annotations:
            name=instances[row['instance_token']]
            annotations[row['sample_token']].append({**row,'category':name,
                'label':1 if name.startswith('vehicle.') else (2 if name.startswith('human.') else -1)})
    model=StagedDrivingModel(cfg,'perception',saved['ego_dropout']).cpu().eval()
    model.load_state_dict(checkpoint['model'],strict=True)
    del checkpoint
    caches={}; report=dict(checkpoint=args.checkpoint,device='cpu',threads=4,
        selection_sha256=manifest['selection_sha256'],split_sha256=manifest['split_sha256'],
        model_config=asdict(cfg),sampling='Equally spaced rounded indices, fixed before seeing predictions.',
        indices=chosen,metrics={},interventions={},notes=[
            'Subset estimates do not replace the full 4819-sequence validation metric.',
            'Threshold maxima use validation labels and are diagnostic upper bounds only.',
            'Tolerant grid and object coverage are not IoU or object detection AP/recall.'])
    rows_by_split={}; examples=[]; audit_raw=[]
    for split in ('val','train'):
        base=DrivingDataset(saved['root'],saved['version'],split,tuple(manifest['image_size']),
            cfg.future_steps,saved['split_file'],depth_bins=cfg.depth_bins,
            depth_min=cfg.depth_min,depth_max=cfg.depth_max,input_mode='current')
        dataset=PerceptionTargets(apply_selection(base,tokens[split]),base,annotations,cfg.grid())
        probabilities=[]; targets=[]; scenes=[]; coverages=[]
        started=time.monotonic(); max_human=-1
        for i,batch in enumerate(make_loader(dataset,1,workers=0)):
            result=model(batch)
            probabilities.append(result['bev_logits'].softmax(1).cpu())
            targets.append(batch['bev_target']); scenes.extend(batch['scene'])
            coverages.append(model.visual.camera_bev.last_visibility[:,0].clone())
            if split=='val' and i<args.intervention_samples:
                audit_raw.append(batch)
            if split=='val':
                humans=int((batch['bev_target']==2).sum())
                example=dict(index=i,selection_index=chosen[split][i],batch=batch,
                             probability=probabilities[-1],target=targets[-1])
                if i==0:
                    examples.append(example)
                if humans>max_human:
                    max_human=humans
                    if len(examples)==1: examples.append(example)
                    else: examples[1]=example
            if (i+1)%8==0:
                print(json.dumps(dict(split=split,processed=i+1,total=len(dataset),seconds=time.monotonic()-started)),flush=True)
        cache=dict(probability=torch.cat(probabilities),target=torch.cat(targets),
                   tokens=tokens[split],scenes=scenes,geometric_coverage=torch.cat(coverages))
        caches[split]=cache
        report['metrics'][split],rows_by_split[split]=analyze(cache,cfg,base,annotations)
        report['metrics'][split]['geometric_foreground_coverage']={CLASS_NAMES[c]:float(
            ((cache['target']==c)&cache['geometric_coverage']).sum()/(cache['target']==c).sum()) for c in (1,2)}
        (output/'partial_report.json').write_text(json.dumps(report,indent=2))
    query=model.visual.camera_bev.queries
    xy=cfg.grid().target_metric_points(1,'cpu',torch.float32)[0,...,:2].reshape(-1,2)
    scale=xy.new_tensor([max(abs(cfg.x_min),abs(cfg.x_max)),max(abs(cfg.y_min),abs(cfg.y_max))])
    q=(query+model.visual.camera_bev.position(xy/scale))[None]
    report['architecture']=dict(spatial_layers=cfg.spatial_layers,channels=cfg.channels,
        image_independent_first_layer_queries=True,
        image_conditioned_resampling_layers=max(0,cfg.spatial_layers-1),
        depth_logits_used_in_spatial_sampling=False,
        learned_offset_feature_pixel_bound=2,
        first_layer_offset_std=float(model.visual.camera_bev.spatial[0].offsets(q).std()))
    for variant in ('base','zero_images','swap_images','position_only'):
        counts=torch.zeros(3,3,dtype=torch.long)
        hooks=[]
        if variant=='position_only':
            hooks=[layer.output.register_forward_hook(lambda module,inputs,out:torch.zeros_like(out))
                   for layer in model.visual.camera_bev.spatial]
        try:
            for i,batch in enumerate(audit_raw):
                if variant=='base': probability=caches['val']['probability'][i:i+1]
                else:
                    altered={**batch}
                    if variant=='zero_images': altered['images']=torch.zeros_like(batch['images'])
                    if variant=='swap_images': altered['images']=audit_raw[(i+1)%len(audit_raw)]['images']
                    probability=model(altered)['bev_logits'].softmax(1)
                target=batch['bev_target']
                counts+=confusion(target,probability.argmax(1),target!=255)
        finally:
            for hook in hooks: hook.remove()
        report['interventions'][variant]=summarize_confusion(counts)
        print(json.dumps(dict(variant=variant,foreground_miou=report['interventions'][variant]['foreground_miou'])),flush=True)
    report['intervention_indices']=chosen['val'][:len(audit_raw)]
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'objects.json').write_text(json.dumps(rows_by_split,indent=2)+'\n')
    torch.save(dict(caches=caches,examples=examples,model_config=asdict(cfg),image_size=manifest['image_size']),output/'maps.pt')
    print(json.dumps(dict(phase='complete',output=str(output))),flush=True)


if __name__=='__main__':
    main()
