"""Prepare a fixed scene-disjoint development experiment; no GPU work."""
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
import torch
from resworld.driving_model import ModelConfig
from resworld.staged_driving import StagedDrivingModel


def round_robin(groups, count, rng):
    names = sorted(groups)
    rng.shuffle(names)
    remaining = {name:list(groups[name]) for name in names}
    for rows in remaining.values(): rng.shuffle(rows)
    result = []
    while len(result)<count:
        before = len(result)
        for name in names:
            if remaining[name]: result.append(remaining[name].pop())
            if len(result)==count: break
        if len(result)==before: raise ValueError('not enough development samples')
    return result


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(PROJECT/'runs/city_phase1/perception_improvements_v1'))
    args=parser.parse_args()
    root=Path(args.output).resolve()
    root.mkdir(parents=True,exist_ok=False)
    prepared=root/'prepared'; prepared.mkdir()
    source_path=PROJECT/'runs/city_phase1/full_selection.json'
    source=json.loads(source_path.read_text())
    official_path=PROJECT/'configs/nuscenes_splits.json'
    official=json.loads(official_path.read_text())
    groups=defaultdict(list)
    for row in source['records']['train']: groups[row['scene']].append(row)
    rng=random.Random(20261015)
    available=sorted(name for name,rows in groups.items() if len(rows)>=8)
    rng.shuffle(available)
    dev_scenes=set(available[:32])
    train=round_robin({k:v for k,v in groups.items() if k not in dev_scenes},2048,rng)
    dev=round_robin({k:v for k,v in groups.items() if k in dev_scenes},256,rng)
    assert not ({r['scene'] for r in train}&{r['scene'] for r in dev})
    assert not (dev_scenes&set(official['val']))
    splits=dict(train=sorted(set(official['train'])-dev_scenes),val=sorted(dev_scenes),
                source='Official training scenes partitioned for development; official validation untouched.')
    selection=dict(version=source['version'],seed=20261015,
        train=[r['token'] for r in train],val=[r['token'] for r in dev],
        definition='Scene round-robin from existing city-condition official training selection.',
        records=dict(train=train,val=dev),
        source_selection_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        source_official_split_sha256=hashlib.sha256(official_path.read_bytes()).hexdigest())
    for name,data in [('selection.json',selection),('splits.json',splits)]:
        (prepared/name).write_text(json.dumps(data,indent=2)+'\n')
    cfg=ModelConfig(**json.loads((PROJECT/'configs/bev_attention_city60_resnet50.json').read_text()))
    cfg=replace(cfg,use_ego_state=False,kinematic_prior=False)
    variants=[]
    for name,changes,image_size,focal,center in [
        ('baseline',{},[256,448],False,False),
        ('spatial3',{'spatial_layers':3},[256,448],False,False),
        ('resolution384',{},[384,672],False,False),
        ('focal_dice',{},[256,448],True,False),
        ('channels64',{'channels':64},[256,448],False,False),
        ('center_aux',{},[256,448],False,True)]:
        variants.append(dict(name=name,model_config=asdict(replace(cfg,**changes)),
            allowed_config_differences=sorted(changes),image_size=image_size,
            focal=focal,center_aux=center,center_weight=.1))
    torch.set_num_threads(4); torch.manual_seed(0)
    model=StagedDrivingModel(cfg,'perception')
    weights=PROJECT/'weights/resnet50-11ad3fa6.pth'
    digest=model.visual.camera_bev.backbone.load_imagenet(weights)
    torch.save(dict(model=model.state_dict(),model_config=asdict(cfg),seed=0,
                    imagenet_sha256=digest,trained_checkpoint=False),prepared/'initialization.pt')
    source_manifest=json.loads((PROJECT/'runs/city_phase1/perception_resnet50_ce_dice_v1/config.json').read_text())
    plan=dict(root=str(root),dataset_root=source_manifest['args']['root'],version=source['version'],
        dependency=str(PROJECT/'runs/city_phase1/resnet50_staged_v2'),
        selection_file=str(prepared/'selection.json'),split_file=str(prepared/'splits.json'),
        full_selection_file=str(source_path),official_split_file=str(official_path),
        initialization=str(prepared/'initialization.pt'),imagenet_sha256=digest,
        variants=variants,seed=0,data_seed=12345,
        screen=dict(epochs=4,micro_batch=2,accumulation_steps=2,effective_batch=4,updates=2048),
        full=dict(epochs=4,micro_batch=2,accumulation_steps=4,effective_batch=8,updates=11240),
        max_full_promotions=2,
        promotion_rule=dict(min_foreground_miou_gain=0.,min_human_iou_gain=0.,
            max_vehicle_iou_drop=.01,max_human_precision_drop=.02),
        notes=['Every candidate is compared with the new, budget-matched control, not the historical full-data model.',
            'Screen development data come solely from original official training scenes.',
            'Dice is computed per fixed micro-batch and its gradients averaged consistently across all candidates.',
            'Architecture changes use explicit compatible initialization with every skipped/new tensor recorded.',
            'All screen checkpoints receive official validation evaluation; promotions are chosen using development only.',
            'At most two development winners and a matched baseline receive full-data retraining.'])
    (root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    summary=dict(phase='prepared',train=len(train),dev=len(dev),
        train_scenes=len({r['scene'] for r in train}),dev_scenes=len(dev_scenes),
        dev_routes=dict(Counter(r['command'] for r in dev)),
        variants=[r['name'] for r in variants])
    (root/'status.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
