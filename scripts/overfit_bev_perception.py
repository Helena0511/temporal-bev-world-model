"""Controlled training-set fit tests for the perception objective/backbone.

Eight scene-diverse training examples, including human-positive frames, are fixed
before optimization. These scores measure training fit, never held-out accuracy.
"""
import argparse
from dataclasses import replace,asdict
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from resworld.city_selection import apply_selection
from resworld.driving_data import DrivingDataset
from resworld.driving_model import ModelConfig
from resworld.perception_targets import PerceptionTargets,load_annotations
from resworld.staged_driving import StagedDrivingModel,perception_loss
from scripts.audit_bev_perception import summarize_confusion
from scripts.train_driving import make_loader,move


@torch.no_grad()
def evaluate_fit(model,batches):
    model.eval()
    confusion=torch.zeros(3,3,dtype=torch.long)
    zero_confusion=torch.zeros_like(confusion)
    for batch in batches:
        target=batch['bev_target']
        valid=target!=255
        for zeros,total in ((False,confusion),(True,zero_confusion)):
            inputs={**batch,'images':torch.zeros_like(batch['images'])} if zeros else batch
            prediction=model(inputs)['bev_logits'].argmax(1)
            total+=torch.bincount((target[valid]*3+prediction[valid]).cpu(),minlength=9).reshape(3,3)
    return dict(normal=summarize_confusion(confusion),zero_images=summarize_confusion(zero_confusion))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',default='runs/city_phase1/perception_full_gpu1/best.pt')
    parser.add_argument('--output',required=True)
    parser.add_argument('--steps',type=int,default=200)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--cases',nargs='+',choices=('balanced','ce_dice','r50_low','r50_natural'),default=['balanced','ce_dice'])
    parser.add_argument('--weights',default='weights/resnet50-11ad3fa6.pth')
    parser.add_argument('--warm-fit',help='continue one identical fit case with a fresh optimizer')
    args=parser.parse_args()
    torch.set_num_threads(4)
    if args.steps<1:raise ValueError('positive steps required')
    if args.warm_fit and len(args.cases)!=1:
        raise ValueError('a fit warm start requires exactly one matching case')
    if args.device.startswith('cuda') and (not torch.cuda.is_available() or torch.cuda.device_count()!=1):
        raise RuntimeError('exactly one visible GPU is required')
    root=Path(args.output)
    if root.exists():raise ValueError('preserving existing fit experiment')
    root.mkdir(parents=True)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    saved=checkpoint['manifest']['args']
    cfg=ModelConfig(**checkpoint['manifest']['model_config'])
    pilot=json.loads(Path('runs/city_phase1/selection.json').read_text())
    source_selection=json.loads(Path(saved['selection_file']).read_text())
    assert set(pilot['train'])<=set(source_selection['train'])
    annotations=load_annotations(saved['root'],saved['version'],pilot['train'])
    humans=[t for t in pilot['train'] if any(a['label']==2 for a in annotations[t])]
    tokens=humans[:4]
    remaining=[t for t in pilot['train'] if t not in tokens]
    indices=torch.linspace(0,len(remaining)-1,8-len(tokens)).round().long().tolist()
    tokens += [remaining[i] for i in indices]
    manifest=dict(args=vars(args),tokens=tokens,selection_rule='four human-positive frames then evenly spread remaining pilot training scenes',
                  source_checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),results={},
                  interpretation='All reported scores are fixed training-set fit, not validation/generalization.')
    for case in args.cases:
        torch.manual_seed(0)
        resnet=case.startswith('r50')
        current_cfg=replace(cfg,image_backbone='resnet50') if resnet else cfg
        size=(256,448) if case=='r50_natural' else (saved['image_height'],saved['image_width'])
        base=DrivingDataset(saved['root'],saved['version'],'train',size,cfg.future_steps,saved['split_file'],
            depth_bins=cfg.depth_bins,depth_min=cfg.depth_min,depth_max=cfg.depth_max,input_mode='current')
        dataset=PerceptionTargets(apply_selection(base,tokens),base,annotations,cfg.grid())
        batches=[move(b,args.device) for b in make_loader(dataset,2,workers=2)]
        model=StagedDrivingModel(current_cfg,'perception').to(args.device)
        if resnet:
            state=model.state_dict()
            copied=[]
            for name,value in checkpoint['model'].items():
                if name.startswith(('visual.camera_bev.backbone.','visual.camera_bev.pyramid.')):
                    continue
                if name not in state or state[name].shape!=value.shape:
                    raise ValueError(f'unexpected incompatible transfer: {name}')
                state[name]=value
                copied.append(name)
            model.load_state_dict(state,strict=True)
            digest=model.visual.camera_bev.backbone.load_imagenet(args.weights)
        else:
            model.load_state_dict(checkpoint['model'],strict=True)
            digest=None
        if args.warm_fit:
            warm=torch.load(args.warm_fit,map_location='cpu',weights_only=False)
            warm_cfg=ModelConfig(**warm['model_config'])
            if (json.dumps(asdict(warm_cfg),sort_keys=True)!=json.dumps(asdict(current_cfg),sort_keys=True)
                    or tuple(warm['image_size'])!=tuple(size) or warm['tokens']!=tokens
                    or warm['objective']!=('balanced' if case=='balanced' else 'ce_dice')):
                raise ValueError('fit warm-start geometry, tokens, image size or objective mismatch')
            model.load_state_dict(warm['model'],strict=True)
            manifest['warm_fit_sha256']=hashlib.sha256(Path(args.warm_fit).read_bytes()).hexdigest()
        groups=[]
        for body in (False,True):
            parameters=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('visual.camera_bev.backbone.body.')==body]
            if parameters:groups.append(dict(params=parameters,lr=3e-5 if body else 3e-4))
        optimizer=torch.optim.AdamW(groups,weight_decay=1e-4)
        objective='balanced' if case=='balanced' else 'ce_dice'
        before=evaluate_fit(model,batches)
        started=time.monotonic()
        curve=[]
        for step in range(args.steps):
            model.train()
            batch=batches[step%len(batches)]
            optimizer.zero_grad(set_to_none=True)
            terms=perception_loss(model(batch),batch,objective)
            if not torch.isfinite(terms['total']):raise RuntimeError('non-finite fit loss')
            terms['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step()
            if (step+1)%50==0 or step+1==args.steps:
                metrics=evaluate_fit(model,batches)
                row=dict(case=case,step=step+1,losses={k:v.item() for k,v in terms.items()},metrics=metrics)
                curve.append(row)
                with (root/'progress.jsonl').open('a') as file:
                    file.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
        result=dict(model_config=asdict(current_cfg),image_size=size,objective=objective,
                    imagenet_sha256=digest,before=before,after=curve[-1]['metrics'],curve=curve,
                    seconds=time.monotonic()-started)
        manifest['results'][case]=result
        (root/'results.json').write_text(json.dumps(manifest,indent=2))
        torch.save(dict(model=model.state_dict(),model_config=asdict(current_cfg),tokens=tokens,
                        image_size=size,objective=objective,source_checkpoint=args.checkpoint),root/(case+'.pt'))
        del model,optimizer,batches
        if args.device.startswith('cuda'):torch.cuda.empty_cache()
    print(root/'results.json',flush=True)


if __name__=='__main__':
    main()
