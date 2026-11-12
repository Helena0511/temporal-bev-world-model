"""Train/evaluate one registered perception intervention on exactly one GPU.

Experimental code is separate from the running historical training entrypoint.
Micro-batch size, gradient accumulation, initialization and data order are fixed
across candidates. Checkpoints support optimizer/RNG recovery at update boundaries.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, default_collate

from resworld.city_selection import apply_selection
from resworld.driving_data import DrivingDataset, iter_json_rows, pose_matrix
from resworld.driving_model import ModelConfig
from resworld.perception_ablation import PerceptionCandidate, initialize_candidate, candidate_loss, make_center_targets
from resworld.perception_targets import CLASS_NAMES, footprint_target
from scripts.audit_bev_perception import summarize_confusion
from scripts.audit_resnet_bev_quality import dilate_disk
from scripts.train_driving import move


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n'); temporary.replace(path)


def load_labels(root, version, tokens):
    directory=Path(root)/version
    categories={r['token']:r['name'] for r in json.loads((directory/'category.json').read_text())}
    instances={r['token']:categories[r['category_token']] for r in json.loads((directory/'instance.json').read_text())}
    annotations={token:[] for token in tokens}
    for r in iter_json_rows(directory/'sample_annotation.json'):
        if r['sample_token'] in annotations:
            name=instances[r['instance_token']]
            label=1 if name.startswith('vehicle.') else (2 if name.startswith('human.') else -1)
            annotations[r['sample_token']].append({k:r[k] for k in ('translation','rotation','size')} | dict(
                label=label,visibility=int(r['visibility_token'] or 0),
                sensor_points=r['num_lidar_pts']+r['num_radar_pts']))
    return annotations


class ExperimentTargets(Dataset):
    def __init__(self, base, tokens, annotations, config, center_aux):
        self.base=base; self.dataset=apply_selection(base,tokens)
        self.annotations=annotations; self.grid=config.grid(); self.center_aux=center_aux

    def __len__(self): return len(self.dataset)

    @lru_cache(maxsize=8192)
    def supervision(self, token):
        pose=self.base.reference_pose(token)
        target=footprint_target(self.annotations[token],pose,self.grid).byte()
        inverse=torch.linalg.inv(pose_matrix(pose)); objects=[]
        for annotation in self.annotations[token]:
            if annotation['label'] not in (1,2): continue
            transform=inverse@pose_matrix(annotation)
            x,y=transform[:2,3].tolist()
            if not (self.grid.x_min<=x<=self.grid.x_max and self.grid.y_min<=y<=self.grid.y_max): continue
            objects.append(dict(xy=[x,y],label=annotation['label'],distance=math.hypot(x,y),
                visibility=annotation['visibility'],sensor_points=annotation['sensor_points']))
        return target,objects

    def __getitem__(self, index):
        batch=self.dataset[index]
        target,objects=self.supervision(batch['token'])
        batch.update(bev_target=target.long(),objects=objects)
        if self.center_aux:
            batch.update(make_center_targets(objects,self.grid,target!=255))
        return batch


def collate(items):
    result=default_collate([{k:v for k,v in item.items() if k!='objects'} for item in items])
    result['objects']=[item['objects'] for item in items]
    return result


def make_loader(dataset, batch_size, workers, seed, start_batch=0, shuffle=False):
    if shuffle:
        order=torch.randperm(len(dataset),generator=torch.Generator().manual_seed(seed)).tolist()
        dataset=Subset(dataset,order[start_batch*batch_size:])
    elif start_batch:
        dataset=Subset(dataset,list(range(start_batch*batch_size,len(dataset))))
    return DataLoader(dataset,batch_size=batch_size,num_workers=workers,collate_fn=collate,
        generator=torch.Generator().manual_seed(seed),persistent_workers=False,pin_memory=False)


def accumulation_samples(batch_index, total_samples, micro_batch, accumulation_steps):
    start=(batch_index//accumulation_steps)*accumulation_steps*micro_batch
    return min(accumulation_steps*micro_batch,total_samples-start)


@torch.no_grad()
def evaluate_candidate(model, loader, device):
    model.eval(); cfg=model.config
    xy=cfg.grid().target_metric_points(1,device,torch.float32)[0,...,:2]
    distance=xy.norm(dim=-1)
    region=(distance>=20).long()+(distance>=40).long()
    counts=torch.zeros(3,3,3,dtype=torch.long)
    covered={c:{r:0 for r in (1,2)} for c in (1,2)}
    objects={name:{CLASS_NAMES[c]:dict(total=0,center_within2m=0) for c in (1,2)}
             for name in ('all','0-20m','20-40m','40+m','supported_visible_20-40m')}
    depth_error=depth_count=sequences=0
    for raw in loader:
        batch=move(raw,device); output=model(batch)
        target=batch['bev_target']; prediction=output['bev_logits'].argmax(1); valid=target!=255
        ids=region[None]*9+target*3+prediction
        counts+=torch.bincount(ids[valid],minlength=27).reshape(3,3,3).cpu()
        depth_prediction=output['depth_logits'][:,0].argmax(2)
        depth_target=batch['depth_targets'][:,-1]; known=depth_target>=0
        depth_error+=(depth_prediction[known]-depth_target[known]).abs().sum().item()
        depth_count+=int(known.sum()); sequences+=len(target)
        for c in (1,2):
            actual=(target==c)&valid; predicted=(prediction==c)&valid
            for radius in (1,2):
                covered[c][radius]+=int((actual&dilate_disk(predicted,radius)).sum())
        for i,rows in enumerate(raw['objects']):
            for c in (1,2):
                selected=[row for row in rows if row['label']==c]
                if not selected: continue
                predicted_xy=xy[prediction[i]==c]
                if len(predicted_xy):
                    centres=xy.new_tensor([row['xy'] for row in selected])
                    near=torch.cdist(centres,predicted_xy).amin(1).cpu().tolist()
                else: near=[float('inf')]*len(selected)
                for row,nearest in zip(selected,near):
                    names=['all','0-20m' if row['distance']<20 else ('20-40m' if row['distance']<40 else '40+m')]
                    if 20<=row['distance']<40 and row['visibility']>=3 and row['sensor_points']>0:
                        names.append('supported_visible_20-40m')
                    for name in names:
                        acc=objects[name][CLASS_NAMES[c]]
                        acc['total']+=1; acc['center_within2m']+=int(nearest<=2)
    all_counts=counts.sum(0); summary=summarize_confusion(all_counts)
    for groups in objects.values():
        for row in groups.values(): row['coverage']=row['center_within2m']/row['total'] if row['total'] else None
    return dict(sequences=sequences,**summary,
        ranges={name:summarize_confusion(counts[i]) for i,name in enumerate(('0-20m','20-40m','40+m'))},
        tolerant_grid_recall={CLASS_NAMES[c]:{str(r):covered[c][r]/max(1,int(all_counts[c].sum())) for r in (1,2)} for c in (1,2)},
        object_center_coverage=objects,
        object_metric_definition='Any same-class predicted grid cell within 2m; no instance matching, not detection recall.',
        depth_quantized_mae_m=depth_error/max(1,depth_count)*(cfg.depth_max-cfg.depth_min)/(cfg.depth_bins-1),
        depth_pixels=depth_count)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite-root',required=True)
    parser.add_argument('--variant',required=True)
    parser.add_argument('--phase',choices=('screen','full'),default='screen')
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--eval-only',action='store_true')
    parser.add_argument('--eval-split',choices=('dev','official'),default='dev')
    parser.add_argument('--checkpoint')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--cuda-limit-mib',type=int,default=10000)
    args=parser.parse_args()
    suite=Path(args.suite_root).resolve(); plan=json.loads((suite/'plan.json').read_text())
    recipe=next((r for r in plan['variants'] if r['name']==args.variant),None)
    if recipe is None: raise ValueError('variant is not registered')
    if args.eval_only != bool(args.checkpoint): raise ValueError('evaluation requires a checkpoint; training uses the shared fresh initialization')
    if args.smoke and (args.resume or args.eval_only): raise ValueError('smoke uses fresh weights')
    torch.set_num_threads(args.threads); torch.manual_seed(plan['seed'])
    random.seed(plan['seed']); np.random.seed(plan['seed'])
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise RuntimeError('exactly one visible GPU required')
        torch.cuda.set_per_process_memory_fraction(args.cuda_limit_mib*2**20/torch.cuda.get_device_properties(0).total_memory)
    output=Path(args.output).resolve()
    if args.resume:
        if not (output/'last.pt').is_file(): raise ValueError('resume requires last.pt')
    else: output.mkdir(parents=True,exist_ok=False)
    def record(row):
        print(json.dumps(row),flush=True)
        with (output/'metrics.jsonl').open('a') as handle: handle.write(json.dumps(row)+'\n')
    cfg=ModelConfig(**recipe['model_config']); schedule=plan[args.phase]
    official=args.eval_only and args.eval_split=='official'
    full=args.phase=='full' or official
    selection_path=Path(plan['full_selection_file'] if full else plan['selection_file'])
    split_path=Path(plan['official_split_file'] if full else plan['split_file'])
    selection=json.loads(selection_path.read_text())
    project_root=Path(__file__).resolve().parents[1]
    source_files=sorted((project_root/'resworld').glob('*.py'))+sorted((project_root/'scripts').glob('*.py'))
    manifest=dict(variant=args.variant,phase=args.phase,recipe=recipe,schedule=schedule,
        plan_sha256=digest(suite/'plan.json'),initialization_sha256=digest(plan['initialization']),
        selection_sha256=digest(selection_path),split_sha256=digest(split_path),
        seed=plan['seed'],data_seed=plan['data_seed'],args=vars(args),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        cuda_device_count=torch.cuda.device_count() if args.device.startswith('cuda') else 0,
        image_size=recipe['image_size'],model_config=asdict(cfg),
        source_sha256={str(path.relative_to(project_root)):digest(path) for path in source_files})
    model=PerceptionCandidate(cfg,recipe['center_aux'])
    initial=torch.load(plan['initialization'],map_location='cpu',weights_only=False)
    manifest['initialization_transfer']=initialize_candidate(model,initial,recipe['allowed_config_differences'])
    del initial
    checkpoint=None
    if args.eval_only or args.resume:
        checkpoint=torch.load(args.checkpoint if args.eval_only else output/'last.pt',map_location='cpu',weights_only=False)
        for key in ('recipe','plan_sha256','initialization_sha256'):
            if checkpoint['manifest'][key]!=manifest[key]: raise ValueError(f'checkpoint mismatch: {key}')
        if args.resume:
            for key in ('schedule','selection_sha256','split_sha256','source_sha256'):
                if checkpoint['manifest'][key]!=manifest[key]: raise ValueError(f'resume mismatch: {key}')
        model.load_state_dict(checkpoint['model'],strict=True)
        if args.eval_only:
            manifest['evaluation_checkpoint_sha256']=digest(args.checkpoint)
            manifest['training_checkpoint_manifest']=checkpoint['manifest']
    model=model.to(args.device)
    active_splits=('val',) if args.eval_only else ('train','val')
    chosen={s:selection[s][:4] if args.smoke else selection[s] for s in active_splits}
    record(dict(phase='loading_data',sizes={s:len(ts) for s,ts in chosen.items()},variant=args.variant))
    annotations=load_labels(plan['dataset_root'],plan['version'],[t for ts in chosen.values() for t in ts])
    datasets={}
    for split,tokens in chosen.items():
        base=DrivingDataset(plan['dataset_root'],plan['version'],split,tuple(recipe['image_size']),
            cfg.future_steps,str(split_path),check_files=False,depth_bins=cfg.depth_bins,
            depth_min=cfg.depth_min,depth_max=cfg.depth_max,input_mode='current')
        datasets[split]=ExperimentTargets(base,tokens,annotations,cfg,recipe['center_aux'])
    manifest['sizes']={s:len(ds) for s,ds in datasets.items()}
    write_json(output/'config.json',manifest)
    eval_loader=make_loader(datasets['val'],schedule['micro_batch'],args.workers,plan['data_seed'])
    if args.eval_only:
        validation=evaluate_candidate(model,eval_loader,args.device)
        write_json(output/'result.json',dict(mode='evaluation',checkpoint=args.checkpoint,
            evaluation_split=args.eval_split,manifest=manifest,validation=validation))
        record(dict(phase='evaluation_finished',validation=validation)); return
    groups={}
    for name,p in model.named_parameters():
        if p.requires_grad:
            scale=.1 if name.startswith('core.visual.camera_bev.backbone.body.') else 1.
            groups.setdefault(scale,[]).append(p)
    optimizer=torch.optim.AdamW([dict(params=ps,lr=3e-4*scale,lr_scale=scale) for scale,ps in groups.items()],weight_decay=1e-4)
    start_epoch=start_batch=step=0; best=-float('inf')
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch=checkpoint['epoch_index']; start_batch=checkpoint['next_batch']; step=checkpoint['step']; best=checkpoint['best_miou']
        torch.set_rng_state(checkpoint['torch_rng']); random.setstate(checkpoint['python_rng']); np.random.set_state(checkpoint['numpy_rng'])
        if args.device.startswith('cuda'): torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    del checkpoint
    def save(name,epoch_index,next_batch):
        value=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),manifest=manifest,
            epoch_index=epoch_index,next_batch=next_batch,step=step,best_miou=best,
            torch_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else None)
        temporary=output/(name+'.tmp'); torch.save(value,temporary); temporary.replace(output/name)
    epochs=1 if args.smoke else schedule['epochs']
    smoke_updates=2
    for epoch in range(start_epoch,epochs):
        model.train(); optimizer.zero_grad(set_to_none=True)
        if args.device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
        for group in optimizer.param_groups:
            group['lr']=3e-4*group['lr_scale']*(.1+.9*(1+np.cos(np.pi*epoch/schedule['epochs']))/2)
        n=len(datasets['train']); begin=start_batch if epoch==start_epoch else 0
        loader=make_loader(datasets['train'],schedule['micro_batch'],args.workers,
            plan['data_seed']+epoch,begin,shuffle=True)
        total_batches=math.ceil(n/schedule['micro_batch']); started=time.monotonic()
        if args.smoke:
            # Two independent optimizer updates on the same real micro-batch.
            raw=next(iter(loader)); batches=[raw]*smoke_updates*schedule['accumulation_steps']
            total_batches=len(batches); n=total_batches*len(raw['images']); begin=0
        else: batches=loader
        for batch_index,raw in enumerate(batches,begin):
            batch=move(raw,args.device); outputs=model(batch)
            terms=candidate_loss(outputs,batch,recipe['focal'],recipe['center_weight'])
            if not torch.isfinite(terms['total']): raise RuntimeError('non-finite loss')
            normalizer=accumulation_samples(batch_index,n,schedule['micro_batch'],schedule['accumulation_steps'])
            (terms['total']*len(batch['images'])/normalizer).backward()
            boundary=(batch_index+1)%schedule['accumulation_steps']==0 or batch_index+1==total_batches
            if not boundary: continue
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step(); optimizer.zero_grad(set_to_none=True); step+=1
            if step%12==0 or args.smoke:
                record(dict(epoch=epoch+1,step=step,batch_index=batch_index+1,
                    losses={k:v.detach().item() for k,v in terms.items()},seconds=time.monotonic()-started))
            del outputs,terms,batch
            if step%200==0 and not args.smoke: save('last.pt',epoch,batch_index+1)
        validation=evaluate_candidate(model,eval_loader,args.device)
        improved=validation['foreground_miou']>best
        best=max(best,validation['foreground_miou'])
        peak=torch.cuda.max_memory_allocated()/2**20 if args.device.startswith('cuda') else None
        result=dict(epoch=epoch+1,step=step,validation=validation,seconds=time.monotonic()-started,
                    cuda_peak_allocated_mib=peak,smoke=args.smoke)
        record(result)
        if not args.smoke:
            save('last.pt',epoch+1,0)
            if improved: save('best.pt',epoch+1,0)
        write_json(output/'result.json',result)
    record(dict(phase='complete',step=step,best_miou=best,smoke=args.smoke))


if __name__=='__main__': main()
