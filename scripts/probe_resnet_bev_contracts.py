"""Verify architecture dependencies and quantify IoU sensitivity on saved maps."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from resworld.driving_model import ModelConfig
from resworld.staged_driving import StagedDrivingModel
from scripts.audit_bev_perception import summarize_confusion
from scripts.audit_resnet_bev_quality import confusion


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',default='runs/city_phase1/resnet50_metric_audit_v2')
    args=parser.parse_args()
    torch.set_num_threads(4)
    root=Path(args.audit)
    report=json.loads((root/'report.json').read_text())
    maps=torch.load(root/'maps.pt',map_location='cpu',weights_only=False)
    checkpoint=torch.load(report['checkpoint'],map_location='cpu',weights_only=False)
    cfg=ModelConfig(**checkpoint['manifest']['model_config'])
    model=StagedDrivingModel(cfg,'perception').cpu().eval()
    model.load_state_dict(checkpoint['model'],strict=True)
    del checkpoint
    batch=maps['examples'][0]['batch']
    recorded={}; outputs={}
    for variant in ('normal','zero_images'):
        captured={name:[] for name in ('offsets','weights','view_weights')}
        hooks=[]
        for name in captured:
            hooks.append(getattr(model.visual.camera_bev.spatial[0],name).register_forward_hook(
                lambda module,inputs,out,key=name:captured[key].append(out.clone())))
        try:
            altered=batch if variant=='normal' else {**batch,'images':torch.zeros_like(batch['images'])}
            outputs[variant]=model(altered)
        finally:
            for hook in hooks: hook.remove()
        recorded[variant]=captured
    result=dict(same_geometry_different_images={name:all(torch.equal(a,b) for a,b in zip(
        recorded['normal'][name],recorded['zero_images'][name])) for name in recorded['normal']},
        image_change_bev_logit_mean_abs=float((outputs['normal']['bev_logits']-outputs['zero_images']['bev_logits']).abs().mean()))
    hook=model.visual.camera_bev.depth_head.register_forward_hook(lambda module,inputs,out:torch.zeros_like(out))
    try:
        altered=model(batch)
    finally:
        hook.remove()
    result['zeroing_depth_logits_leaves_bev_exactly_equal']=torch.equal(altered['bev_logits'],outputs['normal']['bev_logits'])
    result['depth_logits_original_abs_mean']=float(outputs['normal']['depth_logits'].abs().mean())
    result['depth_logits_altered_abs_max']=float(altered['depth_logits'].abs().max())
    truth=maps['caches']['val']['target']
    valid=truth!=255
    cleaned=truth.clone(); cleaned[~valid]=0
    shifted={}
    for name in ('1m_forward','1m_left'):
        prediction=torch.zeros_like(cleaned)
        if name=='1m_forward': prediction[:,:-1,:]=cleaned[:,1:,:]
        else: prediction[:,:,:-1]=cleaned[:,:,1:]
        shifted[name]=summarize_confusion(confusion(truth,prediction,valid))
    result['perfect_gt_mask_translated_one_meter']=shifted
    result['interpretation']='Translated GT masks isolate strict metric sensitivity. Offset and attention weights are image-independent in the single spatial layer; output features still use image content. Depth shares backbone supervision but is not used explicitly for spatial sampling.'
    (root/'probes.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
