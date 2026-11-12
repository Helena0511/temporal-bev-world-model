"""Inspect visible missed pedestrians and their actual model-input pixel size."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import torch
from resworld.driving_data import iter_json_rows, load_metadata, pose_matrix
from resworld.driving_model import ModelConfig
from resworld.nuscenes_io import CAMERAS


def best_projection(annotation,token,records,calibrations,poses,image_size):
    width,length,height=annotation['size']
    xyz=torch.tensor([[x*length/2,y*width/2,z*height/2,1.]
                      for x in (-1,1) for y in (-1,1) for z in (-1,1)],dtype=torch.float64)
    world=xyz@pose_matrix(annotation).T
    candidates=[]
    for camera in CAMERAS:
        row=records[token][camera]
        calibration=calibrations[row['calibrated_sensor_token']]
        global_from_camera=pose_matrix(poses[row['ego_pose_token']])@pose_matrix(calibration)
        points=(world@torch.linalg.inv(global_from_camera).T)[:,:3]
        if (points[:,2] <= .1).any():
            continue
        pixels=points@torch.tensor(calibration['camera_intrinsic'],dtype=torch.float64).T
        uv=pixels[:,:2]/pixels[:,2:]
        low=uv.amin(0).clamp_min(0)
        high=torch.minimum(uv.amax(0),torch.tensor([row['width'],row['height']]))
        size=(high-low).clamp_min(0)
        if size.prod() <= 0:
            continue
        model_width=float(size[0])*image_size[1]/row['width']
        model_height=float(size[1])*image_size[0]/row['height']
        candidates.append(dict(camera=camera,filename=row['filename'],
            native_bbox=[*low.tolist(),*high.tolist()],native_size=[row['width'],row['height']],
            width_px=model_width,height_px=model_height,area=model_width*model_height))
    return max(candidates,key=lambda row:row['area']) if candidates else None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',default='runs/city_phase1/resnet50_metric_audit_v2')
    args=parser.parse_args()
    torch.set_num_threads(4)
    output=Path(args.audit)
    report=json.loads((output/'report.json').read_text())
    maps=torch.load(output/'maps.pt',map_location='cpu',weights_only=False)
    objects=json.loads((output/'objects.json').read_text())['val']
    checkpoint=torch.load(report['checkpoint'],map_location='cpu',weights_only=False)
    saved=checkpoint['manifest']['args']; del checkpoint
    _,calibrations,records,lidars,poses,_=load_metadata(saved['root'],saved['version'])
    wanted={row['annotation_token'] for row in objects}
    annotations={r['token']:r for r in iter_json_rows(Path(saved['root'])/saved['version']/'sample_annotation.json') if r['token'] in wanted}
    enriched=[]
    for row in objects:
        projection=best_projection(annotations[row['annotation_token']],row['token'],records,
                                   calibrations,poses,maps['image_size'])
        enriched.append({**row,'projection':projection})
    result=dict(groups={},examples=[])
    for name in ('vehicle','human'):
        result['groups'][name]={}
        for group,lo,hi in [('0-20m',0,20),('20-40m',20,40),('40+m',40,float('inf'))]:
            selected=[r for r in enriched if r['class_name']==name and lo<=r['distance_m']<hi]
            projected=[r for r in selected if r['projection']]
            good=[r for r in selected if r['visibility']>=3 and r['sensor_points']>0]
            result['groups'][name][group]=dict(objects=len(selected),projectable=len(projected),
                model_input_width_px_quantiles=np.quantile([r['projection']['width_px'] for r in projected],[.1,.5,.9]).tolist() if projected else None,
                model_input_height_px_quantiles=np.quantile([r['projection']['height_px'] for r in projected],[.1,.5,.9]).tolist() if projected else None,
                high_visibility_supported_objects=len(good),
                high_visibility_supported_any_footprint_hits=sum(r['footprint_recalled_cells']>0 for r in good),
                high_visibility_supported_center_within2m=sum(r['nearest_predicted_cell_to_center_m'] is not None and r['nearest_predicted_cell_to_center_m']<=2 for r in good))
    cache=maps['caches']['val']; cfg=ModelConfig(**maps['model_config'])
    for label,lo,hi in [('near',0,20),('mid',20,40)]:
        candidates=[r for r in enriched if r['class_name']=='human' and lo<=r['distance_m']<hi
                    and r['visibility']==4 and r['sensor_points']>0 and r['footprint_recalled_cells']==0
                    and r['projection']]
        if not candidates: continue
        chosen=sorted(candidates,key=lambda r:(-r['sensor_points'],r['annotation_token']))[0]
        projection=chosen['projection']; token=chosen['token']
        annotation=annotations[chosen['annotation_token']]
        ego_from_global=torch.linalg.inv(pose_matrix(poses[lidars[token]['ego_pose_token']]))
        center=(ego_from_global@torch.tensor([*annotation['translation'],1.],dtype=torch.float64))[:2]
        image=Image.open(Path(saved['root'])/projection['filename']).convert('RGB')
        x0,y0,x1,y1=projection['native_bbox']; bw=x1-x0; bh=y1-y0
        crop_box=(max(0,x0-bw),max(0,y0-bh*.5),min(image.width,x1+bw),min(image.height,y1+bh*.5))
        crop=image.crop(tuple(int(round(v)) for v in crop_box))
        resized=image.resize((maps['image_size'][1],maps['image_size'][0]),Image.Resampling.BILINEAR)
        fig,axes=plt.subplots(1,3,figsize=(15,4.8))
        axes[0].imshow(crop); axes[0].axis('off'); axes[0].set_title('Native-resolution crop')
        axes[0].add_patch(Rectangle((x0-crop_box[0],y0-crop_box[1]),bw,bh,fill=False,color='#f08c00',linewidth=2))
        axes[1].imshow(resized); axes[1].axis('off')
        sx,sy=resized.width/image.width,resized.height/image.height
        axes[1].add_patch(Rectangle((x0*sx,y0*sy),bw*sx,bh*sy,fill=False,color='#f08c00',linewidth=1))
        axes[1].set_title(f"Model input: {projection['width_px']:.1f} x {projection['height_px']:.1f} px")
        index=cache['tokens'].index(token)
        probability=cache['probability'][index,2]
        x,y=center.tolist()
        extent=(cfg.y_max+.5,cfg.y_min-.5,cfg.x_min-.5,cfg.x_max+.5)
        heat=axes[2].imshow(probability,vmin=0,vmax=1,cmap='magma',origin='upper',extent=extent)
        axes[2].scatter([y],[x],marker='*',c='#2bff88',s=100,label='GT center')
        axes[2].set_xlim(y+6,y-6); axes[2].set_ylim(x-6,x+6)
        axes[2].set_title('Predicted human probability / local BEV')
        axes[2].set_xlabel('Left y (m)'); axes[2].set_ylabel('Forward x (m)'); axes[2].legend()
        fig.colorbar(heat,ax=axes[2],fraction=.046)
        rr,cc=cfg.grid().metric_to_row_col(x,y)
        center_probability=float(probability[rr,cc])
        fig.suptitle(f"Visible missed pedestrian: {chosen['distance_m']:.1f}m / visibility 80-100% / {chosen['sensor_points']} sensor points / p(human) at GT center {center_probability:.4f}\nSelection: most sensor points among missed high-visibility pedestrians in this distance band",fontsize=10)
        fig.tight_layout(rect=(0,0,1,.88))
        fig.savefig(output/f'missed_human_{label}.png',dpi=160); plt.close(fig)
        result['examples'].append({**chosen,'ego_center_xy_m':center.tolist(),'center_human_probability':center_probability})
    result['notes']=['Pixel dimensions use the largest clipped camera projection of each GT box.',
        'Visibility is annotation metadata; selected examples are diagnostic and not a frequency estimate.',
        'Orange rectangles are GT annotation projections, not model detections.']
    (output/'projection_sizes.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
