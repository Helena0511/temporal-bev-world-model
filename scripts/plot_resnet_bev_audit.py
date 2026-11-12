"""Render saved CPU audit predictions and diagnostics, without model inference."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
from resworld.nuscenes_io import CAMERAS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',default='runs/city_phase1/resnet50_metric_audit_v2')
    args=parser.parse_args()
    root=Path(args.audit)
    report=json.loads((root/'report.json').read_text())
    saved=torch.load(root/'maps.pt',map_location='cpu',weights_only=False)
    cfg=saved['model_config']
    cmap=ListedColormap(['#f1f3f5','#1976d2','#f08c00','#8b8b8b'])
    error_cmap=ListedColormap(['#f1f3f5','#2b8a3e','#e03131','#9c36b5','#f7c948','#8b8b8b'])
    extent=(cfg['y_max']+.5,cfg['y_min']-.5,cfg['x_min']-.5,cfg['x_max']+.5)
    mean=torch.tensor([.485,.456,.406])[:,None,None]
    std=torch.tensor([.229,.224,.225])[:,None,None]
    for i,example in enumerate(saved['examples']):
        batch=example['batch']
        target=example['target'][0].clone()
        predicted=example['probability'][0].argmax(0)
        valid=target!=255
        error=torch.zeros_like(target)
        error[(target>0)&(target<3)&(predicted==target)]=1
        error[(target>0)&(target<3)&(predicted==0)]=2
        error[(target==0)&(predicted>0)]=3
        error[(target>0)&(target<3)&(predicted>0)&(predicted!=target)]=4
        error[~valid]=5
        target[~valid]=3
        fig,axes=plt.subplots(3,3,figsize=(16,12),gridspec_kw={'height_ratios':[1,1,1.6]})
        for j in range(6):
            image=(batch['images'][0,-1,j]*std+mean).clamp(0,1).permute(1,2,0)
            axis=axes.flatten()[j]
            axis.imshow(image)
            axis.set_title(CAMERAS[j],fontsize=10)
            axis.axis('off')
        for axis,raster,title,colors,maximum in (
            (axes[2,0],target,'GT box footprints',cmap,3),
            (axes[2,1],predicted,'ResNet50 camera BEV',cmap,3),
            (axes[2,2],error,'Green hit / Red missed / Purple false positive',error_cmap,5)):
            axis.imshow(raster,cmap=colors,vmin=0,vmax=maximum,extent=extent,origin='upper',interpolation='nearest')
            axis.scatter([0],[0],marker='x',c='black',s=30)
            axis.set_xlabel('Left y (m)'); axis.set_ylabel('Forward x (m)')
            axis.set_title(title,fontsize=10)
        rule='first fixed validation sample' if i==0 else 'most human GT cells among the fixed 128 samples'
        fig.suptitle(f"{batch['scene'][0]} / val index {example['selection_index']} / {rule}\nBlue: vehicle; orange: human; gray: ignored",fontsize=12)
        fig.tight_layout(rect=(0,0,1,.95))
        fig.savefig(root/f'example_{i}.png',dpi=150)
        plt.close(fig)
    val=report['metrics']['val']
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    regions=['0-20m','20-40m','40+m']
    for c,name in enumerate(['vehicle','human']):
        axes[0].plot(regions,[100*val['ranges'][r]['classes'][name]['iou'] for r in regions],
                     marker='o',label=name)
        tolerance=val['tolerant_grid_coverage'][name]
        axes[1].plot([0,1,2],[100*tolerance[str(r)]['recall'] for r in [0,1,2]],marker='o',label=name)
        curve=val['calibration'][name]['curve']
        axes[2].plot([100*r['recall'] for r in curve],[100*r['precision'] for r in curve],marker='.',label=name)
    axes[0].set_ylabel('Strict footprint IoU (%)'); axes[0].set_xlabel('Distance from ego')
    axes[1].set_ylabel('GT cells covered by a nearby same-class prediction (%)')
    axes[1].set_xlabel('Euclidean tolerance in raster-cell metres')
    axes[2].set_ylabel('One-vs-rest grid precision (%)'); axes[2].set_xlabel('Grid recall (%)')
    for axis in axes:
        axis.set_ylim(0,100); axis.grid(alpha=.2); axis.legend()
    fig.suptitle('Fixed 128 held-out samples; tolerance and threshold sweeps are diagnostics, not benchmark scores',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.94))
    fig.savefig(root/'diagnostics.png',dpi=150)
    plt.close(fig)
    print(root)


if __name__=='__main__':
    main()
