"""Alternative BEV supervision; retain class-mean CE for historical controls."""
import torch
from torch.nn import functional as F


def bev_segmentation_loss(logits, target, mode='balanced'):
    valid = target!=255
    pixel_loss = F.cross_entropy(logits,target,ignore_index=255,reduction='none')
    if not valid.any():
        return logits.sum()*0
    if mode == 'balanced':
        means = [pixel_loss[target==i].mean() for i in range(logits.shape[1]) if (target==i).any()]
        return torch.stack(means).mean()
    if mode != 'ce_dice':
        raise ValueError('unknown BEV segmentation objective')
    # Pixel-frequency CE constrains false positives. Foreground Dice keeps
    # sparse vehicle/human masks in the objective without inverse-frequency CE.
    ce = pixel_loss[valid].mean()
    probability = logits.softmax(1)*valid[:,None]
    one_hot = F.one_hot(target.clamp_max(logits.shape[1]-1),logits.shape[1]).permute(0,3,1,2)
    one_hot = one_hot.to(logits)*valid[:,None]
    intersection = (probability*one_hot).sum((0,2,3))
    volume = probability.sum((0,2,3))+one_hot.sum((0,2,3))
    dice = (2*intersection[1:]+1)/(volume[1:]+1)
    return ce+(1-dice).mean()
