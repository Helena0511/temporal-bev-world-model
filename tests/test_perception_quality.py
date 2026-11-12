"""Known-coordinate sampling, loss calibration and pretrained-backbone contracts."""
import unittest

import torch
from torch import nn

from resworld.bev_attention import SpatialCrossAttention
from resworld.driving_model import ModelConfig
from resworld.image_backbone import ResNet50FPN
from resworld.perception_objectives import bev_segmentation_loss
from resworld.staged_driving import StagedDrivingModel


class PerceptionQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads=torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_spatial_samples_correct_pixel_camera_head_and_level(self):
        layer=SpatialCrossAttention(4,2,2,1,points=1,chunk_size=1)
        layer.output=nn.Identity()
        layer.norm=nn.Identity()
        layer.ffn_norm=nn.Identity()
        layer.ffn=nn.Linear(4,4,bias=False)
        with torch.no_grad():
            for module in (layer.offsets,layer.weights,layer.view_weights,layer.ffn):
                for p in module.parameters(): p.zero_()
        features=[]
        for stride,size in ((8,4),(16,3)):
            yy,xx=torch.meshgrid(torch.arange(size),torch.arange(size),indexing='ij')
            physical=xx*stride+10*yy*stride
            channels=torch.arange(4)[:,None,None]*1000
            views=torch.arange(6)[:,None,None,None]*10000
            features.append((physical+channels+views).float()[None])
        uv=torch.tensor([8.,16.]).expand(1,6,1,1,2)
        visible=torch.zeros(1,6,1,1,dtype=torch.bool)
        visible[:,0]=True
        output,_=layer(torch.zeros(1,1,4),features,uv,visible,(8,16))
        torch.testing.assert_close(output[0,0],torch.tensor([168.,1168.,2168.,3168.]))

    def test_class_mean_ce_has_a_uniform_constant_optimum_despite_rare_labels(self):
        target=torch.tensor([0]*1000+[1]*10+[2]).reshape(1,1,-1)
        constant=torch.zeros(1,3,1,1,requires_grad=True)
        loss=bev_segmentation_loss(constant.expand(1,3,1,target.numel()),target,'balanced')
        loss.backward()
        torch.testing.assert_close(constant.grad,torch.zeros_like(constant),atol=1e-7,rtol=0)
        constant.grad=None
        bev_segmentation_loss(constant.expand(1,3,1,target.numel()),target,'ce_dice').backward()
        self.assertLess(constant.grad[0,0,0,0].item(),0)
        self.assertGreater(constant.grad[0,2,0,0].item(),0)

    def test_ce_dice_ignores_unknown_cells_and_handles_an_empty_target(self):
        logits=torch.randn(1,3,2,2,requires_grad=True)
        target=torch.tensor([[[0,1],[2,255]]])
        loss=bev_segmentation_loss(logits,target,'ce_dice')
        changed=logits.detach().clone()
        changed[:,:,1,1]=10000
        torch.testing.assert_close(loss,bev_segmentation_loss(changed,target,'ce_dice'))
        loss.backward()
        torch.testing.assert_close(logits.grad[:,:,1,1],torch.zeros(1,3))
        empty=bev_segmentation_loss(logits,torch.full_like(target,255),'ce_dice')
        self.assertEqual(empty.item(),0)
        self.assertTrue(torch.isfinite(empty))

    def test_resnet_pyramid_stride_and_frozen_batchnorm(self):
        backbone=ResNet50FPN(8).train()
        self.assertEqual(sum(p.numel() for p in backbone.body.parameters()),23508032)
        before=backbone.body.layer2[0].bn1.running_mean.clone()
        outputs=backbone(torch.randn(2,3,64,96))
        self.assertEqual([tuple(t.shape) for t in outputs],[(2,8,8,12),(2,8,4,6)])
        torch.testing.assert_close(backbone.body.layer2[0].bn1.running_mean,before,rtol=0,atol=0)
        sum(t.square().mean() for t in outputs).backward()
        self.assertIsNone(backbone.body.conv1.weight.grad)
        self.assertGreater(backbone.body.layer2[0].conv1.weight.grad.abs().sum().item(),0)

    def test_joint_stage_preserves_permanently_frozen_imagenet_parameters(self):
        cfg=ModelConfig(channels=8,tokens=2,heads=2,bev_height=8,bev_width=8,
                        depth_bins=4,perception='bev_query',image_backbone='resnet50')
        model=StagedDrivingModel(cfg,'joint').train()
        backbone=model.visual.camera_bev.backbone
        self.assertTrue(all(not p.requires_grad for p in backbone.frozen_parameters()))
        self.assertTrue(backbone.body.layer2[0].conv1.weight.requires_grad)
        self.assertFalse(backbone.body.layer2[0].bn1.training)


if __name__=='__main__':
    unittest.main()
