"""Native ResNet-50 feature pyramid compatible with official TorchVision weights.

Architecture/weight naming follows the documented ResNet-50 v1.5 layout:
https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
No torchvision binary operators are required. Classification weights are omitted.
"""
import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, incoming, width, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(incoming, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, width*4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(width*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        value = self.relu(self.bn1(self.conv1(x)))
        value = self.relu(self.bn2(self.conv2(value)))
        return self.relu(self.bn3(self.conv3(value))+identity)


class ResNet50Features(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        incoming = 64
        for stage, (width, blocks, stride) in enumerate(((64,3,1),(128,4,2),(256,6,2),(512,3,2)),1):
            downsample = nn.Sequential(nn.Conv2d(incoming,width*4,1,stride=stride,bias=False),
                                       nn.BatchNorm2d(width*4))
            layers = [Bottleneck(incoming,width,stride,downsample)]
            layers.extend(Bottleneck(width*4,width) for _ in range(blocks-1))
            setattr(self,'layer'+str(stage),nn.Sequential(*layers))
            incoming = width*4
        for module in self.modules():
            if isinstance(module,nn.Conv2d):
                nn.init.kaiming_normal_(module.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(module,nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, images):
        feature = self.maxpool(self.relu(self.bn1(self.conv1(images))))
        feature = self.layer1(feature)
        c3 = self.layer2(feature)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c3,c4,c5


class ResNet50FPN(nn.Module):
    """Stride-8/16 pyramid, including stride-32 semantic context from the body.

    All ImageNet BN statistics/affine parameters, stem and layer1 stay frozen.
    Small driving batches therefore never update ImageNet running statistics.
    """
    def __init__(self, channels):
        super().__init__()
        self.body = ResNet50Features()
        self.lateral = nn.ModuleList(nn.Conv2d(c,channels,1) for c in (512,1024,2048))
        self.output = nn.ModuleList(nn.Sequential(nn.Conv2d(channels,channels,3,padding=1),
            nn.GroupNorm(4,channels),nn.SiLU()) for _ in range(2))
        self.apply_freezing()
        self.train(self.training)

    def frozen_parameters(self):
        frozen = {id(p):p for module in (self.body.conv1,self.body.bn1,self.body.layer1)
                  for p in module.parameters()}
        for module in self.body.modules():
            if isinstance(module,nn.BatchNorm2d):
                frozen.update({id(p):p for p in module.parameters()})
        return list(frozen.values())

    def apply_freezing(self):
        for parameter in self.frozen_parameters():
            parameter.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        for module in self.body.modules():
            if isinstance(module,nn.BatchNorm2d):
                module.eval()
        return self

    def forward(self, images):
        c3,c4,c5 = self.body(images)
        p5 = self.lateral[2](c5)
        p4 = self.lateral[1](c4)+F.interpolate(p5,size=c4.shape[-2:],mode='nearest')
        p3 = self.lateral[0](c3)+F.interpolate(p4,size=c3.shape[-2:],mode='nearest')
        return self.output[0](p3),self.output[1](p4)

    def load_imagenet(self, path):
        path = Path(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = {'resnet50-11ad3fa6.pth':'11ad3fa6','resnet50-0676ba61.pth':'0676ba61'}.get(path.name)
        if expected and not digest.startswith(expected):
            raise ValueError('ImageNet weight checksum mismatch')
        weights = torch.load(path,map_location='cpu',weights_only=True)
        weights = {name:value for name,value in weights.items() if not name.startswith('fc.')}
        self.body.load_state_dict(weights,strict=True)
        return digest
