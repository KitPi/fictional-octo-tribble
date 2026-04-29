import torch
import torch.nn as nn
import torchvision.models as models

net = models.segmentation.fcn_resnet50(pretrained=False, num_classes=2, pretrained_backbone=False)
net.backbone.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

class FloodMaskModel(nn.Module):
    def __init__(self):
        super(FloodMaskModel, self).__init__()
        self.model = net

    def forward(self, x):
        return self.model(x)['out']