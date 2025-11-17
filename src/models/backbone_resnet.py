from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetEncoder(nn.Module):
    def __init__(self, output_dim: int = 512, pretrained: bool = False):
        super().__init__()
        resnet = models.resnet18(weights=None if not pretrained else models.ResNet18_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]  # remove classification layer
        self.encoder = nn.Sequential(*modules)
        self.feat_dim = resnet.fc.in_features
        self.fc: Optional[nn.Linear] = None
        if output_dim != self.feat_dim:
            self.fc = nn.Linear(self.feat_dim, output_dim)
            self.feat_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        feats = feats.view(feats.size(0), -1)
        if self.fc is not None:
            feats = self.fc(feats)
        return feats
