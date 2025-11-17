from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone_resnet import ResNetEncoder
from .predictor import MLPPredictor
from .projector import MLPProjector


def _update_moving_average(target: nn.Module, online: nn.Module, momentum: float):
    with torch.no_grad():
        for tgt_param, online_param in zip(target.parameters(), online.parameters()):
            tgt_param.data = momentum * tgt_param.data + (1.0 - momentum) * online_param.data


class JEPAModel(nn.Module):
    def __init__(
        self,
        backbone_dim: int = 512,
        projector_dim: int = 256,
        projector_hidden: int = 1024,
        predictor_hidden: int = 512,
        momentum: float = 0.99,
    ):
        super().__init__()
        self.online_encoder = ResNetEncoder(output_dim=backbone_dim)
        self.online_projector = MLPProjector(backbone_dim, [projector_hidden], projector_dim)
        self.online_predictor = MLPPredictor(projector_dim, [predictor_hidden], projector_dim)

        self.target_encoder = ResNetEncoder(output_dim=backbone_dim)
        self.target_projector = MLPProjector(backbone_dim, [projector_hidden], projector_dim)
        self.momentum = momentum
        self._init_target()

    @torch.no_grad()
    def _init_target(self):
        for target_param, online_param in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
            target_param.data.copy_(online_param.data)
        for target_param, online_param in zip(self.target_projector.parameters(), self.online_projector.parameters()):
            target_param.data.copy_(online_param.data)
        self.target_encoder.eval()
        self.target_projector.eval()

    @torch.no_grad()
    def update_target(self):
        _update_moving_average(self.target_encoder, self.online_encoder, self.momentum)
        _update_moving_average(self.target_projector, self.online_projector, self.momentum)

    def forward(self, context_view: torch.Tensor, target_view: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        online_feat = self.online_encoder(context_view)
        online_proj = self.online_projector(online_feat)
        pred = self.online_predictor(online_proj)

        with torch.no_grad():
            target_feat = self.target_encoder(target_view)
            target_proj = self.target_projector(target_feat)

        pred_norm = F.normalize(pred, dim=-1)
        target_norm = F.normalize(target_proj.detach(), dim=-1)
        per_sample = 2 - 2 * (pred_norm * target_norm).sum(dim=-1)
        stats = {
            "online_feat_norm": online_feat.norm(dim=-1).mean().item(),
            "target_feat_norm": target_feat.norm(dim=-1).mean().item(),
        }
        return per_sample, stats
