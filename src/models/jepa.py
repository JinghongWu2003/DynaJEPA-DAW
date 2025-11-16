"""Simplified Joint-Embedding Predictive Architecture."""

from __future__ import annotations

from copy import deepcopy
from typing import Tuple

import torch
from torch import nn

from .encoder_backbone import ConvEncoder


def _projection_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )


def _prediction_mlp(dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, dim),
    )


class JEPAModel(nn.Module):
    def __init__(self, feature_dim: int = 256, projector_hidden: int = 512, predictor_hidden: int = 512, ema_decay: float = 0.99):
        super().__init__()
        self.online_encoder = ConvEncoder(feature_dim=feature_dim)
        self.online_projector = _projection_mlp(feature_dim, projector_hidden, feature_dim)
        self.predictor = _prediction_mlp(feature_dim, predictor_hidden)

        # target network
        self.target_encoder = deepcopy(self.online_encoder)
        self.target_projector = deepcopy(self.online_projector)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

        self.ema_decay = ema_decay
        self.loss_fn = nn.CosineSimilarity(dim=1)

    @torch.no_grad()
    def _update_target(self) -> None:
        for online, target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target.data = target.data * self.ema_decay + online.data * (1 - self.ema_decay)
        for online, target in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            target.data = target.data * self.ema_decay + online.data * (1 - self.ema_decay)

    def forward(self, context: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        online_feat = self.online_encoder(context)
        online_proj = self.online_projector(online_feat)
        pred = self.predictor(online_proj)

        with torch.no_grad():
            target_feat = self.target_encoder(target)
            target_proj = self.target_projector(target_feat)

        pred_norm = nn.functional.normalize(pred, dim=1)
        target_norm = nn.functional.normalize(target_proj, dim=1)
        cos_sim = self.loss_fn(pred_norm, target_norm)
        per_sample_loss = 1 - cos_sim  # cosine distance
        mean_loss = per_sample_loss.mean()
        return per_sample_loss, mean_loss

    def update_moving_average(self) -> None:
        self._update_target()

