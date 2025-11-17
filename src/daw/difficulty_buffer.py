from typing import Dict

import torch


class DifficultyBuffer:
    def __init__(self, dataset_size: int, alpha: float = 0.9, device: str = "cpu"):
        self.alpha = alpha
        self.device = device
        self.ema_loss = torch.zeros(dataset_size, device=device)

    @torch.no_grad()
    def update(self, indices: torch.Tensor, per_sample_loss: torch.Tensor):
        indices = indices.to(self.device)
        values = per_sample_loss.to(self.device)
        self.ema_loss[indices] = self.alpha * self.ema_loss[indices] + (1 - self.alpha) * values

    @torch.no_grad()
    def get_difficulty(self, indices: torch.Tensor, mode: str = "instant", per_sample_loss: torch.Tensor | None = None):
        indices = indices.to(self.device)
        if mode == "instant":
            if per_sample_loss is None:
                raise ValueError("per_sample_loss required for instant difficulty")
            return per_sample_loss.detach().to(self.device)
        if mode == "ema":
            return self.ema_loss[indices].detach()
        raise ValueError(f"Unknown difficulty mode: {mode}")

    def state_dict(self) -> Dict:
        return {"ema_loss": self.ema_loss.cpu(), "alpha": self.alpha}

    def load_state_dict(self, state: Dict):
        self.alpha = state.get("alpha", self.alpha)
        self.ema_loss = state["ema_loss"].to(self.device)
