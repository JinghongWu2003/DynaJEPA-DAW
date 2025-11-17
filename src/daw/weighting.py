import torch


def compute_weights(difficulty: torch.Tensor, gamma: float = 1.0, w_min: float = 0.5, w_max: float = 2.0) -> torch.Tensor:
    if difficulty.dim() != 1:
        difficulty = difficulty.view(-1)
    diff = difficulty - difficulty.mean()
    std = difficulty.std(unbiased=False)
    if std > 0:
        diff = diff / (std + 1e-6)
    weights = torch.exp(gamma * diff)
    weights = torch.clamp(weights, min=w_min, max=w_max)
    weights = weights / (weights.mean() + 1e-6)
    return weights.detach()
