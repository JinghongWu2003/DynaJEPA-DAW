import os
from typing import Dict, Optional

import torch


def save_checkpoint(
    path: str,
    model_state: Dict,
    optimizer_state: Dict,
    epoch: int,
    global_step: int,
    buffer_state: Optional[Dict] = None,
):
    state = {
        "model": model_state,
        "optimizer": optimizer_state,
        "epoch": epoch,
        "global_step": global_step,
    }
    if buffer_state is not None:
        state["buffer"] = buffer_state
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, device: str = "cpu") -> Dict:
    state = torch.load(path, map_location=device)
    return state
