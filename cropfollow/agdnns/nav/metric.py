from typing import Dict, List

import numpy as np
import torch
from torch import Tensor


def percentile_95(losses: Tensor):
    device = losses.device
    losses = losses.detach().cpu().numpy()
    return torch.tensor(np.percentile(losses, 95), device=device)


def compose(*functions):
    import functools

    return functools.reduce(lambda f, g: lambda x: g(f(x)), functions, lambda x: x)


class NavNetLosses:
    reducer = {
        "l1_mean": compose(torch.abs, torch.mean),
        "l2_mean": compose(lambda x: x ** 2, torch.mean),
        "bias": torch.mean,
    }

    mode_index = {"heading": 0, "distance": 1, "combined": None}

    def __init__(self, mode: str):
        self.index = self.mode_index[mode]
        self.mode = mode

    def __call__(
        self, model_output: Tensor, targets: List[Tensor]
    ) -> Dict[str, Tensor]:
        if self.index is not None:
            assert model_output.size(1) == 1
            # model_output: batch_size x 1
            error = model_output.squeeze(1) - targets[self.index].float()
            return self._get_loss(self.mode, error)
        assert model_output.size(1) == 2
        # model_output: batch_size x 2
        targets_t = torch.stack(targets, dim=-1).float()
        errors = model_output - targets_t
        losses = {
            "combined/l1_mean": self.reducer["l1_mean"](errors),
            "combined/l2_mean": self.reducer["l2_mean"](errors),
            **self._get_loss("heading", errors[:, 0]),
            **self._get_loss("distance", errors[:, 1]),
        }
        return losses

    @classmethod
    def _get_loss(cls, kind: str, errors: Tensor):
        return {
            f"{kind}/{loss_name}": func(errors)
            for loss_name, func in cls.reducer.items()
        }
