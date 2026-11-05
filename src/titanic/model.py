"""The PyTorch classifier: a small fully connected network (MLP)."""

from __future__ import annotations

import torch
from torch import nn


class TitanicMLP(nn.Module):
    """Input -> [Linear -> ReLU -> Dropout] x len(hidden) -> Linear -> 1 logit.

    The network returns a raw score (logit). Training uses BCEWithLogitsLoss,
    which applies the sigmoid internally in a numerically stable way; at
    prediction time, ``predict_proba`` turns the logit into a probability.

    With ``hidden=()`` the network has no hidden layer and is exactly a
    logistic regression: useful as a baseline to check what the hidden
    layers add.
    """

    def __init__(self, n_features: int, hidden: tuple[int, ...] = (32, 16), dropout: float = 0.2) -> None:
        super().__init__()
        self.config = {"n_features": n_features, "hidden": list(hidden), "dropout": dropout}
        layers: list[nn.Module] = []
        width = n_features
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Survival probability for each row. Switches to eval mode (dropout off)."""
        self.eval()
        return torch.sigmoid(self(x))

    @classmethod
    def from_config(cls, config: dict) -> "TitanicMLP":
        return cls(n_features=config["n_features"], hidden=tuple(config["hidden"]), dropout=config["dropout"])
