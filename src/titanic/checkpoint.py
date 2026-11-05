"""Saving and loading the trained model.

One file holds everything needed for prediction: the network weights, the
network shape, and the fitted preprocessing values. The checkpoint contains
only tensors, numbers, strings, lists and dicts, so it loads with
``torch.load(weights_only=True)``, which refuses to run arbitrary code from
the file.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .model import TitanicMLP
from .preprocessing import TitanicPreprocessor

FORMAT_VERSION = 1


class CheckpointError(ValueError):
    """The model file is missing, damaged, or from an incompatible version."""


def save_checkpoint(path: Path | str, model: TitanicMLP, preprocessor: TitanicPreprocessor, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "model_config": model.config,
            "model_state_dict": model.state_dict(),
            "preprocessor": preprocessor.to_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(path: Path | str) -> tuple[TitanicMLP, TitanicPreprocessor, dict]:
    """Return (model in eval mode, fitted preprocessor, metadata)."""
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"Model file not found: {path}. Run `python train.py` first.")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        raise CheckpointError(f"{path} could not be loaded as a model checkpoint: {e}") from e

    if not isinstance(ckpt, dict) or ckpt.get("format_version") != FORMAT_VERSION:
        raise CheckpointError(f"{path} is not a checkpoint written by this version of train.py.")
    try:
        model = TitanicMLP.from_config(ckpt["model_config"])
        model.load_state_dict(ckpt["model_state_dict"])
        preprocessor = TitanicPreprocessor.from_dict(ckpt["preprocessor"])
    except (KeyError, RuntimeError, ValueError) as e:
        raise CheckpointError(f"{path} does not match the current model code: {e}") from e
    model.eval()
    return model, preprocessor, ckpt.get("metadata", {})
