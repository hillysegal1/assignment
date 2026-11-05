"""Training: one training loop, 5-fold cross-validation to choose the epoch count, final fit.

Workflow (used by train.py):

1. ``cross_validate`` splits the training rows into stratified folds. In each
   round the preprocessor is fitted on the training folds only, a model is
   trained for ``max_epochs``, and the held-out fold is scored after every
   epoch.
2. ``best_epoch`` picks the epoch with the lowest mean held-out loss across
   folds.
3. ``fit_final`` refits the preprocessor and trains one model on all training
   rows for that many epochs.

The validation split is never used here; it is only scored after training.
"""

from __future__ import annotations

import logging
import random
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import TARGET
from .model import TitanicMLP
from .preprocessing import TitanicPreprocessor

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    hidden: tuple[int, ...] = (32, 16)
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 200
    folds: int = 5
    seed: int = 42

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hidden"] = list(self.hidden)
        return d


def set_seed(seed: int) -> None:
    """Make runs repeatable: same seed, same weights, same batches, same result."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def scores(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Log loss, accuracy at threshold 0.5, and ROC AUC."""
    proba = np.clip(proba, 1e-7, 1 - 1e-7)
    return {
        "loss": float(log_loss(y_true, proba, labels=[0, 1])),
        "accuracy": float(accuracy_score(y_true, proba >= 0.5)),
        "auc": float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) == 2 else float("nan"),
    }


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    config: TrainConfig,
    epochs: int,
    X_eval: np.ndarray | None = None,
    y_eval: np.ndarray | None = None,
) -> tuple[TitanicMLP, list[dict]]:
    """Train one network with mini-batch AdamW. Optionally score an evaluation set after each epoch."""
    set_seed(config.seed)
    model = TitanicMLP(X.shape[1], hidden=config.hidden, dropout=config.dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))),
        batch_size=config.batch_size, shuffle=True, generator=generator,
    )
    X_eval_t = torch.from_numpy(X_eval) if X_eval is not None else None

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        row = {"epoch": epoch, "train_loss": total / len(X)}
        if X_eval_t is not None:
            proba = model.predict_proba(X_eval_t).numpy()
            row.update({f"eval_{k}": v for k, v in scores(y_eval, proba).items()})
        history.append(row)
    return model, history


def cross_validate(train_df: pd.DataFrame, config: TrainConfig) -> list[list[dict]]:
    """Per-epoch held-out scores for each fold. Preprocessing is fitted inside each fold."""
    y_all = train_df[TARGET].to_numpy(dtype=int)
    folds = StratifiedKFold(n_splits=config.folds, shuffle=True, random_state=config.seed)
    histories = []
    for i, (fit_idx, held_idx) in enumerate(folds.split(train_df, y_all), start=1):
        fit_rows, held_rows = train_df.iloc[fit_idx], train_df.iloc[held_idx]
        pre = TitanicPreprocessor().fit(fit_rows)
        _, history = train_model(
            pre.transform(fit_rows), y_all[fit_idx], config, config.max_epochs,
            X_eval=pre.transform(held_rows), y_eval=y_all[held_idx],
        )
        best = min(history, key=lambda r: r["eval_loss"])
        logger.info("Fold %d/%d: best held-out loss %.3f at epoch %d (accuracy %.3f, AUC %.3f)",
                    i, config.folds, best["eval_loss"], best["epoch"], best["eval_accuracy"], best["eval_auc"])
        histories.append(history)
    return histories


def best_epoch(histories: list[list[dict]]) -> tuple[int, pd.DataFrame]:
    """Epoch with the lowest mean held-out loss across folds, plus the per-epoch mean and std table."""
    frames = [pd.DataFrame(h).assign(fold=i) for i, h in enumerate(histories, start=1)]
    curves = pd.concat(frames).groupby("epoch")[["train_loss", "eval_loss", "eval_accuracy", "eval_auc"]]
    summary = curves.mean().join(curves.std().add_suffix("_std"))
    return int(summary["eval_loss"].idxmin()), summary


def fit_final(train_df: pd.DataFrame, config: TrainConfig, epochs: int) -> tuple[TitanicMLP, TitanicPreprocessor, list[dict]]:
    """Fit the preprocessor and train the final model on all training rows."""
    pre = TitanicPreprocessor().fit(train_df)
    model, history = train_model(pre.transform(train_df), train_df[TARGET].to_numpy(dtype=int), config, epochs)
    return model, pre, history
