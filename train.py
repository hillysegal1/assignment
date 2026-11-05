"""Train the Titanic survival classifier.

    python train.py                              # fetch train.csv from Kaggle
    python train.py --data-path data/train.csv   # or use a local copy (offline)

Steps:
  1. Load train.csv and split it 80/20 (stratified, fixed seed) into training
     and validation rows. Only train.csv is used.
  2. Run 5-fold cross-validation on the training rows to choose the number of
     epochs. Preprocessing is fitted inside each fold.
  3. Fit the preprocessor and train the final model on all training rows.
  4. Save to the output folder:
       model.pt           weights + network shape + fitted preprocessing
       validation.csv     the held-out rows, for evaluation in the Streamlit app
       training_report.json   settings, cross-validation results, learning curves

The validation rows are not used for any training decision; they are scored
only in the app (streamlit run ds_app.py).

This file holds the training loop and cross-validation; data loading, preprocessing,
the network and checkpointing live in src/titanic/, shared with the app.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from titanic.checkpoint import save_checkpoint  # noqa: E402
from titanic.data import SEED, TARGET, VAL_SIZE, DataError, load_dataset, split_train_val  # noqa: E402
from titanic.model import TitanicMLP  # noqa: E402
from titanic.preprocessing import TitanicPreprocessor  # noqa: E402

logger = logging.getLogger("train")


# ===================================================================== training
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


# ========================================================================== CLI

def parse_args() -> argparse.Namespace:
    defaults = TrainConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-path", type=Path, default=None,
                   help="local train.csv instead of downloading from Kaggle")
    p.add_argument("--output-dir", type=Path, default=Path("artifacts"), help="where to save the results")
    p.add_argument("--seed", type=int, default=SEED, help="seed for the split, folds and training")
    p.add_argument("--val-size", type=float, default=VAL_SIZE, help="share of rows held out for validation")
    p.add_argument("--hidden", type=int, nargs="*", default=list(defaults.hidden),
                   help="hidden layer sizes; pass no values for a logistic regression")
    p.add_argument("--dropout", type=float, default=defaults.dropout)
    p.add_argument("--lr", type=float, default=defaults.lr, help="learning rate")
    p.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size)
    p.add_argument("--max-epochs", type=int, default=defaults.max_epochs,
                   help="longest training run tried during cross-validation")
    p.add_argument("--folds", type=int, default=defaults.folds, help="number of cross-validation folds")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    torch.set_num_threads(1)    # tiny network: one thread is fastest and fully reproducible
    start = time.time()

    config = TrainConfig(
        hidden=tuple(args.hidden), dropout=args.dropout, lr=args.lr, weight_decay=args.weight_decay,
        batch_size=args.batch_size, max_epochs=args.max_epochs, folds=args.folds, seed=args.seed,
    )

    # 1. Data
    try:
        df = load_dataset(args.data_path)
    except DataError as e:
        logger.error("%s", e)
        return 1
    train_df, val_df = split_train_val(df, val_size=args.val_size, seed=args.seed)
    logger.info("Split: %d training rows, %d validation rows (survival %.1f%% / %.1f%%)",
                len(train_df), len(val_df), 100 * train_df["Survived"].mean(), 100 * val_df["Survived"].mean())

    # 2. Cross-validation to choose the number of epochs
    logger.info("Cross-validation: %d folds, up to %d epochs, hidden layers %s",
                config.folds, config.max_epochs, list(config.hidden) or "none (logistic regression)")
    histories = cross_validate(train_df, config)
    epochs, cv_curve = best_epoch(histories)
    cv_best = cv_curve.loc[epochs]
    logger.info("Chosen: %d epochs. Cross-validation at that epoch: accuracy %.3f ± %.3f, AUC %.3f ± %.3f, loss %.3f ± %.3f",
                epochs, cv_best["eval_accuracy"], cv_best["eval_accuracy_std"],
                cv_best["eval_auc"], cv_best["eval_auc_std"], cv_best["eval_loss"], cv_best["eval_loss_std"])

    # 3. Final model on all training rows
    model, preprocessor, final_history = fit_final(train_df, config, epochs)
    logger.info("Final model trained on %d rows for %d epochs (final training loss %.3f)",
                len(train_df), epochs, final_history[-1]["train_loss"])

    # 4. Save
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "config": config.to_dict(),
        "epochs": epochs,
        "val_size": args.val_size,
        "n_train": len(train_df),
        "n_validation": len(val_df),
        "cv_accuracy": float(cv_best["eval_accuracy"]),
        "cv_accuracy_std": float(cv_best["eval_accuracy_std"]),
        "cv_auc": float(cv_best["eval_auc"]),
        "cv_auc_std": float(cv_best["eval_auc_std"]),
        "torch_version": str(torch.__version__),   # plain string: the checkpoint must hold only basic types
    }
    save_checkpoint(out / "model.pt", model, preprocessor, metadata)
    val_df.to_csv(out / "validation.csv", index=False)

    report = {
        **metadata,
        "cv_folds": [
            {"fold": i, **min(h, key=lambda r: r["eval_loss"])} for i, h in enumerate(histories, start=1)
        ],
        "cv_curve": cv_curve.reset_index().round(5).to_dict(orient="records"),
        "final_training_curve": final_history,
        "preprocessing": preprocessor.to_dict(),
    }
    (out / "training_report.json").write_text(json.dumps(report, indent=2))

    logger.info("Saved %s, %s and %s (%.0f s)", out / "model.pt", out / "validation.csv",
                out / "training_report.json", time.time() - start)
    logger.info("Next: streamlit run ds_app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
