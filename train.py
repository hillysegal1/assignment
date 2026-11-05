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
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch  # noqa: E402

from titanic.checkpoint import save_checkpoint  # noqa: E402
from titanic.data import SEED, VAL_SIZE, DataError, load_dataset, split_train_val  # noqa: E402
from titanic.training import TrainConfig, best_epoch, cross_validate, fit_final  # noqa: E402

logger = logging.getLogger("train")


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
