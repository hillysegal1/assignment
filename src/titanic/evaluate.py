"""Predictions, metrics, baselines and plots for a labeled (or unlabeled) Titanic CSV.

Used by the Streamlit app. Every plot function returns a matplotlib Figure.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # no display needed: figures are rendered by Streamlit

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score, log_loss,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)

from .data import TARGET  # noqa: E402
from .model import TitanicMLP  # noqa: E402
from .preprocessing import TitanicPreprocessor  # noqa: E402

C_SURVIVED = "#2a78d6"
C_DIED = "#eb6834"
C_NEUTRAL = "#8a8986"

plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.titleweight": "bold",
})


def predict(model: TitanicMLP, preprocessor: TitanicPreprocessor, df: pd.DataFrame) -> np.ndarray:
    """Survival probability for every row."""
    X = torch.from_numpy(preprocessor.transform(df))
    return model.predict_proba(X).numpy()


def has_labels(df: pd.DataFrame) -> bool:
    return TARGET in df.columns and df[TARGET].notna().all()


def metrics(y: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    out = {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "Log loss": log_loss(y, np.clip(proba, 1e-7, 1 - 1e-7), labels=[0, 1]),
        "Brier score": brier_score_loss(y, proba),
    }
    out["ROC AUC"] = roc_auc_score(y, proba) if len(np.unique(y)) == 2 else float("nan")
    return out


def bootstrap_accuracy(y: np.ndarray, pred: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% interval for accuracy by resampling passengers: how much the score could move with other passengers."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    accs = (pred[idx] == y[idx]).mean(axis=1)
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def baseline_table(df: pd.DataFrame, proba: np.ndarray, threshold: float = 0.5) -> pd.DataFrame:
    """The model next to two simple rules: 'everyone died' and 'women survive'."""
    y = df[TARGET].to_numpy(dtype=int)
    rules = {
        "Neural network": (proba >= threshold).astype(int),
        "Everyone died": np.zeros_like(y),
        "Women survive": (df["Sex"].astype(str).str.strip().str.lower() == "female").to_numpy(dtype=int),
    }
    rows = []
    for name, pred in rules.items():
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y, pred),
            "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0),
        })
    return pd.DataFrame(rows).set_index("Model")


def group_table(df: pd.DataFrame, proba: np.ndarray, threshold: float = 0.5) -> pd.DataFrame:
    """Accuracy, real survival rate and predicted survival rate per sex and class."""
    pred = (proba >= threshold).astype(int)
    d = df.assign(_pred=pred, _correct=pred == df[TARGET].to_numpy(dtype=int))
    g = d.groupby(["Sex", "Pclass"]).agg(
        Passengers=("_correct", "size"),
        Accuracy=("_correct", "mean"),
        Survived=(TARGET, "mean"),
        Predicted_survived=("_pred", "mean"),
    )
    return g.rename(columns={"Predicted_survived": "Predicted survived"})


# Original columns to shuffle. "Name" is shown as Title, since the title is all the model takes from it.
IMPORTANCE_FEATURES = {
    "Sex": ["Sex"],
    "Title (from Name)": ["Name"],
    "Pclass": ["Pclass"],
    "Fare": ["Fare"],
    "Age": ["Age"],
    "Cabin recorded": ["Cabin"],
    "SibSp": ["SibSp"],
    "Parch": ["Parch"],
    "Embarked": ["Embarked"],
    "Ticket (not used, check)": ["Ticket"],
}
# Overlapping columns shuffled together: shuffling one alone understates them, because the model
# can still read the same information from the other.
IMPORTANCE_GROUPS = {
    "Sex + Title together": ["Sex", "Name"],
    "Class + Fare + Cabin together": ["Pclass", "Fare", "Cabin"],
}


def permutation_importance(
    model: TitanicMLP,
    preprocessor: TitanicPreprocessor,
    df: pd.DataFrame,
    n_repeats: int = 20,
    seed: int = 0,
) -> pd.DataFrame:
    """How much the model's log loss and accuracy get worse when a column's values are shuffled.

    Each original column (or group of columns) is shuffled between passengers, which destroys its
    information while keeping its distribution; the model is then scored again. The larger the
    increase in log loss, the more the model relies on that column. Columns in a group are shuffled
    with the same permutation, so they stay consistent with each other.
    """
    y = df[TARGET].to_numpy(dtype=int)
    base = predict(model, preprocessor, df)
    base_loss = log_loss(y, np.clip(base, 1e-7, 1 - 1e-7), labels=[0, 1])
    base_acc = accuracy_score(y, base >= 0.5)

    rng = np.random.default_rng(seed)
    rows = []
    for kind, features in [("single", IMPORTANCE_FEATURES), ("group", IMPORTANCE_GROUPS)]:
        for name, cols in features.items():
            loss_up, acc_down = [], []
            for _ in range(n_repeats):
                order = rng.permutation(len(df))
                shuffled = df.copy()
                for c in cols:
                    shuffled[c] = df[c].to_numpy()[order]
                p = predict(model, preprocessor, shuffled)
                loss_up.append(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]) - base_loss)
                acc_down.append(base_acc - accuracy_score(y, p >= 0.5))
            rows.append({
                "feature": name, "kind": kind,
                "loss_increase": float(np.mean(loss_up)), "loss_increase_std": float(np.std(loss_up)),
                "accuracy_drop": float(np.mean(acc_down)), "accuracy_drop_std": float(np.std(acc_down)),
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ plots

def importance_chart(imp: pd.DataFrame):
    """Interactive horizontal bar chart (Altair) of permutation importance.

    Single columns first (sorted), then the related-column groups, distinguished by colour.
    The Ticket sanity check is left out of the chart; the app reports it in text.
    """
    import altair as alt

    part = imp[~imp["feature"].str.startswith("Ticket")].copy()
    part["low"] = part["loss_increase"] - part["loss_increase_std"]
    part["high"] = part["loss_increase"] + part["loss_increase_std"]
    part["label"] = part["loss_increase"].map(lambda v: f"{v:+.3f}")
    part["Shuffled"] = part["kind"].map({"single": "one column on its own", "group": "related columns together"})
    order = (part[part["kind"] == "single"].sort_values("loss_increase", ascending=False)["feature"].tolist()
             + part[part["kind"] == "group"].sort_values("loss_increase", ascending=False)["feature"].tolist())

    y = alt.Y("feature:N", sort=order, title=None,
              axis=alt.Axis(labelFontSize=13, labelLimit=280, ticks=False, domain=False, labelPadding=10))
    x = alt.X("loss_increase:Q", title="Increase in log loss when shuffled (higher = model relies on it more)",
              scale=alt.Scale(domain=[min(0.0, float(part["low"].min())), float(part["high"].max()) * 1.12]),
              axis=alt.Axis(labelFontSize=12, titleFontSize=12, titleFontWeight="normal",
                            gridOpacity=0.35, tickCount=5))
    color = alt.Color("Shuffled:N", title=None,
                      scale=alt.Scale(domain=["one column on its own", "related columns together"],
                                      range=["#2a78d6", "#8fbbee"]),
                      legend=alt.Legend(orient="top", labelFontSize=12, symbolType="square"))
    tooltip = [alt.Tooltip("feature:N", title="Feature"),
               alt.Tooltip("loss_increase:Q", title="Log loss increase", format="+.3f"),
               alt.Tooltip("loss_increase_std:Q", title="± std over 20 shuffles", format=".3f"),
               alt.Tooltip("accuracy_drop:Q", title="Accuracy drop", format="+.1%")]

    base = alt.Chart(part)
    bars = base.mark_bar(cornerRadiusEnd=4).encode(y=y, x=x, color=color, tooltip=tooltip)
    spread = base.mark_rule(color="#333", opacity=0.5, strokeWidth=1.2).encode(y=y, x="low:Q", x2="high:Q")
    labels = base.mark_text(align="left", dx=6, fontSize=12, color="#444").encode(y=y, x="high:Q", text="label:N")
    return (bars + spread + labels).properties(height=alt.Step(36)).configure_view(strokeWidth=0)


def plot_confusion(y: np.ndarray, proba: np.ndarray, threshold: float = 0.5):
    cm = confusion_matrix(y, (proba >= threshold).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm, cmap="Blues", vmin=0)
    labels = [["Correct: died", "False alarm"], ["Missed survivor", "Correct: survived"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n{labels[i][j]}", ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > cm.max() / 2 else "#222")
    ax.set_xticks([0, 1], ["Died", "Survived"])
    ax.set_yticks([0, 1], ["Died", "Survived"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    ax.grid(False)
    fig.tight_layout()
    return fig


def plot_roc(y: np.ndarray, proba: np.ndarray, threshold: float = 0.5):
    fpr, tpr, thr = roc_curve(y, proba)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.plot(fpr, tpr, color=C_SURVIVED, lw=2, label=f"model (AUC {roc_auc_score(y, proba):.3f})")
    ax.plot([0, 1], [0, 1], color=C_NEUTRAL, ls="--", lw=1, label="random guessing")
    i = int(np.argmin(np.abs(thr - threshold)))
    ax.scatter([fpr[i]], [tpr[i]], s=60, color="#222", zorder=3, label=f"threshold {threshold:.2f}")
    ax.set_xlabel("False positive rate (deaths predicted as survival)")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curve")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def plot_precision_recall(y: np.ndarray, proba: np.ndarray, threshold: float = 0.5):
    """Precision against recall for every threshold, with the current threshold marked."""
    precision, recall, thr = precision_recall_curve(y, proba)
    ap = average_precision_score(y, proba)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.plot(recall, precision, color=C_SURVIVED, lw=2, drawstyle="steps-post",
            label=f"model (average precision {ap:.3f})")
    ax.axhline(y.mean(), color=C_NEUTRAL, ls="--", lw=1, label=f"random guessing ({y.mean():.0%} survived)")
    i = int(np.argmin(np.abs(thr - threshold)))           # thr has one entry fewer than precision/recall
    ax.scatter([recall[i]], [precision[i]], s=60, color="#222", zorder=3, label=f"threshold {threshold:.2f}")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall (share of survivors caught)")
    ax.set_ylabel("Precision (share of predicted survivors correct)")
    ax.set_title("Precision–recall curve")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    return fig


def plot_groups(df: pd.DataFrame, proba: np.ndarray, threshold: float = 0.5):
    """Per sex and class: actual vs predicted survival rate (left) and accuracy (right)."""
    g = group_table(df, proba, threshold).reset_index()
    labels = [f"{s}\nclass {c}\nn={n}" for s, c, n in zip(g["Sex"], g["Pclass"], g["Passengers"])]
    x = np.arange(len(g))

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    ax = axes[0]
    w = 0.38
    bars = ax.bar(x - w / 2, g["Survived"], w, color=C_SURVIVED, label="actually survived")
    ax.bar_label(bars, labels=[f"{v:.0%}" for v in g["Survived"]], padding=2, fontsize=8)
    bars = ax.bar(x + w / 2, g["Predicted survived"], w, color=C_NEUTRAL, label="predicted to survive")
    ax.bar_label(bars, labels=[f"{v:.0%}" for v in g["Predicted survived"]], padding=2, fontsize=8)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Share of passengers")
    ax.set_title("Survival: actual vs predicted")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    overall = float(np.mean((proba >= threshold).astype(int) == df[TARGET].to_numpy(dtype=int)))
    bars = ax.bar(x, g["Accuracy"], color=C_SURVIVED)
    ax.bar_label(bars, labels=[f"{v:.0%}" for v in g["Accuracy"]], padding=2, fontsize=8)
    ax.axhline(overall, color="#222", ls="--", lw=1, label=f"overall {overall:.0%}")
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy per group")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def plot_probabilities(proba: np.ndarray, y: np.ndarray | None = None, threshold: float = 0.5):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    bins = np.linspace(0, 1, 21)
    if y is None:
        ax.hist(proba, bins=bins, color=C_SURVIVED, alpha=0.8)
    else:
        for outcome, color, label in [(0, C_DIED, "Actually died"), (1, C_SURVIVED, "Actually survived")]:
            ax.hist(proba[y == outcome], bins=bins, color=color, alpha=0.35, histtype="stepfilled",
                    ec=color, lw=1.2, label=label)
        ax.legend(fontsize=8)
    ax.axvline(threshold, color="#222", ls="--", lw=1)
    ax.set_xlabel("Predicted survival probability")
    ax.set_ylabel("Passengers")
    ax.set_title("Predicted probabilities")
    fig.tight_layout()
    return fig


def plot_calibration(y: np.ndarray, proba: np.ndarray, n_bins: int = 8):
    frac, mean_pred = calibration_curve(y, proba, n_bins=n_bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.plot([0, 1], [0, 1], color=C_NEUTRAL, ls="--", lw=1, label="perfectly calibrated")
    ax.plot(mean_pred, frac, color=C_SURVIVED, lw=2, marker="o", label="model")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted survival probability")
    ax.set_ylabel("Share who actually survived")
    ax.set_title("Calibration")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def plot_threshold(y: np.ndarray, proba: np.ndarray, threshold: float = 0.5):
    ts = np.linspace(0.05, 0.95, 91)
    acc = [accuracy_score(y, proba >= t) for t in ts]
    prec = [precision_score(y, proba >= t, zero_division=0) for t in ts]
    rec = [recall_score(y, proba >= t, zero_division=0) for t in ts]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(ts, acc, color="#222", lw=2, label="accuracy")
    ax.plot(ts, prec, color=C_SURVIVED, lw=2, label="precision")
    ax.plot(ts, rec, color=C_DIED, lw=2, label="recall")
    ax.axvline(threshold, color="#555", ls="--", lw=1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Effect of the decision threshold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_cv_curve(cv_curve: list[dict], epochs: int):
    c = pd.DataFrame(cv_curve)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(c["epoch"], c["train_loss"], color=C_SURVIVED, lw=2, label="training folds")
    ax.plot(c["epoch"], c["eval_loss"], color=C_DIED, lw=2, label="held-out fold")
    if "eval_loss_std" in c:
        ax.fill_between(c["epoch"], c["eval_loss"] - c["eval_loss_std"], c["eval_loss"] + c["eval_loss_std"],
                        color=C_DIED, alpha=0.15, label="held-out ± 1 std")
    ax.axvline(epochs, color="#555", ls="--", lw=1, label=f"chosen: {epochs} epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Cross-validation learning curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
