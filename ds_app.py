"""Streamlit app: load the trained model, score a CSV, show evaluation results.

    streamlit run ds_app.py

1. Enter the path of a trained model (default: artifacts/model.pt, written by train.py).
2. Enter the path of a Titanic-format CSV (default: artifacts/validation.csv, the
   held-out split) or upload one.
3. Click "Run inference". With a Survived column you get metrics and plots;
   without one, predictions only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from titanic.checkpoint import CheckpointError, load_checkpoint  # noqa: E402
from titanic.data import TARGET, DataError, load_csv, validate  # noqa: E402
from titanic import evaluate as ev  # noqa: E402

DEFAULT_MODEL = ROOT / "artifacts" / "model.pt"
DEFAULT_DATA = ROOT / "artifacts" / "validation.csv"

st.set_page_config(page_title="Titanic survival", page_icon="🚢", layout="wide")


# ---------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading model...")
def get_model(path: str, modified: float):
    """Cached per file path and modification time, so a retrained model is picked up."""
    return load_checkpoint(path)


@st.cache_data(show_spinner="Reading dataset...")
def get_csv(path: str, modified: float) -> pd.DataFrame:
    return load_csv(path)


def read_report(model_path: Path) -> dict | None:
    report = model_path.parent / "training_report.json"
    try:
        return json.loads(report.read_text()) if report.is_file() else None
    except (OSError, json.JSONDecodeError):
        return None


def show(fig) -> None:
    st.pyplot(fig, width="stretch")
    plt.close(fig)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Inputs")
    model_path = Path(st.text_input("Model file", str(DEFAULT_MODEL.relative_to(ROOT))))
    source = st.radio("Dataset", ["File path", "Upload a CSV"], horizontal=True)
    if source == "File path":
        data_path = st.text_input("CSV path", str(DEFAULT_DATA.relative_to(ROOT)))
        upload = None
    else:
        data_path = None
        upload = st.file_uploader("Titanic-format CSV", type="csv")
    run = st.button("Run inference", type="primary", width="stretch")

    st.divider()
    threshold = st.slider(
        "Decision threshold", 0.05, 0.95, 0.50, 0.01,
        help="A passenger is predicted to survive when the predicted probability is at or above this value. "
             "The model was trained and selected at 0.50.",
    )
    st.caption(
        "The model gives each passenger a **survival probability**. The threshold turns it into a yes/no "
        "answer: at or above it, *survived*; below it, *died*. Lower it to catch more survivors at the cost "
        "of more false alarms; raise it for the opposite. It changes the predictions, not the model."
    )

st.title("Titanic survival: model evaluation")
st.caption("A PyTorch neural network trained on Kaggle's train.csv. "
           "Choose a model and a dataset in the sidebar, then run inference.")

with st.expander("How to read this page", expanded="result" not in st.session_state):
    st.markdown(
        """
**What the model produces.** For every passenger, a *survival probability* between 0 and 1. The
**decision threshold** (sidebar, default 0.50) turns it into a prediction: probability ≥ threshold means
*survived*, below means *died*. Moving the threshold changes the predictions and the scores that depend
on them, not the model itself.

**The scores** (shown when the CSV has a `Survived` column):

| Score | Question it answers |
|---|---|
| Accuracy | What share of passengers did the model get right? |
| Precision | Of the passengers it predicted would survive, how many did? |
| Recall | Of the passengers who survived, how many did it catch? |
| F1 | One number balancing precision and recall |
| ROC AUC | How well does it rank survivors above non-survivors, at any threshold? 0.5 = guessing, 1 = perfect |
| Log loss, Brier score | How trustworthy are the probabilities themselves? Lower is better |

**Calibration** asks whether the probabilities can be taken literally: of all passengers the model gives
about 70%, do about 70% actually survive? A well-calibrated model's probabilities mean what they say.

**Two simple rules for comparison:** *everyone died* (always predict the majority outcome) and
*women survive* (predict survival for every woman). A useful model should beat both.
"""
    )

# ---------------------------------------------------------------- run
if run:
    st.session_state.pop("result", None)
    try:
        mp = model_path if model_path.is_absolute() else ROOT / model_path
        if not mp.is_file():
            raise CheckpointError(f"Model file not found: {mp}. Run `python train.py` first.")
        model, preprocessor, meta = get_model(str(mp), mp.stat().st_mtime)

        if upload is not None:
            try:
                df = validate(pd.read_csv(upload), source=upload.name)
            except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
                raise DataError(f"{upload.name} could not be read as a CSV file: {e}") from e
            name = upload.name
        elif data_path:
            dp = Path(data_path)
            dp = dp if dp.is_absolute() else ROOT / dp
            if not dp.is_file():
                raise DataError(f"File not found: {dp}")
            df = get_csv(str(dp), dp.stat().st_mtime)
            name = dp.name
        else:
            raise DataError("Choose a CSV file first.")

        proba = ev.predict(model, preprocessor, df)
        st.session_state["result"] = {
            "df": df, "proba": proba, "meta": meta, "name": name, "report": read_report(mp),
            "model": model, "preprocessor": preprocessor,
        }
    except (CheckpointError, DataError) as e:
        st.error(str(e))
    except Exception as e:  # anything unexpected: show it instead of a blank page
        st.error(f"Unexpected error: {type(e).__name__}: {e}")

result = st.session_state.get("result")
if result is None:
    if not run:
        st.info("Set the model and dataset paths in the sidebar and click **Run inference**. "
                "The defaults use the model and validation split written by `python train.py`.")
    st.stop()

df, proba, meta = result["df"], result["proba"], result["meta"]
pred = (proba >= threshold).astype(int)
labeled = ev.has_labels(df)

# ---------------------------------------------------------------- header
c1, c2, c3, c4 = st.columns(4)
c1.metric("Passengers scored", f"{len(df)}")
c2.metric("Predicted survivors", f"{pred.sum()} ({pred.mean():.0%})")
if labeled:
    c3.metric("Actual survivors", f"{int(df[TARGET].sum())} ({df[TARGET].mean():.0%})")
if meta.get("cv_accuracy") is not None:
    c4.metric("Cross-validation accuracy (training)",
              f"{meta['cv_accuracy']:.1%}",
              help=f"± {meta.get('cv_accuracy_std', float('nan')):.1%} across folds; "
                   f"{meta.get('epochs', '?')} epochs, {meta.get('n_train', '?')} training rows")
st.caption(f"Dataset: **{result['name']}** · threshold {threshold:.2f}" +
           ("" if labeled else " · no `Survived` column, so predictions only"))

if threshold != 0.5 and labeled:
    st.warning("The threshold differs from the 0.50 the model was selected with. If this dataset is the "
               "validation split, choosing a threshold by looking at its results makes them optimistic.")

# ---------------------------------------------------------------- tabs
tab_names = (["Metrics", "Plots", "By group", "Features"] if labeled else []) + ["Predictions", "Training"]
tabs = dict(zip(tab_names, st.tabs(tab_names)))

if labeled:
    y = df[TARGET].to_numpy(dtype=int)
    m = ev.metrics(y, proba, threshold)
    lo, hi = ev.bootstrap_accuracy(y, pred)

    with tabs["Metrics"]:
        cols = st.columns(4)
        cols[0].metric("Accuracy", f"{m['Accuracy']:.1%}", help=f"95% bootstrap interval {lo:.1%} to {hi:.1%}")
        cols[1].metric("Precision", f"{m['Precision']:.1%}", help="Of passengers predicted to survive, the share who did")
        cols[2].metric("Recall", f"{m['Recall']:.1%}", help="Of passengers who survived, the share the model caught")
        cols[3].metric("F1", f"{m['F1']:.3f}", help="Balance of precision and recall")
        cols = st.columns(4)
        cols[0].metric("ROC AUC", f"{m['ROC AUC']:.3f}", help="Ranking quality, independent of the threshold")
        cols[1].metric("Log loss", f"{m['Log loss']:.3f}", help="Quality of the probabilities (lower is better)")
        cols[2].metric("Brier score", f"{m['Brier score']:.3f}", help="Mean squared error of the probabilities")
        cols[3].metric("Accuracy, 95% interval", f"{lo:.0%} – {hi:.0%}",
                       help="Resampling the passengers 2,000 times: how much accuracy could move by chance")

        st.subheader("Compared with simple rules")
        st.dataframe(ev.baseline_table(df, proba, threshold).style.format("{:.1%}"), width="stretch")
        st.caption("'Everyone died' is the score of always predicting the majority outcome; "
                   "'women survive' predicts survival for every female passenger.")

    with tabs["Plots"]:
        a, b = st.columns(2)
        with a:
            show(ev.plot_confusion(y, proba, threshold))
            st.caption("Every passenger falls in one of four boxes: the diagonal holds the correct predictions, "
                       "the other two boxes the two kinds of mistake. Changes with the threshold.")
        with b:
            show(ev.plot_probabilities(proba, y, threshold))
            st.caption("Each passenger's predicted probability, coloured by what actually happened. Orange to the "
                       "left and blue to the right of the dashed threshold line are correct predictions.")
        a, b = st.columns(2)
        with a:
            show(ev.plot_roc(y, proba, threshold))
            st.caption("How well the model ranks survivors above non-survivors, for every possible threshold. "
                       "The closer the curve hugs the top-left corner, the better; the diagonal is random "
                       "guessing. The dot marks the current threshold.")
        with b:
            show(ev.plot_precision_recall(y, proba, threshold))
            st.caption("The trade-off for the survivors only: each point is one threshold. Moving right catches "
                       "more survivors (recall), moving down means more of the predicted survivors are wrong "
                       "(precision). The dashed line is what random guessing would give.")
        a, b = st.columns(2)
        with a:
            show(ev.plot_threshold(y, proba, threshold))
            st.caption("How accuracy, precision and recall change as the threshold moves. Lowering it catches "
                       "more survivors (recall rises) but adds false alarms (precision falls).")
        with b:
            show(ev.plot_calibration(y, proba))
            with st.expander("Reading the calibration plot", expanded=True):
                st.markdown(
                    """
Can the probabilities be taken literally? The passengers are sorted by predicted probability and split
into groups of about equal size. Each dot is one group:

- **horizontal position:** the group's average *predicted* survival probability
- **vertical position:** the share of the group that *actually* survived

On the dashed diagonal, predictions mean exactly what they say: of passengers given about 30%,
about 30% survived. Dots **below** the line mean the model is too optimistic in that range;
dots **above** mean too pessimistic. With under 200 passengers each group holds only about
20 people, so small deviations from the line are expected noise.
"""
                )

    with tabs["By group"]:
        show(ev.plot_groups(df, proba, threshold))
        st.caption("Left: for each sex and class, the share who actually survived next to the share the model "
                   "predicts will survive. Right: how often the model is correct in each group. Accuracy is "
                   "lowest where survival was close to a coin toss (3rd-class women, 1st-class men): within "
                   "those groups the available features cannot separate who survived.")
        with st.expander("Show as a table"):
            groups = ev.group_table(df, proba, threshold)
            st.dataframe(groups.style.format({"Accuracy": "{:.0%}", "Survived": "{:.0%}",
                                              "Predicted survived": "{:.0%}"}), width="stretch")

    with tabs["Features"]:
        if "importance" not in result:           # computed once per run, not on every slider move
            with st.spinner("Shuffling each column 20 times..."):
                result["importance"] = ev.permutation_importance(result["model"], result["preprocessor"], df)
        imp = result["importance"]
        st.subheader("Which columns does the model rely on?")
        st.caption("A longer bar means the model's predictions get worse when that column is scrambled. "
                   "Light bars scramble related columns together: columns that carry the same information "
                   "share their credit when scrambled alone. Hover a bar for exact values; the thin line "
                   "shows the spread over 20 shuffles.")
        st.altair_chart(ev.importance_chart(imp), width="stretch")

        ticket = imp.loc[imp["feature"].str.startswith("Ticket"), "loss_increase"]
        if len(ticket):
            st.caption(f"Check: Ticket is not a model input, and scrambling it changes the log loss by "
                       f"{float(ticket.iloc[0]):+.3f}, as it should.")

        with st.expander("How this is measured"):
            st.markdown(
                """
This is **permutation importance**. One original column at a time, its values are shuffled between
passengers, so the column no longer carries real information while every other column stays
intact. The model then scores the same passengers again, and the bar shows how much the
**log loss** (the quality of its probabilities) gets worse. This is repeated 20 times per column.

- **A bar near zero** means the model barely uses the column, *or* finds the same information in
  another column. Fare is the example: on its own it adds nothing, because Pclass already tells the
  model the same thing.
- **Sex looks small on its own** because Title carries the same information (Mr, Mrs, Miss, Master).
  Scrambled together, they are by far the most important.

This explains the trained model; it does not change it.
"""
            )
        with st.expander("Show as a table"):
            st.dataframe(
                imp.rename(columns={"loss_increase": "log loss increase", "loss_increase_std": "± std",
                                    "accuracy_drop": "accuracy drop", "accuracy_drop_std": "± std "})
                   .set_index("feature").round(3),
                width="stretch",
            )

with tabs["Predictions"]:
    out = df.copy()
    out.insert(0, "Survival probability", proba.round(3))
    out.insert(1, "Predicted", np.where(pred == 1, "Survived", "Died"))
    if labeled:
        out.insert(2, "Correct", pred == df[TARGET].to_numpy(dtype=int))
    only_wrong = labeled and st.toggle("Show only wrong predictions")
    shown = out[~out["Correct"]] if only_wrong else out
    st.dataframe(shown, width="stretch", hide_index=True)
    st.download_button("Download predictions (CSV)", out.to_csv(index=False).encode(),
                       file_name="predictions.csv", mime="text/csv")
    if not labeled:
        show(ev.plot_probabilities(proba, None, threshold))

with tabs["Training"]:
    report = result["report"]
    cfg = meta.get("config", {})
    st.markdown(
        f"- **Network:** {meta.get('config', {}).get('hidden', '?')} hidden units, dropout {cfg.get('dropout', '?')}\n"
        f"- **Training:** {meta.get('epochs', '?')} epochs on {meta.get('n_train', '?')} rows, "
        f"chosen by {cfg.get('folds', '?')}-fold cross-validation\n"
        f"- **Optimiser:** AdamW, learning rate {cfg.get('lr', '?')}, weight decay {cfg.get('weight_decay', '?')}, "
        f"batch size {cfg.get('batch_size', '?')}"
    )
    if report and report.get("cv_curve"):
        show(ev.plot_cv_curve(report["cv_curve"], report["epochs"]))
        folds = pd.DataFrame(report["cv_folds"])[["fold", "epoch", "eval_loss", "eval_accuracy", "eval_auc"]]
        st.dataframe(folds.rename(columns={"epoch": "best epoch", "eval_loss": "loss",
                                           "eval_accuracy": "accuracy", "eval_auc": "AUC"}).round(3),
                     width="stretch", hide_index=True)
    else:
        st.info("No training_report.json next to the model file, so the learning curves are not available.")
