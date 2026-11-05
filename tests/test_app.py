"""Smoke tests for the Streamlit app. Run with: pytest (needs artifacts/model.pt from train.py)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "artifacts" / "model.pt").exists(), reason="run train.py first")


def run_app(model: str | None = None, data: str | None = None) -> AppTest:
    at = AppTest.from_file(str(ROOT / "ds_app.py"), default_timeout=60).run()
    if model is not None:
        at.sidebar.text_input[0].set_value(model)
    if data is not None:
        at.sidebar.text_input[1].set_value(data)
    return at.sidebar.button[0].click().run()


def test_labeled_csv_shows_metrics():
    at = run_app(data="data/sample.csv")
    assert not at.exception and not at.error
    assert "Accuracy" in [m.label for m in at.metric]


def test_unlabeled_csv_shows_predictions_only():
    at = run_app(data="data/sample_unlabeled.csv")
    assert not at.exception and not at.error
    assert [t.label for t in at.tabs] == ["Predictions", "Training"]


def test_bad_paths_give_clear_errors():
    assert "File not found" in run_app(data="does_not_exist.csv").error[0].value
    assert "Model file not found" in run_app(model="artifacts/nothing.pt").error[0].value


def test_feature_importance_behaves_sensibly():
    import sys
    import pandas as pd
    sys.path.insert(0, str(ROOT / "src"))
    from titanic.checkpoint import load_checkpoint
    from titanic.evaluate import permutation_importance

    model, pre, _ = load_checkpoint(ROOT / "artifacts" / "model.pt")
    imp = permutation_importance(model, pre, pd.read_csv(ROOT / "data" / "sample.csv"), n_repeats=3).set_index("feature")
    assert imp.loc["Ticket (not used, check)", "loss_increase"] == 0          # not a model input
    assert imp.loc["Sex + Title together", "loss_increase"] > imp.loc["Sex", "loss_increase"]


def test_features_tab_appears_for_labeled_data():
    at = run_app(data="data/sample.csv")
    assert "Features" in [t.label for t in at.tabs]
