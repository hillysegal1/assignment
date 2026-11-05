"""Tests for the data checks, preprocessing and checkpoint. Run with: pytest"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from titanic.checkpoint import CheckpointError, load_checkpoint, save_checkpoint  # noqa: E402
from titanic.data import DataError, validate  # noqa: E402
from titanic.model import TitanicMLP  # noqa: E402
from titanic.preprocessing import FEATURE_NAMES, TitanicPreprocessor, extract_title  # noqa: E402


def make_rows(**overrides) -> pd.DataFrame:
    base = {
        "PassengerId": [1, 2, 3, 4, 5, 6],
        "Survived": [0, 1, 1, 0, 1, 0],
        "Pclass": [3, 1, 3, 2, 1, 3],
        "Name": ["Braund, Mr. Owen", "Cumings, Mrs. John", "Heikkinen, Miss. Laina",
                 "Rice, Master. Eric", "Rothes, the Countess. of", "Byles, Rev. Thomas"],
        "Sex": ["male", "female", "female", "male", "female", "male"],
        "Age": [22.0, 38.0, np.nan, 4.0, 33.0, np.nan],
        "SibSp": [1, 1, 0, 4, 0, 0],
        "Parch": [0, 0, 2, 1, 0, 0],
        "Ticket": ["A/5 21171", "PC 17599", "3101282", "382652", "110152", "244310"],
        "Fare": [7.25, 71.28, 7.92, 0.0, 86.5, 13.0],
        "Cabin": [np.nan, "C85", np.nan, np.nan, "B77", np.nan],
        "Embarked": ["S", "C", "S", "Q", np.nan, "S"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_titles_are_mapped_to_four_groups():
    rows = make_rows()
    title = extract_title(rows["Name"], rows["Sex"])
    assert title.tolist() == ["Mr", "Mrs", "Miss", "Master", "Mrs", "Mr"]   # Countess -> Mrs, Rev -> Mr


def test_output_shape_and_no_missing_values():
    rows = make_rows()
    X = TitanicPreprocessor().fit_transform(rows)
    assert X.shape == (len(rows), len(FEATURE_NAMES)) == (6, 17)
    assert not np.isnan(X).any()


def test_filling_values_come_from_training_rows_only():
    train, other = make_rows(), make_rows(Age=[80.0] * 6, Fare=[500.0] * 6)
    pre = TitanicPreprocessor().fit(train)
    before = pre.to_dict()
    pre.transform(other)
    assert pre.to_dict() == before          # transform never changes the learned values


def test_zero_fare_is_filled_with_class_median():
    pre = TitanicPreprocessor().fit(make_rows())
    frame = pre._fill(make_rows())
    filled_fare = np.expm1(frame.loc[3, "LogFare"])        # row 3: 2nd class, fare 0
    assert filled_fare == pytest.approx(13.0)              # median of the real 2nd-class fares


def test_unseen_values_do_not_break_transform():
    pre = TitanicPreprocessor().fit(make_rows())
    odd = make_rows(Name=["Smith, Prof. John"] * 6, Embarked=["X"] * 6, Age=[np.nan] * 6, Fare=[np.nan] * 6)
    frame = pre.transform_frame(odd)
    assert not frame.isna().any().any()
    assert (frame[[c for c in FEATURE_NAMES if c.startswith("Embarked_")]].sum(axis=1) == 0).all()
    assert frame.loc[0, "Title_Mr"] == 1                   # unknown title, male -> Mr


def test_validation_rejects_bad_files():
    with pytest.raises(DataError, match="missing column"):
        validate(make_rows().drop(columns=["Fare"]))
    with pytest.raises(DataError, match="Pclass"):
        validate(make_rows(Pclass=[1, 2, 3, 4, 1, 2]))
    with pytest.raises(DataError, match="non-numeric"):
        validate(make_rows(Age=["22", "x", "1", "2", "3", "4"]))


def test_checkpoint_round_trip(tmp_path):
    rows = make_rows()
    pre = TitanicPreprocessor().fit(rows)
    model = TitanicMLP(len(FEATURE_NAMES))
    save_checkpoint(tmp_path / "m.pt", model, pre, {"epochs": 1})

    loaded_model, loaded_pre, meta = load_checkpoint(tmp_path / "m.pt")
    X = torch.from_numpy(pre.transform(rows))
    assert torch.allclose(model.predict_proba(X), loaded_model.predict_proba(X))
    assert np.array_equal(pre.transform(rows), loaded_pre.transform(rows))
    assert meta == {"epochs": 1}


def test_missing_checkpoint_gives_clear_error(tmp_path):
    with pytest.raises(CheckpointError, match="Run `python train.py`"):
        load_checkpoint(tmp_path / "nothing.pt")
