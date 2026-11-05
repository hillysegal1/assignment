"""Preprocessing: the EDA decisions, as code.

``TitanicPreprocessor.fit`` learns every value it needs (filling medians,
scaling statistics) from the training data only. ``transform`` applies those
stored values to any data: training, validation, or a CSV loaded into the
Streamlit app. The learned state is a plain dict of numbers and strings, so it
is saved inside the model checkpoint and the app fills gaps exactly the way
training did.

Model input (17 columns), see the "Preprocessing decisions" table in the EDA:

    standardised   Age, LogFare, SibSp, Parch                     4
    binary flags   IsFemale, AgeMissing, HasCabin                 3
    one-hot        Pclass (1, 2, 3), Embarked (S, C, Q),
                   Title (Mr, Mrs, Miss, Master)                  10
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TITLES = ["Mr", "Mrs", "Miss", "Master"]
# Alternative spellings of the four common titles.
TITLE_ALIASES = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
PCLASSES = [1, 2, 3]
PORTS = ["S", "C", "Q"]

SCALED_COLUMNS = ["Age", "LogFare", "SibSp", "Parch"]
FLAG_COLUMNS = ["IsFemale", "AgeMissing", "HasCabin"]
FEATURE_NAMES = (
    SCALED_COLUMNS
    + FLAG_COLUMNS
    + [f"Pclass_{c}" for c in PCLASSES]
    + [f"Embarked_{p}" for p in PORTS]
    + [f"Title_{t}" for t in TITLES]
)

MISS_WITH_FAMILY = "Miss (Parch > 0)"
MISS_WITHOUT_FAMILY = "Miss (Parch = 0)"


def extract_title(name: pd.Series, sex: pd.Series) -> pd.Series:
    """Title from the name, mapped to Mr, Mrs, Miss or Master.

    "Braund, Mr. Owen Harris" -> "Mr". Mlle and Ms become Miss, Mme becomes Mrs.
    Every other title (Dr, Rev, Col, Lady, the Countess, ...) and any title not
    seen before goes to the adult title for the passenger's sex: Mr for men,
    Mrs for women. Rare-title passengers are adults, and groups of 1-7 people
    are too small to learn from separately.
    """
    raw = name.fillna("").astype(str).str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    title = raw.replace(TITLE_ALIASES)
    is_female = sex.fillna("").astype(str).str.strip().str.lower() == "female"
    adult_title = pd.Series(np.where(is_female, "Mrs", "Mr"), index=name.index)
    return title.where(title.isin(TITLES), adult_title)


def age_group(title: pd.Series, parch: pd.Series) -> pd.Series:
    """Key used to fill missing ages: the title, with Miss split by whether Parch > 0.

    Miss is the only title that mixes girls and adult women; travelling with
    parents or children separates them (median ages 8 vs 26 in the EDA).
    """
    group = title.copy()
    is_miss = title == "Miss"
    group[is_miss & (parch > 0)] = MISS_WITH_FAMILY
    group[is_miss & (parch == 0)] = MISS_WITHOUT_FAMILY
    return group


class TitanicPreprocessor:
    """Fit on training rows, then transform any rows into the 17-column model input."""

    def __init__(self) -> None:
        self.state: dict | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame) -> "TitanicPreprocessor":
        """Learn filling values and scaling statistics from training rows only."""
        title = extract_title(df["Name"], df["Sex"])
        groups = age_group(title, df["Parch"])

        age_overall = float(df["Age"].median())
        age_by_group = {g: float(v) for g, v in df["Age"].groupby(groups).median().dropna().items()}

        fare = df["Fare"].where(df["Fare"] > 0)              # a fare of 0 is treated as missing
        fare_overall = float(fare.median())
        fare_by_class = {str(int(c)): float(v) for c, v in fare.groupby(df["Pclass"]).median().dropna().items()}

        port_mode = df["Embarked"].dropna().mode()

        self.state = {
            "age_by_group": age_by_group,
            "age_overall": age_overall,
            "fare_by_class": fare_by_class,
            "fare_overall": fare_overall,
            "embarked_mode": str(port_mode.iloc[0]) if len(port_mode) else "S",
            "means": {},
            "stds": {},
        }

        # Scaling statistics are computed after filling, so filled values are scaled like real ones.
        filled = self._fill(df)
        for col in SCALED_COLUMNS:
            self.state["means"][col] = float(filled[col].mean())
            std = float(filled[col].std(ddof=0))
            self.state["stds"][col] = std if std > 0 else 1.0
        return self

    # ------------------------------------------------------------ transform
    def _fill(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill gaps and build the unscaled columns, using the stored training values."""
        s = self.state
        out = pd.DataFrame(index=df.index)
        title = extract_title(df["Name"], df["Sex"])
        groups = age_group(title, df["Parch"])

        age_fill = groups.map(s["age_by_group"]).fillna(s["age_overall"])
        out["Age"] = df["Age"].fillna(age_fill)
        out["AgeMissing"] = df["Age"].isna().astype(float)

        fare = df["Fare"].where(df["Fare"] > 0)
        fare_fill = df["Pclass"].astype(int).astype(str).map(s["fare_by_class"]).fillna(s["fare_overall"])
        out["LogFare"] = np.log1p(fare.fillna(fare_fill))

        out["SibSp"] = df["SibSp"].astype(float)
        out["Parch"] = df["Parch"].astype(float)

        out["IsFemale"] = (df["Sex"].fillna("").astype(str).str.strip().str.lower() == "female").astype(float)
        out["HasCabin"] = df["Cabin"].notna().astype(float)

        port = df["Embarked"].astype("string").str.strip().str.upper().fillna(s["embarked_mode"])
        for c in PCLASSES:
            out[f"Pclass_{c}"] = (df["Pclass"] == c).astype(float)
        for p in PORTS:
            out[f"Embarked_{p}"] = (port == p).astype(float)   # an unknown port gives all zeros
        for t in TITLES:
            out[f"Title_{t}"] = (title == t).astype(float)
        return out

    def transform_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """The model input as a labeled DataFrame (handy for inspection)."""
        if self.state is None:
            raise RuntimeError("The preprocessor must be fitted before transform().")
        out = self._fill(df)
        for col in SCALED_COLUMNS:
            out[col] = (out[col] - self.state["means"][col]) / self.state["stds"][col]
        return out[FEATURE_NAMES].astype(np.float32)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.transform_frame(df).to_numpy(dtype=np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    # -------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        if self.state is None:
            raise RuntimeError("Nothing to save: the preprocessor has not been fitted.")
        return {"state": self.state, "feature_names": list(FEATURE_NAMES)}

    @classmethod
    def from_dict(cls, d: dict) -> "TitanicPreprocessor":
        if list(d.get("feature_names", [])) != FEATURE_NAMES:
            raise ValueError("The saved preprocessor uses different features than this code version.")
        p = cls()
        p.state = d["state"]
        return p

    @property
    def n_features(self) -> int:
        return len(FEATURE_NAMES)
