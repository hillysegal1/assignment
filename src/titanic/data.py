"""Loading, validating and splitting the Titanic data.

Only ``train.csv`` from the Kaggle "titanic" competition is used. The
validation split is carved out of it with a fixed seed, so ``train.py``, the
EDA notebook and the Streamlit app all see exactly the same partition.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

KAGGLE_COMPETITION = "titanic"
KAGGLE_FILE = "train.csv"
TARGET = "Survived"

# Columns the preprocessing reads. PassengerId and Survived are optional, so an
# unlabeled CSV can still be scored by the app.
FEATURE_COLUMNS = ["Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]

SEED = 42
VAL_SIZE = 0.2


class DataError(ValueError):
    """A dataset could not be downloaded, read or validated. The message is meant for the user."""


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_env_file(path: Path | str = PROJECT_ROOT / ".env") -> None:
    """Load KEY=value lines from a .env file into the environment, if the file exists.

    This is how the Kaggle token reaches kagglehub without ever appearing in the code: put
    KAGGLE_API_TOKEN=... in .env (which .gitignore keeps out of git). Variables already set in
    the environment win over the file.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:          # python-dotenv is optional: the token can also come from the environment
        return
    load_dotenv(path, override=False)


def fetch_from_kaggle() -> Path:
    """Download train.csv from the Kaggle Titanic competition and return its local path.

    kagglehub finds the API token itself: the KAGGLE_API_TOKEN environment
    variable (also read from a .env file), ~/.kaggle/access_token, or (on Colab)
    a Colab secret. The competition rules must be accepted on kaggle.com first.
    Files are cached, so repeated runs do not download again.
    """
    load_env_file()
    try:
        import kagglehub
    except ImportError as e:
        raise DataError("kagglehub is not installed. Run: pip install -r requirements.txt") from e

    logger.info("Fetching %s from the Kaggle competition '%s'...", KAGGLE_FILE, KAGGLE_COMPETITION)
    try:
        path = Path(kagglehub.competition_download(KAGGLE_COMPETITION, path=KAGGLE_FILE))
    except Exception as e:
        raise DataError(
            "Could not download train.csv from Kaggle. Check that:\n"
            "  1. Your API token is set: KAGGLE_API_TOKEN in a .env file (copy .env.example),\n"
            "     as an environment variable, or saved in ~/.kaggle/access_token\n"
            "  2. You accepted the rules at https://www.kaggle.com/competitions/titanic/rules\n"
            "  3. You are online (or pass --data-path with a local copy of train.csv)\n"
            f"Original error: {type(e).__name__}: {e}"
        ) from e

    if path.is_dir():
        path = path / KAGGLE_FILE
    if not path.is_file():
        raise DataError(f"Kaggle download finished, but {KAGGLE_FILE} was not found at {path}")
    return path


def load_csv(path: Path | str, require_target: bool = False) -> pd.DataFrame:
    """Read a Titanic-format CSV and check its schema. Raises DataError with a readable message."""
    path = Path(path)
    if not path.exists():
        raise DataError(f"File not found: {path}")
    if path.is_dir():
        raise DataError(f"Expected a CSV file, got a folder: {path}")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as e:
        raise DataError(f"{path} is empty.") from e
    except (pd.errors.ParserError, UnicodeDecodeError) as e:
        raise DataError(f"{path} could not be read as a CSV file: {e}") from e
    return validate(df, require_target=require_target, source=path.name)


def validate(df: pd.DataFrame, require_target: bool = False, source: str = "dataset") -> pd.DataFrame:
    """Check columns, types and allowed values. Returns a copy with numeric columns coerced."""
    if df.empty:
        raise DataError(f"{source} has no rows.")

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(
            f"{source} is missing column(s): {', '.join(missing)}. "
            f"Expected the Kaggle Titanic columns: {', '.join(FEATURE_COLUMNS)}."
        )
    if require_target and TARGET not in df.columns:
        raise DataError(f"{source} has no '{TARGET}' column, which is needed for training.")

    df = df.copy()
    for col in ["Pclass", "Age", "SibSp", "Parch", "Fare"]:
        numeric = pd.to_numeric(df[col], errors="coerce")
        bad = numeric.isna() & df[col].notna()
        if bad.any():
            raise DataError(f"Column '{col}' in {source} has non-numeric values, e.g. {df.loc[bad, col].iloc[0]!r}.")
        df[col] = numeric

    for col in ["Pclass", "SibSp", "Parch"]:
        if df[col].isna().any():
            raise DataError(f"Column '{col}' in {source} has missing values; it must be filled in for every row.")
    if not df["Pclass"].isin([1, 2, 3]).all():
        raise DataError(f"Column 'Pclass' in {source} may only contain 1, 2 or 3.")
    for col in ["Age", "Fare", "SibSp", "Parch"]:
        if (df[col].dropna() < 0).any():
            raise DataError(f"Column '{col}' in {source} has negative values.")

    sex = df["Sex"].dropna().astype(str).str.strip().str.lower()
    unexpected = sorted(set(sex) - {"male", "female"})
    if unexpected:
        raise DataError(f"Column 'Sex' in {source} has unexpected values: {unexpected}.")

    if TARGET in df.columns:
        y = pd.to_numeric(df[TARGET], errors="coerce")
        if require_target and y.isna().any():
            raise DataError(f"Column '{TARGET}' in {source} has {int(y.isna().sum())} missing value(s).")
        if not y.dropna().isin([0, 1]).all():
            raise DataError(f"Column '{TARGET}' in {source} may only contain 0 and 1.")
        df[TARGET] = y.astype("Int64")

    return df


def load_dataset(data_path: Path | str | None = None) -> pd.DataFrame:
    """The full labeled train.csv: from Kaggle by default, or from a local file if a path is given."""
    path = Path(data_path) if data_path is not None else fetch_from_kaggle()
    df = load_csv(path, require_target=True)
    logger.info("Loaded %d rows x %d columns from %s", len(df), df.shape[1], path)
    return df


def split_train_val(df: pd.DataFrame, val_size: float = VAL_SIZE, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/validation split. The same seed always gives the same split."""
    train_df, val_df = train_test_split(df, test_size=val_size, stratify=df[TARGET], random_state=seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
